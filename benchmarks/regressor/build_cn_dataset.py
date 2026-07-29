"""
build_cn_dataset.py — paired (conditioning, target, caption) dataset for the MF-ControlNet.

Design (see PHASE_B.md). Conditioning carries EXACT labels only:
  * UNIFORM samples — the whole map is the image's own measured (c1,c2). Exact by definition,
    so `cumulants(conditioning) == measured cumulants(target)` is an identity: the img2img
    fade trap cannot be inherited.
  * MOSAIC samples  — 2x2 tiles from images with DIFFERENT statistics; each map region carries
    that tile's own measured label. This is what forces the ControlNet to read the map
    LOCALLY instead of collapsing it to a global scalar.

Why not per-cell local estimation: measured, it is noise-dominated (a statistically
homogeneous field produces map sd >= that of real photographs at every window/grid tried).
See local_cumulants.py.

Crop-scale augmentation is mandatory and every crop is RE-LABELLED after cropping, so the
caption cannot predict the statistics (R^2(prompt->c1)=0.856 in the raw corpus).

Captions deliberately contain NO numbers (numbers would route the signal through the
pretrained text encoder and starve the ControlNet branch).

Output (diffusers ImageFolder layout):
  cn_dataset/images/<id>.png            target
  cn_dataset/conditioning/<id>.png      conditioning
  cn_dataset/metadata.jsonl             {"image","conditioning_image","text", + labels}
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from build_corpus import measure_c2, to_canonical_gray          # noqa: E402
import local_cumulants as LC                                     # noqa: E402

CROP_FRACS = [1.0, 0.75, 0.5, 0.35]        # 0.25 excluded: label scatter too large
SRC_WEIGHT = {"macro": 3.5, "scaffold": 2.0, "diffusion": 1.0}


def caption_for(group, family_hint=""):
    """Scene/substrate noun phrase, no numbers, no statistics."""
    fam = str(group).split(":")[1] if ":" in str(group) else str(group)
    fam = fam.replace("_", " ")
    return f"{fam}, natural texture, photorealistic, fine detail"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(HERE / "corpus" / "manifest.csv"))
    ap.add_argument("--out", default=str(HERE / "cn_dataset"))
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--mosaic-frac", type=float, default=0.30, help="fraction of mosaic samples")
    ap.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "conditioning").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = pd.read_csv(args.manifest)
    d = d[d.source.isin(SRC_WEIGHT)]                       # 8-bit photo sources only
    d = d[(d.c1 > 0.6) & (d.c1 < 2.4) & (d.c2 > -1.1) & (d.c2 < 0.1)].reset_index(drop=True)
    print(f"pool: {len(d)} images, {d.group.nunique()} groups, sources {dict(d.source.value_counts())}")

    # sampling weights over source
    w = d.source.map(SRC_WEIGHT).to_numpy(float); w /= w.sum()

    n_plan = args.limit or len(d) * len(CROP_FRACS)
    n_mos = int(n_plan * args.mosaic_frac)
    n_uni = n_plan - n_mos
    print(f"planning {n_uni} uniform + {n_mos} mosaic = {n_plan} samples")

    meta, k, t0, fails = [], 0, time.time(), 0

    def load_rgb(path):
        p = Path(path)
        return Image.open(p if p.is_absolute() else ROOT / p).convert("RGB")

    def crop_resize(img, frac, rng):
        W, H = img.size
        s = int(min(W, H) * frac)
        x = int(rng.integers(0, W - s + 1)); y = int(rng.integers(0, H - s + 1))
        return img.crop((x, y, x + s, y + s)).resize((args.res, args.res), Image.BICUBIC)

    # ---------- uniform samples ----------
    for _ in range(n_uni):
        i = int(rng.choice(len(d), p=w))
        r = d.iloc[i]
        frac = CROP_FRACS[int(rng.integers(len(CROP_FRACS)))]
        try:
            img = crop_resize(load_rgb(r["path"]), frac, rng)
            c1, c2 = measure_c2(img)                        # RE-LABEL after the crop
        except Exception:
            fails += 1; continue
        if not (0.5 < c1 < 2.6 and -1.4 < c2 < 0.2):
            continue
        c1m, c2m = LC.uniform_map(c1, c2)
        cond = LC.encode(c1m, c2m, res=args.res)
        idx = f"u{k:06d}"; k += 1
        img.save(out / "images" / f"{idx}.png"); cond.save(out / "conditioning" / f"{idx}.png")
        meta.append(dict(image=f"images/{idx}.png", conditioning_image=f"conditioning/{idx}.png",
                         text=caption_for(r["group"]), c1=round(c1, 4), c2=round(c2, 4),
                         kind="uniform", group=str(r["group"])))
        if k % 200 == 0:
            print(f"  [{k}/{n_plan}] {(time.time()-t0)/60:.1f}m", flush=True)
            (out / "metadata.jsonl").write_text("\n".join(json.dumps(m) for m in meta))

    # ---------- mosaic samples ----------
    for _ in range(n_mos):
        idxs = rng.choice(len(d), size=4, p=w, replace=False)
        parts = []
        try:
            for i in idxs:
                r = d.iloc[int(i)]
                frac = CROP_FRACS[int(rng.integers(len(CROP_FRACS)))]
                sub = crop_resize(load_rgb(r["path"]), frac, rng)
                g = to_canonical_gray(sub)                  # 512 float field
                c1, c2 = measure_c2(sub)                    # exact label for this tile
                if not (0.5 < c1 < 2.6 and -1.4 < c2 < 0.2):
                    raise ValueError("out of range")
                parts.append((g, c1, c2, sub, str(r["group"])))
        except Exception:
            fails += 1; continue
        # build the mosaic image at full res + the exact per-region map
        res, half = args.res, args.res // 2
        canvas = Image.new("RGB", (res, res))
        c1m = np.zeros((LC.GRID, LC.GRID)); c2m = np.zeros((LC.GRID, LC.GRID))
        gh = LC.GRID // 2
        for t, (g, c1, c2, sub, _grp) in enumerate(parts):
            ti, tj = divmod(t, 2)
            canvas.paste(sub.resize((half, half), Image.BICUBIC), (tj * half, ti * half))
            c1m[ti * gh:(ti + 1) * gh, tj * gh:(tj + 1) * gh] = c1
            c2m[ti * gh:(ti + 1) * gh, tj * gh:(tj + 1) * gh] = c2
        cond = LC.encode(c1m, c2m, res=res)
        idx = f"m{k:06d}"; k += 1
        canvas.save(out / "images" / f"{idx}.png"); cond.save(out / "conditioning" / f"{idx}.png")
        meta.append(dict(image=f"images/{idx}.png", conditioning_image=f"conditioning/{idx}.png",
                         text=caption_for(parts[0][4]), c1=round(float(c1m.mean()), 4),
                         c2=round(float(c2m.mean()), 4), kind="mosaic", group="mosaic"))
        if k % 200 == 0:
            print(f"  [{k}/{n_plan}] {(time.time()-t0)/60:.1f}m", flush=True)
            (out / "metadata.jsonl").write_text("\n".join(json.dumps(m) for m in meta))

    (out / "metadata.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    lab = np.array([[m["c1"], m["c2"]] for m in meta])
    print(f"\nDONE {len(meta)} samples ({sum(m['kind']=='mosaic' for m in meta)} mosaic), "
          f"{fails} skipped, {(time.time()-t0)/60:.1f} min")
    print(f"  c1 {lab[:,0].min():.2f}–{lab[:,0].max():.2f}  c2 {lab[:,1].min():+.2f}–{lab[:,1].max():+.2f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
