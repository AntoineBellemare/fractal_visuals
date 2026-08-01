"""
surreal_forms.py — CREATIVE APPLICATION 5: apparitions made only of fractal dimension.

The surrealist use of §5.2.3. A form — an eye, a crescent, a keyhole — is present in the image
with **no outline, no colour change and matched luminance**. It exists solely as a region of
different fractal dimension inside a substrate that is otherwise continuous. Nothing is drawn; the
texture simply *organises* differently there, the way a face seems to organise itself out of bark.

That is a pareidolia device rather than an illustration device, which is what makes it worth
building in this project specifically: the viewer resolves the form out of statistics, and the
statistics are prescribed and measured rather than hoped for.

WHAT IS ALREADY KNOWN, and constrains the design:
  * it must run on the c1 axis — the c2 attempt was 4x BELOW its own noise floor;
  * the endpoints must be amplitude-matched and calibrated, or the blend is dominated by the
    low-c1 component and no gradient exists at all;
  * the contrast must clear a THRESHOLD of dc1 ~ 1.2. Below that there is no signal whatsoever —
    measured, not assumed — so a faint, genuinely invisible apparition is not available. The form
    is legible as texture; it is simply not drawn.

Every render is measured (patches inside vs outside the form, against a matched flat-field null)
and the local-c1 map is correlated with the mask, so "the form is really there" is a number.

Usage:
  python surreal_forms.py --mode field      # CPU: check the masks and the field
  python surreal_forms.py --mode gen        # GPU
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
from cascade_calibration import Calibration                              # noqa: E402
from spatial_c2 import zstd                                              # noqa: E402
from hidden_shapes import patch_stats, c1_map, mask_grid                 # noqa: E402
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN  # noqa: E402

TAIL = ", sharp focus, fine detail, natural light, photorealistic, high resolution, 8K"


def form_mask(kind, n=1024):
    """Solid-interior forms. Thin ones (spirals, script) leave no room for a clean patch, so the
    measurement would be undefined — every form here is fat enough to sample."""
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    c = n / 2
    if kind == "eye":                       # vesica: intersection of two big circles
        r = n * 0.62; dy = n * 0.42
        a = Image.new("L", (n, n), 0); ImageDraw.Draw(a).ellipse(
            [c - r, c - dy - r + n * 0.10, c + r, c - dy + r + n * 0.10], fill=255)
        b = Image.new("L", (n, n), 0); ImageDraw.Draw(b).ellipse(
            [c - r, c + dy - r - n * 0.10, c + r, c + dy + r - n * 0.10], fill=255)
        img = Image.fromarray(np.minimum(np.asarray(a), np.asarray(b)))
    elif kind == "crescent":
        r = n * 0.36
        d.ellipse([c - r, c - r, c + r, c + r], fill=255)
        d.ellipse([c - r + n * 0.16, c - r - n * 0.03, c + r + n * 0.16, c + r - n * 0.03], fill=0)
    elif kind == "keyhole":
        # head must be big enough to contain a 192 px measurement patch AFTER edge softening —
        # at r = 0.17 the patch corners fell just outside and every sample was discarded.
        r = n * 0.24
        d.ellipse([c - r, c - r - n * 0.08, c + r, c + r - n * 0.08], fill=255)
        d.polygon([(c - n * 0.13, c + n * 0.02), (c + n * 0.13, c + n * 0.02),
                   (c + n * 0.20, c + n * 0.38), (c - n * 0.20, c + n * 0.38)], fill=255)
    elif kind == "star":
        pts = []
        for i in range(10):
            ang = -np.pi / 2 + i * np.pi / 5
            rr = n * (0.40 if i % 2 == 0 else 0.17)
            pts.append((c + rr * np.cos(ang), c + rr * np.sin(ang)))
        d.polygon(pts, fill=255)
    elif kind == "door":                    # a doorway: the Magritte move
        d.rounded_rectangle([c - n * 0.20, c - n * 0.30, c + n * 0.20, c + n * 0.34],
                            radius=int(n * 0.19), fill=255)
    else:
        raise ValueError(kind)
    return np.asarray(img, float) / 255.0


def soft(m, n, sigma_frac=1 / 52):
    return np.clip(gaussian_filter(m, sigma=n * sigma_frac), 0, 1)


FORMS = ["eye", "crescent", "keyhole", "star", "door"]

GROUNDS = [
    ("bark",    "the bark of an ancient oak, deep fissured ridges, close up"),
    ("lichen",  "crustose lichen and mineral staining spreading across slate"),
    ("water",   "the dark surface of a still lake at dusk, faint ripples"),
    ("cloud",   "a dense bank of storm cloud filling the frame, soft and turbulent"),
    ("sand",    "rippled desert sand seen from directly above at low sun"),
]


def build(c1_bg, c1_sh, kind, seed, n, cal, c2=-0.45, flat=False):
    lo = zstd(cal.cascade(c1_bg, c2, n=n, seed=seed))
    hi = zstd(cal.cascade(c1_bg if flat else c1_sh, c2, n=n, seed=seed))
    m = soft(form_mask(kind, n), n)
    return lo * (1 - m) + hi * m, m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["field", "gen"], default="field")
    # dc1 = 1.20 is the measured threshold; below it there is no signal at all.
    ap.add_argument("--c1-bg", type=float, default=0.30)
    ap.add_argument("--c1-sh", type=float, default=1.50)
    ap.add_argument("--c2", type=float, default=-0.45)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--cn-scale", type=float, default=0.9)
    ap.add_argument("--guidance-end", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=38)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--forms", default=None)
    ap.add_argument("--grounds", default=None)
    ap.add_argument("--out", default=str(HERE / "creative" / "surreal_forms"))
    args = ap.parse_args()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    cal = Calibration.load()

    forms = [f for f in FORMS if not args.forms or f in args.forms.split(",")]
    grounds = [g for g in GROUNDS if not args.grounds or g[0] in args.grounds.split(",")]

    pipe = None
    if args.mode == "gen":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
        pipe = build_pipe(DEFAULT_CN, device)

    pairs = [(f, g, p) for f in forms for g, p in grounds] if args.mode == "gen" \
        else [(f, "field", "") for f in forms]
    rows = []; t0 = time.time(); k = 0
    n_tot = len(pairs) * args.seeds * 2
    for (kind, gname, prompt) in pairs:
        for s in range(args.seeds):
            for flat in (False, True):
                k += 1
                fld, m = build(args.c1_bg, args.c1_sh, kind, s, args.res, cal, args.c2, flat)
                if args.mode == "field":
                    target = fld
                else:
                    img = generate(pipe, prompt + TAIL, field_to_control(fld), args.cn_scale,
                                   guidance_end=args.guidance_end, steps=args.steps,
                                   cfg=args.cfg, seed=s, res=args.res, device=device)
                    img.save(out / "img" / f"{gname}_{kind}_"
                                           f"{'flat' if flat else 'form'}_s{s}.png", optimize=True)
                    target = np.asarray(img.convert("L"), float) / 255.0
                st = patch_stats(target, m)
                if st is None:
                    print(f"    (skip {kind}: no clean patches)"); continue
                M, ys, xs, win = c1_map(target)
                gm = mask_grid(m, ys, xs, win)
                rec = float(np.corrcoef(M.ravel(), gm.ravel())[0, 1]) if gm.std() > 1e-9 else np.nan
                rows.append(dict(form=kind, ground=gname, seed=s,
                                 variant="flat" if flat else "form",
                                 **{kk: (round(v, 4) if isinstance(v, float) else v)
                                    for kk, v in st.items()}, recover_r=round(rec, 4)))
                print(f"  [{k}/{n_tot}] {gname:8s} {kind:9s} "
                      f"{'flat' if flat else 'FORM':4s}  dc1 {st['dc1']:+.3f}  "
                      f"r {rec:+.3f}  dlum {st['dlum']:+.4f}  ({(time.time()-t0)/60:.1f}m)",
                      flush=True)

    fp = out / "forms.csv"
    with fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import pandas as pd
    from scipy import stats as st_
    d = pd.DataFrame(rows)
    fo = d[d.variant == "form"]; fl = d[d.variant == "flat"]
    for nm in ("dc1", "recover_r", "dlum"):
        t, p = st_.ttest_ind(fo[nm].dropna(), fl[nm].dropna())
        print(f"  {nm:10s} form {fo[nm].mean():+.4f}   flat {fl[nm].mean():+.4f}   "
              f"t={t:+.2f} p={p:.5f}")
    print(f"\n  d' = {(fo.dc1.mean()-fl.dc1.mean())/fl.dc1.std(ddof=1):.2f}   "
          f"recoverable in {int((fo.recover_r > 0.3).sum())}/{len(fo)}")
    print(f"-> {fp}")


if __name__ == "__main__":
    main()
