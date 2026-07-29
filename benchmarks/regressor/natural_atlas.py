"""
natural_atlas.py — a wide atlas of natural images across SCALES with CONTROLLED fractal
dimension (FD) and multifractality.

Uses the method that actually works: joint (c1,c2) classifier guidance with the latent
joint regressor in SDXL's denoising loop (joint_montage.joint_guided). For every
(family x target) it records the MEASURED cumulants of the output, so every image ships
with ground truth.

  c1  = mean roughness / Hölder exponent   ->  FD = 3 - c1   (surface fractal dimension)
  c2  = multifractality / intermittency    ->  0 = monofractal, more negative = more multifractal

Targets are the corners + centre of the perceptual plane, so each family is rendered
smooth<->busy (FD) x uniform<->clustered (multifractality).

Outputs (benchmarks/regressor/natural_atlas/):
  img/<scale>/<family>_c1{t}_c2{t}.png     all images
  atlas.csv                                family, scale, targets, measured c1/c2, FD
  sheet_<scale>.png                        contact sheet per scale band

Usage:
  python natural_atlas.py                       # all families, 5 targets
  python natural_atlas.py --scales plant,rock --targets corners
  python natural_atlas.py --families diatoms,moss --seeds 2
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import guided_sample as gs                        # noqa: E402
import joint_montage as jm                        # noqa: E402
from build_corpus import measure_c2               # noqa: E402
import prompts_natural_scales as PN               # noqa: E402

# (c1, c2) targets: 4 corners of the perceptual plane + centre
TARGETS = {
    "corners": [(0.9, -0.70), (0.9, -0.25), (1.7, -0.70), (1.7, -0.25)],
    "full":    [(0.9, -0.70), (0.9, -0.25), (1.7, -0.70), (1.7, -0.25), (1.3, -0.45)],
    "fd":      [(0.8, -0.45), (1.1, -0.45), (1.4, -0.45), (1.7, -0.45)],   # FD ladder only
    "mf":      [(1.3, -0.15), (1.3, -0.40), (1.3, -0.65), (1.3, -0.90)],   # multifractality ladder
}


def fd_of(c1):
    """Surface fractal dimension from the first cumulant (D = 3 - H, H ~ c1)."""
    return 3.0 - float(c1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=None, help="comma list; default = all")
    ap.add_argument("--scales", default=None, help="comma list of scale bands to include")
    ap.add_argument("--targets", default="full", choices=list(TARGETS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--scale-guid", type=float, default=150.0, help="guidance strength")
    ap.add_argument("--warmup", type=int, default=14)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--low-vram", action="store_true",
                    help="attention/VAE slicing — avoids the host-RAM spill when other GPU "
                         "apps leave <2GB headroom (big net speedup on a busy desktop)")
    ap.add_argument("--out", default=str(HERE / "natural_atlas"))
    args = ap.parse_args()

    fams = args.families.split(",") if args.families else list(PN.PROMPTS)
    if args.scales:
        keep = set(args.scales.split(","))
        fams = [f for f in fams if PN.SCALE_OF.get(f) in keep]
    tgts = TARGETS[args.targets]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = out / "atlas.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(fams)} families x {len(tgts)} targets x {args.seeds} seeds = "
          f"{len(fams)*len(tgts)*args.seeds} images", flush=True)
    pipe = gs.build_pipe(device)
    model = jm.load_joint(device)
    if args.low_vram:
        # Windows/WDDM spills to host RAM when VRAM is over-committed (~6x slowdown).
        # Slicing keeps the peak well under the cap so we stay resident.
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        torch.cuda.empty_cache()
        print(f"low-vram mode on; VRAM now {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    # resume: skip rows already done
    done = set()
    rows = []
    if manifest.exists():
        for r in csv.DictReader(manifest.open()):
            rows.append(r); done.add(r["path"])

    n_total = len(fams) * len(tgts) * args.seeds
    k = 0; t0 = time.time()
    for fam in fams:
        scale = PN.SCALE_OF.get(fam, "other")
        prompt = PN.PROMPTS[fam][0]
        (out / "img" / scale).mkdir(parents=True, exist_ok=True)
        for (c1t, c2t) in tgts:
            for s in range(args.seeds):
                k += 1
                fp = out / "img" / scale / f"{fam}_c1{c1t:.2f}_c2{c2t:+.2f}_s{s}.png"
                rel = fp.relative_to(out).as_posix()
                if rel in done or fp.exists():
                    continue
                img = jm.joint_guided(pipe, model, prompt, c1t, c2t,
                                      steps=args.steps, scale=args.scale_guid,
                                      w1=1.0, w2=1.0, warmup=args.warmup,
                                      res=args.res, seed=s, device=device)
                c1m, c2m = measure_c2(img)
                img.save(fp, optimize=True)
                rows.append(dict(family=fam, scale=scale, c1_target=c1t, c2_target=c2t,
                                 seed=s, c1_meas=round(c1m, 4), c2_meas=round(c2m, 4),
                                 fd_target=round(fd_of(c1t), 3), fd_meas=round(fd_of(c1m), 3),
                                 path=rel))
                if k % 5 == 0 or k == 1:
                    el = (time.time() - t0) / 60
                    eta = el / max(k, 1) * (n_total - k)
                    print(f"  [{k}/{n_total}] {scale:10s} {fam:18s} "
                          f"t=(c1 {c1t:.1f}, c2 {c2t:+.2f}) -> (c1 {c1m:.2f}, c2 {c2m:+.2f}) "
                          f"FD {fd_of(c1m):.2f}  el={el:.0f}m eta={eta:.0f}m", flush=True)
                    _flush(manifest, rows)
    _flush(manifest, rows)
    _sheets(out, rows, tgts)
    print(f"\nDONE {len(rows)} images -> {out}", flush=True)


def _flush(manifest, rows):
    if not rows:
        return
    keys = ["family", "scale", "c1_target", "c2_target", "seed", "c1_meas", "c2_meas",
            "fd_target", "fd_meas", "path"]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _sheets(out, rows, tgts):
    """Contact sheet per scale band: rows = family, cols = target."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    by_scale = defaultdict(list)
    for r in rows:
        by_scale[r["scale"]].append(r)
    for scale, rs in by_scale.items():
        fams = list(dict.fromkeys(r["family"] for r in rs))
        nr, nc = len(fams), len(tgts)
        if nr == 0:
            continue
        fig, ax = plt.subplots(nr, nc, figsize=(nc * 2.3, nr * 2.5))
        ax = np.atleast_2d(ax)
        for i, fam in enumerate(fams):
            for j, (c1t, c2t) in enumerate(tgts):
                cell = [r for r in rs if r["family"] == fam
                        and abs(float(r["c1_target"]) - c1t) < 1e-6
                        and abs(float(r["c2_target"]) - c2t) < 1e-6]
                a = ax[i][j]; a.set_xticks([]); a.set_yticks([])
                if cell:
                    r0 = cell[0]
                    a.imshow(Image.open(out / r0["path"]).resize((300, 300)))
                    a.set_title(f"FD={float(r0['fd_meas']):.2f}  c2={float(r0['c2_meas']):+.2f}",
                                fontsize=7, pad=2)
                if j == 0:
                    a.set_ylabel(fam, fontsize=8, rotation=0, ha="right", va="center")
                    a.yaxis.set_label_coords(-0.03, 0.5)
        for j, (c1t, c2t) in enumerate(tgts):
            ax[0][j].text(0.5, 1.30, f"target FD={3-c1t:.1f}\nc2={c2t:+.2f}",
                          transform=ax[0][j].transAxes, ha="center", fontsize=8, fontweight="bold")
        fig.suptitle(f"Natural atlas — {scale}: controlled fractal dimension x multifractality",
                     fontsize=13, y=1.0)
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])
        fig.savefig(out / f"sheet_{scale}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  sheet -> sheet_{scale}.png", flush=True)


if __name__ == "__main__":
    main()
