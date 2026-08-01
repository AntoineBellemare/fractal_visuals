"""
hidden_shapes.py — CREATIVE APPLICATION 3: a shape that is invisible to the eye but recoverable
from the image statistics.

WHY REVISIT A DOCUMENTED NEGATIVE. This repo already recorded hidden shapes as "not achieved":
shape-vs-background |dc2| = 0.039 against a ~0.15 local-c2 noise floor, ~4x below detectability.
Two things have changed since, and both point the same way:

  1. That attempt varied **c2**, which is the axis that barely transmits spatially. c1 (fractal
     dimension) transmits through the conditioning path far better — mean |dc1| 0.32 over 18
     substrates — so a hidden FD shape is a much better bet than a hidden multifractality shape.
  2. The blend was `fbg*(1-m) + fsh*m` over two cascades of different amplitude — the same
     construction bug that made spatial c2 look impossible until it was amplitude-matched
     (see spatial_c2.py). `shape_field` does hist_match the two components, which helps, but it
     does not equalise the c1 that the endpoints actually deliver.

So: hidden shape on the **c1 axis**, with amplitude-matched and CALIBRATED endpoints.

THE DESIGN. The whole claim is "invisible but recoverable", so it needs two measurements, not one:

  DETECTABILITY  square patches lying entirely inside the shape vs entirely outside it, measured
                 with the ground-truth estimator. The null is the SAME measurement on a matched
                 flat-field render (no shape at all), which gives the noise floor honestly instead
                 of assuming one. d' = dc1 / sd(null).
  RECOVERABILITY can the shape be LOCALISED, not just detected? A coarse sliding-window c1 map is
                 correlated against the shape mask. Chance level comes from the flat control.

And the aesthetic requirement is that the shape must NOT be visible. The mask edge is therefore
softened (a hard edge would show as a seam, which is what made earlier spatial work unusable), and
brightness is equalised across the boundary by construction because both endpoints are z-scored.

Usage:
  python hidden_shapes.py --mode field          # CPU: does the FIELD carry it? no GPU
  python hidden_shapes.py --mode gen            # GPU: does the rendered IMAGE carry it?
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import mfractal as mf                                                    # noqa: E402
from cascade_calibration import Calibration                              # noqa: E402
from spatial_c2 import zstd                                              # noqa: E402
import mf_creative as mc                                                 # noqa: E402
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN  # noqa: E402

TAIL = ", sharp focus, fine detail, natural light, photorealistic, high resolution, 8K"

SHAPES = [
    ("circle",   dict(kind="circle")),
    ("triangle", dict(kind="triangle")),
    ("heart",    dict(kind="heart")),
    ("cross",    dict(kind="cross")),
]

# Substrates that carry an FD ramp well (from FEASIBILITY.md / fd_gradient_gallery).
SUBJECTS = [
    ("moss",    "a bed of dense moss and small ferns on forest floor, top-down"),
    ("lichen",  "crustose lichen and mineral staining spreading across slate"),
    ("sand",    "rippled desert sand seen from directly above"),
    ("canopy",  "dense green forest canopy seen from above, packed crowns"),
]


def soft_mask(spec, n, sigma_frac=1 / 48):
    """Shape mask with a softened edge — a hard edge renders as a visible seam."""
    m = np.asarray(mc.shape_mask(spec), float)
    if m.max() > 1.5:
        m = m / 255.0
    if m.shape[0] != n:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((n, n)), float) / 255.0
    return np.clip(gaussian_filter(m, sigma=n * sigma_frac), 0, 1)


def shape_c1_field(c1_bg, c1_sh, spec, seed, n=1024, cal=None, c2=-0.45, flat=False):
    """Field whose FRACTAL DIMENSION differs inside the shape, at matched brightness."""
    cal = cal or Calibration.load()
    lo = zstd(cal.cascade(c1_bg, c2, n=n, seed=seed))
    hi = zstd(cal.cascade(c1_bg if flat else c1_sh, c2, n=n, seed=seed))
    m = soft_mask(spec, n)
    return lo * (1 - m) + hi * m, m


def patch_stats(arr, mask, patch=192, stride=64, thresh=0.92):
    """Mean (c1,c2) over square patches lying ENTIRELY inside / entirely outside the shape.

    Irregular regions can't be handed to the estimator directly, so we sample squares that are
    unambiguously on one side. Patches straddling the boundary are discarded.
    """
    a = np.asarray(arr, float)
    a = (a - a.min()) / (np.ptp(a) + 1e-12)
    h, w = a.shape
    ins, out = [], []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            mm = mask[y:y + patch, x:x + patch]
            if mm.min() > thresh:
                tag = ins
            elif mm.max() < 1 - thresh:
                tag = out
            else:
                continue
            p = a[y:y + patch, x:x + patch]
            r = mf.wavelet_leaders_2d(p)
            # visibility proxies: mean luminance, RMS contrast, and mean gradient magnitude.
            # A shape is only "hidden" if these MATCH across the boundary while c1 differs —
            # the eye segments texture on local energy, not on the cumulant.
            gy, gx = np.gradient(p)
            tag.append((float(r["c1"]), float(r["c2"]), p.mean(), p.std(),
                        float(np.hypot(gx, gy).mean())))
    if not ins or not out:
        return None
    ins = np.array(ins); out = np.array(out)
    return dict(n_in=len(ins), n_out=len(out),
                c1_in=ins[:, 0].mean(), c1_out=out[:, 0].mean(),
                c2_in=ins[:, 1].mean(), c2_out=out[:, 1].mean(),
                dc1=ins[:, 0].mean() - out[:, 0].mean(),
                dc2=ins[:, 1].mean() - out[:, 1].mean(),
                dlum=ins[:, 2].mean() - out[:, 2].mean(),
                dcontrast=ins[:, 3].mean() - out[:, 3].mean(),
                dgrad=ins[:, 4].mean() - out[:, 4].mean(),
                sd_in=ins[:, 0].std(ddof=1) if len(ins) > 1 else np.nan,
                sd_out=out[:, 0].std(ddof=1) if len(out) > 1 else np.nan)


def c1_map(arr, win=192, stride=96):
    """Coarse local-c1 map for the recoverability test."""
    a = np.asarray(arr, float); a = (a - a.min()) / (np.ptp(a) + 1e-12)
    h, w = a.shape
    ys = list(range(0, h - win + 1, stride)); xs = list(range(0, w - win + 1, stride))
    M = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            M[i, j] = mf.wavelet_leaders_2d(a[y:y + win, x:x + win])["c1"]
    return M, ys, xs, win


def mask_grid(mask, ys, xs, win):
    return np.array([[mask[y:y + win, x:x + win].mean() for x in xs] for y in ys])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["field", "gen"], default="field")
    ap.add_argument("--c1-bg", type=float, default=0.30, help="background: low c1 = HIGH FD = busy")
    ap.add_argument("--c1-sh", type=float, default=1.50, help="shape: high c1 = LOW FD = smooth")
    ap.add_argument("--c2", type=float, default=-0.45)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--cn-scale", type=float, default=0.9)
    ap.add_argument("--guidance-end", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shapes", default=None, help="comma list; default = all")
    ap.add_argument("--subjects", default=None, help="comma list; default = all")
    ap.add_argument("--level", type=float, default=None,
                    help="label recorded in the CSV; used by the dc1 sweep")
    ap.add_argument("--out", default=str(HERE / "creative" / "hidden"))
    args = ap.parse_args()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    cal = Calibration.load()

    pipe = None
    if args.mode == "gen":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
        pipe = build_pipe(DEFAULT_CN, device)

    rows = []
    t0 = time.time(); k = 0
    shp = [x for x in SHAPES if not args.shapes or x[0] in args.shapes.split(',')]
    sbj = [x for x in SUBJECTS if not args.subjects or x[0] in args.subjects.split(',')]
    combos = ([(sn, sp, subj, pr) for sn, sp in shp for subj, pr in sbj]
              if args.mode == "gen" else [(sn, sp, "field", "") for sn, sp in SHAPES])
    n_tot = len(combos) * args.seeds * 2
    for (sname, spec, subj, prompt) in combos:
        for s in range(args.seeds):
            for flat in (False, True):                 # flat = matched null, no shape
                k += 1
                fld, m = shape_c1_field(args.c1_bg, args.c1_sh, spec, seed=s, n=args.res,
                                        cal=cal, c2=args.c2, flat=flat)
                if args.mode == "field":
                    target = fld
                else:
                    img = generate(pipe, prompt + TAIL, field_to_control(fld), args.cn_scale,
                                   guidance_end=args.guidance_end, steps=args.steps,
                                   cfg=args.cfg, seed=s, res=args.res, device=device)
                    tag = "flat" if flat else "shape"
                    img.save(out / "img" / f"{subj}_{sname}_{tag}_s{s}.png", optimize=True)
                    target = np.asarray(img.convert("L"), float) / 255.0
                st = patch_stats(target, m)
                if st is None:
                    print(f"    (skip {sname}: no clean patches)"); continue
                M, ys, xs, win = c1_map(target)
                gm = mask_grid(m, ys, xs, win)
                rec = float(np.corrcoef(M.ravel(), gm.ravel())[0, 1]) if gm.std() > 1e-9 else np.nan
                rows.append(dict(shape=sname, subject=subj, seed=s,
                                 level=(args.level if args.level is not None
                                        else round(args.c1_sh - args.c1_bg, 3)),
                                 variant="flat" if flat else "shape",
                                 **{kk: (round(v, 4) if isinstance(v, float) else v)
                                    for kk, v in st.items()},
                                 recover_r=round(rec, 4)))
                print(f"  [{k}/{n_tot}] {subj:8s} {sname:9s} s{s} "
                      f"{'flat' if flat else 'SHAPE':5s}  dc1 {st['dc1']:+.3f}  "
                      f"recover r {rec:+.3f}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    fp = out / "hidden.csv"
    with fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import pandas as pd
    from scipy import stats as st_
    d = pd.DataFrame(rows)
    sh = d[d.variant == "shape"]; fl = d[d.variant == "flat"]
    print(f"\n{'':22s}{'shape':>18s}{'flat (null)':>18s}")
    for nm in ("dc1", "dc2", "recover_r"):
        a, b = sh[nm].dropna(), fl[nm].dropna()
        t, p = st_.ttest_ind(a, b)
        print(f"  {nm:20s}{a.mean():+10.3f} ± {a.std(ddof=1):.3f}"
              f"{b.mean():+10.3f} ± {b.std(ddof=1):.3f}   t={t:+.2f} p={p:.5f}")
    dp = (sh.dc1.mean() - fl.dc1.mean()) / fl.dc1.std(ddof=1)
    print(f"\n  DETECTABILITY d' = {dp:.2f}  (shape dc1 in units of the flat-field noise floor)")
    print(f"  RECOVERABILITY  shape r = {sh.recover_r.mean():+.3f} vs null {fl.recover_r.mean():+.3f}")
    print(f"-> {fp}")


if __name__ == "__main__":
    main()
