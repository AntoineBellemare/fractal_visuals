"""
spatial_c2_gallery.py — aesthetic gallery from the FIXED spatial-c2 pipeline.

This is the visual counterpart to spatial_c2.py's measurement run. Same construction
(amplitude-matched + calibrated endpoints, smoothstep_mid ramp) fed to the zero-shot tile
ControlNet — no fine-tuned ControlNet is involved; Phase B remains a failed branch.

What the images should show: multifractality rising across the frame at roughly constant
roughness — one side uniform and evenly textured, the other side the same "busyness"
concentrated into sparse strong bursts on smoother ground. Crucially there is NO brightness
ramp and no seam; the old construction produced both (see figure 58).

Every tile is measured with the ground-truth wavelet-leader estimator and the number is
printed on it, so the sheet cannot drift into unverified claims.

Modes:
  --mode c2   ramp multifractality at fixed roughness      (the new capability)
  --mode fd   ramp fractal dimension                       (the established one, for contrast)
  --mode grid one substrate, c2 ramp at 3 global c1 levels (shows the axes are separable)

Usage:
  python spatial_c2_gallery.py --mode c2 --out spatial_c2/gallery
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
from cascade_calibration import Calibration                              # noqa: E402
from spatial_c2 import c2_gradient_field, measure_halves                 # noqa: E402
from gradient_gallery import fd_gradient_field                           # noqa: E402
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN   # noqa: E402

TAIL = (", rich colour, sharp focus, fine detail, dramatic natural light, photorealistic, "
        "high resolution, 8K")

# Weighted towards substrates that measurably carry a c2 ramp (ferrofluid 93 %, ink_water 70 %,
# dye_plume 53 %, firestorm 43 %), plus visually distinct ones for range.
SUBJECTS = [
    ("ferrofluid",    "ferrofluid spikes under a magnet, black iridescent peaks on a mirror surface", "h"),
    ("ink_water",     "black ink diffusing in clear water, turbulent tendrils and curling filaments", "h"),
    ("dye_plume",     "vivid magenta and cyan dye dispersing in water, mushrooming vortices", "h"),
    ("firestorm",     "swirling firestorm, glowing embers and flame filaments against dark smoke", "h"),
    ("ink_bloom",     "indigo ink blooming in water, feathered translucent edges", "d"),
    ("smoke_backlit", "backlit incense smoke, laminar threads breaking into fine vortices", "h"),
    ("frost_fern",    "window frost ferns, dendritic ice crystals on cold blue glass", "radial"),
    ("dendrite",      "manganese dendrites on pale limestone, black fern-like mineral growth", "h"),
    ("cracked_glaze", "crazed ceramic glaze, fine craquelure network over deep celadon", "d"),
    ("mycelium",      "white fungal mycelium threading through dark forest substrate", "h"),
    ("lightning",     "branching electrical discharge in a dark chamber, violet plasma filaments", "v"),
    ("nebula",        "deep space nebula, filaments of ionised gas and dark dust lanes", "radial"),
]


def build_field(mode, name, direction, seed, args, cal):
    if mode == "fd":
        return fd_gradient_field(args.c1_lo, args.c1_hi, -0.45, direction, seed=seed, n=args.res)
    return c2_gradient_field(args.c1, args.c2_lo, args.c2_hi, direction, seed,
                             n=args.res, cal=cal, amp_match=True, calibrated=True,
                             mask_shape=args.mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["c2", "fd", "grid"], default="c2")
    ap.add_argument("--c1", type=float, default=1.3)
    ap.add_argument("--c2-lo", type=float, default=-0.20)
    ap.add_argument("--c2-hi", type=float, default=-1.00)
    ap.add_argument("--c1-lo", type=float, default=0.2)
    ap.add_argument("--c1-hi", type=float, default=1.5)
    ap.add_argument("--mask", default="smoothstep_mid")
    ap.add_argument("--cn-scale", type=float, default=0.85,
                    help="0.9 measures best but desaturates; 0.85 keeps colour and still ramps")
    ap.add_argument("--guidance-end", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--grid-subject", default="ferrofluid")
    ap.add_argument("--out", default=str(HERE / "spatial_c2" / "gallery"))
    args = ap.parse_args()

    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    cal = Calibration.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)

    subs = SUBJECTS
    if args.subjects:
        keep = set(args.subjects.split(","))
        subs = [s for s in SUBJECTS if s[0] in keep]

    jobs = []
    if args.mode == "grid":
        name, prompt, _ = next(s for s in SUBJECTS if s[0] == args.grid_subject)
        for i, c1 in enumerate((0.9, 1.3, 1.7)):
            jobs.append((f"{name}_c1{c1:.1f}", prompt, "h", i, dict(c1=c1)))
    else:
        for i, (name, prompt, d) in enumerate(subs):
            jobs.append((name, prompt, d, i, {}))

    rows, tiles = [], []
    t0 = time.time()
    for k, (name, prompt, direction, seed, over) in enumerate(jobs):
        a = argparse.Namespace(**{**vars(args), **over})
        fld = build_field(args.mode if args.mode != "grid" else "c2",
                          name, direction, seed, a, cal)
        ctrl = field_to_control(fld)
        (fc1A, fc2A), (fc1B, fc2B) = measure_halves(fld, direction)
        img = generate(pipe, prompt + TAIL, ctrl, args.cn_scale,
                       guidance_end=args.guidance_end, steps=args.steps, cfg=args.cfg,
                       seed=seed, res=args.res, device=device)
        g = np.asarray(img.convert("L"), float) / 255.0
        (c1A, c2A), (c1B, c2B) = measure_halves(g, direction)
        img.save(out / "img" / f"{name}.png", optimize=True)
        ctrl.convert("L").save(out / "img" / f"{name}_cond.png", optimize=True)
        rows.append(dict(subject=name, direction=direction, mode=args.mode,
                         c1_A=round(c1A, 3), c1_B=round(c1B, 3),
                         c2_A=round(c2A, 3), c2_B=round(c2B, 3),
                         dc2=round(c2B - c2A, 3), dc1=round(c1B - c1A, 3),
                         field_dc2=round(fc2B - fc2A, 3)))
        tiles.append((name, direction, img, c1A, c1B, c2A, c2B))
        print(f"  [{k+1}/{len(jobs)}] {name:14s} {direction:6s} "
              f"c2 {c2A:+.2f} -> {c2B:+.2f} (d {c2B-c2A:+.2f})  "
              f"c1 {c1A:.2f} -> {c1B:.2f}   ({(time.time()-t0)/60:.1f}m)", flush=True)

    with (out / "gallery.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    _sheet(out, tiles, args)
    d = np.array([r["dc2"] for r in rows])
    print(f"\n{len(rows)} images | mean dc2 {d.mean():+.3f} "
          f"(requested {args.c2_hi - args.c2_lo:+.2f}) | {int((d < 0).sum())}/{len(d)} correct sign")


def _sheet(out, tiles, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nc = 4 if len(tiles) > 6 else 3
    nr = (len(tiles) + nc - 1) // nc
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 3.5, nr * 3.85))
    ax = np.atleast_1d(ax).ravel()
    for a, (name, direction, img, c1A, c1B, c2A, c2B) in zip(ax, tiles):
        a.imshow(img.resize((430, 430))); a.set_xticks([]); a.set_yticks([])
        ok = "✓" if c2B < c2A else "·"
        a.set_title(f"{name}  ({direction})\nc2 {c2A:+.2f} → {c2B:+.2f}  (Δ {c2B-c2A:+.2f}) {ok}\n"
                    f"c1 {c1A:.2f} → {c1B:.2f}", fontsize=8.5, pad=3)
    for a in ax[len(tiles):]:
        a.axis("off")
    what = ("MULTIFRACTALITY (c2) rising across the frame at ~constant roughness"
            if args.mode != "fd" else "FRACTAL DIMENSION (c1) ramp")
    fig.suptitle(f"Spatial gradient gallery — {what}\n"
                 f"amplitude-matched + calibrated field, smoothstep ramp, zero-shot tile "
                 f"ControlNet; every value MEASURED per region", fontsize=12.5, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out / "gallery_sheet.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"sheet -> {out/'gallery_sheet.png'}")


if __name__ == "__main__":
    main()
