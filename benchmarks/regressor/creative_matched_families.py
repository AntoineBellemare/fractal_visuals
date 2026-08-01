"""
creative_matched_families.py — CREATIVE APPLICATION 2: statistically matched families.

The idea. Build a set of images that share ONE measured texture signature (c1, c2) while their
CONTENT and their MEDIUM vary as widely as possible — an oil painting of a forest, a copperplate
engraving of waves, an electron micrograph of pollen, a linocut of feathers. They look nothing
alike, yet they carry the same "visual busyness". That is the thing a prompt cannot do: you can
ask for "detailed" or "minimal" and get an unpredictable amount of it, and the amount you get is
entangled with what you asked to depict.

For art and design this is an asset-pack / brand-system tool. For research it is the more
important of the two: it is the **statistics-matched, content-varied** arm of the pareidolia
design — it dissociates low-level image statistics from semantics, which is exactly the confound
that is otherwise very hard to control.

Method: joint (c1,c2) classifier guidance in the SDXL denoising loop (the validated global
control — NOT the ControlNet route, which is for spatial targets). Every image is measured with
the ground-truth wavelet-leader estimator.

THE CONTROL THAT MAKES IT A CLAIM: each family is also rendered UNGUIDED at the same seed and
prompt (`scale=0`, an exact matched control). If guidance works, the guided set's spread in
(c1,c2) collapses relative to the unguided set. Spread reduction is the measurable, not the
individual values.

Usage:
  python creative_matched_families.py --c1 1.3 --c2 -0.45 --out creative/matched
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import guided_sample as gs                                   # noqa: E402
import joint_montage as jm                                   # noqa: E402
from build_corpus import measure_c2                          # ground-truth estimator  # noqa: E402

# Content AND medium varied deliberately: photography, painting, printmaking, drawing,
# scientific imaging, 3D. If they all land on one (c1,c2) the point is made.
FAMILIES = [
    ("forest_oil",     "oil painting of a dense forest canopy from above, impasto brushwork"),
    ("waves_engrave",  "copperplate engraving of breaking ocean waves, fine cross-hatching"),
    ("sandstone_3d",   "octane 3D render of eroded sandstone, studio lighting"),
    ("clouds_water",   "loose watercolour of storm clouds, granulation and blooms on cold-press paper"),
    ("lichen_photo",   "macro photograph of lichen spreading on granite, natural light"),
    ("smoke_charcoal", "charcoal drawing of billowing smoke, smudged graphite on toned paper"),
    ("feathers_lino",  "linocut print of overlapping feathers, bold carved marks, two-colour"),
    ("patina_photo",   "macro photograph of oxidised copper patina, blue-green corrosion"),
    ("rain_woodblock", "japanese woodblock print of rain falling on a river, ukiyo-e"),
    ("pollen_sem",     "scanning electron micrograph of pollen grains, false colour"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c1", type=float, default=1.3)
    ap.add_argument("--c2", type=float, default=-0.45)
    ap.add_argument("--scale", type=float, default=150.0)
    ap.add_argument("--warmup", type=int, default=14)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--no-control", action="store_true", help="skip the unguided arm")
    ap.add_argument("--out", default=str(HERE / "creative" / "matched"))
    args = ap.parse_args()

    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = gs.build_pipe(device)
    model = jm.load_joint(device)

    arms = [("guided", args.scale)] + ([] if args.no_control else [("unguided", 0.0)])
    rows = []
    t0 = time.time(); k = 0
    n_tot = len(FAMILIES) * len(arms) * args.seeds
    for fam, prompt in FAMILIES:
        for s in range(args.seeds):
            for arm, sc in arms:
                k += 1
                img = jm.joint_guided(pipe, model, prompt, args.c1, args.c2,
                                      steps=args.steps, scale=sc, w1=1.0, w2=1.0,
                                      warmup=args.warmup, res=args.res, seed=s, device=device)
                c1m, c2m = measure_c2(img)
                img.save(out / "img" / f"{fam}_{arm}_s{s}.png", optimize=True)
                rows.append(dict(family=fam, arm=arm, seed=s,
                                 c1_target=args.c1, c2_target=args.c2,
                                 c1_meas=round(c1m, 4), c2_meas=round(c2m, 4),
                                 fd_meas=round(3 - c1m, 3),
                                 err_c1=round(c1m - args.c1, 4), err_c2=round(c2m - args.c2, 4)))
                print(f"  [{k}/{n_tot}] {fam:16s} {arm:8s} -> c1 {c1m:.2f} c2 {c2m:+.2f}  "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)

    with (out / "matched.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print()
    for arm, _ in arms:
        a = [r for r in rows if r["arm"] == arm]
        c1 = np.array([r["c1_meas"] for r in a]); c2 = np.array([r["c2_meas"] for r in a])
        print(f"  {arm:8s} n={len(a):3d}  c1 {c1.mean():.3f} +- {c1.std(ddof=1):.3f}   "
              f"c2 {c2.mean():+.3f} +- {c2.std(ddof=1):.3f}   "
              f"RMSE to target: c1 {np.sqrt(((c1-args.c1)**2).mean()):.3f} "
              f"c2 {np.sqrt(((c2-args.c2)**2).mean()):.3f}")
    if len(arms) == 2:
        g = [r for r in rows if r["arm"] == "guided"]
        u = [r for r in rows if r["arm"] == "unguided"]
        for nm, key in (("c1", "c1_meas"), ("c2", "c2_meas")):
            sg = np.std([r[key] for r in g], ddof=1); su = np.std([r[key] for r in u], ddof=1)
            print(f"  SPREAD in {nm}: unguided {su:.3f} -> guided {sg:.3f}  "
                  f"({(1-sg/su)*100:.0f}% reduction)")
    print(f"-> {out/'matched.csv'}")


if __name__ == "__main__":
    main()
