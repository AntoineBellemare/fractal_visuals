"""
creative_detail_falloff.py — CREATIVE APPLICATION 1: complexity as a compositional device.

The idea. Painters and photographers concentrate detail where they want the eye to land and
simplify everywhere else — "detail falloff". Depth of field does this with *blur*, which is a
crude instrument: it destroys information and reads as an optical effect. A radial FD ramp does
it with *fractal dimension* instead, so the periphery stays sharp and in focus while becoming
statistically simpler. That is a knob no prompt exposes.

Concretely: a radial `prescribed_cascade` ramp with high FD (low c1) at the centre and low FD
(high c1) at the edge, through the zero-shot tile ControlNet.

Why this is the right capability to build on: the FD ramp is the ROBUST one — mean |dc1| 0.32
over 18 substrates (figure 55) — where spatial c2 is only partial. And unlike everything tested
so far, this asks it to work on FIGURATIVE content (an owl, a face, a building) rather than
texture, and across non-photoreal STYLES where the tile ControlNet's photo prior fights the look.
Both are genuine open questions, so every image is measured centre-vs-corner and reported.

Expected sign: centre detailed (low c1) -> corner simplified (high c1), so dc1 = c1_corner -
c1_centre should be POSITIVE.

Usage:
  python creative_detail_falloff.py --out creative/falloff
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
from spatial_c2 import measure_halves                                    # noqa: E402
from gradient_gallery import fd_gradient_field                           # noqa: E402
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN  # noqa: E402

# Figurative subjects — deliberately NOT textures.
SUBJECTS = [
    ("owl",       "a great horned owl perched on a bare branch, facing the viewer"),
    ("fisherman", "the weathered face of an old fisherman, deep wrinkles, salt-white beard"),
    ("cathedral", "a gothic cathedral facade, rose window, carved spires and gargoyles"),
    ("dragon",    "a coiled dragon with layered scales and folded leathery wings"),
]

# Generation styles — the tile ControlNet was trained on photographs, so the painterly and
# line-art styles are the adversarial cases for this method.
STYLES = [
    ("photo",     "photorealistic, 85mm lens, soft natural light, fine detail"),
    ("oil",       "oil painting, thick impasto brushwork, visible canvas weave, old master palette"),
    ("ink",       "sumi-e ink wash painting on rice paper, sparse confident strokes, muted"),
    ("engraving", "copperplate engraving, fine cross-hatching, monochrome line art, antique print"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c1-lo", type=float, default=0.2, help="centre (low c1 = HIGH FD = detailed)")
    ap.add_argument("--c1-hi", type=float, default=1.5, help="edge (high c1 = LOW FD = simplified)")
    ap.add_argument("--cn-scale", type=float, default=0.85,
                    help="0.9 ramps hardest but flattens painterly styles toward photo")
    ap.add_argument("--guidance-end", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--control", action="store_true",
                    help="also render each cell with NO ramp (flat field) as a matched control")
    ap.add_argument("--out", default=str(HERE / "creative" / "falloff"))
    args = ap.parse_args()

    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)

    rows = []
    t0 = time.time()
    n_tot = len(SUBJECTS) * len(STYLES) * (2 if args.control else 1)
    k = 0
    for si, (sub, sprompt) in enumerate(SUBJECTS):
        for yi, (sty, styp) in enumerate(STYLES):
            seed = si * 10 + yi
            prompt = f"{sprompt}, {styp}"
            variants = [("ramp", False)] + ([("flat", True)] if args.control else [])
            for tag, flat in variants:
                k += 1
                if flat:                      # matched control: same c1 everywhere, no ramp
                    fld = fd_gradient_field((args.c1_lo + args.c1_hi) / 2,
                                            (args.c1_lo + args.c1_hi) / 2, -0.45,
                                            "radial", seed=seed, n=args.res)
                else:
                    fld = fd_gradient_field(args.c1_lo, args.c1_hi, -0.45,
                                            "radial", seed=seed, n=args.res)
                img = generate(pipe, prompt, field_to_control(fld), args.cn_scale,
                               guidance_end=args.guidance_end, steps=args.steps,
                               cfg=args.cfg, seed=seed, res=args.res, device=device)
                g = np.asarray(img.convert("L"), float) / 255.0
                (c1C, c2C), (c1E, c2E) = measure_halves(g, "radial")   # A=centre, B=corner
                img.save(out / "img" / f"{sub}_{sty}_{tag}.png", optimize=True)
                rows.append(dict(subject=sub, style=sty, variant=tag, seed=seed,
                                 c1_centre=round(c1C, 3), c1_edge=round(c1E, 3),
                                 dc1=round(c1E - c1C, 3),
                                 fd_centre=round(3 - c1C, 3), fd_edge=round(3 - c1E, 3),
                                 c2_centre=round(c2C, 3), c2_edge=round(c2E, 3)))
                print(f"  [{k}/{n_tot}] {sub:10s} {sty:10s} {tag:4s} "
                      f"FD centre {3-c1C:.2f} -> edge {3-c1E:.2f}  (dc1 {c1E-c1C:+.2f})  "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)

    with (out / "falloff.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    r = [x for x in rows if x["variant"] == "ramp"]
    d = np.array([x["dc1"] for x in r])
    print(f"\n{len(r)} ramped images | mean dc1 {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f} "
          f"(SE) | {int((d > 0).sum())}/{len(d)} correct sign")
    if args.control:
        c = np.array([x["dc1"] for x in rows if x["variant"] == "flat"])
        print(f"  flat control: mean dc1 {c.mean():+.3f} +- {c.std(ddof=1)/np.sqrt(len(c)):.3f}")
    for sty in dict.fromkeys(x["style"] for x in r):
        v = np.array([x["dc1"] for x in r if x["style"] == sty])
        print(f"    {sty:10s} mean dc1 {v.mean():+.3f}  ({int((v>0).sum())}/{len(v)})")
    print(f"-> {out/'falloff.csv'}")


if __name__ == "__main__":
    main()
