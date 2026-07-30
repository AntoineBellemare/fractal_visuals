"""
gradient_gallery.py — high-quality COMPLEXITY-GRADIENT imagery (aesthetic pass).

Honest framing: this is the *visual* capability, not measured spatial control. The scaffold
field is a smooth spatial blend between two prescribed cascades, fed to the zero-shot tile
ControlNet. It yields smooth, artifact-free images that genuinely READ as calm->turbulent
across the frame. Measured c2 transmission through this route is only ~5%, and spatial
guidance (the route that could have given measured control) failed — see spatial_guidance.py.
So: use these for visuals; do not claim measured spatial control from them.

Quality choices (vs the earlier gradient_pass):
  * 40 steps instead of 30;
  * cn_scale ~0.8 — strong enough to shape the frame, low enough to keep the subject photoreal
    (>=1.2 starts overriding content; the earlier 1.3 run looked procedural);
  * control_guidance_end 0.85 so the last steps re-photorealise texture;
  * subjects chosen from the families that rendered best in the atlases (terrain, atmosphere,
    water/ice, creatures) rather than arbitrary scenes.
Every image is still measured (calm vs turbulent third) and written to a CSV, so the gallery
never drifts into unverified claims.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import mfractal as mf                                              # noqa: E402
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN  # noqa: E402
from mf_creative import gradient_field, grad_mask, cascade                 # noqa: E402


def fd_gradient_field(c1_lo, c1_hi, c2, direction, seed=0, n=1024):
    """Spatial gradient in c1 (= FALLING fractal dimension) at fixed c2.

    Why this is the better gradient: the 8-bit conditioning image preserves c1 with slope
    +1.163 (r 0.997) but c2 with only +0.506 -- c1 is a SLOPE of log-leaders and so is
    invariant to the affine rescaling the encoder applies, whereas c2 is a VARIANCE and the
    percentile stretch flattens it. So an FD ramp survives the conditioning path that a
    multifractality ramp cannot."""
    import mfractal as _mf
    import numpy as _np
    lo = _np.asarray(_mf.prescribed_cascade(n=n, seed=seed, c1_target=c1_lo, c2_target=c2), float)
    hi = _np.asarray(_mf.prescribed_cascade(n=n, seed=seed, c1_target=c1_hi, c2_target=c2), float)
    m = grad_mask(direction)
    if m.shape[0] != n:
        from PIL import Image as _I
        m = _np.asarray(_I.fromarray((m * 255).astype('uint8')).resize((n, n)), float) / 255.0
    return lo * (1 - m) + hi * m

# (label, prompt, gradient direction) — drawn from the families that rendered best
SUBJECTS = [
    ("river_delta",   "aerial view of a branching river delta, distributary channels in sediment", "h"),
    ("canyon",        "aerial of a deep eroded canyon system, branching tributary gorges", "d"),
    ("coastline",     "aerial of a rocky coastline, indented bays and headlands", "h"),
    ("mountains",     "aerial of a rugged mountain range, ridges and valleys in relief", "v"),
    ("tundra",        "aerial of arctic patterned ground, ice-wedge polygons", "radial"),
    ("volcanic",      "aerial of a lava field, ropey and blocky flow textures", "h"),
    ("cirrus",        "wispy cirrus clouds in a pale sky, fibrous ice streaks", "h"),
    ("mammatus",      "mammatus cloud undersides, hanging pouched lobes at dusk", "v"),
    ("smoke",         "turbulent smoke plume, billowing eddies and thinning wisps", "d"),
    ("frost",         "window frost ferns, dendritic ice crystals on cold glass", "radial"),
    ("sea_foam",      "sea foam on wet sand from above, lacy dissolving bubbles", "h"),
    ("ice_sheet",     "cracked sea ice from above, fractured floes and leads", "v"),
    ("forest",        "dense green forest canopy seen from above, packed crowns", "h"),
    ("autumn",        "autumn forest treetops from above, mixed orange and gold crowns", "d"),
    ("jellyfish",     "a translucent bioluminescent deep-sea creature with delicate tendrils", "radial"),
    ("coral",         "vibrant coral reef underwater, branching colonies", "h"),
    ("lichen_rock",   "crustose lichen and mineral staining spreading across slate", "d"),
    ("dunes",         "rippled desert sand dunes from above, sinuous crests and shadows", "h"),
]

# Substrates that are PHYSICALLY INTERMITTENT (sparse strong structure on smooth ground) --
# the only ones where a c2 ramp is expressible at all. Everything else can only do FD ramps.
INTERMITTENT = [
    ("smoke",        "turbulent smoke plume, billowing eddies and thinning wisps", "h"),
    ("smoke_backlit","backlit incense smoke, laminar threads breaking into vortices", "d"),
    ("ink_water",    "black ink diffusing in water, turbulent tendrils and curls", "h"),
    ("ferrofluid",   "ferrofluid spikes under a magnet, black peaks on a mirror surface", "radial"),
    ("frost_fern",   "window frost ferns, dendritic ice crystals on cold glass", "h"),
    ("lichen_slate", "crustose lichen and mineral staining spreading across slate", "d"),
    ("cirrus_wisp",  "wispy cirrus clouds in a pale sky, fibrous ice streaks", "h"),
    ("dendrite",     "manganese dendrites on limestone, black fern-like mineral growth", "radial"),
    ("mycelium",     "white fungal mycelium threading through dark substrate", "h"),
    ("dye_plume",    "dye dispersing in water, mushrooming vortices and fine filaments", "v"),
    ("firestorm",    "swirling firestorm, embers and flame filaments against dark smoke", "d"),
    ("cracked_glaze","crazed ceramic glaze, fine craquelure network over celadon", "h"),
]

TAIL = ", sharp focus, fine detail, natural light, photorealistic, high resolution, 8K"


def measure_thirds(img, direction):
    """c2 of the calm third vs the turbulent third (record-keeping, not a control claim)."""
    a = np.asarray(img.convert("L"), float) / 255.0
    h, w = a.shape
    if direction == "v":
        A, B = a[:h // 3], a[2 * h // 3:]
    elif direction == "radial":
        A, B = a[3 * h // 8:5 * h // 8, 3 * w // 8:5 * w // 8], a[:h // 4, :w // 4]
    elif direction == "d":
        # the ramp runs top-left -> bottom-right, so sample those CORNERS. Sampling
        # left/right thirds (as before) only spans half the ramp and understated 'd' subjects.
        A, B = a[:h // 2, :w // 2], a[h // 2:, w // 2:]
    else:                                   # h ramps left -> right
        A, B = a[:, :w // 3], a[:, 2 * w // 3:]
    def m(x):
        s = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((512, 512)), float) / 255
        r = mf.wavelet_leaders_2d(s)
        return float(r["c1"]), float(r["c2"])
    return m(A), m(B)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cn-scale", type=float, default=0.8)
    ap.add_argument("--guidance-end", type=float, default=0.60)   # 0.85 held control too long and washed out content
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--c1", type=float, default=1.3)
    ap.add_argument("--c2-calm", type=float, default=-0.20)
    ap.add_argument("--c2-turb", type=float, default=-0.90)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--subjects", default=None, help="comma list; default = all")
    ap.add_argument("--mode", choices=["mf", "fd"], default="mf",
                    help="mf = ramp c2 (only works on intermittent substrates); "
                         "fd = ramp c1/fractal dimension (survives the 8-bit path, works broadly)")
    ap.add_argument("--c1-lo", type=float, default=0.8)
    ap.add_argument("--c1-hi", type=float, default=1.9)
    ap.add_argument("--intermittent", action="store_true",
                    help="use only substrates that can physically express a c2 ramp")
    ap.add_argument("--out", default=str(HERE / "gradient_gallery"))
    args = ap.parse_args()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)

    subs = INTERMITTENT if args.intermittent else SUBJECTS
    if args.subjects:
        keep = set(args.subjects.split(","))
        subs = [s for s in SUBJECTS if s[0] in keep]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)

    rows, tiles = [], []
    import time; t0 = time.time()
    for i, (name, prompt, direction) in enumerate(subs):
        if args.mode == "fd":
            field = fd_gradient_field(args.c1_lo, args.c1_hi, -0.45, direction, seed=i, n=args.res)
        else:
            field = gradient_field(args.c1, args.c2_calm, args.c2_turb, direction, seed=i)
        ctrl = field_to_control(field)
        img = generate(pipe, prompt + TAIL, ctrl, args.cn_scale,
                       guidance_end=args.guidance_end, steps=args.steps, cfg=args.cfg,
                       seed=i, res=args.res, device=device)
        (c1A, c2A), (c1B, c2B) = measure_thirds(img, direction)
        calm, turb = (c1A, c1B) if args.mode == "fd" else (c2A, c2B)
        fp = out / "img" / f"{name}_{direction}.png"
        img.save(fp, optimize=True)
        rows.append(dict(subject=name, direction=direction, mode=args.mode,
                         c1_A=round(c1A, 3), c1_B=round(c1B, 3),
                         c2_A=round(c2A, 3), c2_B=round(c2B, 3),
                         delta=round(turb - calm, 3)))
        tiles.append((name, direction, img, calm, turb))
        print(f"  [{i+1}/{len(subs)}] {name:13s} {direction:6s} calm {calm:+.2f} -> "
              f"turb {turb:+.2f}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    with (out / "gallery.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # contact sheet
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nc = 3; nr = (len(tiles) + nc - 1) // nc
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 3.4, nr * 3.6))
    ax = np.atleast_1d(ax).ravel()
    for a, (name, direction, img, calm, turb) in zip(ax, tiles):
        a.imshow(img.resize((420, 420))); a.set_xticks([]); a.set_yticks([])
        lab = "FD" if args.mode == "fd" else "c2"
        v0, v1 = (3 - calm, 3 - turb) if args.mode == "fd" else (calm, turb)
        a.set_title(f"{name}  ({direction})\n{lab} {v0:+.2f} → {v1:+.2f}", fontsize=8, pad=3)
    for a in ax[len(tiles):]:
        a.axis("off")
    what = ("FRACTAL DIMENSION (c1) ramp — detailed → smooth" if args.mode == "fd"
            else "MULTIFRACTALITY (c2) ramp — uniform → clustered")
    fig.suptitle("Complexity gradients across the frame: " + what
                 + "  (scaffold + tile ControlNet; values MEASURED per third)",
                 fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out / "gallery_sheet.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    d = np.array([r["delta"] for r in rows])
    print(f"\n{len(rows)} images | mean measured c2 delta {d.mean():+.3f} "
          f"(requested {args.c2_turb - args.c2_calm:+.2f})")
    print(f"sheet -> {out/'gallery_sheet.png'}")


if __name__ == "__main__":
    main()
