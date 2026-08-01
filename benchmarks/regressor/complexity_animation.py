"""
complexity_animation.py — CREATIVE APPLICATION 4: complexity as an axis of MOTION.

The idea. Every parametric axis a generative model gives you for animation is either semantic
(the subject changes) or photographic (exposure, focus, camera). There is no knob for "the same
thing, rendered more or less intricate". Sweeping c1 across frames gives exactly that: a texture
that *breathes* between smooth and busy while the subject, the composition and the lighting stay
put. Fractal dimension becomes a timeline parameter.

WHAT MAKES IT WORK — and the thing that had to be checked first. Coherence comes from holding TWO
seeds fixed: the cascade seed (so the conditioning field deforms continuously rather than being
redrawn) and the diffusion seed. Measured: consecutive conditioning fields correlate at r = 0.92 -
0.995 across the sweep, while the same c1 at a different cascade seed correlates at r = 0.003. So
the field genuinely morphs.

One trap found on the way: asking the calibrator to invert a *different* wanted-c1 each frame makes
it hop between solutions (consecutive-field r collapsed to 0.345 at one step). Fix — invert only
the two endpoints and linearly interpolate the TARGETS between them. Parameter motion is then
smooth by construction, and the achieved c1 is measured and reported rather than assumed.

THE CONTROL. "Coherent" is a claim, so it needs a null: an arm rendered with the same c1 sweep but
a DIFFERENT diffusion seed per frame. That is what an incoherent sequence looks like, and the
frame-to-frame distance of the real sweep is compared against it.

Outputs: PNG frames, an animated GIF (ping-pong, so it loops), a per-frame CSV, and the tracking
and coherence numbers.

Usage:
  python complexity_animation.py --frames 16 --subjects moss,ink
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
import mfractal as mf                                                    # noqa: E402
from cascade_calibration import Calibration                              # noqa: E402
from spatial_c2 import zstd                                              # noqa: E402
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN  # noqa: E402

TAIL = ", sharp focus, fine detail, natural light, photorealistic, high resolution, 8K"

# Weighted towards families with a WIDE achieved c1 span in FEASIBILITY.md — a narrow-span
# substrate (moss: FD 2.21 -> 2.03) barely breathes no matter how hard the field is driven.
SUBJECTS = {
    "ink":        "black ink diffusing in clear water, turbulent tendrils and curling filaments",
    "ferrofluid": "ferrofluid spikes under a magnet, black iridescent peaks on a mirror surface",
    "dye":        "vivid magenta and cyan dye dispersing in water, mushrooming vortices",
    "smoke":      "a turbulent smoke plume, billowing eddies thinning into wisps",
    "firestorm":  "swirling firestorm, glowing embers and flame filaments against dark smoke",
    "frost":      "window frost ferns, dendritic ice crystals on cold blue glass",
    "ice_sheet":  "cracked sea ice seen from above, fractured floes and open leads",
    "snow":       "wind-sculpted snow surface, sastrugi ridges in low raking light",
    "dunes":      "rippled desert sand dunes from above, sinuous crests and shadows",
    "delta":      "aerial view of a branching river delta, distributary channels in sediment",
    "mountains":  "aerial of a rugged mountain range, ridges and valleys in relief",
    "canyon":     "aerial of a deep eroded canyon system, branching tributary gorges",
    "marble":     "polished marble slab, veined and translucent",
    "granite":    "weathered granite surface, feldspar crystals and mineral speckle",
    "lichen":     "crustose lichen and mineral staining spreading across slate",
    "moss":       "a bed of dense moss and small ferns on a forest floor, top-down",
    "canopy":     "dense green forest canopy seen from above, packed crowns",
    "coral":      "vibrant coral reef underwater, branching colonies",
    "jellyfish":  "a translucent bioluminescent deep-sea creature with delicate tendrils",
    "nebula":     "deep space nebula, filaments of ionised gas and dark dust lanes",
}

# A second, deliberately SURREAL set. These are harder for the method than natural textures: the
# scene has to hold together as an image while its texture statistics are swept, and dreamlike
# content gives SDXL more licence to redraw rather than re-texture. Kept because if the axis only
# worked on flat natural texture it would be a curiosity, not a compositional tool.
SURREAL = {
    "melting_clocks": "melting pocket watches draped over a bare desert branch, long shadows",
    "floating_isles": "floating islands with waterfalls pouring into an empty sky",
    "eye_storm":      "a vast human eye opening within a storm cloud above an ocean",
    "bone_cathedral": "a cathedral grown from whale bone and coral, vaulted ribs and buttresses",
    "moth_machine":   "a giant moth with clockwork wings, brass gears between the wing scales",
    "root_city":      "a city whose towers are living tree roots, small windows glowing within",
    "jelly_sky":      "translucent jellyfish drifting through a desert sky above dunes",
    "mirror_desert":  "a desert of shattered mirrors reflecting a second sun",
    "paper_ocean":    "an ocean of folded paper waves, origami spray and creased foam",
    "myco_brain":     "a human brain made of glowing mycelium threads in dark soil",
    "clock_forest":   "a forest where every leaf is a tiny tarnished pocket watch",
    "stone_faces":    "weathered stone faces half-emerging from a cliff, unfinished",
    "feather_storm":  "a storm of falling feathers inside a flooded marble ballroom",
    "coral_ribs":     "a human ribcage overgrown with coral polyps and anemones underwater",
    "ink_birds":      "a flock of birds dissolving into spilled ink at the wingtips",
    "crystal_organs": "translucent crystal organs suspended in dark viscous fluid",
    "sand_figures":   "human figures forming and dissolving out of blowing desert sand",
    "lung_tree":      "a bare tree whose branches are bronchial airways, backlit in mist",
    "egg_moon":       "a cracked eggshell moon leaking pale light over a still black lake",
    "book_terraces":  "a landscape built from open books, their pages terraced like rice fields",
}
SETS = {"natural": SUBJECTS, "surreal": SURREAL}


def measure(a):
    a = np.asarray(a, float); a = (a - a.min()) / (np.ptp(a) + 1e-12)
    r = mf.wavelet_leaders_2d(a)
    return float(r["c1"]), float(r["c2"])


def _path_point(t0, t1, u):
    return t0[0] + (t1[0] - t0[0]) * u, t0[1] + (t1[1] - t0[1]) * u


def uniform_us(c1_lo, c1_hi, c2, frames, cal, probe_n=384, probe_k=24, seeds=3):
    """Reparameterise the target path so MEASURED field c1 advances uniformly per frame.

    Walking the path at constant speed does not give constant *perceptual* speed: the forward map
    saturates, so a linear sweep crawls at one end and stalls at the other (both pilot substrates
    turned over around frame 11). Here the path is probed, the achieved c1 measured along it, and
    the frame positions resampled so equal frame steps mean equal c1 steps.

    Reparameterising is safe where re-inverting is not: it stays on ONE monotone path, so the
    field still morphs continuously instead of hopping between calibrator solutions.
    """
    t0 = cal.invert(c1_lo, c2); t1 = cal.invert(c1_hi, c2)
    us = np.linspace(0, 1, probe_k)
    got = []
    for u in us:
        a, b = _path_point(t0, t1, u)
        v = [measure(np.asarray(mf.prescribed_cascade(n=probe_n, seed=s, c1_target=a,
                                                      c2_target=b), float))[0]
             for s in range(seeds)]
        got.append(float(np.mean(v)))
    got = np.maximum.accumulate(np.array(got))          # enforce monotonicity for inversion
    want = np.linspace(got[0], got[-1], frames)
    return np.interp(want, got, us), got


def frame_fields(c1_lo, c1_hi, c2, frames, n, seed, cal, us=None):
    """Interpolate the CALIBRATED endpoint targets — inverting per frame makes the solver hop."""
    t0 = cal.invert(c1_lo, c2); t1 = cal.invert(c1_hi, c2)
    if us is None:
        us = np.linspace(0, 1, frames)
    out = []
    for u in us:
        a, b = _path_point(t0, t1, u)
        out.append(zstd(np.asarray(mf.prescribed_cascade(n=n, seed=int(seed),
                                                         c1_target=a, c2_target=b), float)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--c1-lo", type=float, default=0.35)
    ap.add_argument("--c1-hi", type=float, default=1.45)
    ap.add_argument("--c2", type=float, default=-0.45)
    ap.add_argument("--subjects", default="moss,ink", help="comma list, or 'all'")
    ap.add_argument("--set", dest="subject_set", default="natural",
                    choices=list(SETS), help="which subject dictionary to draw from")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--cn-scale", type=float, default=0.9)
    ap.add_argument("--guidance-end", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--no-uniform", action="store_true",
                    help="walk the target path linearly; motion then crawls at one end and stalls "
                         "at the other (step-size cv 0.63 vs 0.23 reparameterised)")
    ap.add_argument("--gif-size", type=int, default=512)
    ap.add_argument("--gif-colors", type=int, default=128)
    ap.add_argument("--out", default=str(HERE / "creative" / "anim"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cal = Calibration.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)

    BANK = SETS[args.subject_set]
    subs = ([s for s in args.subjects.split(",") if s in BANK]
            if args.subjects != "all" else list(BANK))
    arms = [("coherent", True)] + ([] if args.no_control else [("control", False)])
    rows = []
    t0 = time.time(); k = 0
    n_tot = len(subs) * len(arms) * args.frames

    us = None
    if not args.no_uniform:
        us, probe = uniform_us(args.c1_lo, args.c1_hi, args.c2, args.frames, cal)
        print(f"uniform reparameterisation: field c1 {probe[0]:.2f} -> {probe[-1]:.2f} "
              f"over {args.frames} frames", flush=True)

    for si, sub in enumerate(subs):
        prompt = BANK[sub] + TAIL
        # a per-subject cascade seed so the 20 clips are not all the same structure
        flds = frame_fields(args.c1_lo, args.c1_hi, args.c2, args.frames, args.res,
                            args.seed + si, cal, us=us)
        for arm, fixed_seed in arms:
            (out / arm / sub).mkdir(parents=True, exist_ok=True)
            imgs = []
            for i, fld in enumerate(flds):
                k += 1
                dseed = (args.seed + si) if fixed_seed else 1000 + i   # control: reseed each frame
                img = generate(pipe, prompt, field_to_control(fld), args.cn_scale,
                               guidance_end=args.guidance_end, steps=args.steps,
                               cfg=args.cfg, seed=dseed, res=args.res, device=device)
                img.save(out / arm / sub / f"f{i:02d}.png", optimize=True)
                imgs.append(img)
                fc1, _ = measure(fld)
                c1m, c2m = measure(np.asarray(img.convert("L"), float) / 255.0)
                rows.append(dict(subject=sub, arm=arm, frame=i,
                                 field_c1=round(fc1, 4), c1_meas=round(c1m, 4),
                                 c2_meas=round(c2m, 4), fd_meas=round(3 - c1m, 3)))
                print(f"  [{k}/{n_tot}] {sub:7s} {arm:9s} f{i:02d}  field c1 {fc1:.2f} -> "
                      f"image c1 {c1m:.2f} (FD {3-c1m:.2f})  ({(time.time()-t0)/60:.1f}m)",
                      flush=True)
            # frame-to-frame distance
            g = [np.asarray(im.convert("L"), float) / 255.0 for im in imgs]
            dif = [float(np.abs(a - b).mean()) for a, b in zip(g[:-1], g[1:])]
            for i, v in enumerate(dif):
                rows[-(len(g)) + i + 1]["frame_dist"] = round(v, 5)
            print(f"    {sub}/{arm}: mean frame-to-frame distance {np.mean(dif):.4f}", flush=True)
            # ping-pong GIF so it loops without a jump
            seq = imgs + imgs[-2:0:-1]
            small = [im.resize((args.gif_size, args.gif_size)).convert(
                "P", palette=Image.ADAPTIVE, colors=args.gif_colors) for im in seq]
            small[0].save(out / f"{sub}_{arm}.gif", save_all=True, append_images=small[1:],
                          duration=110, loop=0, optimize=True)
            print(f"    -> {sub}_{arm}.gif", flush=True)

    fp = out / "anim.csv"
    keys = ["subject", "arm", "frame", "field_c1", "c1_meas", "c2_meas", "fd_meas", "frame_dist"]
    with fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({kk: r.get(kk, "") for kk in keys})

    import pandas as pd
    d = pd.DataFrame(rows)
    print()
    for sub in subs:
        s = d[(d.subject == sub) & (d.arm == "coherent")]
        r = np.corrcoef(s.field_c1, s.c1_meas)[0, 1]
        print(f"  {sub}: tracking corr(field c1, image c1) = {r:+.3f}   "
              f"image c1 span {s.c1_meas.min():.2f} -> {s.c1_meas.max():.2f} "
              f"(FD {3-s.c1_meas.max():.2f} -> {3-s.c1_meas.min():.2f})")
    for arm in [a for a, _ in arms]:
        v = d[d.arm == arm].frame_dist.dropna()
        print(f"  frame-to-frame distance, {arm:9s}: {v.mean():.4f}")
    if len(arms) == 2:
        a = d[d.arm == "coherent"].frame_dist.dropna().mean()
        b = d[d.arm == "control"].frame_dist.dropna().mean()
        print(f"  -> coherent sequence is {b/a:.1f}x smoother than the reseeded control")
    print(f"-> {fp}")


if __name__ == "__main__":
    main()
