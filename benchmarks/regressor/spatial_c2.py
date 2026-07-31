"""
spatial_c2.py — spatial control of MULTIFRACTALITY, after fixing the conditioning field.

WHAT WAS ACTUALLY WRONG. The repo concluded that spatial c2 gradients "fail on all substrates"
and blamed the 8-bit conditioning image (c1 transmits at slope ~1.16, c2 at only ~0.51). That
diagnosis was downstream of the real bug: **the conditioning field never contained a c2 gradient
in the first place.**

  mf_creative.gradient_field does   f = lo*(1-m) + hi*m
  with lo = cascade(c1, c2_lo), hi = cascade(c1, c2_hi).

But cascade amplitude collapses as intermittency rises — the same "energy" concentrates into rarer,
sharper bursts. Measured at c1_target=1.3, n=1024:

    c2_target -0.20  ->  field sd 0.01213
    c2_target -0.90  ->  field sd 0.00147     (8x weaker)
    c2_target -1.20  ->  field sd 0.00129

So in the linear blend the LOW-c2 field dominates the sum everywhere, even at m=1 where it is
supposed to have vanished. Measured c2 across four strips of such a field runs
-0.058, +0.002, +0.308, +0.383 for a requested -0.20 -> -0.90: no gradient, wrong sign, and it
drifts POSITIVE because summing two independent fields pushes towards Gaussian (i.e. monofractal).

Controls that rule out the alternative explanations:
  * the estimator is fine at region scale — pure cascades at c2 -0.20 vs -0.90 separate cleanly
    even in a 170 px region (delta -0.773, d/sd -2.55);
  * cropping is fine — a third-crop of a pure c2=-0.90 field still reads -1.285;
  * so the gradient's absence is a property of the CONSTRUCTION, not of measurement.

THE FIX — two parts, both necessary:
  1. AMPLITUDE MATCHING. Standardise each cascade to unit sd before blending, so the high-c2 field
     is actually present where the mask says it should be. Alone this takes the measured field
     gradient from +0.026 +- 0.295 (nothing) to -0.344 +- 0.126 (~49% of the requested -0.70,
     right sign, and less than half the seed noise).
  2. CALIBRATED ENDPOINTS. prescribed_cascade does not deliver independent cumulants — requesting
     c2 = -1.20 at c1_target = 1.3 yields measured c1 = 2.04. So a naive c2 ramp is also a large c1
     ramp, and since c1 is what the eye reads as complexity, the "multifractality gradient" was
     substantially a fractal-dimension gradient. cascade_calibration inverts the forward map so
     both endpoints share the SAME measured c1 and differ only in c2.

DESIGN — PAIRED POLARITY. SDXL puts its own left-right structure in a frame (light, composition),
which is exactly the size of the effect we are chasing. So each subject is rendered TWICE from the
same seed, with the ramp running forward and reversed, and the statistic is the difference of
differences:

    effect = (c2_B - c2_A)|forward  -  (c2_B - c2_A)|reversed

Under the null this is 0 regardless of any fixed asymmetry the substrate or the model imposes;
under real control it is twice the transmitted gradient. A single-ramp measurement — which is what
every earlier run in this repo used — cannot tell those apart.

Usage:
  python spatial_c2.py --mode field                 # CPU: validate the construction, no GPU
  python spatial_c2.py --mode gen --subjects smoke,ink_water   # GPU: end-to-end paired test
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import mfractal as mf                                                    # noqa: E402
from cascade_calibration import Calibration, measure_field               # noqa: E402

TAIL = ", sharp focus, fine detail, natural light, photorealistic, high resolution, 8K"

# Substrates that are physically intermittent — sparse strong structure on smooth ground.
# A c2 ramp is only expressible where the substrate can actually be bursty.
SUBJECTS = [
    ("smoke",         "turbulent smoke plume, billowing eddies and thinning wisps"),
    ("ink_water",     "black ink diffusing in water, turbulent tendrils and curls"),
    ("ferrofluid",    "ferrofluid spikes under a magnet, black peaks on a mirror surface"),
    ("frost_fern",    "window frost ferns, dendritic ice crystals on cold glass"),
    ("dendrite",      "manganese dendrites on limestone, black fern-like mineral growth"),
    ("dye_plume",     "dye dispersing in water, mushrooming vortices and fine filaments"),
    ("firestorm",     "swirling firestorm, embers and flame filaments against dark smoke"),
    ("cracked_glaze", "crazed ceramic glaze, fine craquelure network over celadon"),
    ("mycelium",      "white fungal mycelium threading through dark substrate"),
    ("cirrus_wisp",   "wispy cirrus clouds in a pale sky, fibrous ice streaks"),
]


def zstd(a):
    """Unit-sd standardisation. Affine, so it preserves BOTH cumulants exactly; its only job is
    to stop the weaker-amplitude cascade from being swamped in the blend."""
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-12)


def ramp_mask(direction, n, reverse=False, shape="smoothstep_mid"):
    """Ramp profile across the frame.

    Shape matters far more than anything downstream. With a LINEAR ramp the measured thirds each
    contain a lot of the other endpoint, so the field only delivers dc2 -0.324 of a -0.80 request.
    Concentrating the transition into the middle 50% with a smoothstep leaves the outer thirds
    nearly pure and delivers -0.737 -- 92% of the request, and essentially the hard-step ceiling
    (-0.768) -- while staying C1-continuous, so it does NOT produce the visible seam that made the
    earlier hard-block experiments unusable.
    """
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    if direction == "h":
        t = x
    elif direction == "v":
        t = y
    elif direction == "d":
        t = (x + y) / 2
    else:
        t = np.clip(np.hypot(x - 0.5, y - 0.5) / 0.707, 0, 1)
    if shape == "linear":
        m = t
    elif shape == "smoothstep":
        u = np.clip(t, 0, 1); m = u * u * (3 - 2 * u)
    else:                                    # smoothstep_mid (default)
        u = np.clip((t - 0.25) / 0.5, 0, 1); m = u * u * (3 - 2 * u)
    return 1.0 - m if reverse else m


def c2_gradient_field(c1_want, c2_lo, c2_hi, direction, seed, n=1024,
                      cal=None, amp_match=True, calibrated=True, reverse=False,
                      mask_shape="smoothstep_mid"):
    """Conditioning field whose MULTIFRACTALITY varies across the frame at ~constant roughness.

    amp_match / calibrated are switchable so the script can reproduce the old broken behaviour
    and quantify exactly what each fix contributes.
    """
    if calibrated:
        cal = cal or Calibration.load()
        lo = cal.cascade(c1_want, c2_lo, n=n, seed=seed)
        hi = cal.cascade(c1_want, c2_hi, n=n, seed=seed)
    else:
        lo = np.asarray(mf.prescribed_cascade(n=n, seed=seed, c1_target=c1_want,
                                              c2_target=c2_lo), float)
        hi = np.asarray(mf.prescribed_cascade(n=n, seed=seed, c1_target=c1_want,
                                              c2_target=c2_hi), float)
    if amp_match:
        lo, hi = zstd(lo), zstd(hi)
    m = ramp_mask(direction, n, reverse=reverse, shape=mask_shape)
    return lo * (1 - m) + hi * m


def measure_halves(arr, direction, frac=3):
    """(c1,c2) of region A vs region B along the ramp. Regions are resized to a common 512 so
    every measurement sees the same number of octaves."""
    a = np.asarray(arr, float)
    h, w = a.shape
    k = h // frac
    if direction == "v":
        A, B = a[:k], a[h - k:]
    elif direction == "d":
        A, B = a[:h // 2, :w // 2], a[h // 2:, w // 2:]
    elif direction == "radial":
        A, B = a[3 * h // 8:5 * h // 8, 3 * w // 8:5 * w // 8], a[:h // 4, :w // 4]
    else:
        A, B = a[:, :k], a[:, w - k:]

    def m(z):
        z = (z - z.min()) / (np.ptp(z) + 1e-12)
        # Resize in FLOAT ('F' mode). Going through uint8 here would 8-bit-quantise a heavy-tailed
        # cascade and read c2 as tens-of-negative — the known pathology this repo warns about. The
        # generated-image path is already 8-bit, so float resize is the consistent choice for both.
        z = np.asarray(Image.fromarray(z.astype(np.float32), mode="F").resize((512, 512)), float)
        r = mf.wavelet_leaders_2d(z)
        return float(r["c1"]), float(r["c2"])
    return m(A), m(B)


# ------------------------------------------------------------------ CPU: field validation
def cmd_field(args):
    """Ablate the two fixes on the CONDITIONING FIELD alone. No GPU, no diffusion."""
    cal = Calibration.load()
    variants = [("baseline (repo)", False, False),
                ("amp-match only", True, False),
                ("calibrated only", False, True),
                ("amp-match + calibrated", True, True)]
    req = args.c2_hi - args.c2_lo
    print(f"requested dc2 = {req:+.2f} at c1 = {args.c1};  {args.seeds} seeds, n={args.n}\n")
    print(f"{'construction':>24} | {'dc2 (field)':>18} | {'dc1 (want ~0)':>16} | transmit")
    print("-" * 78)
    rows = []
    for name, amp, calib in variants:
        D2, D1 = [], []
        for s in range(args.seeds):
            f = c2_gradient_field(args.c1, args.c2_lo, args.c2_hi, "h", s, n=args.n,
                                  cal=cal, amp_match=amp, calibrated=calib,
                                  mask_shape=args.mask)
            (c1A, c2A), (c1B, c2B) = measure_halves(f, "h")
            D2.append(c2B - c2A); D1.append(c1B - c1A)
        t = np.mean(D2) / req * 100
        print(f"{name:>24} | {np.mean(D2):+.3f} +- {np.std(D2):.3f} | "
              f"{np.mean(D1):+.3f} +- {np.std(D1):.3f} | {t:5.0f}%")
        rows.append(dict(construction=name, amp_match=amp, calibrated=calib,
                         dc2=round(float(np.mean(D2)), 4), dc2_sd=round(float(np.std(D2)), 4),
                         dc1=round(float(np.mean(D1)), 4), dc1_sd=round(float(np.std(D1)), 4),
                         requested=req, transmit_pct=round(float(t), 1)))
    out = HERE / "spatial_c2"; out.mkdir(parents=True, exist_ok=True)
    with (out / "field_ablation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n-> {out/'field_ablation.csv'}")


# ------------------------------------------------------------------ GPU: end-to-end paired test
def cmd_gen(args):
    import torch
    from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN
    cal = Calibration.load()
    subs = SUBJECTS
    if args.subjects:
        keep = set(args.subjects.split(","))
        subs = [s for s in SUBJECTS if s[0] in keep]
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)

    req = args.c2_hi - args.c2_lo
    rows = []
    t0 = time.time()
    for i, (name, prompt) in enumerate(subs):
        per = {}
        for rev in (False, True):
            fld = c2_gradient_field(args.c1, args.c2_lo, args.c2_hi, args.direction, i,
                                    n=args.res, cal=cal,
                                    amp_match=not args.no_amp_match,
                                    calibrated=not args.no_calibrate, reverse=rev,
                                    mask_shape=args.mask)
            (fc1A, fc2A), (fc1B, fc2B) = measure_halves(fld, args.direction)
            img = generate(pipe, prompt + TAIL, field_to_control(fld), args.cn_scale,
                           guidance_end=args.guidance_end, steps=args.steps, cfg=args.cfg,
                           seed=i, res=args.res, device=device)
            tag = "rev" if rev else "fwd"
            img.save(out / "img" / f"{name}_{tag}.png", optimize=True)
            g = np.asarray(img.convert("L"), float) / 255.0
            (c1A, c2A), (c1B, c2B) = measure_halves(g, args.direction)
            per[tag] = dict(dc2=c2B - c2A, dc1=c1B - c1A, field_dc2=fc2B - fc2A)
        # difference of differences — cancels any fixed left/right asymmetry
        effect = (per["fwd"]["dc2"] - per["rev"]["dc2"]) / 2.0
        effect_c1 = (per["fwd"]["dc1"] - per["rev"]["dc1"]) / 2.0
        rows.append(dict(subject=name,
                         fwd_dc2=round(per["fwd"]["dc2"], 3), rev_dc2=round(per["rev"]["dc2"], 3),
                         field_dc2=round(per["fwd"]["field_dc2"], 3),
                         effect_dc2=round(effect, 3), effect_dc1=round(effect_c1, 3),
                         requested=req,
                         transmit_pct=round(effect / req * 100, 1)))
        print(f"  [{i+1}/{len(subs)}] {name:14s} fwd {per['fwd']['dc2']:+.3f}  "
              f"rev {per['rev']['dc2']:+.3f}  -> effect {effect:+.3f} "
              f"({effect/req*100:4.0f}%)  ({(time.time()-t0)/60:.1f}m)", flush=True)

    with (out / "paired.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    e = np.array([r["effect_dc2"] for r in rows])
    n = len(e)
    se = e.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"\n{n} subjects | mean paired effect {e.mean():+.3f} +- {se:.3f} (SE) "
          f"for requested {req:+.2f} -> {e.mean()/req*100:.0f}% transmission")
    print(f"  t = {e.mean()/se:+.2f}" if n > 1 else "")
    print(f"-> {out/'paired.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["field", "gen"], default="field")
    ap.add_argument("--c1", type=float, default=1.3)
    ap.add_argument("--c2-lo", type=float, default=-0.20)
    ap.add_argument("--c2-hi", type=float, default=-0.90)
    ap.add_argument("--direction", default="h")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--cn-scale", type=float, default=0.9)
    ap.add_argument("--guidance-end", type=float, default=0.60)
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--mask", default="smoothstep_mid",
                    choices=["linear", "smoothstep", "smoothstep_mid"],
                    help="ramp profile; smoothstep_mid keeps the measured regions nearly pure")
    ap.add_argument("--no-amp-match", action="store_true", help="reproduce the old broken blend")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--out", default=str(HERE / "spatial_c2"))
    args = ap.parse_args()
    (cmd_field if args.mode == "field" else cmd_gen)(args)


if __name__ == "__main__":
    main()
