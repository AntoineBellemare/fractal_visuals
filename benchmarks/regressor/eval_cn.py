"""
eval_cn.py — Phase-B gates for the fine-tuned MF-ControlNet.

Everything is measured with the TRUE wavelet-leader estimator (build_corpus.measure_c2),
never with joint_model.pt (which classifier guidance adversarially attacks).

Baselines to beat (all previously measured in this repo):
  zero-shot tile CN : texture target c2=-0.70 -> best achieved -0.45
                      scene spatial c2 gradient: ~5% of the requested span transmitted
  classifier guidance (ground truth, n=63): dC1/dT1 = +0.58, dC2/dT2 = +0.49

GATES
  G1  gradient transmission on scenes  >= 40%      (vs ~5% zero-shot)
  G2  impose c2 <= -0.70 on FIRE                    (guidance is stuck at ~-0.58)
  G3  reach the empty corner c1~1.6, c2~-0.90 within |dc1|<=0.15, |dc2|<=0.08
  G4  dC1/dT1 >= 0.58  (must not regress vs guidance)

Usage:
  python eval_cn.py --controlnet benchmarks/regressor/cn_model
  python eval_cn.py --controlnet benchmarks/regressor/cn_model/checkpoint-1000/controlnet
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
from build_corpus import measure_c2                     # ground truth  # noqa: E402
from mf_scaffold import NEG, TAIL, MODEL_ID             # noqa: E402
import local_cumulants as LC                            # noqa: E402

VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

SUBSTRATES = {
    "marble":      "polished marble slab, natural texture, photorealistic, fine detail",
    "forest":      "dense forest canopy seen from above, natural texture, photorealistic, fine detail",
    "mountain":    "rocky mountain terrain, natural texture, photorealistic, fine detail",
    "firestorm":   "raging fire and flames, natural texture, photorealistic, fine detail",
    "frost_fern":  "frost fern crystals on cold glass, natural texture, photorealistic, fine detail",
}
# (c1, c2) grid — deliberately includes the empty corner (1.6, -0.90)
GRID = [(1.0, -0.25), (1.0, -0.60), (1.3, -0.45), (1.6, -0.25), (1.6, -0.90), (1.3, -0.90)]


def build(cn_path, device):
    from diffusers import (StableDiffusionXLControlNetPipeline, ControlNetModel,
                           AutoencoderKL, EulerDiscreteScheduler)
    cn = ControlNetModel.from_pretrained(cn_path, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    p = StableDiffusionXLControlNetPipeline.from_pretrained(
        MODEL_ID, controlnet=cn, vae=vae, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True)
    p.scheduler = EulerDiscreteScheduler.from_config(p.scheduler.config)
    p.to(device); p.set_progress_bar_config(disable=True)
    return p


def gen(pipe, prompt, cond, scale, seed, res, steps=35, cfg=6.0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return pipe(prompt=prompt, negative_prompt=NEG, image=cond, height=res, width=res,
                num_inference_steps=steps, guidance_scale=cfg,
                controlnet_conditioning_scale=float(scale),
                control_guidance_start=0.0, control_guidance_end=1.0,
                generator=g).images[0]


def half_c2(img, direction="h"):
    """(c2 of calm third, c2 of turbulent third) for a gradient image."""
    a = np.asarray(img.convert("L"), float) / 255.0
    h, w = a.shape
    if direction == "h":
        left, right = a[:, :w // 3], a[:, 2 * w // 3:]
    else:
        left, right = a[:h // 3], a[2 * h // 3:]
    def m(x):
        s = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((512, 512)), float) / 255
        import mfractal as mf
        return float(mf.wavelet_leaders_2d(s)["c2"])
    return m(left), m(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlnet", default=str(HERE / "cn_model"))
    ap.add_argument("--cn-scale", default="0.8,1.1")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "cn_eval"))
    args = ap.parse_args()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    scales = [float(x) for x in args.cn_scale.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading fine-tuned ControlNet: {args.controlnet}", flush=True)
    pipe = build(args.controlnet, device)

    rows = []
    # ---------- uniform target grid ----------
    print("\n=== uniform (c1,c2) targets ===", flush=True)
    for name, prompt in SUBSTRATES.items():
        for (c1t, c2t) in GRID:
            cond = LC.encode(*LC.uniform_map(c1t, c2t), res=args.res)
            for s in scales:
                img = gen(pipe, prompt, cond, s, args.seed, args.res, device=device)
                c1m, c2m = measure_c2(img)
                img.save(out / "img" / f"{name}_c1{c1t:.2f}_c2{c2t:+.2f}_s{s:.1f}.png")
                rows.append(dict(kind="uniform", substrate=name, c1_target=c1t, c2_target=c2t,
                                 cn_scale=s, c1_meas=round(c1m, 4), c2_meas=round(c2m, 4)))
                print(f"  {name:11s} t=({c1t:.1f},{c2t:+.2f}) cn={s:.1f} -> ({c1m:.2f},{c2m:+.2f})", flush=True)

    # ---------- spatial gradient ----------
    print("\n=== spatial c2 gradient (-0.20 -> -0.90) ===", flush=True)
    for name in ("forest", "mountain", "marble"):
        c1m_, c2m_ = LC.gradient_map(1.3, 1.3, -0.20, -0.90, direction="h")
        cond = LC.encode(c1m_, c2m_, res=args.res)
        for s in scales:
            img = gen(pipe, SUBSTRATES[name], cond, s, args.seed, args.res, device=device)
            calm, turb = half_c2(img, "h")
            img.save(out / "img" / f"grad_{name}_s{s:.1f}.png")
            transmitted = (turb - calm) / (-0.90 + 0.20) * 100
            rows.append(dict(kind="gradient", substrate=name, c1_target=1.3, c2_target=np.nan,
                             cn_scale=s, c1_meas=np.nan, c2_meas=np.nan,
                             calm=round(calm, 4), turb=round(turb, 4),
                             transmitted_pct=round(transmitted, 1)))
            print(f"  {name:11s} cn={s:.1f}: calm {calm:+.3f} turb {turb:+.3f} "
                  f"-> {transmitted:.0f}% transmitted", flush=True)

    with (out / "results.csv").open("w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})

    # ---------- gates ----------
    import pandas as pd
    d = pd.DataFrame([r for r in rows if r["kind"] == "uniform"])
    g = pd.DataFrame([r for r in rows if r["kind"] == "gradient"])
    print("\n" + "=" * 62 + "\nGATES (true estimator)\n" + "=" * 62)
    s1 = np.polyfit(d.c1_target, d.c1_meas, 1)[0]
    s2 = np.polyfit(d.c2_target, d.c2_meas, 1)[0]
    print(f"  slopes: dC1/dT1 = {s1:+.2f}   dC2/dT2 = {s2:+.2f}   (guidance: +0.58 / +0.49)")
    t = g.transmitted_pct.max() if len(g) else float("nan")
    print(f"  G1 gradient transmission  best {t:.0f}%  (>=40%)   "
          f"{'PASS' if t >= 40 else 'FAIL'}   [zero-shot was ~5%]")
    fire = d[d.substrate == "firestorm"].c2_meas.min()
    print(f"  G2 fire c2 reached        {fire:+.2f}  (<=-0.70)  {'PASS' if fire <= -0.70 else 'FAIL'}")
    corner = d[(d.c1_target == 1.6) & (d.c2_target == -0.90)]
    ok3 = ((corner.c1_meas - 1.6).abs() <= 0.15) & ((corner.c2_meas + 0.90).abs() <= 0.08)
    best = corner.assign(err=(corner.c1_meas - 1.6).abs() + (corner.c2_meas + 0.90).abs())
    b = best.loc[best.err.idxmin()] if len(best) else None
    print(f"  G3 empty corner           best ({b.c1_meas:.2f},{b.c2_meas:+.2f}) "
          f"{'PASS' if ok3.any() else 'FAIL'}" if b is not None else "  G3 n/a")
    print(f"  G4 c1 not regressed       dC1/dT1 {s1:+.2f} (>=0.58)  {'PASS' if s1 >= 0.58 else 'FAIL'}")
    print(f"\nresults -> {out/'results.csv'}")


if __name__ == "__main__":
    main()
