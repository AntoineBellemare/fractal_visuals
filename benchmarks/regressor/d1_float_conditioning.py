"""
D1 — does conditioning as a FLOAT TENSOR (bypassing uint8) restore faithful c2 transfer?

Established: the 8-bit conditioning image is the bottleneck (a field at c2=-0.80 measures
-0.356 once percentile-stretched to uint8; no 8-bit encoding avoids this). The diffusers
pipelines accept a torch.Tensor for `image=`, so we can hand over the field in float and skip
quantisation entirely.

Compares, at fixed img2img strength, the achieved c2 for:
  uint8-percentile (current scaffold_init)  |  float-linear  |  float-log
against the field's own raw c2. Slope 1.0 = the pipeline faithfully transfers the field.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import mfractal as mf                                       # noqa: E402
from mf_scaffold import scaffold_init, MODEL_ID, NEG, TAIL   # noqa: E402
from build_corpus import measure_c2                          # noqa: E402


def to_float_tensor(field, mode="linear"):
    """Field -> (1,3,H,W) float tensor in [0,1]; NO 8-bit quantisation anywhere."""
    f = np.asarray(field, np.float32)
    if mode == "linear":
        g = (f - f.min()) / (np.ptp(f) + 1e-9)
    elif mode == "log":
        L = np.log(f + 1e-3)
        g = (L - L.min()) / (np.ptp(L) + 1e-9)
    else:
        raise ValueError(mode)
    t = torch.from_numpy(g)[None, None].repeat(1, 3, 1, 1)
    return t.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2-targets", default="-0.30,-0.60,-0.90,-1.20")
    ap.add_argument("--c1-target", type=float, default=1.3)
    ap.add_argument("--strength", type=float, default=0.30)
    ap.add_argument("--substrate", default="polished marble slab")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "d1_float"))
    args = ap.parse_args()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    targets = [float(x) for x in args.c2_targets.split(",")]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler, AutoencoderKL
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(dev); pipe.set_progress_bar_config(disable=True)

    rows = []
    print(f"substrate: {args.substrate} | strength {args.strength}\n")
    print(f"{'c2 req':>7} {'field raw':>10} | {'uint8':>8} {'float-lin':>10} {'float-log':>10}")
    for c2t in targets:
        field = np.asarray(mf.prescribed_cascade(n=args.res, seed=args.seed,
                                                 c1_target=args.c1_target, c2_target=c2t), float)
        raw = float(mf.wavelet_leaders_2d(field)["c2"])
        inits = {}
        inits["uint8"], _ = scaffold_init(c2t, args.res, args.seed, c1_target=args.c1_target)
        inits["float-lin"] = to_float_tensor(field, "linear")
        inits["float-log"] = to_float_tensor(field, "log")
        got = {}
        for name, init in inits.items():
            g = torch.Generator(device=dev).manual_seed(args.seed)
            img = pipe(prompt=args.substrate + TAIL, negative_prompt=NEG, image=init,
                       strength=args.strength, num_inference_steps=args.steps,
                       guidance_scale=args.cfg, generator=g).images[0]
            c1m, c2m = measure_c2(img)
            img.save(out / "img" / f"c2{c2t:+.2f}_{name}.png")
            got[name] = c2m
            rows.append(dict(c2_target=c2t, field_raw=round(raw, 3), encoding=name,
                             c1_meas=round(c1m, 3), c2_meas=round(c2m, 3)))
        print(f"{c2t:7.2f} {raw:10.2f} | {got['uint8']:8.3f} {got['float-lin']:10.3f} "
              f"{got['float-log']:10.3f}", flush=True)

    with (out / "d1.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 56)
    print("SLOPE of achieved c2 vs the field's RAW c2 (1.0 = faithful)")
    print("=" * 56)
    for name in ("uint8", "float-lin", "float-log"):
        sub = [r for r in rows if r["encoding"] == name]
        x = np.array([r["field_raw"] for r in sub]); y = np.array([r["c2_meas"] for r in sub])
        sl = np.polyfit(x, y, 1)[0]
        print(f"  {name:10s} slope {sl:+.3f}   achieved range {y.min():+.2f} .. {y.max():+.2f}")
    print("\nbaseline: uint8 path previously measured ~0.45 of the requested c2")
    print(f"data -> {out/'d1.csv'}")


if __name__ == "__main__":
    main()
