"""
fade_vs_strength.py — at what img2img strength does the scaffold stop fading?

This decides the Phase-B pairing question. A ControlNet needs pairs where
cumulants(conditioning) == measured cumulants(target). Two ways to get that:
  * synthesise the field at the target's MEASURED value  -> exact, but the field is a
    different realisation, so there is NO spatial correspondence;
  * generate the target FROM the field                   -> spatial correspondence, but the
    statistics fade (measured slope 0.382 at strength 0.45 over 926 corpus rows).
If a low enough strength gives slope ~1.0 while still looking photographic, we get BOTH.

Measures achieved c2 (true wavelet-leader estimator) vs the field's target, per strength.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import mfractal as mf                                    # noqa: E402
from mf_scaffold import scaffold_init, MODEL_ID, NEG, TAIL  # noqa: E402
from build_corpus import measure_c2                       # noqa: E402

SUBSTRATES = {
    "marble": "polished marble slab",
    "forest": "dense forest canopy seen from above",
    "frost":  "frost fern crystals on cold glass",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", default="0.20,0.30,0.40,0.50")
    ap.add_argument("--c2-targets", default="-0.20,-0.50,-0.80,-1.10")
    ap.add_argument("--c1-target", type=float, default=1.3)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "fade_test"))
    args = ap.parse_args()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    strengths = [float(x) for x in args.strengths.split(",")]
    targets = [float(x) for x in args.c2_targets.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler, AutoencoderKL
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device); pipe.set_progress_bar_config(disable=True)

    rows = []
    for name, sub in SUBSTRATES.items():
        for c2t in targets:
            init, c2_field = scaffold_init(c2t, args.res, args.seed, c1_target=args.c1_target)
            for s in strengths:
                g = torch.Generator(device=device).manual_seed(args.seed)
                img = pipe(prompt=sub + TAIL, negative_prompt=NEG, image=init, strength=s,
                           num_inference_steps=args.steps, guidance_scale=args.cfg,
                           generator=g).images[0]
                c1m, c2m = measure_c2(img)
                img.save(out / "img" / f"{name}_c2{c2t:+.2f}_st{s:.2f}.png")
                rows.append(dict(substrate=name, c2_target=c2t, c2_field=round(c2_field, 3),
                                 strength=s, c1_meas=round(c1m, 3), c2_meas=round(c2m, 3)))
                print(f"  {name:7s} target c2={c2t:+.2f} (field {c2_field:+.2f}) "
                      f"strength {s:.2f} -> achieved c2 {c2m:+.3f}", flush=True)

    with (out / "fade.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 58)
    print("FADE SLOPE per strength (achieved c2 vs requested c2; 1.0 = no fade)")
    print("=" * 58)
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["strength"]].append((r["c2_target"], r["c2_meas"]))
    for s in sorted(by):
        a = np.array(by[s])
        sl = np.polyfit(a[:, 0], a[:, 1], 1)[0]
        rr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        print(f"  strength {s:.2f}: slope {sl:+.3f}  r {rr:+.2f}   "
              f"{'<-- usable for exact pairs' if sl > 0.8 else ''}")
    print("\nreference: strength 0.45 over 926 corpus rows gave slope 0.382")
    print(f"data -> {out/'fade.csv'}")


if __name__ == "__main__":
    main()
