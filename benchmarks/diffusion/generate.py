"""
Diffusion-generated natural texture sweep.

Loads SG161222/RealVisXL_V4.0 (an SDXL fine-tune optimized for photorealism)
and generates `images_per_family` images per family, distributed across the
5 prompt variations in prompts.py.

Outputs: out/<family>/<prompt_idx>_s<seed>.png
Manifest: out/manifest.csv  (family, prompt_idx, seed, path)
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

from prompts import PROMPTS, NEG, ALL_FAMILIES

MODEL_ID = "SG161222/RealVisXL_V4.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output dir")
    ap.add_argument("--families", default="all",
                    help="comma-separated family names, or 'all'")
    ap.add_argument("--images-per-family", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned work, don't generate")
    args = ap.parse_args()

    families = ALL_FAMILIES if args.families == "all" else args.families.split(",")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    plan = []
    for fam in families:
        prompts = PROMPTS[fam]
        n_prompts = len(prompts)
        for i in range(args.images_per_family):
            pi = i % n_prompts            # round-robin across prompts
            seed = args.seed_base + i + hash(fam) % 10000
            plan.append((fam, pi, prompts[pi], seed))

    print(f"Plan: {len(plan)} images across {len(families)} families "
          f"(images_per_family={args.images_per_family})")
    if args.dry_run:
        for fam, pi, prompt, seed in plan[:6]:
            print(f"  [{fam}] p{pi} s{seed} -> {prompt[:80]}...")
        return

    print(f"Loading {MODEL_ID} (fp16, CUDA)...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    manifest_rows = []
    started = time.time()
    for k, (fam, pi, prompt, seed) in enumerate(plan, 1):
        fam_dir = out / fam; fam_dir.mkdir(parents=True, exist_ok=True)
        fname = f"p{pi}_s{seed}.png"
        fpath = fam_dir / fname
        if fpath.exists():
            print(f"  [{k}/{len(plan)}] {fam:24s} {fname} (cached)")
            manifest_rows.append(dict(family=fam, prompt_idx=pi, seed=seed,
                                      prompt=prompt, path=str(fpath.relative_to(out))))
            continue
        gen = torch.Generator(device="cuda").manual_seed(seed)
        t_im = time.time()
        out_img = pipe(
            prompt=prompt,
            negative_prompt=NEG,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.resolution, height=args.resolution,
            generator=gen,
        ).images[0]
        out_img.save(fpath, optimize=True)
        dt = time.time() - t_im
        elapsed = time.time() - started
        eta = elapsed / k * (len(plan) - k)
        print(f"  [{k}/{len(plan)}] {fam:24s} p{pi} s{seed} "
              f"t={dt:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
        manifest_rows.append(dict(family=fam, prompt_idx=pi, seed=seed,
                                  prompt=prompt, path=str(fpath.relative_to(out))))

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "prompt_idx", "seed", "prompt", "path"])
        w.writeheader(); w.writerows(manifest_rows)
    print(f"\nWrote {len(manifest_rows)} stimuli + {manifest}")
    print(f"Total wall time: {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
