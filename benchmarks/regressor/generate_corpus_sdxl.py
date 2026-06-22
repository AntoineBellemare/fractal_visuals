"""
generate_corpus_sdxl.py  --  Stage 2: c2-BALANCED SDXL data for the regressor.

Not "generate 5k images blindly". The diffusion c2 distribution is clustered (most
families sit in [-0.3, -0.05]; only clouds/fluids reach -0.7..-1.0), so blind
generation over-samples the middle and starves the extremes the guidance system needs.

Instead: generate -> measure c2 -> keep only if its c2 bin is under-filled, until every
bin across the target range holds `--per-bin` images (rejection sampling). An adaptive
family picker biases generation toward families whose running-mean c2 sits near the most
under-filled bin, so we waste fewer rejected samples than uniform sampling.

Images are real 8-bit photos, so c2 is measured the 8-bit way (build_corpus.measure_c2),
identical to how the rest of the diffusion corpus is labelled. Rows are appended straight
into corpus/manifest.csv (source=diffusion, group=prompt) so train_stage1.py picks them up.

Cost: at the measured ~9.9 s/img (28 steps, 1024², RealVisXL on a 4090 Laptop), filling
10 bins x 150 = 1500 KEPT images is ~6-10 h wall (more attempts than kept near the tails).
Checkpointable: re-running skips bins already full and existing files.

Usage:
  python generate_corpus_sdxl.py --per-bin 150
  python generate_corpus_sdxl.py --per-bin 150 --c2-lo -1.0 --c2-hi -0.02 --bins 10
  python generate_corpus_sdxl.py --dry-run            # plan only, no model load
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
from build_corpus import measure_c2, ROOT as _ROOT  # noqa: E402

MODEL_ID = "SG161222/RealVisXL_V4.0"


def load_prompts(module_names):
    """Merge one or more prompt modules (e.g. 'prompts', 'prompts_intricate').
    Returns (PROMPTS, NEG, ALL_FAMILIES). NEG taken from the first module."""
    import importlib
    merged, neg = {}, None
    for name in module_names:
        m = importlib.import_module(name)
        if neg is None:
            neg = m.NEG
        for fam, ps in m.PROMPTS.items():
            merged[fam] = ps
    return merged, neg, list(merged.keys())


def bin_of(c2, edges):
    """Index of the bin containing c2, or None if outside [edges[0], edges[-1]]."""
    if c2 < edges[0] or c2 >= edges[-1]:
        return None
    return int(np.searchsorted(edges, c2, side="right") - 1)


def family_for_bin(target_center, fam_hist, families, rng):
    """Pick the family whose running-mean measured c2 is closest to target_center.
    Families with no history yet are explored first (round-robin)."""
    unseen = [f for f in families if not fam_hist[f]]
    if unseen:
        return unseen[int(rng.integers(len(unseen)))]
    means = {f: float(np.mean(fam_hist[f])) for f in families}
    return min(families, key=lambda f: abs(means[f] - target_center))


def load_existing(manifest):
    if not manifest.exists():
        return []
    import pandas as pd
    return pd.read_csv(manifest).to_dict("records")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "corpus" / "images" / "diffusion_balanced"))
    ap.add_argument("--manifest", default=str(HERE / "corpus" / "manifest.csv"))
    ap.add_argument("--per-bin", type=int, default=150)
    ap.add_argument("--c2-lo", type=float, default=-1.0)
    ap.add_argument("--c2-hi", type=float, default=-0.02)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--max-attempts", type=int, default=6000)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps-choices", default="30,50")
    ap.add_argument("--cfg-choices", default="5,7,9")
    ap.add_argument("--families", default="all")
    ap.add_argument("--prompts-module", default="prompts",
                    help="comma-separated prompt modules to merge, e.g. "
                         "'prompts,prompts_intricate' or just 'prompts_intricate'")
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    PROMPTS, NEG, ALL_FAMILIES = load_prompts(args.prompts_module.split(","))
    families = ALL_FAMILIES if args.families == "all" else args.families.split(",")
    edges = np.linspace(args.c2_lo, args.c2_hi, args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    steps_choices = [int(x) for x in args.steps_choices.split(",")]
    cfg_choices = [float(x) for x in args.cfg_choices.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)

    # seed bin counts from any diffusion rows already in the manifest
    existing = load_existing(manifest)
    counts = [0] * args.bins
    for r in existing:
        if r.get("source") == "diffusion":
            b = bin_of(float(r["c2"]), edges)
            if b is not None:
                counts[b] += 1
    print("target bins (c2):", " ".join(f"[{edges[i]:.2f},{edges[i+1]:.2f})"
                                        for i in range(args.bins)))
    print("already filled  :", counts)
    print(f"goal: {args.per_bin}/bin  | families={len(families)}  "
          f"steps={steps_choices} cfg={cfg_choices}")
    if args.dry_run:
        return

    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    print(f"loading {MODEL_ID} (fp16, cuda)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    rng = np.random.default_rng(0)
    fam_hist: dict[str, list] = defaultdict(list)
    new_rows, attempts, kept = [], 0, 0
    t0 = time.time()
    while attempts < args.max_attempts and min(counts) < args.per_bin:
        # most under-filled bin -> aim a family at its center
        target_bin = int(np.argmin(counts))
        fam = family_for_bin(centers[target_bin], fam_hist, families, rng)
        prompts = PROMPTS[fam]
        pi = int(rng.integers(len(prompts)))
        steps = steps_choices[int(rng.integers(len(steps_choices)))]
        cfg = cfg_choices[int(rng.integers(len(cfg_choices)))]
        seed = args.seed_base + attempts
        attempts += 1

        gen = torch.Generator(device="cuda").manual_seed(seed)
        img = pipe(prompt=prompts[pi], negative_prompt=NEG, num_inference_steps=steps,
                   guidance_scale=cfg, width=args.resolution, height=args.resolution,
                   generator=gen).images[0]
        c1, c2 = measure_c2(img)
        fam_hist[fam].append(c2)
        b = bin_of(c2, edges)
        status = "drop"
        if b is not None and counts[b] < args.per_bin:
            counts[b] += 1; kept += 1; status = f"bin{b}"
            fpath = out / f"{fam}_p{pi}_s{seed}_cfg{cfg:g}_st{steps}.png"
            img.save(fpath, optimize=True)
            new_rows.append(dict(path=Path(fpath).resolve().relative_to(_ROOT).as_posix(),
                                 c1=round(c1, 5), c2=round(c2, 5), source="diffusion",
                                 group=f"diffusion:{fam}:{pi}"))
        if attempts % 20 == 0 or status != "drop":
            el = time.time() - t0
            print(f"  [{attempts:4d}] {fam:20s} c2={c2:+.3f} -> {status:5s} "
                  f"kept={kept} min_bin={min(counts)} el={el/60:.1f}m")
        # periodic flush so long runs are crash-safe
        if status != "drop" and kept % 25 == 0:
            _flush(manifest, existing, new_rows)

    _flush(manifest, existing, new_rows)
    print(f"\nDONE: kept {kept} in {attempts} attempts ({time.time()-t0:.0f}s). "
          f"bins: {counts}")
    if min(counts) < args.per_bin:
        print(f"WARNING: not all bins reached {args.per_bin} within --max-attempts; "
              f"the tail bins are hard for this prompt set. Raise --max-attempts or "
              f"widen prompts/CFG/steps.")


def _flush(manifest, existing, new_rows):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # de-dup by path (existing + new), keep last
    by_path = {r["path"]: r for r in existing}
    for r in new_rows:
        by_path[r["path"]] = r
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "c1", "c2", "source", "group"])
        w.writeheader()
        w.writerows(by_path.values())


if __name__ == "__main__":
    main()
