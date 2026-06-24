"""
mf_scaffold.py — MF-ControlNet-lite: impose an EXACT target c2 on ANY pareidolic
substrate by using a procedural multifractal field as the img2img init.

The idea (decouple substrate from multifractality, the structural way):
  1. `prescribed_cascade(c2_target=t)` synthesizes a field whose c2 IS t (validated).
     That field is the multifractal "scaffold" — sparse bright filaments on smooth voids
     for strong t, gentle texture for weak t.
  2. SDXL img2img with a substrate prompt (lichen, ferrofluid, smoke...) repaints the
     scaffold photorealistically. At the right `--strength`, the output keeps the
     scaffold's intermittency (so its measured c2 stays near t) while looking like the
     substrate. This works even for substrates that are natively FLAT (lichen, coral),
     which guidance alone can't push — the c2 comes from the scaffold, not the prompt.

This is the lightweight version of the PLAN's MF-ControlNet (no ControlNet training):
a procedural scaffold + img2img. Sweep `--strength` to find the c2-preserving sweet spot.

Usage:
  python mf_scaffold.py --substrate "lichen colony on slate" --c2-target -0.7 \
      --strength-sweep 0.3,0.45,0.6,0.75
  python mf_scaffold.py --prompts-module prompts_intricate --family ferrofluid \
      --c2-target -0.8 --strength 0.5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import mfractal as mf                                  # noqa: E402
from build_corpus import measure_c2, measure_c2_field  # noqa: E402

MODEL_ID = "SG161222/RealVisXL_V4.0"
NEG = ("low quality, blurry, smooth, flat, distorted, watermark, text, logo, frame, "
       "border, illustration, cartoon, human, person, face, figure")
TAIL = ", extreme macro photograph, sharp focus, natural light, 8K, photorealistic"


def robust_unit(f, lo=1.0, hi=99.0):
    a, b = np.percentile(f, lo), np.percentile(f, hi)
    return np.clip((f - a) / (b - a + 1e-8), 0, 1)


def scaffold_init(c2_target, n, seed, c1_target=None):
    """Procedural field with exact (c1,)c2 -> spread-for-display RGB init image.
    prescribed_cascade targets BOTH cumulants, so passing c1_target lets the scaffold
    place the init anywhere in the (c1,c2) plane -- including the busy+strong corner SDXL
    can't reach."""
    kw = {} if c1_target is None else {"c1_target": float(c1_target)}
    raw = mf.prescribed_cascade(n=n, seed=seed, c2_target=float(c2_target), **kw)
    # measure faithfully at native res (don't 8-bit-downsize a heavy-tailed field)
    c2_scaffold = float(mf.wavelet_leaders_2d(np.asarray(raw, float))["c2"])
    disp = robust_unit(np.asarray(raw, float))         # spread so structure is visible
    img = Image.fromarray((disp * 255).astype(np.uint8), "L").convert("RGB")
    return img, c2_scaffold


def build_scaffold_corpus(args):
    """Generate scaffold images across substrates x c2 targets x seeds; label each by
    its MEASURED output c2; append to corpus/manifest.csv as source=scaffold. The
    scaffold fixes the c2 regime, so this densely populates the strong-c2 end that plain
    SDXL and guidance struggle to reach."""
    import importlib
    pm = importlib.import_module(args.corpus_families)
    fams = list(pm.PROMPTS.keys())
    targets = [float(x) for x in args.corpus_targets.split(",")]
    c1_targets = ([float(x) for x in args.corpus_c1_targets.split(",")]
                  if args.corpus_c1_targets else [None])
    img_dir = HERE / "corpus" / "images" / "scaffold"; img_dir.mkdir(parents=True, exist_ok=True)
    manifest = HERE / "corpus" / "manifest.csv"
    import pandas as pd
    existing = pd.read_csv(manifest).to_dict("records") if manifest.exists() else []
    by_path = {r["path"]: r for r in existing}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device); pipe.set_progress_bar_config(disable=True)

    import time
    n_total = len(fams) * len(targets) * len(c1_targets) * args.corpus_seeds
    k = 0; t0 = time.time()
    for fam in fams:
        prompt = pm.PROMPTS[fam][0] + TAIL
        for c1t in c1_targets:
            for t in targets:
                for s in range(args.corpus_seeds):
                    k += 1
                    init_img, _ = scaffold_init(t, args.res, s, c1_target=c1t)
                    g = torch.Generator(device=device).manual_seed(s)
                    img = pipe(prompt=prompt, negative_prompt=NEG, image=init_img,
                               strength=args.corpus_strength, num_inference_steps=args.steps,
                               guidance_scale=args.cfg, generator=g).images[0]
                    c1, c2 = measure_c2(img)
                    if not np.isfinite(c2) or abs(c2) > 3.0:
                        continue
                    c1tag = "" if c1t is None else f"_c1{c1t:+.2f}"
                    fp = img_dir / f"{fam}{c1tag}_t{t:+.2f}_s{s}.png"
                    img.save(fp, optimize=True)
                    rel = fp.resolve().relative_to(ROOT).as_posix()
                    band = f"{round(c1,1):+.1f},{round(c2,1):+.1f}"
                    by_path[rel] = dict(path=rel, c1=round(c1, 5), c2=round(c2, 5),
                                        source="scaffold", group=f"scaffold:{fam}:{band}")
                    if k % 10 == 0 or k == 1:
                        el = (time.time() - t0) / 60
                        print(f"  [{k}/{n_total}] {fam:14s} c1t={c1t} t={t:+.2f} -> "
                              f"({c1:.2f},{c2:+.3f})  el={el:.1f}m", flush=True)
                        with manifest.open("w", newline="") as f:   # periodic flush
                            w = csv.DictWriter(f, fieldnames=["path", "c1", "c2", "source", "group"])
                            w.writeheader(); w.writerows(by_path.values())
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "c1", "c2", "source", "group"])
        w.writeheader(); w.writerows(by_path.values())
    scaf = [r for r in by_path.values() if r["source"] == "scaffold"]
    c2s = np.array([r["c2"] for r in scaf])
    print(f"\nscaffold corpus: {len(scaf)} rows, c2 mean {c2s.mean():+.3f} "
          f"[{c2s.min():+.2f},{c2s.max():+.2f}], strong(<=-0.5)={int((c2s<=-0.5).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--substrate", default=None, help="free-text substrate prompt")
    ap.add_argument("--prompts-module", default=None)
    ap.add_argument("--family", default=None, help="family name from --prompts-module")
    ap.add_argument("--c2-target", type=float, default=-0.7)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--strength-sweep", default=None,
                    help="comma-separated strengths to sweep, e.g. 0.3,0.45,0.6,0.75")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "scaffold_out"))
    # corpus-enrichment mode: generate a labelled batch across substrates x c2 targets
    ap.add_argument("--corpus", action="store_true",
                    help="generate a labelled batch and append to corpus/manifest.csv "
                         "(source=scaffold) -- enriches the strong-c2 end for retraining")
    ap.add_argument("--corpus-targets", default="-1.1,-0.9,-0.7,-0.5,-0.3",
                    help="scaffold c2 targets to sweep (strong-weighted)")
    ap.add_argument("--corpus-c1-targets", default=None,
                    help="if set, joint-fill: sweep c1 x c2 to cover the whole plane, "
                         "e.g. '0.7,1.0,1.3,1.6,1.9'")
    ap.add_argument("--corpus-seeds", type=int, default=3)
    ap.add_argument("--corpus-strength", type=float, default=0.5)
    ap.add_argument("--corpus-families", default="prompts_intricate",
                    help="prompt module whose families become the substrates")
    args = ap.parse_args()

    if args.corpus:
        return build_scaffold_corpus(args)

    # resolve the substrate prompt
    substrate = args.substrate
    if substrate is None and args.prompts_module and args.family:
        import importlib
        pm = importlib.import_module(args.prompts_module)
        substrate = pm.PROMPTS[args.family][0]
    if substrate is None:
        ap.error("give --substrate TEXT or --prompts-module M --family F")
    prompt = substrate + TAIL
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_img, c2_scaffold = scaffold_init(args.c2_target, args.res, args.seed)
    init_img.save(out / f"scaffold_t{args.c2_target:+.2f}.png")
    print(f"scaffold target c2={args.c2_target:+.3f}  measured={c2_scaffold:+.3f}")

    from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device); pipe.set_progress_bar_config(disable=True)

    strengths = ([float(x) for x in args.strength_sweep.split(",")]
                 if args.strength_sweep else [args.strength])
    rows = []
    print(f"substrate: {substrate[:60]}...")
    for s in strengths:
        g = torch.Generator(device=device).manual_seed(args.seed)
        img = pipe(prompt=prompt, negative_prompt=NEG, image=init_img, strength=s,
                   num_inference_steps=args.steps, guidance_scale=args.cfg,
                   generator=g).images[0]
        c2_out = measure_c2(img)[1]
        fp = out / f"t{args.c2_target:+.2f}_s{s:.2f}.png"
        img.save(fp)
        err = c2_out - args.c2_target
        rows.append(dict(c2_target=args.c2_target, strength=s, c2_scaffold=round(c2_scaffold, 3),
                         c2_out=round(c2_out, 3), err=round(err, 3)))
        print(f"  strength {s:.2f} -> output c2 {c2_out:+.3f}  (err {err:+.3f})  {fp.name}")

    with (out / "scaffold_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["c2_target", "strength", "c2_scaffold",
                                          "c2_out", "err"]); w.writeheader(); w.writerows(rows)
    best = min(rows, key=lambda r: abs(r["err"]))
    print(f"\nbest: strength {best['strength']:.2f} -> c2 {best['c2_out']:+.3f} "
          f"(target {args.c2_target:+.2f}, err {best['err']:+.3f})")
    print("As strength rises, SDXL overrides the scaffold and output c2 drifts toward the "
          "substrate's native c2; low strength keeps c2 near target but looks more procedural.")


if __name__ == "__main__":
    main()
