# Diffusion-generated natural-texture sweep

A SDXL fine-tune (RealVisXL_V4.0) generates 5 prompt variations x 10 images
per family for all 32 families in the toolbox. Multifractality (c1, c2) is
measured on every output with the same `mfractal.wavelet_leaders_2d`
estimator used in the procedural sweep, so the two are directly comparable.

## Files

```
prompts.py    32 families x 5 prompts, plus the photorealism prefix/suffix
generate.py   SDXL pipeline + planner + saves PNG + manifest.csv
analyze.py    measure c1/c2 per image, build figures + REPORT.md
out/          generated images (gitignored), manifest.csv, analysis.csv, report/
```

## Reproducing

Requires a 12+ GB CUDA GPU and a fresh venv with torch / diffusers /
transformers / accelerate / pillow installed.

```bash
# 1. Generate (320 images, ~45 min on a 4090 Laptop)
python benchmarks/diffusion/generate.py --images-per-family 10

# 2. Measure + report (~5 min)
python benchmarks/diffusion/analyze.py
```

`generate.py` is idempotent: if `out/<family>/<prompt_idx>_s<seed>.png`
exists it is skipped. To regenerate, delete the files.

## Outputs

After `analyze.py`:

- `out/analysis.csv` -- per-image c1, c2, mean, std (320 rows)
- `out/report/per_family_c2.png` -- violins of c2 per family, colored by domain
- `out/report/domain_c2.png` -- c2 by domain
- `out/report/diffusion_vs_procedural.png` -- diffusion mean c2 vs procedural
  mean c2 per family (does the diffusion model land in the same intermittency
  ballpark as the procedural generator that bears the same name?)
- `out/report/REPORT.md` -- human-readable summary table + figure links
