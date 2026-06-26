# c2 regressor

Train an image→c2 regressor accurate enough to classifier-guide SDXL toward a target
multifractal intermittency (c2). See [PLAN.md](PLAN.md) for the full roadmap and the
record of design corrections.

## Quickstart (CPU for Stage 0, GPU for Stage 1)

```bash
# 0. Build the labelled corpus (run where the diffusion PNGs live; they are gitignored).
python build_corpus.py --all                       # procedural+prescribed+diffusion+pareidolia
python build_corpus.py --macro /path/to/photos --append   # your real macro textures
python check_c2_invariance.py                      # confirm the augmentation set on your data

# 1. Train DINOv2 + head (needs torch + transformers + a GPU).
python train_stage1.py --epochs 300

# 2. c2-balanced SDXL data (rejection sampling into under-filled c2 bins), then retrain.
python generate_corpus_sdxl.py --per-bin 150
python train_stage1.py --epochs 300

# 3. Latent-space regressor (cache VAE latents, train, compare to pixel model).
python latent_regressor.py --cache-latents --vae-res 1024
python latent_regressor.py --epochs 300

# 4. Classifier-guided sampling + the measured-vs-target validation grid.
python guided_sample.py --grid --guidance-scale 200
```

Stages 2–4 need `diffusers` + SDXL/VAE weights (and RealVisXL_V4.0) in addition to the
Stage-1 deps. Stage 4's `--guidance-scale` always needs tuning — run `--grid` first.

Outputs: `corpus/manifest.csv` (the labels — tracked), `corpus/images/*.npy` (fields —
gitignored), `pixel_model.pt`, `metrics_v2.json`.

## The one invariant to respect

**c2 is measured on the exact field the network is trained on, at one fixed resolution,
and only flip/rotate augmentations are applied.** Everything else (8-bit quantization of
HDR fields, crops, resizes, brightness/contrast, `punch`) measurably moves c2 and turns
the regression target into noise — `check_c2_invariance.py` quantifies each. Generated
fields are therefore stored as float `.npy`; only genuine photos are read as 8-bit.

## Dependencies

`build_corpus.py` / `check_c2_invariance.py`: numpy, scipy, pywt, pandas, Pillow, mfractal
(CPU only). `train_stage1.py`: + torch, transformers (GPU).
