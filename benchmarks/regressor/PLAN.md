# c2 regressor improvement plan — GPU phase

## Current state (CPU MVP, this commit)

- **Backbone:** frozen ImageNet ResNet18 (512-d features)
- **Head:** MLP 128 → 64 → 1, dropout 0.2
- **Training data:** 912 labeled images (486 procedural + 320 diffusion + 106 pareidolia)
- **Test metrics (80/10/10 split, stratified by source):**
  - overall Pearson r = **+0.88**, R² = **0.77**, RMSE = **0.149**
  - procedural-only: r = +0.97, RMSE = **0.07** (nailed)
  - diffusion-only:  r = +0.82, RMSE = **0.22** (bottleneck)
  - pareidolia-only: r = +0.59, RMSE = 0.14 (only n=10, noisy estimate)
- **Bottleneck:** val MSE plateaus at 0.22 from epoch 50 while train MSE keeps dropping — in-domain overfit + cross-domain gap. Classifier-guidance needs **diffusion RMSE < 0.10** to be a clean signal.

## Goal

A regressor good enough to drive classifier-guidance in SDXL with target c2 in [-0.5, -0.02].

Acceptance: diffusion-domain RMSE < 0.10, latent-space variant within 0.02 RMSE of the
pixel-space variant. End-to-end demo: (prompt × seed × c2_target) grid where measured c2 of the
SDXL output correlates with c2_target at slope ~0.8, r > 0.85.

## Stage 1 — better backbone with existing data (1–2h GPU)

**Why first:** zero new data, isolates whether the backbone is the bottleneck.

- Swap ResNet18 → **DINOv2-small** (`facebook/dinov2-small`, ~22M params, 384-d features).
  - Fallback: CLIP-ViT-B/32 (`openai/clip-vit-base-patch32`).
- Same MLP head architecture, retrain on existing 912 samples.
- Add augmentation (applied on every epoch, not cached):
  - Random crops 224 → variable (192–272) → resized back to 224
  - Brightness/contrast jitter ±10% (c2 invariant)
  - Horizontal + vertical flip (c2 invariant)
  - 90°/180°/270° rotation (c2 invariant)
  - Effective dataset size ~8x
- Stronger regularization: dropout 0.3, weight_decay 1e-3, early stopping on val MSE.
- 200–400 epochs with cosine LR schedule (1e-3 → 1e-5).

**Pass criterion:** diffusion-domain RMSE drops below **0.15**. If yes, backbone is doing real work; proceed to Stage 2. If no, jump straight to Stage 2 (data, not arch, is the limiter).

## Stage 2 — generate more diffusion training data (4–6h GPU)

**Why:** 320 diffusion images is the load-bearing limit. The diffusion-domain visual variance is ~10x richer than the procedural domain.

- Generate **~3000–5000 SDXL images** unconditionally:
  - ~150 diverse prompts (reuse `benchmarks/diffusion/prompts.py`, broaden with random categories: clouds, water, marble, fire, foam, sand, ice, lichen, fabric, leather, foliage, asphalt, etc.)
  - ~20 seeds per prompt
  - Vary CFG (5, 7, 9) and steps (30, 50) to broaden the c2 distribution naturally
- Label each with `mfractal.wavelet_leaders_2d` at 512×512 (matches procedural sweep convention). CPU; ~30 min.
- Train at 1024×1024 native (SDXL output res) → DINOv2 expects 224 — center-crop then random-crop in training.
- Merge with the existing 912 → ~4000–6000 labeled samples; 80/10/10 stratified split as before.
- Retrain backbone+head with the same Stage-1 recipe.

**Pass criterion:** diffusion-domain RMSE < **0.10**. If hit, regressor is production-ready.

### Stage 2b — OOD validation (out-of-distribution generalization probe)

**Why:** the regressor is a function of *pixels*, not prompts, but it only sees the visual distribution shaped by the natural-texture training prompts. Before committing to classifier-guidance we want to know how far it transfers beyond that domain — and whether the natural-texture prompt corpus actually constrains the system at sampling time.

- Generate ~50 SDXL images spanning categories *outside* natural textures:
  - Portraits (10) — faces in studio / outdoor light
  - Landscapes with structure (10) — mountains, cityscapes, architecture
  - Still-life objects (10) — fruit, glassware, books
  - Indoor scenes (10) — kitchens, libraries, workshops
  - Stylized/abstract photorealism (10) — long-exposure light trails, motion blur, fog
- Label with `wavelet_leaders_2d` (CPU, ~5 min).
- Run the Stage-2 regressor on this set. Report Pearson r and RMSE, compared against the natural-texture test set.

**Decision matrix:**

| OOD RMSE / in-domain RMSE | Interpretation | Action |
|:---:|:---|:---|
| < 1.5x | Regressor generalizes well — low-level texture features carry across content types. | Stick with natural-texture training. Classifier-guidance will work broadly. |
| 1.5–3x | Partial generalization — reliable on texture-heavy content, noisy on semantically-rich content. | OK for natural-texture stimulus research. Broaden corpus only if downstream use needs broader steering. |
| > 3x | Regressor is content-coupled — gradient is unreliable outside natural textures. | Stage 2c: expand prompts to ~500 spanning textures + scenes + objects + faces. ~3-5x compute. |

Cheap (~30 min total). High information value — tells us whether the prompt corpus is a fundamental limit on the classifier-guidance system or just shapes the convenient sweet spot.

## Stage 3 — latent-space regressor (2–3h GPU)

**Why:** at SDXL sampling time, decoding the latent through the VAE every denoising step is slow (~50 steps × VAE decode + backward). A regressor that operates **directly on the SDXL VAE latent** (4×128×128 for 1024×1024 output) skips the decode entirely. This is the architectural call that makes classifier-guidance affordable.

- Encode the Stage-2 training images through SDXL VAE (`stabilityai/sdxl-vae`, fp16) → 4×128×128 latents. Cache as `.npz`.
- Build a small CNN regressor in latent space:
  - Input: (4, 128, 128) latent
  - Architecture: 3–4 conv blocks (stride 2 each) + global avg pool + MLP head
  - ~2M params, fp16-friendly
  - Trained on (latent, c2) pairs using the same MSE loss
- Critical: pixel-space and latent-space regressors trained on the same c2 labels, so they're directly comparable.

**Pass criterion:** latent-space RMSE within **0.02** of pixel-space RMSE. If yes, latent-space is the deployment artifact. If the gap is bigger, the VAE compression is losing texture info — fall back to pixel-space + VAE decode at sampling time (slower but works).

## Stage 4 — classifier-guidance integration (3–4h GPU)

**The actual experiment.** Hook the latent regressor into SDXL's denoising loop:

```python
# At each denoising step t:
with torch.enable_grad():
    latent.requires_grad_(True)
    eps = unet(latent, t, prompt_embeds)
    x0_pred = (latent - sigma_t * eps) / alpha_t      # predicted clean latent
    c2_pred = regressor(x0_pred)
    loss = ((c2_pred - c2_target) ** 2).mean()
    grad = torch.autograd.grad(loss, latent)[0]

eps_guided = eps + guidance_scale * sigma_t * grad
# proceed with eps_guided
```

- Tunables: `c2_target ∈ [-0.5, -0.02]`, `guidance_scale ∈ [50, 500]` (typical classifier-guidance range, will need to be tuned), prompt, seed.
- Use a deterministic sampler (DDIM or Euler) so the same (prompt, seed, c2_target) is reproducible.
- Validation experiment: pick 4 prompts × 3 seeds × 5 c2_targets = 60 images. Measure c2 of each output. Plot measured c2 vs target c2 — should be near-linear with slope > 0.7, r > 0.85.

**Failure modes to watch:**
- Guidance too weak: output c2 doesn't shift. Crank `guidance_scale`.
- Guidance too strong: image distorts, oversharpened, loses photorealism. Back off `guidance_scale`.
- Out-of-distribution c2: e.g., c2 = -1 on a cumulus prompt may be infeasible — model can't produce that combination. Stay within the c2 range your training data covers.

## Compute budget

| Stage | Time   | GPU mem peak | Storage |
|------:|:-------|:-------------|:--------|
| 1     | 1–2 h  | ~6 GB        | + ~0    |
| 2     | 4–6 h  | ~12 GB       | + ~10 GB (5k SDXL PNGs) |
| 2b    | 0.5 h  | ~12 GB       | + ~0.5 GB (50 OOD SDXL PNGs) |
| 3     | 2–3 h  | ~10 GB       | + ~100 MB (latents cache) |
| 4     | 3–4 h  | ~12 GB       | + ~1 GB (demo grid)     |
| **Total** | **10–16 h** | within 16 GB | ~11 GB |

Fits comfortably on the 4090 Laptop (16 GB VRAM).

## Decision points

- After Stage 1: if diffusion RMSE ≥ 0.18 with DINOv2, the regressor architecture isn't the limiter — go straight to Stage 2 + retrain DINOv2 with the larger dataset.
- After Stage 2: if diffusion RMSE ≥ 0.12 even with 5k images, the regressor may need a more domain-adapted backbone (consider an SDXL-VAE-tuned encoder or a small from-scratch CNN at 512×512). Reconvene before committing more compute.
- After Stage 3: if latent gap > 0.05, fall back to pixel-space + VAE decode in the loop (3x slower at sampling time but works). If latent gap < 0.02, latent-space is the deployment path.

## What lives in this repo when done

```
benchmarks/regressor/
  c2_regressor.py             current CPU MVP (Stage 0)
  c2_regressor_v2.py          Stage 1: DINOv2 + augmentation (overwrite OK)
  generate_diffusion_corpus.py Stage 2: SDXL unconditional generation + c2 labeling
  latent_regressor.py         Stage 3: latent-space CNN regressor
  guided_sample.py            Stage 4: SDXL sampling with classifier-guidance
  features.npz                cached pixel features (Stage 1–2)
  latents.npz                 cached VAE latents (Stage 3)
  pixel_model.pt              best pixel-space regressor
  latent_model.pt             best latent-space regressor
  metrics_v{1,2,3,4}.json     per-stage metrics
  guided_demo.png             Stage 4 demo grid
```

## Open questions for Phil / Bounzaï

- **GPU access:** when can the hostspace 4090 take a 10–15 hour intermittent load? (Stages can be checkpointed between sessions.)
- **Prompt corpus:** keep restricted to `benchmarks/diffusion/prompts.py` (natural textures only) or broaden to anything photorealistic for richer c2 coverage?
- **SDXL variant:** stick with RealVisXL_V4 (current `prompts.py` is tuned for it) or try a base SDXL for cleaner generalization?
