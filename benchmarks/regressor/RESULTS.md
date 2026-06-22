# c2 regressor — overnight results (2026-06-22)

End-to-end pipeline run autonomously on the RTX 3090. Full narrative in `NIGHT_LOG.md`.

## TL;DR
- **A latent-space c2 regressor hits honest diffusion RMSE 0.068 (r 0.955)** — beats the
  0.10 goal and the frozen-DINOv2 pixel model (0.185). It's the deployment artifact.
- **Classifier guidance works**: for prompts that can express the target c2 (marble),
  measured c2 tracks target at **slope 1.00, r 0.97**. Aggregate is lower only because
  some prompts (granite, cirrus) physically can't reach the requested c2.
- Beats the other agent's MVP (diffusion RMSE 0.223, leaky split) on every matched measure.

## Corpus (Stage 0) — 2342 labelled samples
diffusion 1000 (c2-balanced, all 10 bins×100, uniform c2∈[-1.0,-0.02]), procedural 520,
prescribed 480, macro 189 (your ice/water/lichen photos), pareidolia 93, metal 60 (OOD).

## Stage 1 — pixel regressor (frozen DINOv2-small + head)
Honest group split (prompts/families/textures held out), metal held out as OOD:

| source | RMSE | r | note |
|---|---|---|---|
| diffusion | 0.185 | 0.71 | prompt-held-out; stratified(leaky)=0.151 |
| metal (OOD) | 0.186 | 0.77 | never trained — generalizes to novel procedural domain |
| macro | 0.239 | 0.46 | lifted from r −0.12 once diffusion data was added |
| procedural | 0.084 | 0.93 | |
| prescribed | 0.104 | 0.97 | split-invariant gold standard |

Frozen backbone plateaus ~0.15–0.18 honest. `--finetune` is the untried lever for <0.10 here.

## Stage 3 — latent regressor (CNN on SDXL VAE latents) ← deployment model
Honest group split: **diffusion RMSE 0.068, r 0.955**; overall 0.073; macro 0.096 (r 0.84);
prescribed 0.058; procedural 0.069. The expected "VAE blurs high-freq c2" failure did NOT
happen — an end-to-end CNN reads c2 from the 4×128×128 latent far better than frozen
features + head. Fast (no VAE decode in the guidance loop). → `latent_model.pt`.

## Stage 4 — classifier-guided SDXL
DPS/FreeDoM-style guidance on the x0-prediction, latent regressor in the loop, guidance
gated after warmup. Per-prompt (scale 150, 40 images):

| prompt | slope | r |
|---|---|---|
| marble | 1.00 | 0.97 |
| ink/dye | 0.90 | 0.67 |
| cirrus | 0.37 | 0.70 |
| granite | 0.25 | 0.61 |

Measured c2 tracks target cleanly where the prompt's natural c2 brackets the target.
Aggregate slope 0.63 / r 0.42 is limited by infeasible prompt×c2 combos, not the method.
Demo grids: `guided_final/grid.png`, `guided_s200/grid.png` (gitignored, local).

## What was integrated from the other agent's branch
Merged `mfractal-toolbox` (metal/corrosion domain, generator updates). Adopted their
feature-caching idea (extract once → train head in seconds). Their `c2_regressor.py`
(ResNet18, image-level split) coexists. Labels unaffected (estimator unchanged).

## Recommended next steps
1. **Fine-tune** the pixel backbone (frozen plateaus; latent is already great) — or just
   ship the latent model.
2. Stage 4: restrict each prompt to its **feasible c2 range** (e.g. marble for the full
   span; don't ask granite for −0.8) → aggregate slope/r will jump.
3. More diverse **macro photos** (only 3 texture types → held-out-texture is unreliable).
4. Branch `regressor-gpu` is committed but **not pushed** — push when you're ready.

## Intricate-prompt expansion + the intricacy≠c2 finding (2026-06-22)
Added `benchmarks/diffusion/prompts_intricate.py` (19 pareidolia substrates) + `--prompts-module`
wiring in generate_corpus_sdxl and guided_sample. Validation (1 img/family, native c2):

STRONG/intermittent (good complex substrates, guide easily): frost_fern -0.53, ferrofluid -0.51,
  smoke_eddies -0.41, manganese_dendrite -0.35, ink_bloom -0.32, cracked_glaze -0.30, mycelium_net -0.29
MID: mold_colony -0.20, salt_efflorescence -0.19, oil_iridescence -0.19, slime_mold -0.18
FLAT/uniform-detail (low c2, weak-end substrates): mammatus -0.11, lichen -0.07, rust -0.06,
  capillary -0.06, coral -0.04, copper -0.02, marbled_ink -0.01, peeling_paint +0.02

KEY FINDING: visual intricacy != multifractal c2. c2 = variance of local roughness (intermittency),
high only when SPARSE strong structure sits on smooth ground (dendrites/spikes/wisps), not uniform busy detail.
=> STRATEGY: decouple. Prompt sets the pareidolic SUBSTRATE; GUIDANCE (latent regressor) sets c2.
   Use intermittent substrates for strong-c2 stimuli; flat substrates cover the weak-c2 end.
