# Plan: MF-ControlNet

The 320-image diffusion sweep (this folder) showed that:

- the model RealVisXL_V4.0 can produce photoreal natural textures at ~8 s/image
  on a 16 GB GPU,
- the c2 of those textures ranges from ~−1.0 (cirrus, dye_diffusion) to ~0
  (gneiss, riven_oak), spanning the relevant intermittency space,
- but the *procedural toolbox's* c2 only weakly predicts the *diffusion model's*
  c2 (Spearman ρ ≈ +0.14 across families) — i.e. asking the diffusion model for
  "marble" or "cumulus" is not a c2-control knob.

The next step is to train a ControlNet that takes a multifractal conditioning
signal and produces photoreal natural-texture images at the requested
intermittency.

## Why the current 320-set is not enough

| dimension                | what we have       | what ControlNet needs |
|---|---|---|
| volume                   | 320 imgs           | 2 k (LoRA-CN floor) — 15 k (proper SDXL CN) |
| balance                  | family-stratified  | c2-stratified (~250 imgs per c2 bin)  |
| range                    | [-1.0, -0.02] ✓    | same span, but evenly filled  |
| diversity per c2         | low                | many distinct visual categories per c2 |
| conditioning signal      | global c2 scalar   | dense local map (e.g. `mfractal.local_fd_map` or sliding-window c2) |

The 320-set is a good **prior** on the prompt→c2 mapping. We can re-use it to
choose prompts that hit underrepresented c2 ranges in the next phase.

## Phase 1 — assemble a c2-balanced dataset (~8.5 k images)

**Path B: free baseline from DTD (~1 h, no GPU).**

- Download the Describable Textures Dataset (5640 photoreal texture images,
  47 categories). Resize to 1024 × 1024, convert to grayscale for analysis
  (keep RGB for training).
- Run `mfractal.wavelet_leaders_2d` on each → per-image c2.
- Bin into 20 bins across [-1.0, 0.0]. Inspect coverage.

**Path A: targeted SDXL generation (~8 h GPU).**

- Use the 320-set as a prompt → c2 prior. For each underrepresented bin,
  identify prompts that historically land there.
- Expand the prompt vocabulary beyond our 32 families:
  - more solid-surface categories near c2 ≈ 0 (polished metals, ice, plastic,
    fabric, paper, leather, calm water)
  - more fine-detail categories near c2 ≈ -0.8 to -1.0 (lichen, frost, capillary
    veins, frost flowers, foam, smoke striations, dust nebulae)
  - mid-range fillers if needed
- Target ≈ 3 k generated images, c2-balanced across the 20 bins.
- Combine with DTD → ~ 8.5 k images total.

**Output:** `datasets/mf_natural/` with images + `manifest.csv` (path, source,
c2, c1, mean, std).

## Phase 2 — choose a conditioning signal

Two options for the per-image condition:

1. **Global scalar c2** — simplest. Token-injected via text encoder, or via a
   small MLP into a learned embedding. Lower expressiveness; easier to train.
2. **Dense local FD / c2 map** — compute `local_fd_map(image, window=64)`
   downsampled to a 64×64 single-channel image, used as the ControlNet
   conditioning input directly. Higher expressiveness; matches the ControlNet
   architecture's spatial-conditioning paradigm.

**Recommendation:** start with option 1 for a LoRA-CN proof of concept, then
move to option 2 for the full ControlNet.

## Phase 3 — train

- Base model: SDXL 1.0 (or RealVisXL_V4.0 — same backbone, photoreal init).
- Architecture: SDXL-ControlNet with the chosen conditioning signal.
- Training: ~6-12 h on the 4090 Laptop (LoRA-style ControlNet ≈ 6 h, full
  ≈ 12 h).
- Validation: hold out ~10 % of the 8.5 k. Measure c2 on generated outputs
  for each conditioning value. Target Spearman ρ(condition c2, output c2) > 0.8.

## Phase 4 — release artifact

- ControlNet weights (LoRA or full).
- Demo notebook: condition → image at a chosen c2 sweep.
- Per-family qualitative comparison: procedural at cx → diffusion at matching
  measured-c2.
- Update README roadmap section with results.

## Open questions for Bounzaï

- **Target use case** for the ControlNet — is it for the pareidolia stimulus
  design (need precise c2 control), or for general generative art (just need
  the c2 knob to work)? This affects how tight the validation bar should be.
- **Local vs global conditioning** — pick a side, or run both as a comparison?
- **Backbone** — stick with SDXL/RealVis, or revisit FLUX now that we know
  the pipeline works?
- **Volume budget** — 8.5 k is the recommended floor; we could go to 15 k for
  more confidence at +5 h GPU.

## Resource estimate

| step | wall time | GPU | disk |
|---|---|---|---|
| DTD download + c2 measurement (Phase 1B) | ~ 1 h | none | ~ 5 GB |
| Targeted diffusion generation (Phase 1A) | ~ 8 h | yes | ~ 6 GB |
| LoRA-ControlNet training (Phase 3) | ~ 6 h | yes | ~ 1 GB |
| Full SDXL-ControlNet training (Phase 3) | ~ 12 h | yes | ~ 3 GB |
| **Total to LoRA-CN deliverable** | **~ 15 h** | | **~ 12 GB** |
| **Total to full-CN deliverable** | **~ 21 h** | | **~ 14 GB** |

Hostspace currently has ~ 60 GB free (94 % used). Cleanup or external storage
might be sensible before kicking off the full plan.
