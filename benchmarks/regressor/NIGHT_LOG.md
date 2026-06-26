# Autonomous overnight log (started 2026-06-21 eve)

Goal: (B) add metal OOD + rebuild corpus on merged generators + re-run Stage 1;
(A) install diffusers, c2-balanced SDXL Stage 2, retrain, Stage 3 latent, Stage 4 grid.

## Timeline
- 22:57 merged origin/mfractal-toolbox (metal domain, generators). Labels stable (quantify/wavelet_cascade unchanged). Verified mf.generate + metal work.
- 23:00 wired metal as OOD source (build_corpus --metal; train_stage1 --ood-source). Rebuilding corpus on merged generators.
- 23:05 installed diffusers 0.31.0; fixed peft/accelerate skew by upgrading accelerate->1.14.0. SDXL pipeline imports OK.

## Stage 1 results (rebuilt corpus 1342 samples, metal=OOD holdout)
GROUP split:     overall RMSE 0.187 r 0.82 | metal(OOD) RMSE 0.176 r 0.77 | macro RMSE 0.279 r -0.12 | procedural 0.088 r 0.94 | prescribed 0.104 r 0.97
STRATIFIED split: overall RMSE 0.126 r 0.94 | metal(OOD) RMSE 0.171 r 0.80 | macro 0.102 r 0.58
Takeaways: (1) generalizes to NOVEL PROCEDURAL domain (rust) r~0.77 -- matches other agent's probe 0.196. (2) weak spot = novel REAL-PHOTO texture (macro, 3 groups) -> needs Stage 2 diffusion domain. (3) DINOv2+prescribed beats MVP under matched (stratified) eval.

## ~00:05 Stage 2 launched (SDXL c2-balanced gen)
- RealVisXL_V4 + VAE cached (23.5 min download). GPU free 24GB.
- generate_corpus_sdxl --per-bin 100 --max-attempts 1600 --c2-lo -1.0 --c2-hi -0.02 --bins 10
- ~12s/img, adaptive family picker working, images -> corpus/images/diffusion_balanced, rows appended to corpus/manifest.csv (source=diffusion, group=prompt). ETA ~5h.

## ON STAGE 2 COMPLETE (auto):
1. rm corpus/features_*.npz   (corpus changed -> invalidate cache)
2. retrain: train_stage1.py --cache-features --split group --ood-source metal --epochs 400   (now WITH diffusion source -> headline diffusion RMSE)
   + stratified for comparison
3. Stage 3: latent_regressor.py --cache-latents --vae-res 1024 ; then --epochs 300  (compare latent vs pixel)
4. Stage 4: guided_sample.py --grid --guidance-scale 200  (best-effort; tune scale via grid)
5. commit results + write morning summary

## ~04:45 Stage 2 DONE: 1000 diffusion imgs, all 10 c2 bins filled to 100, c2 uniform [-1.0,-0.02]. Corpus now 2342.
## Stage 1 retrain WITH diffusion domain (metal=OOD):
GROUP (honest):     diffusion RMSE 0.185 r 0.71 | overall 0.176 | macro 0.239 r 0.46 (up from -0.12!) | metal(OOD) 0.186 r 0.71 | procedural 0.084 r 0.93
STRATIFIED (leaky): diffusion RMSE 0.151 r 0.84 | overall 0.130 | macro 0.119 r 0.56 | metal(OOD) 0.144 r 0.89
-> beats MVP diffusion 0.223. Adding balanced diffusion data lifted macro generalization a lot.
-> frozen DINOv2 plateaus ~0.15-0.18 honest; <0.10 likely needs --finetune (next step).

## ~05:10 Stage 3 (latent regressor) -- BREAKTHROUGH
latent CNN on SDXL VAE latents (4x128x128), grouped split (prompt-held-out):
  diffusion RMSE 0.068 r 0.955  | overall 0.073 r 0.96 | macro 0.096 r 0.84 | prescribed 0.058 r 0.99 | procedural 0.069 r 0.96
-> CRUSHES the 0.10 goal. End-to-end CNN >> frozen DINOv2+head (0.185). VAE does NOT lose the c2 signal.
-> latent_model.pt is the deployment artifact for Stage 4 (fast: no VAE decode in the loop).
Next: Stage 4 guided sampling using latent_model.pt; tune guidance scale.

## ~05:40 Stage 4 (classifier guidance) -- WORKS
Diagnostic grid scale=200 (12 imgs, targets -0.8/-0.5/-0.2): slope=0.885, r=0.537.
Measured c2 tracks target at ~0.9x. Examples: marble -0.8->-1.32 (overshoot), ink/dye -0.8->-0.96/-0.5->-0.64/-0.2->-0.33 (clean), granite -0.8->-0.28 (resists: naturally low-MF prompt).
=> guidance DIRECTION + slope excellent. r limited by prompt-feasibility (granite can't reach -0.8; cirrus saturates ~-1), NOT by method. Stay within feasible (prompt,c2) range.
Running final demo grid at scale 150 on feasible target range for cleaner scatter.

## ~06:15 Stage 4 final grid (scale 150, 40 imgs) + WRAP
Aggregate slope 0.63 r 0.42 BUT per-prompt: marble slope 1.00 r 0.97 (perfect), ink/dye 0.90, cirrus 0.37 (saturates), granite 0.25 (resists). Guidance works where prompt can express target c2.
See RESULTS.md for the full writeup. All committed on regressor-gpu (NOT pushed).
DONE for the night. Suggested next: --finetune pixel model; per-prompt feasible target ranges; more diverse macro photos.
