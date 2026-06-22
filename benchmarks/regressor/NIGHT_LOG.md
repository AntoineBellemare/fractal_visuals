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
