# Percept embedding — confirmed strategies

CPU-only routes for weaving an *embedded percept* (face, tree, figure) into a
controlled-multifractality texture, such that the result both (a) sits at a
prescribed (c1, c2) and (b) shows the percept. The ambiguity band — percept
present enough to recognize, woven enough to stay *in* the texture — is the
pareidolia stimulus.

Full design menu + the parked GPU routes: `../regressor/PLAN_embedded_percepts.md`.
This folder holds what **survived empirical prototyping**.

## Files
- `embed.py`     — the library (the two good routes + c1-lock). Importable.
- `percepts.py`  — procedural percept generators (face, tree). Swap in any
                   (n,n) [0,1] image as a percept.
- `demo.py`      — reproduces both figures. `python benchmarks/percept_embedding/demo.py`
                   (writes to `.attachments/percept_embedding/`, gitignored).

## The key fact that organizes everything
c2 is read off the **wavelet-leader magnitude distribution across scales**, and
the fine-scale **phase correlations** carry it. So any operation that rewrites
fine-scale phase or contrast *fights* c2; operations that preserve each scale's
magnitude marginal (and leave the finest scales alone) keep it. That single fact
predicts every result below.

## ✅ Flagged good

### S1 — luminance (first-order)  `luminance_embed(F, P, eps)`
Percept rides a gentle low-frequency brightness modulation of the base field.
- **Visible:** yes, trivially. **c1/c2:** essentially untouched (Δc2 ≈ +0.004).
- **Caveat:** it's a *luminance* cue, not a texture cue — the percept is "painted
  on" in brightness. Perfectly valid if a first-order percept is acceptable.

### P3 — wavelet-subband + c1-lock (second-order)  `embed_percept(F, P, beta)`
Per coarse/mid wavelet subband, the fractal field's coefficient **magnitudes are
reordered spatially** to follow the percept's energy (the magnitude marginal is
preserved *exactly* → leader stats, hence c2, preserved); fractal signs kept;
finest 2 levels left pure fractal. Then `lock_c1` restores the base c1 with an
isotropic radial spectral tilt `k^{-δ}` (isotropic → doesn't move the percept;
deterministic per-scale → shifts c1 without disturbing c2).
- **Visible:** yes — and it's a genuine *texture-structural* percept, not a
  brightness overlay.
- **c1:** locked to the base to ≤0.003. **c2:** follows the base across the whole
  complexity arc (small residual, ~0.01 mild → ~0.04 at the rough end).
- **beta = presence dial.** beta≈0.25 already reads strongly.
- **Synergy:** percept visibility *scales with base complexity* — a richer
  cascade sculpts a sharper percept at the same beta. Multifractality and
  pareidolia reinforce rather than fight.

Measured (P3 @ beta=0.25, c1-locked):
```
cx     base c1/c2        FACE embed       TREE embed
0.1    1.51 / -0.158     1.51 / -0.144    1.51 / -0.148
0.3    1.45 / -0.205     1.45 / -0.192    1.45 / -0.201
0.5    1.40 / -0.277     1.40 / -0.251    1.40 / -0.266
0.7    1.36 / -0.382     1.35 / -0.345    1.36 / -0.361
0.9    1.32 / -0.510     1.32 / -0.462    1.32 / -0.471
```

## ❌ Dead ends (don't re-run these)
- **S2, texture-defined via spatial compositing** (two fields, same c1 different
  c2, histogram-matched, composited through a mask). Pure version is *invisible*;
  forcing visibility (preserving contrast / scale-contrast) shifts global c1/c2
  hard (contrast-defined face c1 jumped to 2.39). The route that worked is doing
  the same *idea* in the wavelet domain (P3), not the spatial domain.
- **Fourier phase embedding** (fractal amplitude + blended percept phase). Percept
  becomes visible but **c2 collapses to ~0** — blending global phase destroys the
  fine-scale phase correlations c2 lives in. Exactly the failure the key-fact
  predicts.

## Parked (need GPU / assets not in repo)
From `../regressor/PLAN_embedded_percepts.md`, Tier 2/3:
- Compositional diffusion guidance (`../regressor/embed_percept.py`) — needs the
  latent c2 regressor `model.pt` / `latent_model.pt` (on Bounzaï's desktop).
- Dual-condition MF-ControlNet — needs training. Not built.

## Next steps for the continuing agent
1. Try P3 on **real** B&W percepts (photos) and on the mineral/bark texture
   families, not just the synthetic face/tree.
2. The c2 residual at high complexity (~0.04) is from the reorder tempering
   extreme intermittency — worth a c2-lock analogous to the c1-lock if exactness
   matters.
3. Decide whether to promote `embed_percept` into the `mfractal` package as a
   first-class API once the percept-loading interface is settled.
4. Map the full **beta × complexity** operating envelope to pick per-use defaults.
