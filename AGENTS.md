# mfractal — agent guide

Procedural generator of **multifractal pareidolia stimuli** across three natural
domains (rocks, clouds, liquids), plus the code to **measure** their multifractality
and to **build/validate datasets**. Grayscale fields in [0, 1] unless noted. Pure
Python (numpy, scipy, matplotlib; pywavelets optional for the c2 validator).

## Quickstart
```python
import mfractal as mf
img  = mf.marble(512, seed=1, complexity=0.7)          # one stimulus, complexity in [0,1]
disp = mf.punch(img)                                   # percentile-clip + tanh for display
c    = mf.wavelet_leaders_2d(img)                       # {'c1':..., 'c2':...}  c2<0 = multifractal
mf.build_dataset('out/', families=mf.ROCK_FAMILIES, complexities=(0,.25,.5,.75,1),
                 seeds=range(4), n=256, estimator='wavelet')   # writes images/ + manifest.csv
mf.validate_dataset('out/manifest.csv', out_fig='val.png', value='c2')  # per-family c2 vs complexity
```

## Domains (all families: `family(n=512, seed=None, complexity=0.5)`)
- **ROCK_FAMILIES** (15): marble, agate, weathered, granite, cellular_stone, gneiss,
  concentric, breccia, vesicular, convoluted, turbulent, flow_banded, schist,
  serpentinite, veined.
- **CLOUD_FAMILIES** (8): cascade_lognormal, stratified, billow, cirrus, ridged,
  warped_fbm, cloud_mrw, cloud_multifractional.
- **FLUID_FAMILIES** (7): eddies, vorticity, plume  (real Navier-Stokes sims, take
  only (n, seed)) + curl_weave, dye_diffusion, rheoscopic, choppy. The four
  procedural fluids are in **CX_FLUIDS** and take `complexity`; the NS sims do not.

`mf.generate(family, n, complexity, seed)` dispatches across all three domains.

## Flagships (mfractal.flagships)
- `multifractal_cloud(n, granularity, multifractality, smooth, seed, raw)` — 2D MRW,
  genuinely multifractal cloud field.
- `multifractional(n, fd_center, fd_range, scale, seed)` — within-image varying FD.
- `punch(field, low, high, strength)` — display transform.

## Measuring multifractality (mfractal.quantify)
- `wavelet_leaders_2d(image, wavelet='db3')` -> log-cumulants **c1** (~dominant Holder
  exponent / spectrum peak) and **c2** (intermittency: c2≈0 monofractal, more negative
  = more multifractal). This is the ROBUST, monotonic readout — prefer it.
- `mfdfa_2d`, `moments_2d`, `legendre`, `spectrum_summary` (-> delta_alpha, alpha0),
  `local_fd_map`, `heterogeneity`.

## VALIDATION PHILOSOPHY  (important)
`complexity` is a generative/visual control, NOT a guarantee of multifractality.
Always bin experimental stimuli by **measured c2**, not nominal complexity.
- Monofractal anchors (c2≈0 by design): **gneiss** (rock), **warped_fbm** and
  **ridged** (clouds), **rheoscopic** (fluid). Use these as controls.
- Strong multifractal axes: cloud_mrw, stratified, choppy, cascade_lognormal,
  marble/serpentinite/granite (rocks), curl_weave.
- Some families inflate at the SPARSE/low-complexity end (dye_diffusion, ridged):
  the large |c2| there is a degenerate few-regions artifact, not real structure.

## cloudscape() — photorealistic renderer (NOT a validated family)
`mf.cloudscape(n, seed, coverage, puff, n_clusters, sky, sun)` -> (n,n,3) RGB.
Stacked-sphere (metaball) cumulus with diffuse + AO lighting over a sky gradient.
`sky` in {'day','golden','overcast'} (mf.SKY_PALETTES). Aesthetic output only; for
controlled stimuli use the abstract `clouds.py` families.

## Module map
generators.py (fBm/normalize core) · flagships.py · textures.py (rocks) ·
clouds.py · fluids.py · cloudscape.py (renderer) · quantify.py (c1/c2, MFDFA) ·
dataset.py (build/validate/generate) · organic.py · stimuli.py · demo.py · README.md

## Open threads (suggested next work)
1. **c2-balanced cross-domain emitter**: target a grid of measured-c2 bins across
   rocks+clouds+fluids, seed-search each image into its bin, emit a balanced
   c2-matched set + manifest with the monofractal anchors included.
2. **Natural color** per domain (rocks have colorize_natural; add sky/water palettes).
3. **New domains** (designed, not built): terrain/landscape relief, fire & smoke,
   night sky / cosmic web, frost/ice dendrites, vegetation/bark, wood grain, stains.
   Terrain, fire/smoke, and night sky are the highest-value untapped families.
