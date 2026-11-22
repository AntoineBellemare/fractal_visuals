# fractal_visuals

A toolbox for generating and measuring **(multi)fractal natural textures** for
pareidolia research. It covers procedural generators in six domains (rock,
cloud, fluid, bark, fire, metal), SDXL generation guided toward a requested
fractal dimension (c1) and intermittency (c2), and the experimental stimulus
sets built from both.

The earlier monofractal (1/f) work, including the 300 FD stimuli of the online
pareidolia study (`legacy/images/FD12`, `FD14`, `FD16`), lives under
[`legacy/`](legacy/). The current toolbox lives in [`mfractal/`](mfractal/).

---

## What's inside

```
mfractal/                     procedural generators + multifractal estimators
benchmarks/run_validation.py  sweep all families x complexity x seed, measure c2 / Delta_alpha
benchmarks/make_figures.py    catalogs, c2-vs-complexity panels, cross-family scatter, ranks
benchmarks/pymultifracs_check.py  independent c2 cross-check via pymultifracs
benchmarks/results/           per-experiment CSVs (validation, guidance, atlases, gradients)
benchmarks/figures/           benchmark and results figures
benchmarks/diffusion/         SDXL texture sweep + c2 report
benchmarks/regressor/         c1/c2 regressors, guided SDXL, MF-ControlNet experiments, result docs
benchmarks/percept_embedding/ embed a percept into a texture at fixed (c1, c2)
datasets/build_pareidolia.py  candidate pool -> c2 binning -> balanced selection
datasets/pareidolia/          the experimental stimulus set (PNG + manifest.csv + montage.png)
datasets/reach_map.py         where c1 and c2 can be crossed, per domain
datasets/build_pareidolia_rock.py, make_twins.py   crossed c1 x c2 rock set + phase-scrambled twins
datasets/pareidolia_rock/     manifest of the crossed set (images regenerate, gitignored)
legacy/                       previous notebooks, FD-binned image archive, old-dev scripts
```

---

## Quickstart

```bash
pip install -r requirements.txt
# (pymultifracs is optional - only the cross-check script uses it.)
```

```python
import mfractal as mf

# Generate
img = mf.granite(512, seed=0, complexity=0.7)     # rock family, complexity in [0,1]
disp = mf.punch(img)                              # display transform (percentile clip + tanh)

# Quantify
wl = mf.wavelet_leaders_2d(img)                   # robust log-cumulants
print(wl["c1"], wl["c2"])                         # c1 ~ Hurst exponent; c2 << 0 means multifractal

# Dispatch any family by name
img = mf.generate("prescribed_cascade", n=512, complexity=0.5, seed=1)
```

The full generator + estimator reference lives in
[`mfractal/README.md`](mfractal/README.md); the agent-friendly summary is in
[`AGENTS.md`](AGENTS.md).

---

## Domains & curated families

The toolbox ships 35 families across six domains. Started as 30 (rock /
cloud / fluid), shrunk to 21 through empirical c2 validation and a
pixel-similarity prune, then grew to 35 with three new domains and additional
cloud generators added in 2026-06. The **35 curated** families below are the
current set.

| Domain | Curated families | Notes |
|---|---|---|
| **Rock** (11) | marble, agate, weathered, granite, vesicular, turbulent, schist, serpentinite, flow_banded, convoluted, **gneiss** | gneiss = monofractal anchor (rho ~ 0) |
| **Cloud** (10) | cascade_lognormal, billow, cirrus, cloud_multifractional, **warped_fbm**, ridged, billow_smoke, universal_cascade, prescribed_cascade, de_jong_attractor | warped_fbm = monofractal anchor; ridged is a structural-detail axis; billow_smoke = broad billowing smoke; universal_cascade = α-stable Lévy cascade; prescribed_cascade = (c1, c2) targeting (validated, c1 RMSE 0.12 / c2 RMSE 0.08); de_jong_attractor = chaotic-map orbit density (cosmic-dust / wispy reads) |
| **Fluid** (6) | curl_weave, eddies, vorticity, dye_diffusion, rheoscopic, wave_interference | eddies/vorticity = real 2D Navier-Stokes sims; dye_diffusion is sparse at low cx; rheoscopic is a near-monofractal flow-viz texture; wave_interference = sum of random plane waves with optional `crispness` knob |
| **Bark** (2) | burled_oak, riven_oak | intricate-knot wood surfaces; complexity grows knot density / cracks-per-anchor |
| **Fire** (3) | firestorm, lava_pool, volcanic_fissure | drastic spatial silhouettes: chaotic vortices / Voronoi pools / vertical cracks |
| **Metal** (3) | rust_bloom, rust_pitted, rust_dewy | corrosion morphologies: irregular oxidation blooms / dark pits punched through a ferric crust / bright ferric beads on a dark base. cx grows rust coverage / pit density / bead density |

**Dropped** from the original 30 after the c2 validation:
`plume` (estimator artifact, c2 ~ -8); `cellular_stone`, `concentric`,
`breccia` (inverted rho - more "complexity" = more uniform tiling); `veined`
(very high per-seed variance).

**Pruned** in 2026-06 after pixel-similarity audit
([`benchmarks/family_similarity.py`](benchmarks/family_similarity.py)):
`stratified`, `cloud_mrw`, `choppy`, `levy_marble` — each was a thin
post-processing layer over a cascade shared with another family and landed
at pixel correlation > 0.95 at matched seed.

**Added** in 2026-06 to broaden the multifractal vocabulary: three new
domains (bark, fire, metal — see table above) plus four new cloud-domain
generators (`universal_cascade`, `prescribed_cascade`, `de_jong_attractor`,
`wave_interference`). All additions cleared the same pixel-similarity check
(max cross-domain pair ≤ 0.78).

The complete cross-family scatter plot is at
[`benchmarks/figures/04_cross_family_summary.png`](benchmarks/figures/04_cross_family_summary.png)
and the per-family complexity-knob ranking at
[`benchmarks/figures/05_complexity_ranks.png`](benchmarks/figures/05_complexity_ranks.png).

---

## Multifractality estimators

Two complementary 2D estimators ship in [`mfractal.quantify`](mfractal/quantify.py):

- `wavelet_leaders_2d(image)` -> log-cumulants **c1** (~ dominant Holder exponent
  / Hurst) and **c2** (intermittency: c2 ~ 0 monofractal, more negative = more
  multifractal). The Wendt/Abry-style estimator - **robust and monotonic; prefer
  this**.
- `mfdfa_2d(image)` -> 2D multifractal detrended fluctuation analysis (Gu & Zhou
  2006). Pair with `spectrum_summary` -> **Delta_alpha** (singularity-spectrum
  width). Noisier than wavelet leaders, but the spectrum view is interpretable
  and useful for visualization.

Also: `moments_2d` (partition function for positive measures), `legendre`
(transform tau -> singularity spectrum), `local_fd_map` and `heterogeneity` for
the within-image FD-varying flagship.

### Independent cross-check via pymultifracs

`benchmarks/pymultifracs_check.py` runs the
[pymultifracs](https://github.com/neurospin/pymultifracs) 1D wavelet-leader
cumulants on rows and columns of each family's median sample as an independent
implementation. Across the 30 families at n=256, the two estimators agree
strongly: **Pearson r = +0.93, Spearman rho = +0.55** on c2. The off-scale
agreement on the plume outlier (both flag c2 ~ -8 to -3) and on the
monofractal controls (both flag c2 ~ 0 for gneiss and warped_fbm) is
particularly clean.

---

## Validation philosophy

A `complexity ∈ [0,1]` knob is a **generative** control of visual intricacy,
**not** a guarantee of multifractality. Two checks are baked in:

1. **Per-family rho(c2, complexity)** is the readout that says whether the
   knob actually drives multifractality. The ranking is in
   [`benchmarks/figures/05_complexity_ranks.png`](benchmarks/figures/05_complexity_ranks.png).
2. **Bin experimental stimuli by *measured* c2**, not by nominal complexity.
   The pareidolia dataset below uses measured c2 as the multifractal IV.

---

## The pareidolia stimulus dataset

`datasets/build_pareidolia.py` builds a c2-balanced cross-domain set:

1. Sweep the curated 25 families x 5 complexities x 4 seeds at n=512.
2. Measure c2 of each candidate; bin into five levels:
   monofractal . weak . mild . moderate . strong.
3. Round-robin select per (domain x bin), capped at 2 per family per bin for
   diversity.

Result: [`datasets/pareidolia/`](datasets/pareidolia/) - **106 stimuli, 14 MB**,
manifest CSV with c1/c2/cx/seed/control flag per image, plus a
[montage](datasets/pareidolia/montage.png) (one median sample per
domain x bin cell). Rocks and clouds are balanced 8 per bin; fluids are
naturally smaller (8 / 2 / 4 / 4 / 8 across bins) since only 4 fluid families
take complexity and 3 are NS sims.

Stim filenames encode the domain (`r`/`c`/`f`) + a serial index + family, e.g.
`r042_granite.png`. Bind to the manifest to recover all metadata.

17 of the 106 stimuli come from families retired after the set was built
(stratified, cloud_mrw, choppy). `mf.generate(..., allow_retired=True)`
redirects those names to their replacements, and
[`mfractal/retired.py`](mfractal/retired.py) records every retirement.

### Crossed c1 × c2 rock set

[`datasets/pareidolia_rock/`](datasets/pareidolia_rock/) holds the manifest of
96 stimuli, 3 c1 levels × 4 c2 levels × 8 per cell, each cell drawn from 3–4
rock families. Rock is the one domain where c1 and c2 vary independently
(corr +0.26, against −0.84 in cloud and −0.94 in fluid, measured over 2656
fields by [`datasets/reach_map.py`](datasets/reach_map.py)).

| factor | levels | between-level range / within-cell sd |
|---|---|---|
| c1 | 1.276 · 1.426 · 1.576 | 7.7 |
| c2 | −0.176 · −0.109 · −0.042 · +0.024 | 9.4 |

Every stimulus has an IAAFT phase-scrambled twin with the same histogram and
power spectrum. Twins keep 98% of the c1 manipulation and 7% of the c2
manipulation, so an effect present in stimuli and absent in twins isolates
higher-order multifractal structure. RMS contrast correlates with both factors
(0.44 with c1, 0.52 with c2). The manifest records RMS, mean, spectral β and
anisotropy per stimulus for use as covariates.

---

## Reproducing

```bash
# 1. Validation sweep (~4 min)
python benchmarks/run_validation.py

# 2. Figures from the manifest (~30 s)
python benchmarks/make_figures.py

# 3. (optional) pymultifracs cross-check (~50 s)
python benchmarks/pymultifracs_check.py

# 4. Pareidolia dataset (~7-8 min)
python datasets/build_pareidolia.py

# 5. Crossed rock set + phase-scrambled twins (options in the module docstring)
python datasets/build_pareidolia_rock.py --auto-levels --per-cell 8
python datasets/make_twins.py --set datasets/pareidolia_rock
```

Steps 1–4 re-run cleanly from a fresh clone - outputs end up under
`benchmarks/results/`, `benchmarks/figures/`, and `datasets/pareidolia/`.
The diffusion and regressor stages need a CUDA GPU and are documented in
[`benchmarks/diffusion/README.md`](benchmarks/diffusion/README.md) and
[`benchmarks/regressor/README.md`](benchmarks/regressor/README.md).

---

## Diffusion control of c1 and c2

Asking SDXL for a family name gives weak control of c2. Across 32 families the
procedural c2 predicts the diffusion c2 at Spearman ρ = 0.14
([`benchmarks/diffusion/`](benchmarks/diffusion/), 320 RealVisXL images). The
work in [`benchmarks/regressor/`](benchmarks/regressor/) therefore conditions
generation on the cumulants directly.

| approach | status | result |
|---|---|---|
| Classifier guidance with a latent (c1, c2) regressor | main controller | regressor RMSE 0.068 on held-out diffusion c2. Control slopes 0.58 (c1) and 0.49 (c2), ground-truth estimator, n = 63 |
| Zero-shot tile ControlNet fed a `prescribed_cascade` field | works for spatial FD gradients and creative uses | 27% of a requested spatial c2 gradient transmits after the conditioning-field fix |
| Fine-tuned MF-ControlNet (Phase B) | negative result | ignored its conditioning maps after 1000 steps (c2 slope 0.01). A redesign with real-exemplar conditioning is specified in PHASE_B.md |

On photographs and generated images c1 tracks box-counting fractal dimension
(r ≈ −0.9), so controlling c1 controls FD. c2 control depends on the
substrate. It is feasible on 27 of 95 tested substrates and strongest where
sparse structure sits on smooth ground (smoke, frost, ferrofluid, creatures).

Result documents:

- [`benchmarks/diffusion/out/report/REPORT.md`](benchmarks/diffusion/out/report/REPORT.md), diffusion sweep and c2 per family
- [`benchmarks/regressor/RESULTS.md`](benchmarks/regressor/RESULTS.md) and [`NIGHT_LOG.md`](benchmarks/regressor/NIGHT_LOG.md), regressor and guidance
- [`benchmarks/regressor/PHASE_B.md`](benchmarks/regressor/PHASE_B.md), ControlNet fine-tune design, result and diagnosis
- [`benchmarks/regressor/APPLICATIONS.md`](benchmarks/regressor/APPLICATIONS.md), capability matrix with research and art uses
- [`benchmarks/regressor/FEASIBILITY.md`](benchmarks/regressor/FEASIBILITY.md), what each substrate reaches
- [`benchmarks/percept_embedding/README.md`](benchmarks/percept_embedding/README.md), percept embedding at fixed (c1, c2)

The labelled corpus, trained weights (`latent_model.pt`, the ControlNet
checkpoint) and generated images are gitignored and stay on the GPU
workstation.

---

## License

(unchanged from the upstream repo - see git history)
