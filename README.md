# fractal_visuals

A procedural **(multi)fractal image generation + measurement** toolbox for
pareidolia research — generate 2D natural-surface stimuli (rocks, clouds,
fluids) whose intermittency / multifractality can be controlled and
independently quantified, then build c2-balanced experimental datasets.

The historical scripts and notebooks from the earlier monofractal-1/f line of
work live under [`legacy/`](legacy/). The current toolbox lives in
[`mfractal/`](mfractal/).

---

## What's inside

```
mfractal/                     procedural generators + multifractal estimators
benchmarks/run_validation.py  sweep all families x complexity x seed, measure c2 / Delta_alpha
benchmarks/make_figures.py    catalogs, c2-vs-complexity panels, cross-family scatter, ranks
benchmarks/pymultifracs_check.py  independent c2 cross-check via pymultifracs
benchmarks/results/           validation_manifest.csv + pymultifracs_check.csv
benchmarks/figures/           the eight benchmark PNGs
datasets/build_pareidolia.py  candidate pool -> c2 binning -> balanced selection
datasets/pareidolia/          the experimental stimulus set (PNG + manifest.csv + montage.png)
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
img = mf.generate("cloud_mrw", n=512, complexity=0.5, seed=1)
```

The full generator + estimator reference lives in
[`mfractal/README.md`](mfractal/README.md); the agent-friendly summary is in
[`AGENTS.md`](AGENTS.md).

---

## Domains & curated families

The toolbox ships 30 families across three domains. After empirical validation
(see [`benchmarks/`](benchmarks/)) the **25 curated families** below are the
recommended ones for experimental use - they have either a clean monotonic
multifractality knob, or serve as deliberate monofractal controls, or carry a
documented stimulus value despite imperfect knob behaviour.

| Domain | Curated families | Notes |
|---|---|---|
| **Rock** (11) | marble, agate, weathered, granite, vesicular, turbulent, schist, serpentinite, flow_banded, convoluted, **gneiss** | gneiss = monofractal anchor (rho ~ 0) |
| **Cloud** (8) | cascade_lognormal, stratified, billow, cirrus, cloud_mrw, cloud_multifractional, **warped_fbm**, ridged | warped_fbm = monofractal anchor; ridged is a structural-detail axis (bin by measured c2 because of a smooth-low-cx wavelet artifact) |
| **Fluid** (6) | curl_weave, choppy, eddies, vorticity, dye_diffusion, rheoscopic | eddies/vorticity = real 2D Navier-Stokes sims; dye_diffusion is sparse at low cx (bin by measured c2); rheoscopic is a near-monofractal flow-viz texture |

**Dropped** from the original 30 after validation:
`plume` (estimator artifact, c2 ~ -8); `cellular_stone`, `concentric`,
`breccia` (inverted rho - more "complexity" = more uniform tiling); `veined`
(very high per-seed variance).

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
```

All four scripts re-run cleanly from a fresh clone - outputs end up under
`benchmarks/results/`, `benchmarks/figures/`, and `datasets/pareidolia/`.

---

## License

(unchanged from the upstream repo - see git history)
