# mfractal — multifractal & multifractional stimuli for pareidolia research

Generate and quantify 2D fields with controlled higher-order structure, as
successors to the monofractal 1/f stimuli in *Processing visual ambiguity in
fractal patterns* (iScience 2022). Dependencies: numpy, scipy, matplotlib.

```python
from mfractal import multifractal_cloud, multifractional, punch
img = punch(multifractal_cloud(multifractality=1.2, seed=0))   # flagship 2
img = multifractional(fd_center=2.5, fd_range=1.4, seed=0)     # flagship 1
```

## The two flagships

**Flagship 1 — `multifractional()` : within-image varying fractal dimension.**
Full brightness; local roughness/FD varies across the image. A generalization
of the 2022 between-image FD manipulation to *within* image. Controls the
local-FD spectrum (center via `fd_center`, width via `fd_range`, spatial scale
via `scale`). Quantify with `local_fd_map` / `heterogeneity`. It is
multifractional, not multifractal in the moment sense (by design): at any point
it is locally monofractal, so MFDFA reports a flat f(alpha). Report it as a
distribution of local FD, not a wide moment spectrum.

**Flagship 2 — `multifractal_cloud()` : full-brightness multifractal field.**
A volatility-modulated fractional field (2D multifractal random walk): a signed
fractional base modulated by a log-correlated volatility field. The
multifractality lives in the local variance, not the mean intensity, so the
field stays full gray-range (no darkening). The gravest modes of both fields are
high-passed (no single dominant gray patch) and the singularities are spatially
correlated (connected black structures, not scattered dot speckle). Genuinely
multifractal; quantify with mfdfa_2d -> spectrum_summary (Delta_alpha).

### Dials of multifractal_cloud
- granularity (beta): fine/high-FD .. coarse/low-FD. range 2.0 .. 4.5
- multifractality (sigma): moment Delta_alpha. 0 (mono) .. ~1.0-1.6 sweet .. ~2.5
- smooth: connect singularities into filaments vs dot speckle. ~0.3
- distribution: one big calm patch (0) .. fully distributed (>=4)
- calm_scale: big calm chains (low) .. broken/small (high). 2 .. 8
- seed: fresh realization

complexity(c) bundles these into one simple->complex macro-dial.
sample_battery(n) draws a varied set with random parameter combinations.
punch() is the display transform (linear [5,95] + mild S-curve) for legible
black/white -- apply identically to all stimuli.

## Quantifiers
- mfdfa_2d -> generalized Hurst h(q), mass exponent tau(q). Moment spectrum.
- moments_2d -> tau(q), generalized dims D(q). For positive measures.
- legendre(q, tau) -> singularity spectrum (alpha, f(alpha)).
- spectrum_summary -> Delta_alpha (width), asymmetry, alpha0, D0/D1/D2.
- local_fd_map, heterogeneity -> windowed local FD and spread (for flagship 1).
- For publication-grade moment estimates, prefer 2D wavelet leaders
  (pymultifracs, Abry group) over DFA.

## Design rules (learned by looking at the images, not just metrics)
1. Amplitude vs geometry under thresholding. Flagship-2 multifractality is
   amplitude-coded -> equalization / B/W thresholding largely destroys it
   (binary masks ~identical across sigma). Flagship-1 heterogeneity is geometric
   -> partially survives thresholding (boundary roughness varies). Present
   grayscale for flagship 2; flagship 1 tolerates B/W.
2. Linear contrast is safe; rank transforms are not. Pure linear rescaling
   leaves Delta_alpha invariant; clipping mildly erodes it; equalization
   destroys it. punch() stays in the safe zone.
3. Concentration tradeoff. Strong multifractality necessarily concentrates
   activity onto a sparse set -> the rest goes inactive (dark for a measure,
   flat/calm for a signed field). The moderate range is the perceptual sweet
   spot; high sigma reintroduces sparse speckle + large calm areas.
4. Quantify on the field, display with punch. Contrast is a fixed presentation
   choice applied to every condition, not part of the manipulation.
5. The complexity macro-dial confounds factors (co-varies granularity with
   multifractality), so raw Delta_alpha at the simple end mostly reflects the
   granularity-set monofractal floor. To use multifractality as an isolated IV,
   vary multifractality alone at fixed granularity.

## Experimental axes available (all full-brightness)
mean FD; FD heterogeneity (flagship 1); multifractality / f(alpha) width
(flagship 2) -- three structural properties beyond the 2022 FD x contrast design.

## Files
flagships.py (the two generators + punch + complexity + sample_battery),
generators.py (primitives: fBm, cascades, log-normal MRM, universal MF, mBm
bank, etc.), quantify.py (estimators), stimuli.py (contrast/threshold/symmetry/
battery helpers), demo.py (regenerates the showcase).

## Three process-based organic generators (organic.py)
Unlike the spectral flagships, these enact a multiplicative cascade through a
physical process, so they look grown rather than filtered. Display with punch();
quantify dla / branching_network (positive measures) with moments_2d, and
stir_and_fold (a signed field) with mfdfa_2d.

- `dla(n_particles, stick, ...)` — diffusion-limited aggregation by random
  walkers. The cluster's coarse-grained density is a harmonic-measure multifractal
  (Delta_alpha ~ 1.7). Coral / frost / neuron dendrites. `stick`<1 -> denser.
- `stir_and_fold(steps, amp, diffusion, mode, ...)` — chaotic mixing by random-
  phase alternating sine flows. mode='dissipation' (default) renders the scalar-
  gradient field where the multifractality concentrates (Delta_alpha ~ 1.4-1.7),
  striated like marbling / agate / wood grain. mode='scalar' gives the smooth
  marbled field (weakly multifractal).
- `branching_network(depth, angle_spread, curve, decay, ...)` — recursive
  stochastic branching with conserved flux split multiplicatively at each
  bifurcation (a cascade on a tree). Delta_alpha ~ 1.6-2.0. Venation / vessels /
  river delta / lightning. Tip: display sqrt(image) to lift the thin deep veins.

Note: dla and branching_network are intrinsically *strongly* multifractal (the
cascade is built into the process), unlike the spectral flagships whose Delta_alpha
is moderate but finely tunable and whose field stays space-filling/full-brightness.

## Pareidolia surface textures (textures.py)
Ambiguous natural *surfaces* (rather than literal objects) -- the substrate where
people see faces / figures. Display with punch; quantify with mfdfa_2d.

- `rock_texture(amp, granularity, base_mf, iters, ...)` -- domain-warped
  multifractal stone. Repeatedly displacing a multifractal field by fractional
  warp fields folds its strata into organic swirls / mottling. Spans marble ->
  agate -> weathered -> granite via warp strength (`amp`) and grain (`granularity`).
  Presets: `marble()`, `agate()`, `weathered()`, `granite()`.
- `cellular_stone(cells, crack, ...)` -- Worley/Voronoi cells with crack ridges,
  fractal-modulated. Cracked rock / dried mud / cellular mineral.

These are moderately multifractal (Delta_alpha ~ 0.3-0.6) -- the perceptual sweet
spot -- and full gray-range, which is what makes them ambiguous enough to evoke
strong pareidolia rather than reading as a specific object.

### Controlling multifractality in the textures, and color
- `rock_texture(base_mf=...)` -- base_mf sets the spectrum *width* Delta_alpha
  (monotonic: ~0.4 at base_mf=0.3 up to ~1.5 at base_mf=2.3); `granularity` shifts
  *where* the spectrum sits (fine grain -> lower alpha). So multifractality is
  fully tunable while keeping the rock look.
- `colorize(gray, palette, saturation)` -- subtle natural palettes
  (sandstone, slate, iron_oxide, moss, basalt) as a display mapping. Quantify the
  grayscale field first; color rides on luminance and does not change the spectrum.

### Polychromatic natural coloring
- `colorize_natural(gray, palette, weight_beta, warp, sharpness, saturation, seed)`
  -- multiple earth pigments mixed per pixel by independent domain-warped fractal
  fields, so several hues interweave in organic fractal patches (desert varnish,
  lichen, mineral oxide, forest floor, weathered copper). Luminance follows the
  texture, so the multifractal structure still reads. Much more natural/complex
  than the single-ramp `colorize`. NATURAL_PALETTES lists the sets.

### Rock families (twelve)
`ROCK_FAMILIES` lists them, grouped by mechanism (all multifractal, Delta_alpha ~ 0.2-1.4):
- domain-warped multifractal: `marble`, `agate`, `weathered`, `granite`
- banding + warp: `strata` (sedimentary), `gneiss` (folded metamorphic), `concentric` (agate eyes / Liesegang)
- cellular fragmentation: `cellular_stone` (cracked), `breccia` (cemented fragments)
- inclusions / process: `porphyry` (speckled crystalline), `dendritic` (manganese landscape stone), `vesicular` (vuggy/pitted)
The set is open-ended -- the same mechanisms extend to cross-bedding, schist
foliation, coquina, etc. All accept (n, seed) and pair with punch() and the
colorize / colorize_natural palettes.

### Seventeen rock families + multifractality validation
Added 5 geologically-researched families: `stylolite` (pressure-dissolution suture
seams), `mylonite` (sheared foliation + augen 'eyes'), `stromatolite` (domal
laminae), `boudinage` (stretched segmented layers), `liesegang` (rhythmic
precipitation bands). Total 17 in ROCK_FAMILIES.

Validation (2D MFDFA, Delta_alpha mean +/- sd over seeds) -- honest summary:
- Clearly multifractal (above the ~0.36 MFDFA monofractal floor): stylolite (~1.05,
  tightest), granite, vesicular, concentric, stromatolite, boudinage, strata,
  breccia, mylonite, porphyry, weathered, dendritic.
- Not separable from monofractal by this estimator: agate, marble, cellular_stone,
  liesegang, gneiss.
Caveats: the 0.36 floor is itself an MFDFA finite-size artifact (not zero); and 2D
MFDFA's isotropic detrending under-estimates strongly anisotropic/banded textures
(gneiss reads 0.16 < floor, an estimator artifact, not absence of structure). The
estimate is also high-variance on structured textures. For a publication claim use
2D wavelet leaders (pymultifracs) and more seeds; treat MFDFA here as screening.

### Calibrated within-family complexity
marble, agate and granite take a `complexity` in [0,1] that maps (per-family
calibration) to evenly-spaced Delta_alpha across that family's usable range, so
equal steps in complexity give equal steps in multifractality:
- agate   ~0.32 -> 0.92  (most monofractal-leaning; heavy warp floors the spectrum)
- marble  ~0.48 -> 1.39
- granite ~0.32 -> 2.20  (widest range)
`complexity_range(family)` returns the (min, max) Delta_alpha. Use a fixed seed
across complexity levels to hold the macro-structure constant and vary only the
multifractality -- ideal as an experimental IV.

### Fine-tuned within-family complexity (v2)
marble/agate/granite take complexity in [0,1] that co-varies that family's own
perceptually-salient levers -- grain (coarse->fine) and base multifractality (and
fold scale) -- so the visible structure grades smoothly from simple to intricate
while the family identity is kept. granite gives the widest, cleanest range; agate
the narrowest (its heavy warp floors the spectrum). IMPORTANT: 2D MFDFA Delta_alpha
is high-variance on these warped textures (the same complexity on two seeds can
differ by ~0.4), so complexity sets the GENERATIVE structure reliably but the
measured Delta_alpha is noisy per image -- generate across seeds and bin by the
measured Delta_alpha when building matched experimental conditions.

### Liquid / turbulence patterns (fluids module)
`from mfractal import FLUID_FAMILIES, eddies, plume, whirlpool, ink, filaments, kelvin_helmholtz`
A divergence-free velocity field is built from a superposition of power-law vortices
(stream function psi = inverse-Laplacian of vorticity, velocity = curl psi) and a
multifractal dye is advected through it semi-Lagrangianly -- fluid-like flow carrying
genuine multifractality. Validated Delta_alpha (2D MFDFA, 4 seeds):
whirlpool ~1.2, plume ~1.1, eddies ~0.82, filaments ~0.81, ink ~0.53, kelvin_helmholtz ~0.43.
New fluid colour palettes for colorize_natural: "ink", "smoke", "oil_film"
(e.g. colorize_natural(punch(whirlpool(512, seed=3)), "oil_film")).

### Naturalistic turbulence (Navier-Stokes simulation)
eddies, vorticity and plume are now produced by an actual pseudo-spectral 2D
incompressible Navier-Stokes solver (vorticity form, McWilliams random initial
conditions, 2/3 dealiasing, hyperviscosity, Heun stepping):
- vorticity -> the evolved vorticity field: coherent vortices wrapped by filaments
  (Delta_alpha ~1.0-1.2), the classic 2D-turbulence / soap-film look
- eddies -> a passive scalar advected by that flow: naturalistic multifractal
  filaments / dye-in-fluid marbling (Delta_alpha ~0.5-0.65)
- plume -> a buoyant (Boussinesq) scalar rising and rolling into billows
These are SIMULATIONS, so they are slower (~20-30 s at n=256; finer detail at higher
n and steps). whirlpool / ink / filaments / kelvin_helmholtz remain fast
static-flow patterns. All pair with the smoke / ink / oil_film colour palettes.

### Within-family complexity (v3 — co-vary grain + multifractality + warp)
The complexity in [0,1] knob now also opens up the domain warp (gentle->heavy) along
with grain (coarse->fine) and base multifractality (low->high), so the LOW end is
genuinely simple (smooth, few features) and the HIGH end intricate -- a clearly
graded, smooth simple->intricate ramp that keeps family identity. Approx Delta_alpha
spans: marble ~0.5->1.1, agate ~0.4->0.8 (narrowest), granite ~0.55->1.9 (widest,
cleanest). MFDFA Delta_alpha is still high-variance per image (especially agate), so
the complexity knob is the reliable control; measure Delta_alpha per generated
stimulus and bin when matching conditions.

### complexity below 0 + the MFDFA caveat
complexity now accepts values below 0 (and above 1): multifractality and warp are
clamped at 0, so c<=0 yields a smooth, near-MONOFRACTAL field (the simplest end),
ramping to intricate multifractal texture at c=1. IMPORTANT: 2D MFDFA Delta_alpha is
NOT a reliable complexity readout here -- it over-estimates very smooth/strongly
correlated fields, so the smoothest (lowest-complexity) images can show spuriously
HIGH Delta_alpha. What the complexity knob moves *cleanly* is alpha_0 (the f(alpha)
peak position / dominant Holder exponent, ~ fractal dimension), which shifts
monotonically with complexity. For a trustworthy multifractal WIDTH use 2D wavelet
leaders (pymultifracs) and many seeds; treat complexity as the reliable generative
control.

### Dataset builder + multifractal validation (dataset module)
`from mfractal import build_dataset, validate_dataset, generate, wavelet_leaders_2d`
- build_dataset(out_dir, families, complexities, seeds, n, estimator='wavelet') sweeps
  family x complexity x seed, saves a PNG per stimulus, and writes manifest.csv with
  per-image measurements. Rock families take `complexity`; fluids do not.
- wavelet_leaders_2d(image) returns log-cumulants c1 (~ dominant Holder exponent) and
  c2 (intermittency; c2~0 monofractal, more negative = more multifractal). Requires
  PyWavelets. This is the robust descriptor; MFDFA delta_alpha is noisier.
- validate_dataset(manifest.csv, out_fig, value='c2') aggregates mean/std per
  (family, complexity) and reports the Spearman rho of the descriptor vs complexity.

IMPORTANT validation finding: the complexity knob is a GENERATIVE (visual-intricacy)
control, not a guarantee of multifractality. For warped/granular families
(granite, marble, weathered, schist, serpentinite, turbulent, vesicular, agate)
complexity raises multifractality monotonically (rho -0.7..-0.95). gneiss stays
monofractal (control). But for the cellular / vein / banded families
(cellular_stone, breccia, veined, concentric) more "complexity" = more/finer features
= MORE uniform = LESS multifractal (rho positive). Always bin experimental stimuli by
the MEASURED c2, not by the nominal complexity value.

### Cloud families (clouds module)
`from mfractal import CLOUD_FAMILIES`  -- 8 families, each name(n, seed, complexity):
cascade_lognormal, stratified, billow, cirrus, ridged, warped_fbm, mf_cloud, multifractional.
complexity in [0,1] runs simple->intricate with brightness held roughly constant.

Validated c2 (wavelet-leader intermittency) vs complexity over seeds:
- Strong / clean multifractal, monotonic: mf_cloud (0 -> -0.79, the MRW flagship),
  cascade_lognormal (rho -0.96), stratified (rho -0.97), multifractional (rho -0.67),
  billow (-0.26 -> -0.52).
- warped_fbm: MONOFRACTAL control (c2 ~ 0 at all complexity; knob = fractal dimension only).
- cirrus, ridged: crisp STRUCTURAL-detail axes (kept bright/sharp; complexity adds finer
  crisp fibres/folds). c2 is mild and NOT a reliable readout for these -- ridged in
  particular over-reads at the smooth low-complexity end (estimator artefact on near-smooth
  fields). Treat cirrus/ridged as structural-complexity families and bin by measured c2 with care.

Clouds are integrated into build_dataset / validate_dataset (type='cloud'); generate(family,
n, complexity, seed) dispatches rocks, fluids, and clouds.

### Cloud families (clouds module)
`from mfractal import CLOUD_FAMILIES` plus the 8 generators. Each takes
`family(n=512, seed=None, complexity=0.5)`; complexity in [0,1] runs simple ->
intricate with brightness held roughly constant (auto-gamma), so intricacy reads
as structure not darkness.

  cascade_lognormal      log-normal multiplicative cascade (FIF). MULTIFRACTAL, rho(c2,complexity) ~ -0.94
  stratified             anisotropic cascade, layered sky. MULTIFRACTAL ~ -0.97
  billow                 cascade advected by a multi-scale vortex flow. MULTIFRACTAL
  cirrus                 cascade sheared into crisp fibers (+contrast). mildly MULTIFRACTAL ~ -0.92
  ridged                 multi-octave crisp ridge folds. near-MONOFRACTAL crisp structural axis
                         (c2 ~ 0 at the detailed end; large |c2| at c=0 is a few-folds artifact)
  warped_fbm             domain-warped fBm. MONOFRACTAL control (c2 ~ 0); complexity is pure FD/roughness
  cloud_mrw              flagship multifractal_cloud (MRW) wrapped. MULTIFRACTAL, widest range ~ -0.95
  cloud_multifractional  flagship multifractional (within-image varying FD) wrapped ~ -0.96

build_dataset / validate_dataset handle clouds alongside rocks (generate() dispatches
rock vs cloud vs fluid). As with rocks, bin experimental stimuli by MEASURED c2:
warped_fbm and ridged are the (near-)monofractal anchors; cloud_mrw spans the widest
multifractality. Two families (warped_fbm, ridged) are intentionally ~monofractal.

### Fluids module (revised)
`from mfractal import FLUID_FAMILIES, CX_FLUIDS`. Seven families:
  eddies, vorticity, plume   real 2D Navier-Stokes sims (vorticity form); eddies/plume
                             are passive-scalar mixing, vorticity is the vorticity field.
                             Take (n, seed) only. MULTIFRACTAL turbulent mixing.
  curl_weave    dye advected by a multi-scale vortex flow -> woven marbled filaments.
                MULTIFRACTAL (c2 ~ -0.14..-0.29 over complexity).
  dye_diffusion sparse dye sources dispersed + diffused -> ink-in-water tendrils.
                complexity = dye density; multifractal when dense (c2 ~ -0.4),
                degenerate-sparse at low complexity.
  rheoscopic    line-integral-convolution silk along curl-noise streamlines (kalliroscope).
                ~MONOFRACTAL flow-viz texture (c2 ~ 0).
  choppy        single multifractal cascade as a water surface: smooth/calm at low
                complexity, rough with dense crisp glints (figure-ground) at high. Light
                relief + strong contrast. MULTIFRACTAL (c2 ~ -0.08..-0.40); complexity
                raises visual intricacy AND multifractality together.
CX_FLUIDS = [curl_weave, dye_diffusion, rheoscopic, choppy] take a `complexity` arg
(n, seed, complexity=0.5); the NS sims do not. build_dataset/validate_dataset and
generate() handle all three domains (rock / cloud / fluid). Removed in this revision:
whirlpool, ink, filaments, kelvin_helmholtz.
