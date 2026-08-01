# Controllable Multifractality — Toolkit, Capabilities, and Applications

What this project can actually do, what it can't (yet), and what it's good for — in
research and in art. Every capability claim below is backed by a measurement in this repo.

---

## 1. The two parameters

The wavelet-leader formalism (`mfractal.wavelet_leaders_2d`) summarises an image's
texture statistics in two numbers:

| | meaning | perceptual reading | range we work in |
|---|---|---|---|
| **c1** | mean Hölder exponent / roughness | **how busy / detailed** — *this is what the eye reads as "complexity"* | 0.7 → 2.3 |
| **c2** | second cumulant, intermittency | **how clustered** the roughness is (0 = monofractal, more negative = more multifractal) | 0 → −1.3 |

**c1 is fractal dimension — on image-like content.** Validated against Sarkar–Chaudhuri
differential box counting (`validate_fd.py`, `results/fd_validation.csv`, figure 57). The DBC
estimator was first checked against ground truth: on spectral-synthesis fBm of known Hurst H it
gives **r = −1.000**, a flat image reads 2.03 and white noise 2.93.

| sample | n | corr(c1, DBC-FD) | fit |
|---|---|---|---|
| real photographs (`macro`) | 189 | **−0.85** | FD ≈ 2.91 − 0.37·c1 |
| natural atlas | 300 | **−0.90** | FD ≈ 2.85 − 0.29·c1 |
| diffusion corpus | 2 199 | **−0.91** | FD ≈ 2.86 − 0.29·c1 |
| procedural / prescribed float fields | 1 000 | **−0.46 / −0.38** | — |

So **controlling c1 == controlling fractal dimension, for photographic and generated imagery** —
the mapping is monotone and near-deterministic, r ≈ −0.85 … −0.91 across three independent samples.
Two honest caveats. (1) **Box counting compresses hard**: measured FD spans only ~2.0–2.7 and the
fitted slope is ≈ −0.3, not the idealised −1.0 of `FD = 3 − c1` — and this is a property of DBC
itself, which returns slope 0.75 even on ideal fBm. Treat `FD = 3 − c1` as the *ordering*, not as
an absolute scale. (2) The relation **does not hold on raw procedural/prescribed float fields**
(r ≈ −0.4), which occupy a separate cloud; it is a statement about images, not about cascades.

> An earlier version of this section claimed r = −0.91 on "200 photographic corpus images" with no
> artifact in the repo to back it. The table above is the reproduction, now committed.
>
> **Trap:** the `fd_meas` column in `results/natural_atlas.csv` is *not* a measurement — it is
> exactly `3 − c1_meas` by construction. Do not cite it as independent evidence for anything.
> `results/fd_validation.csv` holds the only real box-counting numbers in this repo.

**Key finding (this project):** *perceived* complexity tracks **c1**, not c2. In a
controlled montage, observers saw texture change where c1 moved and saw nothing where only
c2 moved. Corollary: **visual intricacy ≠ multifractality** — uniformly busy textures
(lichen, coral, rust) measure c2 ≈ 0; high c2 needs *sparse strong structure on smooth
ground* (dendrites, spikes, wisps).

---

## 2. The toolkit

| tool | what it does | file |
|---|---|---|
| **Procedural generators** | **40** natural-texture families (rock, cloud, fluid, fire, metal, bark, attractor…) | `mfractal/` |
| **Exact synthesis** | `prescribed_cascade(c1_target, c2_target)` → a field with *exactly* those cumulants | `mfractal/wavelet_cascade.py` |
| **Estimator** | `wavelet_leaders_2d(field)` → (c1, c2); the ground truth | `mfractal/quantify.py` |
| **Joint regressor** | reads (c1, c2) from any image in one forward pass off the SDXL VAE latent | `joint_regressor.py`, `joint_model.pt` |
| **Classifier guidance** | steers SDXL generation toward a target (c1, c2) | `joint_montage.py` |
| **MF-scaffold** | procedural field as img2img init — imposes structure on flat substrates | `mf_scaffold.py` |
| **MF-ControlNet** | field as ControlNet conditioning; supports *spatially varying* targets | `mf_controlnet.py` |
| **Labelled corpus** | 4 467 images, every one with measured (c1, c2) | `corpus/manifest.csv` |
| **Natural atlas** | 60 prompt families across 9 scale bands, rendered across the (FD × MF) plane | `natural_atlas.py` |
| **Complex-content atlas** | creatures / faces / emergent — 80 images, tests whether control survives semantics | `results/creature_atlas.csv` |
| **Cascade calibration** | inverts the synthesiser's forward map so a requested (c1,c2) is actually delivered | `cascade_calibration.py` |
| **Feasibility reference** | per-substrate reachable (c1,c2) box, control slope, ramp polarity — 95 substrates | `FEASIBILITY.md`, `results/feasibility_table.csv` |
| **FD gradient gallery** | spatially ramped fractal dimension inside one image (the working spatial route) | `gradient_gallery.py --mode fd` |
| **Spatial c2** | amplitude-matched, calibrated c2 ramp + paired-polarity measurement | `spatial_c2.py` |
| **Spatial guidance** | per-region pooled guidance — **negative result**, kept so it is not re-attempted | `spatial_guidance.py` |

---

## 3. Capability matrix — what actually works

Be honest about the ceiling; it differs a lot by method and by substrate.

| capability | status | evidence |
|---|---|---|
| **Measure** (c1,c2) of any image, fast | ✅ **excellent** | c1 RMSE **0.046** (r 0.992), c2 RMSE **0.071** (r 0.971), held-out grouped split, n = 398 (`metrics_joint.json`, the shipped `joint_model.pt`) |
| **Synthesise** a field at a target (c1,c2) | ⚠️ **good, and only after calibration** | raw `prescribed_cascade` does **not** deliver independent cumulants — see §3.1. Calibrated: c1 RMSE 0.26, c2 RMSE 0.17 over held-out targets (`cascade_calibration.py`) |
| **Control c1 (= FD)** in generated images | ✅ **strong** | monotonic on **8/8** in-distribution and **8/8** out-of-distribution substrates, per-substrate r ≥ 0.98 (`results/wider_c1_control.csv`, `results/ood_c1_control.csv`) |
| **Control c2** on *feasible* substrates | ✅ **good** | marble slope **1.32** r 0.96 (overshoots); ferrofluid **0.61** r 0.99; frost_fern **0.54** r 0.96 — direction and ordering exact, magnitude 0.5–1.3× of request (`results/c2_control_montage.csv`) |
| **Control c2** on *resistant* substrates | ⚠️ **limited** | fire's c2 is stuck ≈ −0.58 regardless of target; cirrus is pinned at **−0.94 … −1.32** (mean −1.19) and cannot be pulled back toward 0 (`results/reachable_atlas.csv`) |
| **Impose c2 on a flat substrate** (scaffold) | ✅ **works** | lichen natively −0.07 → imposed **−0.57** |
| **Generalise to novel content** | ✅ **yes** | c1 control holds on circuit boards, foil, knitwear, feathers (never trained on) |
| **Spatial gradient of FD (c1) in one image** | ✅ **works** | 18 substrates, mean measured **\|Δc1\| = 0.32**; 10/18 ≥ 0.30, 12/18 ≥ 0.15 (`results/fd_gradient_gallery.csv`, figure 55) |
| **Spatial gradient of c2 in one image** | ✅ **works, partially** — *new* | paired-polarity effect **−0.213** of a requested −0.80 (**27 %**), 9/10 substrates correct sign (p = 0.022). Against the old construction (+0.145, *inverted*) the paired shift is −0.358, t = −3.34, **p = 0.0086** (`spatial_c2.py`, figure 58) |
| **Hidden shape as a pure statistics anomaly** | ❌ **not achieved** | shape vs background \|Δc2\| = **0.039** against a ~0.15 local-c2 noise floor — ~4× below detectability, neither visible nor recoverable |
| **Control c1 (= FD) on *scene* / semantic content** | ✅ **holds** | creatures slope 0.52, faces 0.54, emergent 0.59 — matches natural textures (0.57). Semantics do not impede FD control (`results/creature_atlas.csv`, 80 images) |
| **Control c2 on *scene* content** | ⚠️ **substrate-dependent, not content-dependent** | creatures are the **best c2 carrier measured anywhere** (slope 0.52, reaching c2 −1.60); faces are the worst (0.14, floor −0.62) — uniformly textured skin/fur, the same feasibility law as fire |

**The honest summary:** the dividing line is **c1 vs c2**, not texture vs scene. **c1/FD is
controllable everywhere we have looked** — every natural scale band, novel out-of-distribution
content, semantic content (creatures, faces), and *spatially within a single frame*
(mean \|Δc1\| 0.32). **c2 is controllable globally on physically intermittent substrates, and — as
of this pass — partially in space too** (27 % of a requested ramp, 9/10 substrates correct sign).
Measurement is excellent; synthesis is excellent *once calibrated*. The frontier is closing the
gap on spatial c2, which is now a matter of degree rather than a wall.

### 3.1 The synthesiser itself needed calibrating — and this confounds some earlier "c2" results

`prescribed_cascade(c1_target, c2_target)` was treated throughout this project as delivering
*exactly* those cumulants. It does not. Measured at `c1_target = 1.3`, n = 1024, mean of 5 seeds:

| c2_target | measured c1 | | c2_target | measured c1 |
|---|---|---|---|---|
| −0.20 | 1.301 ✓ | | −0.90 | **1.771** |
| −0.45 | 1.400 | | −1.20 | **2.040** |

Asking for stronger multifractality silently drags roughness up — by **+0.74** at the strong end.
The error is not a constant offset either; it interacts with `c1_target` (the effective c1 slope
falls from ~0.59 at c2 = −0.20 to ~0.21 at c2 = −1.20), so no single additive correction fixes it.

**Why this matters beyond synthesis.** c1 is what the eye reads as complexity (this project's own
central finding). So every "pure c2 ladder" built by sweeping `c2_target` at fixed `c1_target` was
*also* a substantial c1 ladder. Any result in this repo that attributes a perceptual or generative
effect to multifractality alone, when it was produced that way, carries that confound and should be
re-read with it in mind.

**The fix** (`cascade_calibration.py`): measure the forward map on a 9 × 6 grid, fit a quadratic
surface, and invert it numerically to choose the targets that land on what you actually wanted.
Held-out, off-grid: c1 RMSE **0.409 → 0.258**. c2 RMSE is 0.150 → 0.165 overall, but that is
dominated by two requests the synthesiser simply cannot reach; restricted to the feasible region it
is **0.164 → 0.079**.

**The synthesiser has its own feasible region.** It is a diagonal band, not a rectangle — low c1
with strong c2, and high c1 with weak c2, are both unreachable, because measured c1 rises with
|c2| by construction. At any fixed c1 the widest achievable c2 span is about **0.55**, with ~0.26
of residual c1 drift across it. This is the same feasibility law the substrates obey, one level
further down.

### Controllability depends on the substrate class — measured

The natural atlas (300 images, 60 families, 9 scale bands, ground-truth wavelet-leader
measurement — `results/natural_atlas.csv`) gives the cleanest statement of the ceiling.
Slope = measured change per unit of requested change; 1.0 would be perfect.

| scale band | n | **c1 / FD** corr · slope | **c2 / multifractality** corr · slope |
|---|---|---|---|
| terrain (deltas, canyons, coasts) | 35 | **+0.90 · 0.69** | +0.48 · 0.36 |
| atmosphere (cloud, smoke, fog) | 25 | **+0.89 · 0.66** | **+0.52 · 0.51** |
| water / ice | 30 | **+0.88 · 0.67** | +0.38 · **0.48** |
| rock | 35 | **+0.87 · 0.67** | +0.32 · 0.22 |
| microbial | 30 | +0.86 · 0.50 | +0.19 · 0.21 |
| tree / canopy | 30 | +0.75 · 0.49 | +0.41 · 0.23 |
| plant | 50 | +0.72 · 0.52 | +0.19 · 0.15 |
| sand / sediment | 30 | +0.63 · 0.40 | +0.20 · 0.11 |
| microscopy | 35 | +0.61 · 0.49 | +0.11 · 0.19 |
| **overall** | **300** | **+0.73 · 0.57** | +0.21 · 0.26 |

Achieved FD spans **1.07 – 2.31**; achieved c2 spans **−1.33 – +0.01**.

**Read this as the project's central empirical result.** FD (c1) is controllable across
*every* natural scale from microscopy to landscape. Multifractality (c2) is controllable
**only on substrates that are physically intermittent** — atmosphere, water/ice, terrain
(sparse strong structure on smooth ground) — and essentially *not* on uniformly dense ones
(microscopy, plant, sand, microbial). This is the "intricacy ≠ multifractality" finding
appearing as a clean, predictable ordering over substrate classes, not as a model defect.

### Known structural limits
- **Substrate feasibility.** Each texture domain reaches only part of the (c1,c2) plane —
  fire occupies a tiny region, clouds are locked at strong c2, fluids/ice/water sweep wide.
  This is physical, not a model defect.
- **c1–c2 coupling: negative everywhere in nature, sign-flipping under control.** In the
  *natural* corpus every domain co-varies negatively — busier is more intermittent: fluid −1.06,
  lichen −1.00, fire −0.95, rock −0.87, bark −0.66, water −0.44, ice −0.43, and cloud only −0.30
  (r −0.21, effectively independent). But *under guidance* the sign flips by domain: guiding c1
  up drags c2 down in rock (−0.30), fluid (−0.58), fire (−0.41) and water (−1.20), and drags it
  *up* in cloud (**+0.46**) and ice (**+0.28**). So: guiding rock busier makes it **more**
  intermittent; guiding cloud busier makes it **less**. Budget for the drag when you need one
  cumulant held fixed (`results/c1c2_coupling.json`).
- **Exhausted levers (measured, negative):** a large data-enrichment pass did *not* improve
  the regressor (it is data-saturated), and a noise-aware timestep-conditioned regressor did
  *not* beat clean guidance. Gains now live in *conditioning*, not in data or the regressor.

---

## 4. Research applications

### 4.1 Pareidolia and perception (the driving goal)
- **Parametric stimulus ladders.** Same prompt, same seed, graded FD/multifractality — find
  the statistical sweet spot for seeing faces/patterns in noise, with content held fixed.
- **The two orthogonal designs the toolkit uniquely enables:**
  - *content-matched, statistics-varied* — one scene at many (c1,c2) values;
  - *statistics-matched, content-varied* — forests, reefs and marble that all share
    **identical** (c1,c2). This dissociates low-level image statistics from semantics, the
    confound that is otherwise very hard to control.
- **Individual differences.** Pareidolia-proneness is linked to hallucination-proneness,
  schizotypy and Parkinson's; a calibrated stimulus axis makes those studies quantitative.

### 4.2 Neuroimaging / EEG
- Parametric stimulus sets whose *only* varying dimension is a measurable image statistic —
  ideal for regressing neural response against FD or multifractality.
- Because generation is cheap and labelled, you can build hundreds of trials per condition
  with matched luminance/contrast and randomised content.

### 4.3 Visual aesthetics
- Human fractal preference peaks around **FD ≈ 1.3–1.5** for contours (Taylor et al.);
  our tools let you *generate photoreal imagery at a prescribed FD* rather than curating it,
  turning correlational aesthetics work into an experimental design.
- Test whether preference tracks **c1 alone** or needs **c2** — a question this toolkit is
  almost uniquely positioned to answer, because it can hold one constant while varying the other.

### 4.4 Image-statistics / vision science
- **Natural scene statistics**: measure FD and multifractality across a corpus at scale
  (the regressor is one forward pass per image).
- **Generative-model auditing**: do SDXL/Flux outputs reproduce the multifractal statistics
  of real photographs? The regressor makes this cheap to ask at scale. *Not yet answered here* —
  our diffusion corpus was **c2-balanced by construction** (rejection-sampled to fill 10 uniform
  bins over c2 ∈ [−1.0, −0.02]), so its c2 distribution reflects the sampling design, not SDXL.
  A clean audit needs an unfiltered generation run against a matched photographic set.
- **Dataset characterisation**: fingerprint any image dataset by its (c1,c2) distribution.

### 4.5 Domain sciences
- **Geology / remote sensing / ecology**: FD and multifractality are established descriptors
  of terrain, canopy, and sediment. The regressor gives a fast, consistent measurement, and
  the generators give *synthetic ground-truth imagery* at known parameters — useful for
  validating analysis pipelines and for training data augmentation.
- **Microscopy / biomedical**: cell and tissue texture is routinely characterised by fractal
  dimension; the natural atlas includes microscopy-scale substrates.

---

## 5. Arts and design applications

### 5.1 Authoring by perceptual parameter
Specify a texture by *what it does perceptually* — "FD 2.4, mildly multifractal" — instead of
guessing prompts. The procedural scaffold is exact and free; diffusion photorealises it.

### 5.2 Things prompts fundamentally cannot do — with HONEST status

- **Statistically matched families** ✅ **works today.** An asset pack (marble, bark, rust, foam)
  all sharing one multifractal signature, so a game or brand system has coherent "visual
  busyness". Guide each family to the same (c1,c2), then *verify* with the regressor and keep
  what lands in tolerance. Only limit is substrate feasibility (fire cannot reach strong c2).
- **FD (complexity) gradients inside a single image** ✅ **works — measured.** Ramp *c1*: a spatial
  blend of two `prescribed_cascade` fields through the zero-shot tile ControlNet. Measured
  calm-third vs turbulent-third on 18 substrates: **mean \|Δc1\| = 0.32**, 10/18 ≥ 0.30, 12/18 ≥
  0.15 (`results/fd_gradient_gallery.csv`, figure 55). Config: c1 span **0.2 → 1.5**,
  `cn_scale 0.9`, `guidance_end 0.60`. Five substrates (dunes, coral, forest, lichen, volcanic)
  render the ramp with **inverted sign**, reproducibly; flipping the request made the gradient
  *weaker* (0.287 vs 0.374), so record the direction rather than fight it. Five more (cirrus,
  smoke, mammatus, canyon, coastline) stay flat — narrow intrinsic FD range, i.e. feasibility again.
- **c2 (multifractality) gradients inside a single image** ✅ **partially — and the long-standing
  "impossible" verdict was wrong.** See §5.2.1.
- **Hidden multifractal shapes** ❌ **not achieved.** Measured on the generated images:
  shape-region vs background **|Δc2| = 0.039**, against a local-c2 noise floor of **~0.15** at a
  256 px window — i.e. ~4x *below* detectability. The shapes are neither visible nor
  statistically recoverable. (An earlier note in this repo claimed the luminance-matching "fixed"
  this; that was based on the drawn overlay, not on measurement, and was wrong.)

**The limit is one of SCALE, not of the statistic.** Local cumulant estimation is noise-dominated
below whole-image scale (0–47 % signal across every window/grid tested), so a texture statistic
confined to a *small* region is hard to impose *and* hard to verify. At half-image scale it is
neither. So: **coarse spatial control is feasible — for c1 and, as of §5.2.1, for c2; fine spatial
control is not.**

#### 5.2.1 Spatial c2: the "impossible" verdict was a bug in the conditioning field

For most of this project the conclusion was that a c2 ramp transmits at ~5 % on every substrate
and that the cause was the 8-bit conditioning image (c1 is a *slope* and survives the affine
rescale; c2 is a *variance* and the percentile stretch flattens it). **That diagnosis was
downstream of the actual defect.** The conditioning field never contained a c2 gradient at all.

`gradient_field` built the ramp as `f = lo*(1−m) + hi*m`. But cascade amplitude *collapses* as
intermittency rises — the same energy concentrates into rarer, sharper bursts:

| c2_target | field sd |
|---|---|
| −0.20 | 0.01213 |
| −0.90 | 0.00147 (**8× weaker**) |
| −1.20 | 0.00129 |

So the low-c2 field dominated the sum everywhere, *including where the mask said it had vanished*.
Measured across four strips of such a field, c2 ran −0.058, +0.002, +0.308, +0.383 for a requested
−0.20 → −0.90: no gradient, wrong sign, drifting *positive* because summing two independent fields
pushes towards Gaussian. And because the two endpoints also had different mean levels, the 8-bit
conditioning image came out as a **luminance wedge** — column-mean brightness 224 → 18 across the
frame. The tile ControlNet reproduced the wedge faithfully. *That* is why those images looked
"cut in half": the seam was never a c2 boundary, it was a brightness ramp.

Controls that rule out the alternatives: the estimator separates pure c2 −0.20 vs −0.90 cleanly
even in a 170 px region (Δ −0.773, d/sd −2.55); a third-crop of a pure c2 = −0.90 field still reads
−1.285. So the gradient's absence was a property of the construction, not of measurement.

**Two fixes, both necessary** (`spatial_c2.py`, ablation over 8 seeds, requested Δc2 = −0.80):

| construction | field Δc2 | field Δc1 (want ≈ 0) |
|---|---|---|
| baseline (repo) | −0.036 ± 0.255 | +0.230 |
| amplitude-matched only | −0.261 ± 0.122 | +0.242 |
| calibrated only | +0.120 ± 0.189 | +0.236 |
| **both** | **−0.324 ± 0.113** | **+0.003** |

Amplitude-matching (standardise each endpoint to unit sd — an affine map, so it provably preserves
both cumulants) *creates* the gradient; calibration (§3.1) *removes the c1 confound*. Then ramp
**shape** turns out to be the largest single lever: a `smoothstep` concentrated in the middle 50 %
leaves the outer thirds nearly pure and delivers **−0.737 ± 0.198** — 92 % of the request and
essentially the hard-step ceiling (−0.768) — while staying C¹-continuous, so it produces no seam.

And the 8-bit encoder, long blamed for this, **retains 87 % of a real field gradient** (−0.324 →
−0.282, with *lower* variance). It was never the first bottleneck.

**"8-bit" was also the wrong name for the second one.** Decomposing the encoder
(`cond_encoding_sweep.py`, `results/cond_encoding_spatial.csv`) separates its three steps:

| encoder step | global c2 slope |
|---|---|
| none / pure affine min-max, no quantisation | **1.000** (c1 also 1.000, r 1.000) |
| + 1/99 percentile **clip**, still float | 0.589 |
| + 8-bit quantisation | 0.514 |

So a **pure affine map provably preserves both cumulants exactly** — confirmed, not assumed — the
**clip costs 0.411 and the bit depth only 0.075, a 5.5 : 1 ratio.** A true 16-bit encoding
(`bitsplit16`, 0.299) reproduces its own float control (0.299) to three decimals: 65 536 levels buy
*nothing* over 256. If this path is ever revisited, widen or remove the percentile clip; do not
reach for more bits.

Two further results worth recording, both negative:
- **A log transform is catastrophic here** (c2 slope 55.9, 33 distinct grey levels), because
  `prescribed_cascade` returns `_normalize01()` of a **signed** fBm-modulated field, not a positive
  multiplicative measure — so `log()` is applied to the wrong object and blows up the lower tail.
  For a symmetric heavy-tailed variable the right analogue is `asinh` or a rank transform.
- **The measurement was cancelling, not blind.** On the old construction the true c2 signal is
  −0.414 and the amplitude-ramp artifact is **+0.438**; they very nearly annihilate. That is why
  the ramp read as "pure noise on all substrates" — an *absence* was inferred from a *cancellation*
  (figure 59). Null controls: a same-c2 blend reads −0.005 ± 0.026, but an amplitude-ramp-only
  field with zero true c2 gradient reads **+0.438 ± 0.135**, and every global encoding — including
  no encoding at all — reports that spurious positive.

**End-to-end, measured on the rendered image.** Each subject is rendered twice from one seed with
the ramp forward and reversed, and the statistic is ½·[Δc2(fwd) − Δc2(rev)] — a difference of
differences, so any fixed left/right asymmetry SDXL imposes cancels. A single-ramp measurement,
which is what every earlier run in this repo used, cannot separate those.

| arm | paired effect | transmission | correct sign |
|---|---|---|---|
| baseline (old construction) | **+0.145 ± 0.065** | −18 % (*inverted*) | 3/10 |
| fixed, linear ramp | −0.097 ± 0.054 | 12 % | 6/10 |
| **fixed + smoothstep_mid** | **−0.213 ± 0.098** | **27 %** | **9/10** (p = 0.022) |

Paired against the matched baseline: shift **−0.358, t = −3.34, p = 0.0086** (Wilcoxon p = 0.0039,
n = 10). Best substrates: ferrofluid 93 %, ink_water 70 %, dye_plume 53 %, firestorm 43 %. Smoke is
the one reliable failure and inverts. See figure 58.

**Read this honestly.** 27 % is partial control, not the 1:1 fidelity global c2 guidance achieves,
and the single-arm test on its own is only p = 0.059 — it is the *paired* contrast against the
matched baseline that is solid. What has changed is the verdict: spatial c2 is not blocked by a
fundamental limit of the estimator, the encoder or the ControlNet. It was blocked by a field
constructed the wrong way, and it now moves in the right direction on 9 of 10 substrates.

**Spatial guidance was tried and does not work (measured).** Per-region pooling of the
regressor with per-region targets: hard 2×2 blocks gave frost 88 % / forest 27 % / marble −14 %,
but the 88 % was a **hard seam**, not control — disjoint blocks make the loss step between
neighbouring targets, so the image splits into two halves with a visible cut plus yellow-grid and
hallucinated-text artifacts. Smooth overlapping windows remove the seam but collapse transmission
to **6 % / 3 % / −11 %**, and raising the gradient 12× only reaches 16 % (it saturates), while
artifacts return.

**Root cause — the regressor's features are not local enough.** Its body is four 3×3 stride-2
blocks, so each cell of the 8×8 feature map has a receptive field of ~25 % of the image and a
4×4 pooling window sees ~62 %. Neighbouring "regions" read almost the same global information:
the only way to satisfy two different targets is a discontinuity, and smooth overlap washes the
differentiation out entirely. This is the **same limit** as the local cumulant map — c2 cannot be
localised below roughly half an image, by the estimator (noise floor) *or* the regressor
(receptive field). Genuine spatial control needs a regressor **trained for locality**, not a
global one repurposed by pooling.

### 5.2.2 Two applications tested on non-texture content and across artistic media

Everything above was measured on texture-like imagery in a photoreal style. Two creative
applications were then pushed onto **figurative content** and **non-photoreal media** — oil
impasto, sumi-e ink wash, copperplate engraving, watercolour, charcoal, linocut, woodblock, 3D
render, electron micrograph. One unifying result came out of both:

> **The artistic medium behaves as a substrate, with its own feasible region.** Media whose
> mark-making is sparse and variable (ink wash, charcoal) carry complexity structure; media whose
> mark-making is uniform by construction (engraving cross-hatch, linocut, oil impasto's canvas
> weave) resist it — the same feasibility law that governs fire and cirrus, one level up.

Both were then **validated** on fresh seeds with proper inference. Two of the pilot's headline
claims did not survive; the applications themselves did.

**A · Complexity as a compositional device** ✅ **validated** (`creative_detail_falloff.py`,
figure 63). An FD ramp concentrates detail on the subject and simplifies the rest. Unlike depth of
field the simplified region stays **sharp and in focus** — it just becomes statistically simpler,
which is a knob no prompt exposes. Every cell is rendered a second time with a **flat field** as a
matched control, so the statistic is the ramp-minus-control difference, not the raw Δc1.

| ramp geometry | n paired cells | effect | p |
|---|---|---|---|
| radial (pilot + replication) | 48 | **+0.106** | **0.016** (Wilcoxon 0.0014), 36/48 |
| **horizontal** | 16 | **+0.183** | **0.014** (Wilcoxon 0.018) |

**Refinement — use a horizontal or vertical ramp, not radial.** A radial conditioning field also
imposes radial *composition*: in the pilot the cathedral rendered as a dome and the dragon as a
spiral, and where that content leakage was worst it inverted the measurement (cathedral/photo,
Δc1 −0.60). The horizontal ramp has no such confound, and it is also the **stronger** effect
(+0.183 vs +0.106) with a cleaner control (flat −0.004 vs +0.032). Radial remains the better
*art-direction* tool; horizontal is the one to use when the statistic has to mean something.

**Retracted from the pilot:** "ink wash carries the ramp (+0.251, 4/4), oil impasto and engraving
do not" — that was n = 4 per medium and **did not replicate**. On fresh seeds photo led and ink was
middling. Pooled at n = 12 per medium, ink (p = 0.03) and engraving (p = 0.05) clear significance
while photo (p = 0.36) does not, despite a larger mean — so the per-medium ranking is not
established. The one durable part is that **oil impasto is the consistent failure** (+0.024,
p = 0.75): its canvas weave is uniformly textured everywhere, so there is no room for a
complexity gradient. Medium-as-substrate survives as a direction, not as a ranking.

**B · Statistically matched families** — ✅ **validated on c1, ❌ refuted on c2**
(`creative_matched_families.py`, figure 64). Ten media driven toward one (c1,c2) by joint guidance,
each with an **exact unguided control** (same prompt, same seed, `scale = 0`), 4 seeds per family,
n = 40 pairs. Because guided and unguided share prompt *and* seed, the powered test is the **paired
per-family error**:

| | unguided \|err\| | guided \|err\| | paired test |
|---|---|---|---|
| **c1** | 0.412 | **0.246** | t = −6.46, **p < 0.0001**, better in **32/40** |
| **c2** | 0.154 | 0.162 | t = +0.66, **p = 0.51** — no effect |

So guidance **roughly halves the c1 error across radically different media** and does **nothing at
all** for c2. Adding rejection sampling (keep the seed closest to target, the workflow this doc
prescribes) improves both arms a little — guided c1 RMSE 0.293 → 0.242 — but rejection sampling
*alone*, without guidance, does not (0.456 → 0.373, n.s.). Guidance is doing the work.

**Retracted from the pilot:** the "31 %/34 % spread reduction" figures. Spread across only 10
families is hopelessly underpowered — bootstrap 95 % CIs on every spread ratio straddle 1
(guidance alone c1: 0.66 [0.33, 1.45]). Those were point estimates quoted without inference. The
paired-error test above is the claim that actually holds, and it is far stronger.

**Why c2 fails** is feasibility, as elsewhere: the media differ enormously in how much
intermittency they can express (charcoal smoke reaches −0.49; woodblock and linocut sit near
−0.07), so guidance moves each by a different amount. The supporting correlation — between how
negative a medium's c2 can go and how far it lands from target — is **+0.51, p = 0.13, n = 10**:
consistent with the feasibility account but *not* on its own significant.

**Practical consequence.** The **statistics-matched / content-varied** stimulus arm is sound on the
FD axis and must **not** be claimed on the multifractality axis. Check `FEASIBILITY.md` per medium
first. Note also that guidance pushed one medium (charcoal) into visible colour-grid artifacts —
the known symptom of driving toward an infeasible corner.

### 5.3 Time and motion
- **Complexity as an animation axis**: sweep (c1,c2) across frames so a texture "breathes"
  between calm and turbulent while the subject stays fixed — a genuinely new parametric
  dimension for motion and live visuals.

### 5.4 Practical craft
- **Viewing-distance-aware design**: FD predicts how detail reads at scale — useful for
  murals, large-format print and projection.
- **Complexity style transfer**: re-render a subject with the texture statistics of smoke,
  turbulence or cloud, without changing what it depicts.
- **Curation and QC**: score a body of work by its texture statistics; find outliers; keep a
  series visually coherent.

---

## 6. What to reach for, given a goal

| goal | use |
|---|---|
| Measure (c1,c2)/FD of existing images | `joint_regressor.py` + `joint_model.pt` |
| Exact synthetic field at a target | `mfractal.prescribed_cascade` |
| Controlled *texture* stimuli | classifier guidance (`joint_montage.py`) |
| Controlled stimuli on a *flat/resistant* substrate | `mf_scaffold.py` |
| A spatial **FD/complexity gradient** | `gradient_gallery.py --mode fd` (defaults now reproduce figure 55) |
| A spatial **c2 / multifractality gradient** | `spatial_c2.py --mode gen` — 27 % transmission, 9/10 substrates; use `--mask smoothstep_mid` |
| A hidden shape | nothing works — \|Δc2\| 0.039 against a ~0.15 noise floor |
| Look up whether a substrate can reach your target | `FEASIBILITY.md` / `results/feasibility_table.csv` |
| A field that actually hits the (c1,c2) you asked for | `cascade_calibration.py` — **not** raw `prescribed_cascade` |
| A broad labelled stimulus atlas | `natural_atlas.py` |

---

## 7. Current frontier

**Phase B was run and FAILED.** Fine-tuning the ControlNet on exact-label conditioning (1000 steps,
2000 samples, evaluated with the true estimator on 22 images) produced `dC1/dT1 = +0.04` and
`dC2/dT2 = +0.01` against classifier guidance's `+0.58 / +0.49` — the network **ignores the
conditioning map**. All four gates failed. Diagnosed cause: a uniform exact-label map is a flat
colour field, so the ControlNet is asked to learn a purely *global* colour→statistic mapping, which
is not what the architecture is shaped for. The zero-shot path works precisely *because*
`prescribed_cascade` conditioning carries the target statistics as real structure. See `PHASE_B.md`.

**Where the frontier actually is now.** Two things moved this pass, and both were bugs rather than
walls: the synthesiser did not deliver the cumulants it was asked for (§3.1), and the spatial c2
field was constructed so that no gradient existed in it (§5.2.1). Fixing those took spatial c2 from
"inverted, −18 %" to "27 %, 9/10 substrates correct sign". That suggests the remaining agenda is
less exotic than previously assumed:

1. **Close the spatial-c2 gap from 27 % toward 1:1.** The conditioning field now delivers 92 % of
   the request; the loss is in the ControlNet step (~45 % transmission for *both* c1 and c2, so it
   is a generic spatial-transmission ceiling of the tile CN, not something specific to c2). Raising
   `cn_scale`, or a CN whose prior is texture rather than tile/detail, are the obvious levers.
2. **A regressor trained for locality**, if *fine* (sub-half-image) spatial control is ever needed.
   The estimator is noise-dominated there (0–47 % signal) and the regressor's 8×8 feature cells each
   see ~25 % of the frame — this is what killed spatial guidance, and it is unchanged.
3. **Re-examine earlier c2 results for the c1 confound** introduced by uncalibrated synthesis.

Reliable deliverables today: **global (c1,c2) control on texture, creature and face content**,
**spatial FD gradients** (mean \|Δc1\| 0.32 over 18 substrates), and **partial spatial c2**. All
three are already sufficient for the pareidolia and aesthetics experiments described above.
