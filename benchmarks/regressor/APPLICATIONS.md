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

**c1 is fractal dimension.** Validated in this repo against differential box-counting on
200 photographic corpus images: **corr(c1, FD) = −0.91**, `FD ≈ 2.91 − 0.33·c1`
(the idealised relation is `FD = 3 − c1`; box-counting compresses the range but the
mapping is near-deterministic). So **controlling c1 == controlling fractal dimension.**

**Key finding (this project):** *perceived* complexity tracks **c1**, not c2. In a
controlled montage, observers saw texture change where c1 moved and saw nothing where only
c2 moved. Corollary: **visual intricacy ≠ multifractality** — uniformly busy textures
(lichen, coral, rust) measure c2 ≈ 0; high c2 needs *sparse strong structure on smooth
ground* (dendrites, spikes, wisps).

---

## 2. The toolkit

| tool | what it does | file |
|---|---|---|
| **Procedural generators** | 60+ natural-texture families (rock, cloud, fluid, fire, metal…) | `mfractal/` |
| **Exact synthesis** | `prescribed_cascade(c1_target, c2_target)` → a field with *exactly* those cumulants | `mfractal/wavelet_cascade.py` |
| **Estimator** | `wavelet_leaders_2d(field)` → (c1, c2); the ground truth | `mfractal/quantify.py` |
| **Joint regressor** | reads (c1, c2) from any image in one forward pass off the SDXL VAE latent | `joint_regressor.py`, `joint_model.pt` |
| **Classifier guidance** | steers SDXL generation toward a target (c1, c2) | `joint_montage.py` |
| **MF-scaffold** | procedural field as img2img init — imposes structure on flat substrates | `mf_scaffold.py` |
| **MF-ControlNet** | field as ControlNet conditioning; supports *spatially varying* targets | `mf_controlnet.py` |
| **Labelled corpus** | 4 467 images, every one with measured (c1, c2) | `corpus/manifest.csv` |
| **Natural atlas** | 60 families across 9 scale bands, rendered across the (FD × MF) plane | `natural_atlas.py` |

---

## 3. Capability matrix — what actually works

Be honest about the ceiling; it differs a lot by method and by substrate.

| capability | status | evidence |
|---|---|---|
| **Measure** (c1,c2) of any image, fast | ✅ **excellent** | c1 RMSE 0.042 (r 0.99), c2 RMSE 0.060 (r 0.97), held-out grouped split |
| **Synthesise** a field at an exact (c1,c2) | ✅ **excellent** | `prescribed_cascade`, validated c1 RMSE 0.12 / c2 RMSE 0.08 |
| **Control c1 (= FD)** in generated images | ✅ **strong** | monotonic on 8/8 in-distribution and 7/8 out-of-distribution substrates |
| **Control c2** on *feasible* substrates | ✅ **good** | marble slope 1.00 r 0.97; ferrofluid/frost ≈ 1:1 |
| **Control c2** on *resistant* substrates | ⚠️ **limited** | fire's c2 is stuck ≈ −0.58 regardless of target; cirrus saturates ≈ −1.0 |
| **Impose c2 on a flat substrate** (scaffold) | ✅ **works** | lichen natively −0.07 → imposed **−0.57** |
| **Generalise to novel content** | ✅ **yes** | c1 control holds on circuit boards, foil, knitwear, feathers (never trained on) |
| **Spatial gradient of complexity in one image** | ⚠️ **weak** (zero-shot CN) | only ~5 % of a −0.20→−0.85 gradient transmitted onto scenes |
| **Hidden shape as a pure statistics anomaly** | ⚠️ **prototype** | works after luminance histogram-matching, but low contrast |
| **Strong control over *scene* content** | ❌ **not yet** | needs the Phase-B fine-tuned ControlNet |

**The honest summary:** we have **excellent measurement**, **excellent synthesis**, and
**good-to-strong control on texture-like imagery** — especially of c1/FD. Control over
*scene* imagery and *spatially varying* targets is prototype-grade and is the current
research frontier.

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
- **c1–c2 coupling is domain-specific and even flips sign.** Clouds are nearly independent
  (slope −0.12); fluids/rocks/lichen are strongly coupled (≈ −1.0 to −1.2). Making rock
  busier makes it *less* intermittent; making cloud busier makes it *more*.
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
  of real photographs? We already found diffusion outputs sit *systematically more
  multifractal* than procedural counterparts.
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

### 5.2 Things prompts fundamentally cannot do
- **Complexity gradients inside a single image** — calm on one side, turbulent on the other,
  as a continuous spatial field (working; strength to be improved by Phase B).
- **Hidden multifractal shapes** — a letter or figure embedded as a *texture-statistics*
  anomaly with **no luminance edge**, invisible to naive inspection but detectable
  statistically. Both a novel steganography and ideal pareidolia bait.
- **Statistically matched families** — an asset pack (marble, bark, rust, foam) that all share
  one multifractal signature, so a game or brand system has coherent "visual busyness".

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
| Spatially varying targets (gradients, shapes) | `mf_controlnet.py` (weak zero-shot; Phase B pending) |
| A broad labelled stimulus atlas | `natural_atlas.py` |

---

## 7. Current frontier

Phase B: **fine-tune the ControlNet** on conditioning→image pairs whose cumulants genuinely
match, so multifractality can be imposed *spatially* and on *scene* content — the one lever
not yet exhausted. Until then, the reliable deliverable is **texture-domain stimuli with
controlled FD and multifractality**, which is already sufficient for the pareidolia and
aesthetics experiments described above.
