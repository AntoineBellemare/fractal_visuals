# Phase B — fine-tuning the MF-ControlNet

Design settled by an adversarial scoping pass that ran real experiments on this repo's data.
Every number below was measured, not assumed.

---

## Why Phase B at all

Zero-shot ControlNet is insufficient — measured:
- texture, target c2 = −0.7 → best achieved **−0.45** (cn_scale 1.2);
- on **scene** prompts, a conditioning gradient spanning c2 −0.20 → −0.85 transmitted
  **Δc2 = −0.027** into the output (66 % correct direction, n=82) ≈ **5 %** of the intent.

The tile CN was trained to *hallucinate* photographic detail from degraded photos, so it
transfers a coarse envelope but overwrites fine structure with its own texture prior.

---

## 1. The central trap, and why the chosen scheme is immune

A ControlNet learns `conditioning → target image`. For our goal the pairs **must** satisfy
`cumulants(conditioning) == measured cumulants(target)`.

**Our existing scaffold pairs violate this**, and it was quantified over all 926 rows
(targets parsed from filenames):
- `c2_measured − c2_target = +0.270 ± 0.240`; regression slope `measured ~ target` = **0.382**.
- So training on them literally teaches a **0.38× gain** — the exact compression we're escaping.
- Curating only well-matched pairs (`|Δc1|,|Δc2| ≤ 0.10`) leaves **57 rows**, and the survivors
  are *systematically the weak targets* (mean fade +0.539 for c2_target ≤ −0.9, ≈ 0 near c2 −0.1).
  **Curation is dead** — it discards precisely the strong-c2 regime we need.

### Chosen: **Scheme B — local cumulant map computed FROM the target image**

The conditioning is a *measurement of the target*, so cumulant agreement is an **identity, not
an approximation**. Verified: `mean(c1 map)` vs the image's own label → **r = 0.9998, RMSE 0.0061**.

**Encoding** — 3-channel uint8, a 16×16 cell grid (64 px cells) upsampled to 1024:

| ch | content | mapping |
|----|---------|---------|
| R | local **c1** map | `255·(v−0.5)/(2.7−0.5)`, v clipped to [0.5, 2.7] |
| G | local **c2** map | `255·(v+1.8)/(0.4+1.8)`, v clipped to [−1.8, 0.4] |
| B | validity mask | 255 = cell specified, 0 = free / don't care |

Quantisation step ≈ 0.0086 in both cumulants — **5× finer** than the regressor's own RMSE
(0.042 / 0.060), so the encoding is not the bottleneck.

### ⚠️ CORRECTION AFTER IMPLEMENTATION — the per-cell local map is noise-dominated

The scoping pass claimed "90–97 % of the map's spatial variance is real, image-locked
heterogeneity". **On implementation this did not replicate.** Measured on photographic corpus
images, comparing each image's map spatial sd against the sd produced by a *statistically
homogeneous* `prescribed_cascade` field at the **same** global (c1,c2) — i.e. a pure
estimator-noise reference:

| win | grid | real sd c1 | noise c1 | signal | real sd c2 | noise c2 | signal |
|---|---|---|---|---|---|---|---|
| 256 | 16 | 0.148 | 0.182 | **0 %** | 0.149 | 0.153 | **0 %** |
| 256 | 8 | 0.146 | 0.129 | 47 % | 0.153 | 0.159 | **0 %** |
| 384 | 16 | 0.109 | 0.179 | **0 %** | 0.113 | 0.108 | 28 % |
| 512 | 8 | 0.081 | 0.111 | **0 %** | 0.087 | 0.083 | 30 % |

At **no** window/grid setting does spatial signal clearly exceed noise, and the settings that
suppress noise do so by making the window near-global (degenerate — the map becomes constant).
**Sub-image cumulant estimation is not reliable enough to be a per-cell training label.**

What *did* replicate: the map's **mean** is an excellent estimate of the global label —
`corr(map_mean, global)` = **+0.9935** (c1) and **+0.9984** (c2), near-identity affine.

**Revised design — exact labels only.** Training conditioning is never estimated per cell:
- **Uniform samples** — the entire map is the image's own measured (c1,c2). Exact by definition.
- **Mosaic samples** (30 %) — 2×2 tiles from images with *different* statistics, each map region
  carrying that tile's own measured label. This is what forces the ControlNet to read the map
  **locally** rather than collapsing it to a global scalar; verified map spatial sd ≈ 0.17.

This is *stronger* than the original scheme on both counts that matter: labels are exact
(verified identity below), and — because inference maps (uniform / gradient / shape) are smooth
while the rejected local maps were noisy — the revised training distribution actually **matches
inference**, removing the distribution-shift risk rather than adding it. Since the ControlNet is
fully convolutional, a smooth painted map looks locally like a uniform training sample everywhere.

**Verified end-to-end on the built dataset:**
- conditioning encode→decode round-trip error: c1 **0.0046**, c2 **0.0039** (≪ regressor RMSE);
- **identity check** — the target image measured against its conditioning label:
  c1 **0.00003**, c2 **0.00001**. The fade trap is structurally impossible, not merely avoided.

### Why the estimator must not be called per crop (still true, and why)

Naive `wavelet_leaders_2d` on independent W×W crops **fails**, because the estimator auto-picks
its fit band from the input size (`J = max(3, log2(n)−3)`), so the octaves being fitted move with
the window. Measured over 18 corpus images (crops of the canonical 512 field):

| W | c1 bias | c1 within-image sd | c2 within-image sd | runtime |
|---|---|---|---|---|
| 96 | +0.277 | 0.310 | **2.030** | 0.76 ms |
| 128 | +0.255 | 0.260 | **1.985** | 1.03 ms |
| 192 | +0.206 | 0.192 | 1.256 | 1.50 ms |
| 256 | +0.124 | 0.127 | 0.639 | 2.73 ms |

Systematic W-dependent bias plus enormous c2 variance ⇒ **unusable as a label**.

**Correct approach — one shared wavelet pyramid:** compute the leaders `L_j(x)` once with the
same code as `wavelet_leaders_2d`, then per scale form box-filtered local mean/variance of
`log L_j` over a 256 px window, and regress across the **default** `j_fit=(2,5)` per pixel.
Same fit band as the global label ⇒ locally consistent, globally calibrated, and fast.

**The map carries real signal, not noise.** Within-image map variance was compared against a
noise floor (the same map on a statistically homogeneous `prescribed_cascade` field): **90–97 %
of the map's spatial variance is genuine image-locked heterogeneity.** Noise floor grows with
|c2| (c2 sd 0.038 at global c2 −0.13 → 0.221 at −0.89), so weight/threshold accordingly.

---

## 2. Dataset

- **Pool:** diffusion 2110 + scaffold 925 + macro 187 = **3222** images (after filtering to
  c1 ∈ [0.6, 2.4], c2 ∈ [−1.1, 0.1]), across **899 leakage-safe groups**. 8-bit photo sources only.
- **Crop-scale augmentation is mandatory** — it is what decorrelates the caption from the
  statistics. Sample `frac ∈ {1.0, 0.75, 0.5, 0.35}`, random square position, resize to 1024,
  then **relabel after the crop**. → **12 888 triples** on disk; re-randomise crop position each
  epoch for effectively unbounded data. Measured label shift per crop scale:

  | frac | Δc1 | Δc2 |
  |---|---|---|
  | 1.00 | +0.003 ± 0.024 | −0.003 ± 0.012 (identity sanity check) |
  | 0.70 | +0.055 ± 0.109 | −0.016 ± 0.184 |
  | 0.50 | +0.091 ± 0.177 | −0.079 ± 0.323 |
  | 0.35 | +0.167 ± 0.214 | −0.133 ± 0.426 |
  | 0.25 | +0.274 ± 0.267 | −0.278 ± 0.865 | ← too noisy, resolves JPEG artefacts. **Do not use.** |

  The useful quantity is the per-image *scatter*, not the mean shift: the same image+caption
  lands at genuinely different (c1,c2) in different directions, so the model cannot absorb it
  as a constant.
- **Sampling weights:** macro ×3.5, scaffold ×2, diffusion ×1 (over-represent the decorrelated sources).

### Captions must NOT contain the numeric targets

This is counter-intuitive but well-argued:
1. **Redundancy is actively harmful.** Over 275 diffusion prompt groups, `R²(prompt → c1) = 0.856`
   and `R²(prompt → c2) = 0.725`. The text already predicts the statistics; adding numbers routes
   the information through the large pretrained text encoder and **starves the freshly-initialised
   ControlNet branch** — the single most likely way this project fails.
2. CLIP-L / OpenCLIP-bigG tokenise "−0.73" into subword fragments with **no ordinal or metric
   structure**; there is no mechanism to interpolate between −0.70 and −0.75. The 16×16 map is a
   far higher-bandwidth, spatially addressable, genuinely continuous channel.
3. **Inference distribution shift** — real use is "a forest floor at dawn" + a painted map.

---

## 3. Inference — how a user builds the conditioning map

| use case | map |
|---|---|
| uniform global target | fill R,G with the target (c1,c2); B = 255 everywhere |
| **spatial complexity gradient** | R (and/or G) ramps across space; B = 255 |
| **hidden shape** | uniform background; paint the shape region with a different G (c2); B = 255 |
| partial specification | set B = 0 where you don't care — the model is free there |

Because B is a validity mask, the *same* trained model does global control, gradients, hidden
shapes, and partial constraints — no retraining per use case.

---

## 4. Validation gates (all measured with the TRUE estimator, never the regressor)

Baselines to beat: zero-shot CN texture c2 −0.7 → −0.45; scene gradient transmission ≈ 5 %;
guidance (ground-truth measured, n=63): pooled **dC1/dT1 = +0.58**, **dC2/dT2 = +0.49**,
cross-talk dC1/dT2 = −0.21, dC2/dT1 = −0.04.

- **G1 — gradient transmission** on scenes ≥ 40 % (vs 5 % zero-shot).
- **G2 — resistant substrate:** impose c2 ≤ −0.7 on *fire* (guidance is stuck at ≈ −0.58) while
  the output still reads as fire.
- **G3 — the empty corner:** land within |Δc1| ≤ 0.15, |Δc2| ≤ 0.08 of (c1 ≈ 1.6, c2 ≈ −0.9),
  a cell containing **zero** of the 4467 corpus images.
- **G4 — no c1 regression:** dC1/dT1 must not fall below the guidance baseline of 0.58.

---

## 5. Known risks

- **R1 Measurement circularity (fixed).** `mf_controlnet.py` previously reported "achieved" values
  using `joint_model.pt` — the same network whose gradient drives guidance, so on *guided* outputs
  it is adversarially attacked and reads optimistically. It now reports **both** the true estimator
  and the regressor; quote the estimator. (`reachable_atlas.py` and `verify_gradient.py` were
  already correct, so the headline guidance and gradient-transmission numbers stand.)
- **R2 Novelty risk on c1.** `partial r(c1, jpeg_kb | c2) = −0.85` and `R²(c1 | classic proxies)
  = 0.918` — c1 is largely predictable from compressed file size. Any "perceived complexity tracks
  c1" claim needs a **JPEG-size-matched** control condition, and a pre-committed willingness to
  report that c1 adds nothing beyond compressed size if that is what the data show.
- **R3 The fine-tune relearns the fade** — mitigated by construction via Scheme B.

---

## 6. RESULT — checkpoint-1000 FAILED: the ControlNet ignores the map

Trained 1000 steps (batch 1 × accum 2 = 2000 samples ≈ ⅓ epoch, 512 res, warm-started from
xinsir tile, lr 1e-5). Evaluated with the **true** estimator on 22 images, 2 substrates ×
6 (c1,c2) targets × 2 conditioning scales:

| substrate | achieved c1 across ALL targets | achieved c2 across ALL targets |
|---|---|---|
| forest | 1.28 – 1.35 | −0.06 – −0.12 |
| marble | 1.28 – 1.40 | −0.14 – −0.23 |

**Slopes: dC1/dT1 = +0.04, dC2/dT2 = +0.01** (guidance: +0.58 / +0.49). The output depends on
the *prompt* and slightly on cn_scale, and **not at all on the requested statistics**. All four
gates fail. The network learned to treat the conditioning as a no-op.

### Diagnosis — and it points at my own design choice

1. **Under-trained.** 2000 samples seen. Scoping estimated 3–6k *steps* for usable behaviour,
   10k for solid; we ran 1000 steps. Necessary but probably not sufficient on its own.
2. **The conditioning is nearly information-free.** A uniform map is a *flat colour field*.
   For the CN to use it, it must learn a purely **global** colour→texture-statistic mapping,
   which is not what the architecture is shaped for (it injects spatially-structured residuals),
   and the 30 % mosaics were only ~600 samples.
3. **The warm start actively fights us.** The tile CN's prior is "reproduce the structure of the
   reference image". Given a flat reference there is no structure to reproduce, so the prior
   contributes nothing and must be *un*learned.
4. **Ironic root cause:** the zero-shot run partly worked on c1 (target 1.3 → achieved ~1.35)
   precisely because `prescribed_cascade` conditioning carries the target statistics as **real
   structure**. Replacing it with a flat exact-label map removed the only signal the CN could
   actually exploit. I traded a *usable but imprecise* signal for an *exact but unusable* one.

### Recommended next design (combines both, keeps exactness)

Condition on the **`prescribed_cascade` field** (real structure, so the tile prior helps), but fix
the pairing so the target genuinely has the field's statistics — not by measuring the target
(flat map), and not by reusing faded img2img pairs:

> generate targets with the **guidance** method (which already reaches slope ≈ 0.5–0.6),
> **measure** each output, then synthesise the conditioning field at the **measured** (c1,c2).

Conditioning is then structured *and* statistically exact, and the fade trap stays closed
because the field is built from the achieved value, never the requested one. Cost: one
generation pass over the corpus plus a real (≥6k-step) training run.

Also change: **lr 5e-5–1e-4** (teaching new conditioning semantics, not nudging), and consider
**from-scratch CN init** instead of the tile warm start if the structured conditioning still
under-responds.

### Cheap kill-criterion before spending another GPU day
Take 200 structured-conditioning pairs, train 500 steps, and check `dC2/dT2 > 0.1`. If a
structured field cannot beat 0.1 in 500 steps, the ControlNet route is not the answer and
**classifier guidance (already at 0.49–0.58) remains the best available controller.**

## 7. Reframe worth internalising

Guided generation runs at **11.1 s/image** at 1024. A 200-image stimulus set is **40–60 min** of
GPU. **Compute has not been the bottleneck for weeks** — yield-within-tolerance and validation are.
Budget nights of GPU less, and human/validation data more.

---

## 8. ROOT CAUSE FOUND — the "fade" is the 8-bit conditioning image, not diffusion

Four measurements, in order:

1. **Lowering img2img strength does not help.** Fade slope saturates: strength 0.20 -> 0.449,
   0.30 -> 0.437, 0.40 -> 0.388, 0.50 -> 0.177 (r = 0.99 at low strength, i.e. highly systematic).
2. **The SDXL VAE is not the bottleneck.** Pure encode->decode round trip on the conditioning
   image preserves c2 with slope **+0.927** (r = 1.00).
3. **The 8-bit conditioning image is the bottleneck.** A field synthesised at c2 = -0.80 measures
   only **-0.356** once `scaffold_init` percentile-stretches it to uint8 — before diffusion sees it.
   Encoding sweep (8-bit vs raw field): percentile 1/99 **+0.506**, log+sigmoid +0.468,
   rank/hist-eq +0.349; while log-minmax / sqrt / linear preserve dynamic range but then
   8-bit quantisation produces the sparse-field artifact (c2 -> -50). **There is no 8-bit encoding
   that carries a synthetic heavy-tailed cascade faithfully** — spread it and you lose c2,
   preserve it and quantisation destroys it.
4. **img2img is therefore FAITHFUL to what it is given.** Achieved tracks the *8-bit field's own*
   measured c2 roughly 1:1 (8-bit -0.23/-0.46/-0.86 -> achieved -0.284/-0.492/-0.740).

### Consequence: calibration recovers most of the range for free

Because the map is systematic, simply **over-drive the request**:

| requested c2 | 8-bit field | achieved c2 |
|---|---|---|
| -0.5 | -0.23 | -0.284 |
| -1.0 | -0.46 | -0.492 |
| **-1.5** | **-0.86** | **-0.740** |
| -2.0 | -1.75 | -0.595 (field starting to break) |
| -3.0 | -7.44 | -3.23 (artifact) |

Requesting ~2x the desired value reaches **c2 = -0.74** with NO training, versus the -0.45 we
believed was the ceiling. Usable window ends near request -2.0, where the 8-bit encoding
collapses into sparse-field artifacts. Rule of thumb: **request ~ 2 x target, valid to target ~ -0.8.**

### What this means for the ControlNet
The failed fine-tune was trained on conditioning that could only ever carry ~50% of the intended
c2, on top of being flat/structureless. Any future ControlNet must avoid the 8-bit carrier:
either feed the conditioning as a **float tensor** (the diffusers pipeline accepts torch.Tensor
for `image=`, bypassing uint8 entirely) or use **real 8-bit exemplars** that natively possess the
target statistics, rather than synthetic cascades that cannot survive quantisation.
