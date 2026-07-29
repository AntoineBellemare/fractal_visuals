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

### Critical implementation detail: do NOT call the estimator per crop

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

## 6. Reframe worth internalising

Guided generation runs at **11.1 s/image** at 1024. A 200-image stimulus set is **40–60 min** of
GPU. **Compute has not been the bottleneck for weeks** — yield-within-tolerance and validation are.
Budget nights of GPU less, and human/validation data more.
