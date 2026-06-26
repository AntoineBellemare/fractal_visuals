# Strategies for embedding percepts in controlled-multifractality textures

**Goal.** Produce a natural fractal texture that simultaneously (a) hits a target
multifractality (c1/c2) and (b) contains an *embedded percept* — a face, animal,
figure — woven into the texture rather than pasted on top. The percept must be
present enough to trigger recognition yet ambiguous enough to stay *in* the
texture. That ambiguity band *is* the pareidolia stimulus.

This is a **two-constraint generation** problem: one global statistical
constraint (multifractality) + one local structural constraint (the percept).

## Two framing insights

**1. The constraints are largely orthogonal.** c2 is a *global* statistic
(variance of local roughness / intermittency); a percept is a *local* structure.
A subtly embedded face barely moves global c2, so the two can be steered nearly
independently. Same shape as the `RESULTS.md` finding "intricacy != c2 → decouple
substrate from c2", generalized: **semantic content != c2.**

**2. A percept can live in a texture two different ways** — this splits the whole
strategy space:

- **First-order (luminance) embedding** — the percept is faint *brightness*
  structure (light/dark forms shaped like a face). Easy to see, easy to make,
  closest to ordinary pareidolia (clouds, stains).
- **Second-order (statistical / texture-defined) embedding** — the percept is a
  boundary in *local statistics* (a region of different c2 / roughness /
  orientation) with **no mean-luminance cue**. Invisible to first-order
  (brightness) analysis; the form pops out only from a change in texture. This is
  the *pure fractal-pareidolia* stimulus and it's squarely in our wheelhouse —
  we already control c2 spatially.

The deliverable is not one image but a **2-D map in (c2, percept-detectability)
space**; the pareidolia target is the mid-detectability ridge at each c2.

## Operating constraints (2026-06)

- Desktop locked → `latent_model.pt` (the Stage-3 c2 regressor) and the trained
  ControlNet are **both unavailable**, indefinitely.
- We will **not** do a full GPU re-run (corpus regen + retrain) on Phil's machine.
- ⇒ Weight the plan toward strategies that need **neither blocked artifact** —
  ideally pure-CPU procedural ones runnable *today*, then light GPU-inference ones
  that reuse only *downloadable* models.

```
strategy                              tier  compute   needs model.pt  needs training  runnable NOW
1 luminance-field embedding (1st)      1     CPU        no              no              YES
2 texture-defined embedding (2nd)      1     CPU        no              no              YES   <- distinctive
3 frequency-domain / hybrid image      1     CPU        no              no              YES
4 off-the-shelf ControlNet + img2img   2     GPU        no*             no (stock CN)   when GPU
5 SDEdit from a procedural hybrid       2     GPU        no*             no              when GPU
6 compositional guidance (Plan A)      3     GPU        YES             no              parked (artifact)
7 dual-condition MF-ControlNet         3     GPU+train  YES             YES             parked (no re-run)
  * Tier-2 needs c2 only as a *measurement/substrate*, not a guidance critic, so it
    avoids model.pt — c2 comes from the procedural substrate it starts from.
```

---

## Tier 1 — runnable now (pure CPU, procedural toolbox, nothing blocked)

These reuse `mfractal`'s existing c2-controlled fields. No diffusion model, no
`latent_model.pt`, no GPU. We can build and validate all three immediately.

### Strategy 1 — luminance-field embedding (first-order)
Modulate a c2-controlled procedural field by a faint low-frequency percept mask:

```
field' = field * (1 + eps * (percept_lowpass - 0.5))
```

`percept_lowpass` = a blurred luminance map of the target (face silhouette).
Because the modulation is *low-frequency* and small (`eps` ~ 0.05–0.2), it shifts
where the large light/dark forms sit without touching local roughness — so c2 is
preserved (verify with `wavelet_leaders_2d`). `eps` is the embedded-ness dial.
- **Pros:** trivial, deterministic, c2-safe, instant. Good baseline + sanity check.
- **Cons:** first-order cue → can look like a watermark if `eps` too high; least
  "novel" scientifically.

### Strategy 2 — texture-defined / second-order embedding ← the distinctive one
Render **two** fields at slightly different c2 (or roughness/orientation),
matched in mean luminance and contrast, and composite them through the percept
mask:

```
field' = mask * field_c2A + (1 - mask) * field_c2B    # mask = soft face region
```

The face then exists *only* as a change in local fractal statistics — no
brightness boundary. This is texture-segmentation pop-out (a "second-order"
form): a percept that first-order/luminance models literally cannot detect but
the visual system can. Knobs: |c2A − c2B| (the statistical contrast = detectability
dial), mask sharpness, whether the differing axis is c2, c1, or anisotropy.
- **Pros:** scientifically novel, uses our core c2 machinery, a genuinely new
  *kind* of pareidolia stimulus, fully CPU. Strong paper angle.
- **Cons:** harder to make perceptible (needs enough statistical contrast);
  requires careful luminance/contrast matching so no first-order cue leaks in.

### Strategy 3 — frequency-domain / hybrid-image embedding
Classic Oliva-style hybrid: inject the percept's energy into a frequency band
shaped to respect the field's 1/f^β spectrum so the c2 measurement is unmoved,
then verify perceptibility. CPU, controllable.
- **Pros:** precise spectral control; complements 1/2 as a comparison point.
- **Cons:** tends to read as "a filtered photo" rather than an organic texture.

---

## Tier 2 — needs a GPU but only downloadable models (no blocked artifacts)

Resume when a GPU is available; these avoid `latent_model.pt` by getting c2 from
the *procedural substrate* rather than a guidance critic.

### Strategy 4 — off-the-shelf ControlNet + procedural-c2 substrate
Key point: we do **not** need our custom MF-ControlNet for the *percept*. A
**stock** Canny/depth/scribble SDXL ControlNet (downloadable, no training) can
impose a faint percept layout. Feed it a Tier-1 procedural texture (already at the
target c2) via img2img, condition the layout weakly on the percept edge map, and
let the model naturalize it. c2 is set by the substrate + measured after.
- **Pros:** photoreal output, no training, no model.pt; percept layout is explicit
  and spatially precise.
- **Cons:** needs SDXL + a stock ControlNet download + GPU; c2 control is
  open-loop (measure, don't guide) unless we later add the regressor back.

### Strategy 5 — SDEdit / img2img from a procedural hybrid
Make a crude percept⊕texture composite on CPU (Strategy 1 or 2 output), run img2img
at moderate denoise strength so SDXL dissolves the seams into natural texture.
Strength is the embedded-ness dial; substrate sets c2.
- **Pros:** simplest GPU path, no extra models, naturalizes Tier-1 outputs.
- **Cons:** strength trades off percept-survival vs naturalness; c2 drifts under
  heavy denoise (re-measure, pick strength accordingly).

---

## Tier 3 — needs GPU + the blocked artifacts (parked)

### Strategy 6 — compositional classifier guidance (Plan A)
Implemented in **`embed_percept.py`**: SDXL DPS loop with the latent c2 regressor
*and* a CLIP percept critic, gradients normalized separately, ratio
`percept_scale/c2_scale` = the dial. The strongest closed-loop control of both
axes at once. **Blocked**: requires `latent_model.pt` + GPU. Ready to run when both
return.

### Strategy 7 — dual-condition MF-ControlNet (production)
Two conditioning channels (local-c2 map + percept/saliency map), trained on a
corpus generated by the strategies above. Reliable, fast, spatially precise — but
needs the training run we've ruled out for now. Parked.

---

## Evaluation (the actual deliverable)

Two metrics per image: **c2** (wavelet leaders / `measure_c2`) and
**detectability** (CLIP→face sim now; upgrade to a face-detector confidence and a
small human/CLIP recognition study). Note first- vs second-order percepts will
score differently on a luminance-only detector — that contrast is itself a result.
Plot the swept images in (c2, detectability) space; the pareidolia band is the
mid-detectability ridge. Guard against critic=evaluator leakage (don't score with
the same CLIP that guided).

## Recommendation under the current constraints

Start **Tier 1 now** — it needs nothing blocked and produces real stimuli today:
- **Strategy 2 (texture-defined)** as the scientifically distinctive deliverable —
  a percept that exists purely as a fractal-statistics boundary, which is the most
  original thing this project can make and leans entirely on our c2 machinery.
- **Strategy 1 (luminance)** as the fast baseline to calibrate the detectability
  dial against.

Both are pure CPU, deterministic, and reversible. Tier 2 (stock-ControlNet /
SDEdit) is the natural next step the moment any GPU is available, since it still
sidesteps `latent_model.pt`. Tier 3 (Plan A guidance, MF-ControlNet) resumes only
when the desktop returns with the trained artifacts.
