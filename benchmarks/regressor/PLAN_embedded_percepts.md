# Plan: embedded percepts in controlled-multifractality textures

**Goal.** Generate a natural fractal texture that simultaneously (a) hits a
target multifractality (c1/c2) and (b) contains an *embedded percept* — a face,
animal, figure — woven into the texture rather than pasted on top. The percept
must be present enough to trigger recognition yet ambiguous enough to stay "in"
the texture. That ambiguity band *is* the pareidolia stimulus.

This is a **two-constraint generation** problem: one global statistical
constraint (multifractality) and one local semantic constraint (the percept).

## Key insight — the constraints are largely orthogonal

c2 is a *global* statistic (variance of local roughness / intermittency); a
percept is a *local* structure. A subtly embedded face barely moves global c2,
so the two can be controlled largely independently. This is the same shape as
the `RESULTS.md` finding "visual intricacy != c2 → decouple substrate from c2",
now generalized: **semantic content != c2 → guide them with separate critics.**

The deliverable is therefore not a single image but a 2-D map in
`(c2, percept_detectability)` space — the pareidolia target is a *region*
(ambiguous-but-findable), not a point.

## Approach (a) — compositional guidance ← START HERE (implemented)

Training-free. Extends the verified `guided_sample.py` DPS loop with a second
critic. At each denoising step, on the predicted clean latent x0:

```
loss   = c2_scale     * ||regressor(x0)      - c2_target||²     # texture stats
       + percept_scale * (1 - cos(CLIP_img(decode(x0)), CLIP_txt("a face")))  # content
x_t   <- x_t - ( c2_scale·ĝ_c2  +  percept_scale·ĝ_percept )    # grads normalized separately
```

- **c2 critic**: Stage-3 latent regressor (`latent_model.pt`) — reads the latent
  directly, cheap.
- **percept critic**: CLIP image↔text similarity — needs pixels, so VAE-decodes
  x0 each guided step (the heavy part; mitigated by `--percept-every`, VAE
  slicing/tiling, optional region crop).
- **The pareidolia dial** = `percept_scale / c2_scale`. Too high → pasted face;
  too low → pure texture; sweet spot → a face *in* the texture.

Implemented in **`embed_percept.py`**. Reuses `build_pipe`, `load_regressor`,
`_encode_prompt` from `guided_sample.py` so the diffusion/guidance path is
identical to the validated one.

**First runs (once GPU is available):**
1. `--dial --c2-target -0.4 --percept-scales 0,30,60,120,240` — sweep the dial at
   fixed c2. Expect: percept_sim climbs while measured c2 stays ~flat
   (confirms orthogonality) — and find the ambiguity band visually.
2. Repeat at c2 ∈ {-0.7,-0.4,-0.15} to check the dial behaves across the
   intermittency range.
3. `--region 0.3,0.25,0.7,0.75` — localize the percept; verify it concentrates
   the face spatially without wrecking global c2.
4. Tune `c2_scale`/`percept_scale` on a small grid first (scales ALWAYS need
   per-setup tuning), then scale up.

**Substrate choice matters.** Per RESULTS.md, intermittent substrates
(frost_fern, ferrofluid, smoke_eddies, manganese_dendrite) carry strong-c2 well;
flat ones (lichen, marble, peeling_paint) cover the weak-c2 end. The percept
reads most naturally on substrates whose large light/dark forms can *become* the
face — the same forms the visual system already latches onto.

## Approach (b) — structure-conditioned init / layout bias

Inject the percept as a *faint* low-frequency prior (a face silhouette / depth /
edge map) — as a weak Canny/depth ControlNet OR a soft bias in the initial
noise — then let c2-guidance fill the texture so the face *emerges* from the
texture's own forms. "Pareidolia by construction": put the percept at the large
scale where faces are read, fractal detail supplies plausible deniability.
Cheap; complements (a). Needs the trained ControlNet only for the Canny/depth
variant; the noise-bias variant is training-free.

## Approach (c) — dual-condition ControlNet (production)

Two conditioning channels: a multifractal (local-c2) map + a percept/saliency
map. Reliable, fast, spatially precise — but needs images labelled with both.
Best built by using (a) to *generate* a corpus, then distilling into the
ControlNet. Heavy; do it AFTER (a) tells us what "embedded" operationally means.

## Approach (d) — frequency-domain embedding (baseline/sanity)

Hybrid-images / steganography: place face energy in a frequency band shaped to
respect the 1/f^β spectrum so it doesn't break the c2 measurement, then check
it's still perceptible. Elegant and controllable, but tends to read as "a
filtered photo" rather than an organic texture. Keep as a comparison point.

## Evaluation (the actual deliverable)

Two metrics per image:
- **c2** — measured via `measure_c2` (wavelet leaders) / the regressor.
- **detectability** — CLIP cosine sim to the percept text now; upgrade to a
  face-detector confidence and, ideally, a small human/CLIP recognition-rate
  study. (CLIP-as-critic and CLIP-as-metric should eventually be *different*
  models to avoid the guidance optimizing its own evaluator.)

Plot the swept images in `(c2, detectability)` space. The pareidolia band is the
mid-detectability ridge at each c2. That map is what we hand to the perception
side of the project.

## Risks / watch-list

- **VAE-decode memory** at 1024 with backprop through VAE+CLIP alongside SDXL in
  16 GB. Mitigations wired: VAE slicing/tiling, `--percept-every`, region crop;
  fallbacks: decode at 512, or guide a downsampled x0.
- **fp16 VAE decode** can NaN on SDXL; if so decode the VAE in fp32 for guidance.
- **Adversarial percept** — CLIP guidance can produce a high-similarity image
  that doesn't look like a face to a human (the classic critic-fooling failure).
  Cap percept_scale; validate with a held-out detector, not just CLIP.
- **Critic = evaluator leakage** — don't score detectability with the same CLIP
  that guided it.

## Sequencing

Approach (a) is the right first move under *every* strategy: training-free,
reuses the regressor we already trust, produces the continuous embedded-ness
knob directly, and generates the corpus that de-risks (b)/(c). The GPU work
already queued (the regressor) is the foundation for all of this — it's the c2
critic for guidance and the metric for everything downstream.
