# Diffusion natural-texture multifractality report

- Model: **SG161222/RealVisXL_V4.0** (SDXL fine-tune, photorealism)
- Resolution: 1024 x 1024 (measured at 512 x 512 grayscale for c2 consistency with the procedural sweep)
- Sweep size: 32 families x 10 images = **320 stimuli**
- Per family: 5 prompt variations x 2 seeds
- Generation wall time: 52.9 min on RTX 4090 Laptop (16 GB)

## Headline finding

Cross-family **Spearman ρ(procedural c2, diffusion c2) = +0.14** (Pearson r = +0.31)
excluding the `volcanic_fissure` outlier — i.e. **the procedural toolbox's c2
does NOT strongly predict the diffusion model's c2** when both are asked for the
"same" family. The two generators populate different multifractality regimes.

Diffusion outputs land **systematically lower (more multifractal)** than their
procedural counterparts — most points sit below the y=x line in the comparison
plot. Photographs of natural textures carry more nested intermittency than the
procedural simplifications.

This motivates the MF-ControlNet roadmap step: to control photoreal output at a
target c2, condition on c2 directly rather than on family name.

## Domain-level patterns

- **Clouds are the most multifractal regime**: cirrus c2 = -0.92, cascade_lognormal -0.72,
  billow -0.67, cloud_multifractional -0.67. Soft branching atmospheric structures
  carry strong intermittency.
- **Rocks are the least multifractal**: gneiss -0.03, granite/weathered/schist ~-0.07,
  riven_oak -0.02. Solid surfaces have cleaner, less nested texture.
- **Fire and smoke are in between**, with the *billowing* members (billow_smoke
  c2=-0.73, firestorm -0.77) more multifractal than the *structured* members
  (lava_pool -0.19, volcanic_fissure -0.35).
- **dye_diffusion** is the most multifractal family in BOTH generators
  (procedural -0.96, diffusion -0.99) -- ink-in-water turbulence carries a
  signature that survives both procedural and diffusion paths.

## Outlier

`volcanic_fissure`: procedural c2 = -19.16 (sparse-field estimator artifact -
the procedural fissure is mostly black with a few bright cracks, which the
wavelet-leader estimator interprets as extreme intermittency). Diffusion output
c2 = -0.35 (normal). This is the same failure mode flagged for the original
`plume` family in the README's "Dropped" list; the procedural side here is the
unreliable measurement, not the diffusion side.

## c2 summary (mean ± std across 10 images per family)

```
family                   domain   c2_mean   c2_std   proc_c2    delta
marble                   rock      -0.447    0.400    -0.279   -0.168
agate                    rock      -0.288    0.193    -0.171   -0.117
weathered                rock      -0.069    0.047    -0.215   +0.146
granite                  rock      -0.065    0.093    -0.169   +0.103
vesicular                rock      -0.674    1.467    -0.162   -0.511
turbulent                rock      -0.172    0.089    -0.165   -0.007
schist                   rock      -0.068    0.035    -0.192   +0.124
serpentinite             rock      -0.123    0.254    -0.245   +0.122
flow_banded              rock      -0.112    0.092    -0.194   +0.082
convoluted               rock      -0.163    0.084    -0.156   -0.007
gneiss                   rock      -0.031    0.024    -0.021   -0.010
cascade_lognormal        cloud     -0.719    0.191    -0.162   -0.557
stratified               cloud     -0.574    0.399    -0.193   -0.381
billow                   cloud     -0.672    0.209    -0.259   -0.413
cirrus                   cloud     -0.920    0.444    -0.165   -0.755
cloud_mrw                cloud     -0.598    0.160    -0.253   -0.345
cloud_multifractional    cloud     -0.669    0.071    -0.289   -0.381
warped_fbm               cloud     -0.116    0.366    +0.002   -0.118
ridged                   cloud     -0.395    0.112    -0.710   +0.315
curl_weave               fluid     -0.222    0.190    -0.083   -0.140
choppy                   fluid     -0.320    0.170    -0.223   -0.097
eddies                   fluid     -0.262    0.118    -0.121   -0.141
vorticity                fluid     -0.432    0.243    -0.105   -0.327
dye_diffusion            fluid     -0.991    0.378    -0.962   -0.028
rheoscopic               fluid     -0.697    0.279    -0.040   -0.657
burled_oak               bark      -0.108    0.036    -0.050   -0.058
riven_oak                bark      -0.022    0.091    -0.088   +0.066
chimney_plume            smoke     -0.376    0.365    -0.014   -0.363
billow_smoke             smoke     -0.730    0.181    +0.069   -0.799
firestorm                fire      -0.769    0.183    -0.036   -0.733
lava_pool                fire      -0.194    0.104    -0.158   -0.037
volcanic_fissure         fire      -0.346    0.151   -19.160  +18.814  (procedural outlier)
```

## Figures

- `report/montage_all.png` — 32-family final montage with c2 mean ± std on each tile
- `report/per_family_c2.png` — per-family c2 distribution (violins, colored by domain)
- `report/domain_c2.png` — c2 by domain
- `report/diffusion_vs_procedural.png` — scatter of diffusion c2 vs procedural c2 per family
