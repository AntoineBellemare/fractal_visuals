"""
textures.py -- ambiguous natural-surface textures for pareidolia.

These are *surfaces*, not objects: marble / agate / weathered stone / granite /
cracked rock. The ambiguity (swirls, mottling, edges that merely suggest forms)
is what drives pareidolia, unlike the literal objects in organic.py.

  rock_texture()   : domain-warped multifractal. The workhorse. Spans marble ->
                     agate -> weathered -> granite via warp strength and grain.
                     Convenience presets: marble(), agate(), weathered(), granite().
  cellular_stone() : Worley/Voronoi cells modulated by a fractal -> cracked rock,
                     dried mud, cellular mineral.

All return [0,1]; display with flagships.punch. Domain warping a multifractal
keeps it multifractal (quantify with mfdfa_2d) while compounding scales into the
organic swirls that read as faces / landscapes / creatures.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from scipy.spatial import cKDTree

from .generators import _normalize01, _fractional_field
from .flagships import multifractal_cloud


def rock_texture(n=512, amp=44, granularity=2.4, base_mf=1.4, warp_beta=3.6,
                 iters=2, seed=None):
    """
    Domain-warped multifractal stone surface.

    A multifractal base field is repeatedly displaced by fractional warp fields,
    folding its level sets into organic swirls / strata / mottling. Reads as
    polished or weathered rock, full of ambiguous forms (good for pareidolia).

    amp          : warp strength. low (~20) marble veins .. high (~65) agate swirl.
    granularity  : base grain (2.0 fine/granite .. 2.9 coarse/marble).
    base_mf      : multifractality of the base (sigma). ~1.0-1.6.
    iters        : number of successive warps (more = more folded/turbulent).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    base = multifractal_cloud(n, granularity=granularity, multifractality=base_mf,
                              smooth=0.22, distribution=2.0, calm_scale=4.0,
                              seed=seed, raw=True)
    for _ in range(iters):
        wx = _fractional_field(n, beta=warp_beta, seed=int(rng.integers(0, 2**31-1)))
        wy = _fractional_field(n, beta=warp_beta, seed=int(rng.integers(0, 2**31-1)))
        base = map_coordinates(base, [(yy + amp * wy) % n, (xx + amp * wx) % n],
                               order=1, mode="grid-wrap")
    return _normalize01(base)


# convenience presets ---------------------------------------------------------
# Per-family within-family COMPLEXITY (complexity in [0,1]). Each family co-varies
# its own perceptually-salient levers (grain/FD, base multifractality, fold count)
# so the progression is both visible and monotonic in Delta_alpha. Measured ranges:
# approximate Delta_alpha spans (MFDFA is high-variance on warped textures, so
# these are rough; measure per generated stimulus for the experiment).
_COMPLEXITY_RANGE = {
    "marble": (0.30, 1.1), "agate": (0.37, 0.95), "granite": (0.27, 2.0),
    "weathered": (0.26, 1.0), "cellular_stone": (0.30, 0.55), "gneiss": (0.06, 1.1),
    "concentric": (0.45, 0.65), "breccia": (0.40, 0.55), "vesicular": (0.21, 1.65),
    "convoluted": (0.30, 1.6), "turbulent": (0.32, 1.7), "flow_banded": (0.40, 0.75),
    "schist": (0.19, 1.0), "serpentinite": (0.30, 0.85), "veined": (0.25, 0.6),
}


def complexity_range(family):
    """Approximate achievable (min, max) Delta_alpha across complexity in [0,1]."""
    return _COMPLEXITY_RANGE[family]


def _lerp(a, b, t):
    # no clip: complexity may go below 0 (even simpler / near-monofractal) or above 1
    return a + (b - a) * float(t)


def granite(n=512, base_mf=1.5, complexity=None, seed=None):
    """Fine granular crystalline stone. complexity in [0,1] co-varies grain
    (coarse->fine) and base multifractality -> the widest, most visible range
    (Delta_alpha ~0.3 -> 2.0); otherwise base_mf sets it."""
    rng = np.random.default_rng(seed)
    if complexity is not None:
        c = complexity
        b = multifractal_cloud(n, granularity=_lerp(2.7, 1.7, c),
                               multifractality=max(0.0, _lerp(0.05, 2.2, c)),
                               smooth=_lerp(0.14, 0.04, c),
                               distribution=_lerp(2.0, 3.2, c), calm_scale=6.0,
                               seed=seed, raw=True)
        return _normalize01(_warp_base(b, max(0.0, _lerp(0.0, 9, c)), 1, rng))
    b = multifractal_cloud(n, granularity=1.85, multifractality=base_mf, smooth=0.05,
                           distribution=3.0, calm_scale=6.0, seed=seed, raw=True)
    return _normalize01(_warp_base(b, 6, 1, rng))


def marble(n=512, base_mf=1.2, complexity=None, seed=None):
    """Smooth flowing veins. complexity in [0,1] co-varies grain and base
    multifractality (smooth few veins -> fine intricate veining, Delta_alpha
    ~0.3 -> 0.9); otherwise base_mf sets it."""
    rng = np.random.default_rng(seed)
    if complexity is not None:
        c = complexity
        b = multifractal_cloud(n, granularity=_lerp(3.7, 2.3, c),
                               multifractality=max(0.0, _lerp(0.05, 2.0, c)),
                               smooth=_lerp(0.55, 0.25, c), distribution=1.3,
                               calm_scale=3.0, seed=seed, raw=True)
        return _normalize01(_warp_ms(b, [max(0.0, _lerp(0.0, 34, c)),
                                         max(0.0, _lerp(0.0, 16, c))], rng)) ** 0.8
    b = multifractal_cloud(n, granularity=3.1, multifractality=base_mf, smooth=0.45,
                           distribution=1.2, calm_scale=3.0, seed=seed, raw=True)
    return _normalize01(_warp_ms(b, [24, 10], rng)) ** 0.8


def agate(n=512, base_mf=1.3, complexity=None, seed=None):
    """Strongly swirled layered mineral agate. With complexity in [0,1] it uses a
    double warp + grain/multifractality co-variation for a monotonic (if narrow)
    range (Delta_alpha ~0.32 -> 0.72; agate is the most monofractal-leaning
    family). Without complexity it is the original heavy triple-warp swirl."""
    rng = np.random.default_rng(seed)
    if complexity is not None:
        c = complexity
        b = multifractal_cloud(n, granularity=_lerp(3.2, 2.3, c),
                               multifractality=max(0.0, _lerp(0.05, 2.0, c)),
                               smooth=_lerp(0.4, 0.16, c), distribution=2.0,
                               calm_scale=4.0, seed=seed, raw=True)
        return _normalize01(_warp_ms(b, [max(0.0, _lerp(0.0, 60, c)),
                                         max(0.0, _lerp(0.0, 34, c)),
                                         max(0.0, _lerp(0.0, 18, c))], rng))
    return rock_texture(n, amp=65, granularity=2.6, base_mf=base_mf, iters=3, seed=seed)


def weathered(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["weathered"](n, complexity, seed)
    """Blotchy stained surface: medium grain modulated by large soft stains."""
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=2.5, multifractality=1.7, smooth=0.2,
                           distribution=2.4, calm_scale=3.2, seed=seed, raw=True)
    stain = _normalize01(_fractional_field(n, beta=4.2,
                          seed=(None if seed is None else seed + 7)))
    return _normalize01(_normalize01(_warp_ms(b, [34, 14], rng)) * (0.45 + 0.55 * stain))


def cellular_stone(n=512, cells=46, crack=3.0, mod_beta=3.0, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["cellular_stone"](n, complexity, seed)
    """
    Worley/Voronoi cellular stone: distance to nearest feature points gives cell
    interiors; the near-tie ridge (F2-F1) gives crack lines. Modulated by a
    fractal field for organic variation. Cracked rock / dried mud / cellular mineral.

    cells : sqrt of number of feature points (more = finer cells).
    crack : crack-line width.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, n, size=(cells * cells, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    P = np.c_[xx.ravel(), yy.ravel()]
    d, _ = cKDTree(pts, boxsize=n).query(P, k=2)
    F1 = d[:, 0].reshape(n, n); F2 = d[:, 1].reshape(n, n)
    cracks = 1 - np.exp(-(F2 - F1) / crack)
    mod = _normalize01(_fractional_field(n, beta=mod_beta,
                                         seed=(None if seed is None else seed + 3)))
    out = _normalize01(0.6 * cracks + 0.4 * _normalize01(F1)) * (0.6 + 0.4 * mod)
    return _normalize01(out)


# ---------------------------------------------------------------------------
# subtle natural color palettes (pure numpy, no matplotlib needed)
# ---------------------------------------------------------------------------
PALETTES = {
    "sandstone":   [(0.16, 0.11, 0.07), (0.44, 0.29, 0.17), (0.72, 0.55, 0.36),
                    (0.90, 0.80, 0.62), (0.97, 0.93, 0.84)],
    "slate":       [(0.09, 0.10, 0.13), (0.24, 0.27, 0.33), (0.45, 0.49, 0.55),
                    (0.66, 0.70, 0.75), (0.86, 0.89, 0.91)],
    "iron_oxide":  [(0.11, 0.07, 0.05), (0.39, 0.15, 0.10), (0.64, 0.29, 0.16),
                    (0.82, 0.52, 0.30), (0.95, 0.85, 0.70)],
    "moss":        [(0.09, 0.11, 0.08), (0.23, 0.29, 0.17), (0.41, 0.46, 0.28),
                    (0.63, 0.66, 0.48), (0.87, 0.89, 0.79)],
    "basalt":      [(0.05, 0.05, 0.06), (0.19, 0.19, 0.21), (0.42, 0.41, 0.42),
                    (0.67, 0.65, 0.63), (0.93, 0.91, 0.88)],
}


def list_palettes():
    return list(PALETTES)


def colorize(gray, palette="sandstone", saturation=0.65):
    """
    Map a [0,1] grayscale texture through a subtle natural rock palette.
    Returns an RGB array in [0,1]. `saturation` blends toward gray (lower =
    more muted / natural). Color is a display mapping; the multifractal
    structure lives in the luminance, so quantify before colorizing.
    """
    anchors = np.asarray(PALETTES[palette]); m = len(anchors)
    x = np.clip(gray, 0, 1) * (m - 1)
    i0 = np.floor(x).astype(int); i1 = np.minimum(i0 + 1, m - 1)
    t = (x - i0)[..., None]
    rgb = anchors[i0] * (1 - t) + anchors[i1] * t
    g3 = np.stack([gray] * 3, -1)
    return np.clip(rgb * saturation + g3 * (1 - saturation), 0, 1)


# ---------------------------------------------------------------------------
# polychromatic natural coloring: hue from independent fractal pigment fields
# ---------------------------------------------------------------------------
NATURAL_PALETTES = {
    # --- earthy / rock (original set) ---
    "desert_varnish": [(0.30, 0.22, 0.16), (0.55, 0.34, 0.20), (0.68, 0.55, 0.35),
                       (0.42, 0.45, 0.38), (0.85, 0.78, 0.62)],
    "lichen":         [(0.40, 0.43, 0.36), (0.58, 0.60, 0.46), (0.52, 0.40, 0.28),
                       (0.70, 0.68, 0.58), (0.34, 0.40, 0.40)],
    "mineral_oxide":  [(0.34, 0.30, 0.26), (0.55, 0.30, 0.22), (0.38, 0.46, 0.44),
                       (0.66, 0.58, 0.40), (0.46, 0.52, 0.46)],
    "forest_floor":   [(0.28, 0.24, 0.16), (0.46, 0.40, 0.24), (0.40, 0.44, 0.30),
                       (0.58, 0.46, 0.34), (0.72, 0.68, 0.54)],
    "weathered_copper": [(0.26, 0.30, 0.30), (0.30, 0.46, 0.44), (0.50, 0.54, 0.40),
                         (0.56, 0.38, 0.28), (0.78, 0.76, 0.66)],
    "ink":            [(0.05, 0.07, 0.12), (0.12, 0.20, 0.40), (0.25, 0.42, 0.62),
                       (0.55, 0.70, 0.82), (0.92, 0.95, 0.98)],
    "smoke":          [(0.10, 0.10, 0.12), (0.30, 0.30, 0.34), (0.52, 0.50, 0.52),
                       (0.72, 0.70, 0.70), (0.90, 0.89, 0.88)],
    "oil_film":       [(0.18, 0.10, 0.34), (0.10, 0.38, 0.52), (0.20, 0.56, 0.40),
                       (0.74, 0.60, 0.22), (0.62, 0.26, 0.40)],

    # --- cloud / atmosphere ---
    "dawn_sky":       [(0.18, 0.18, 0.28), (0.42, 0.32, 0.42), (0.78, 0.55, 0.52),
                       (0.85, 0.72, 0.62), (0.96, 0.92, 0.88)],
    "storm_cell":     [(0.10, 0.11, 0.14), (0.26, 0.27, 0.32), (0.34, 0.32, 0.40),
                       (0.55, 0.55, 0.58), (0.82, 0.82, 0.84)],
    "golden_hour":    [(0.20, 0.15, 0.15), (0.55, 0.32, 0.25), (0.82, 0.55, 0.32),
                       (0.92, 0.78, 0.55), (0.98, 0.94, 0.86)],

    # --- fluid / water ---
    "ocean_deep":     [(0.04, 0.10, 0.18), (0.10, 0.22, 0.34), (0.18, 0.42, 0.50),
                       (0.42, 0.66, 0.66), (0.82, 0.92, 0.92)],
    "tidepool":       [(0.08, 0.16, 0.18), (0.18, 0.36, 0.32), (0.46, 0.46, 0.30),
                       (0.62, 0.46, 0.32), (0.86, 0.84, 0.74)],
    "marsh":          [(0.10, 0.15, 0.10), (0.26, 0.34, 0.22), (0.42, 0.46, 0.30),
                       (0.60, 0.54, 0.36), (0.82, 0.78, 0.62)],

    # --- bark / wood ---
    "aged_oak":       [(0.12, 0.08, 0.04), (0.32, 0.20, 0.10), (0.58, 0.40, 0.22),
                       (0.78, 0.62, 0.40), (0.92, 0.85, 0.70)],
    "driftwood":      [(0.18, 0.16, 0.14), (0.42, 0.38, 0.32), (0.58, 0.52, 0.44),
                       (0.74, 0.70, 0.62), (0.92, 0.90, 0.84)],
    "cedar":          [(0.14, 0.07, 0.05), (0.42, 0.18, 0.10), (0.66, 0.36, 0.20),
                       (0.82, 0.58, 0.36), (0.95, 0.86, 0.70)],

    # --- smoke / haze ---
    "ash_drift":      [(0.12, 0.12, 0.14), (0.26, 0.26, 0.30), (0.45, 0.42, 0.46),
                       (0.66, 0.62, 0.62), (0.88, 0.86, 0.85)],
    "wildfire_haze":  [(0.16, 0.12, 0.10), (0.38, 0.28, 0.22), (0.62, 0.46, 0.32),
                       (0.76, 0.62, 0.46), (0.92, 0.84, 0.72)],

    # --- fire / lava ---
    "ember_glow":     [(0.06, 0.02, 0.02), (0.32, 0.06, 0.04), (0.68, 0.22, 0.06),
                       (0.92, 0.55, 0.18), (0.98, 0.86, 0.55)],
    "volcanic":       [(0.04, 0.04, 0.05), (0.20, 0.12, 0.10), (0.54, 0.18, 0.10),
                       (0.86, 0.46, 0.16), (0.96, 0.85, 0.55)],
    "hearth":         [(0.10, 0.06, 0.04), (0.36, 0.18, 0.08), (0.62, 0.32, 0.12),
                       (0.84, 0.56, 0.22), (0.94, 0.82, 0.55)],
}


# Recommended palette per family (used by colorize_for_family).
FAMILY_PALETTES = {
    # rock
    "marble":               "desert_varnish",
    "agate":                "mineral_oxide",
    "weathered":            "weathered_copper",
    "granite":              "lichen",
    "vesicular":            "forest_floor",
    "turbulent":            "weathered_copper",
    "schist":               "ink",
    "serpentinite":         "lichen",
    "flow_banded":          "mineral_oxide",
    "convoluted":           "desert_varnish",
    "gneiss":               "smoke",
    "cellular_stone":       "forest_floor",
    # cloud
    "cascade_lognormal":    "storm_cell",
    "stratified":           "dawn_sky",
    "billow":               "golden_hour",
    "cirrus":               "dawn_sky",
    "cloud_mrw":            "storm_cell",
    "cloud_multifractional":"dawn_sky",
    "warped_fbm":           "storm_cell",
    "ridged":               "golden_hour",
    # fluid
    "curl_weave":           "oil_film",
    "choppy":               "ocean_deep",
    "eddies":               "tidepool",
    "vorticity":            "ink",
    "dye_diffusion":        "oil_film",
    "rheoscopic":           "marsh",
    # bark
    "burled_oak":           "aged_oak",
    "riven_oak":            "driftwood",
    # billow_smoke (lives in clouds module): keep its smoky haze palette
    "billow_smoke":         "wildfire_haze",
    # fire
    "firestorm":            "ember_glow",
    "lava_pool":            "volcanic",
    "volcanic_fissure":     "hearth",
}


def colorize_for_family(gray, family, **kw):
    """Colorize `gray` with the recommended palette for `family` via
    colorize_natural. Convenience wrapper: keeps the call site short when
    rendering many families. Pass any colorize_natural kwarg to override
    defaults (warp, sharpness, saturation, seed)."""
    palette = FAMILY_PALETTES.get(family, "desert_varnish")
    return colorize_natural(gray, palette=palette, **kw)


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def colorize_natural(gray, palette="desert_varnish", weight_beta=3.0, warp=32,
                     sharpness=1.6, saturation=0.75, seed=None):
    """
    Polychromatic natural coloring. Several earth "pigments" are mixed per pixel
    with weights from independent, domain-warped fractal fields, so multiple hues
    interweave in organic fractal patches; luminance is then set to follow `gray`
    so the multifractal rock structure still reads. Desaturated for a natural feel.

    palette    : a key of NATURAL_PALETTES, or a list of (r,g,b) pigment anchors.
    weight_beta: grain of the pigment fields (lower = finer color mixing).
    warp       : how organically the color patches swirl.
    sharpness  : higher = more distinct mineral patches; lower = smoother blends.
    saturation : lower = more muted / subtle.

    Color rides on luminance; it does not change the multifractal spectrum.
    """
    pig = np.asarray(NATURAL_PALETTES[palette] if isinstance(palette, str)
                     else palette, float)
    H, W = gray.shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    fields = []
    for _ in range(len(pig)):
        f = _fractional_field(H, beta=weight_beta, seed=int(rng.integers(0, 2**31-1)))
        wx = _fractional_field(H, beta=3.6, seed=int(rng.integers(0, 2**31-1)))
        wy = _fractional_field(H, beta=3.6, seed=int(rng.integers(0, 2**31-1)))
        f = map_coordinates(f, [(yy + warp * wy) % H, (xx + warp * wx) % W],
                            order=1, mode="grid-wrap")
        fields.append(f)
    Wt = np.stack(fields, 0)
    Wt = (Wt - Wt.mean((1, 2), keepdims=True)) / (Wt.std((1, 2), keepdims=True) + 1e-9)
    Wt = np.exp(sharpness * Wt); Wt /= Wt.sum(0, keepdims=True)      # softmax mix
    pigmix = np.einsum("khw,kc->hwc", Wt, pig)
    out = np.clip(pigmix * (gray / (_lum(pigmix) + 1e-3))[..., None], 0, 1)
    L = _lum(out)[..., None]
    return np.clip(out * saturation + L * (1 - saturation), 0, 1)


# ---------------------------------------------------------------------------
# More rock families (banding, cellular fragmentation, inclusions)
# ---------------------------------------------------------------------------
def _warp_base(base, amp, iters, rng, beta=3.6):
    n = base.shape[0]; yy, xx = np.mgrid[0:n, 0:n].astype(float)
    for _ in range(iters):
        wx = _fractional_field(n, beta=beta, seed=int(rng.integers(0, 2**31-1)))
        wy = _fractional_field(n, beta=beta, seed=int(rng.integers(0, 2**31-1)))
        base = map_coordinates(base, [(yy + amp*wy) % n, (xx + amp*wx) % n],
                               order=1, mode="grid-wrap")
    return base


def strata(n=512, freq=7, seed=None):
    """Sedimentary banding: wavy layered beds (warped sinusoidal strata)."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    und = _normalize01(multifractal_cloud(n, granularity=2.8, multifractality=1.0,
                                          smooth=0.3, seed=seed, raw=True))
    bands = 0.5 + 0.5 * np.sin(freq * 2 * np.pi * yy / n + 6 * und)
    det = _normalize01(multifractal_cloud(n, granularity=2.1, multifractality=1.2,
                                          smooth=0.15,
                                          seed=(None if seed is None else seed + 1), raw=True))
    return _normalize01(_warp_base(0.6 * bands + 0.4 * det, 14, 1, rng))


def gneiss(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["gneiss"](n, complexity, seed)
    """Metamorphic folded banding (strongly warped strata) -- migmatite/gneiss."""
    rng = np.random.default_rng(seed)
    return _normalize01(_warp_base(strata(n, freq=9, seed=seed), 55, 3, rng, beta=3.4))


def concentric(n=512, centers=4, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["concentric"](n, complexity, seed)
    """Concentric ring banding (agate eyes / Liesegang / orbicular)."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    und = _normalize01(multifractal_cloud(n, granularity=2.6, multifractality=1.1,
                                          seed=seed, raw=True))
    acc = np.zeros((n, n))
    for _ in range(centers):
        cx, cy = rng.uniform(0.1 * n, 0.9 * n, 2)
        acc += 0.5 + 0.5 * np.sin(np.hypot(xx - cx, yy - cy) / (n * 0.018) + 5 * und)
    return _normalize01(_warp_base(acc, 12, 1, rng))


def breccia(n=512, cells=22, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["breccia"](n, complexity, seed)
    """Cemented angular fragments (breccia / conglomerate) -- Voronoi cells."""
    rng = np.random.default_rng(seed); pts = rng.uniform(0, n, size=(cells * cells, 2))
    yy, xx = np.mgrid[0:n, 0:n]; P = np.c_[xx.ravel(), yy.ravel()]
    d, idx = cKDTree(pts, boxsize=n).query(P, k=2)
    F1 = d[:, 0].reshape(n, n); F2 = d[:, 1].reshape(n, n); cid = idx[:, 0].reshape(n, n)
    shade = rng.uniform(0.2, 0.9, size=len(pts))[cid]
    tex = _normalize01(_fractional_field(n, beta=2.6,
                                         seed=(None if seed is None else seed + 2)))
    return _normalize01((0.6 * shade + 0.4 * tex) * (1 - 0.7 * np.exp(-(F2 - F1) / 2.0)))


def porphyry(n=512, seed=None):
    """Speckled crystalline rock (granite/porphyry): matrix + mineral grains."""
    rng = np.random.default_rng(seed)
    matrix = 0.4 + 0.45 * _normalize01(multifractal_cloud(n, granularity=2.05,
                          multifractality=1.2, smooth=0.08, seed=seed, raw=True))
    light = gaussian_filter((_normalize01(_fractional_field(n, 2.4, seed=int(rng.integers(0, 2**31-1)))) > 0.62).astype(float), 1.0) * 0.4
    dark = gaussian_filter((_normalize01(_fractional_field(n, 2.2, seed=int(rng.integers(0, 2**31-1)))) > 0.68).astype(float), 1.0) * 0.35
    cryst = gaussian_filter((_normalize01(_fractional_field(n, 3.0, seed=int(rng.integers(0, 2**31-1)))) > 0.74).astype(float), 2.5) * 0.3
    return _normalize01(matrix + light - dark + cryst)


def dendritic(n=512, seed=None):
    """Landscape stone: manganese-oxide dendrites on pale limestone."""
    from .organic import dla
    rng = np.random.default_rng(seed)
    base = 0.55 + 0.45 * _normalize01(_warp_base(
        multifractal_cloud(n, granularity=2.8, multifractality=0.9, smooth=0.3,
                           seed=seed, raw=True), 30, 2, rng))
    mask = np.zeros((n, n))
    for _ in range(int(rng.integers(3, 5))):
        g = 181
        dn = _normalize01(dla(g, int(rng.integers(700, 1300)),
                              seed=int(rng.integers(0, 2**31-1))))
        dz = zoom(dn, (n * rng.uniform(0.5, 1.0)) / g, order=1)
        h, w = dz.shape
        oy = int(rng.integers(0, max(1, n - h))); ox = int(rng.integers(0, max(1, n - w)))
        mask[oy:oy+h, ox:ox+w] = np.maximum(mask[oy:oy+h, ox:ox+w], dz)
    return _normalize01(base * (1 - 0.6 * gaussian_filter(mask, 0.8)))


def vesicular(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["vesicular"](n, complexity, seed)
    """Vuggy / pitted volcanic rock: irregular organic vesicles."""
    rng = np.random.default_rng(seed)
    matrix = _normalize01(multifractal_cloud(n, granularity=2.2, multifractality=1.3,
                                             smooth=0.15, seed=seed, raw=True))
    pf = _normalize01(multifractal_cloud(n, granularity=2.5, multifractality=1.0,
                      smooth=0.2, seed=(None if seed is None else seed + 5), raw=True))
    pits = 1.0 / (1.0 + np.exp((pf - 0.32) * 30))
    return _normalize01(matrix * (1 - 0.8 * pits))


ROCK_FAMILIES = [
    "marble", "agate", "weathered", "granite", "cellular_stone",
    "gneiss", "concentric", "breccia", "vesicular",
    "convoluted", "turbulent",
    "flow_banded", "schist", "serpentinite", "veined",
]


# ---------------------------------------------------------------------------
# Geologically-grounded families (researched structures)
# ---------------------------------------------------------------------------
def _prof1d(W, beta, rng):
    f = np.fft.rfft(rng.standard_normal(W)); k = np.arange(len(f)).astype(float); k[0] = 1
    p = np.fft.irfft(f * k ** (-beta / 2), n=W)
    return (p - p.mean()) / (p.std() + 1e-9)


def stylolite(n=512, seams=6, seed=None):
    """Pressure-dissolution suture seams in limestone (self-affine jagged interfaces
    with sharp interpenetration teeth) -- iconic polished-limestone pattern."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    base = 0.5 + 0.4 * _normalize01(_warp_base(
        multifractal_cloud(n, granularity=2.7, multifractality=1.0, smooth=0.3,
                           seed=seed, raw=True), 25, 2, rng))
    img = base.copy()
    for k in range(seams):
        r = (k + 0.5) * n / seams + rng.uniform(-n * 0.03, n * 0.03)
        seam = r + _prof1d(n, 3.0, rng) * n * 0.012 + np.abs(_prof1d(n, 1.2, rng)) * n * 0.02
        dist = np.abs(yy - seam[None, :].repeat(n, 0))
        img *= (1 - 0.85 * np.exp(-(dist ** 2) / (2 * 1.6 ** 2)))
    return _normalize01(img)


def mylonite(n=512, n_augen=7, seed=None):
    """Sheared metamorphic rock: fine flow foliation wrapping lens-shaped feldspar
    porphyroclasts (augen / 'eyes')."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    defl = np.zeros((n, n)); aug = np.zeros((n, n)); rim = np.zeros((n, n))
    for _ in range(n_augen):
        cx, cy = rng.uniform(0.12 * n, 0.88 * n, 2)
        a = rng.uniform(0.04, 0.10) * n; b = a * rng.uniform(0.4, 0.6); th = rng.uniform(-0.3, 0.3)
        X = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        Y = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        e = (X / a) ** 2 + (Y / b) ** 2
        aug = np.maximum(aug, np.exp(-e * 1.5))
        rim = np.maximum(rim, np.exp(-(np.sqrt(e) - 1) ** 2 / 0.05) * (e > 1))
        defl += np.sign(yy - cy) * a * 0.8 * np.exp(-e)
    foli = 0.5 + 0.5 * np.sin(14 * 2 * np.pi * (yy + defl) / n)
    tex = _normalize01(multifractal_cloud(n, granularity=2.2, multifractality=1.2,
                       smooth=0.12, seed=(None if seed is None else seed + 1), raw=True))
    return _normalize01(_warp_base(0.45 * foli + 0.25 * tex + 0.55 * aug - 0.3 * rim, 8, 1, rng))


def stromatolite(n=512, columns=6, seed=None):
    """Microbial laminated build-ups: nested domal laminae draped over columns."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    bump = np.zeros(n)
    for _ in range(columns):
        cx = rng.uniform(0, n); w = rng.uniform(0.05, 0.12) * n
        bump += rng.uniform(0.5, 1.0) * np.exp(-((np.arange(n) - cx) ** 2) / (2 * w ** 2))
    bump = _normalize01(bump) * n * 0.16
    laminae = 0.5 + 0.5 * np.sin(2 * np.pi * (yy - bump[None, :].repeat(n, 0)) / (n * 0.045))
    tex = _normalize01(multifractal_cloud(n, granularity=2.3, multifractality=1.0,
                       smooth=0.2, seed=(None if seed is None else seed + 2), raw=True))
    return _normalize01(_warp_base(0.7 * laminae + 0.3 * laminae * tex, 10, 1, rng))


def boudinage(n=512, layers=4, seed=None):
    """Competent layers stretched and segmented into boudins within foliation."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    base = 0.35 + 0.3 * _normalize01(multifractal_cloud(n, granularity=2.4,
                     multifractality=1.1, smooth=0.2, seed=seed, raw=True))
    base = base * 0.7 + 0.15 * (0.5 + 0.5 * np.sin(26 * 2 * np.pi * yy / n))
    for k in range(layers):
        r = (k + 0.5) * n / layers + rng.uniform(-n * 0.04, n * 0.04)
        seg = rng.uniform(0.10, 0.16) * n
        pinch = np.clip(np.sin(2 * np.pi * xx / seg + rng.uniform(0, 6)) * 1.4, 0, 1) ** 1.5
        band = np.exp(-((yy - r) ** 2) / (2 * (n * 0.018) ** 2)) * pinch
        base = np.maximum(base, 0.9 * band)
    return _normalize01(_warp_base(base, 12, 1, rng))


def liesegang(n=512, bands=9, seed=None):
    """Rhythmic precipitation banding (Liesegang) with sqrt spacing law, warped."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    und = _normalize01(multifractal_cloud(n, granularity=2.7, multifractality=1.0,
                                          seed=seed, raw=True))
    r = (yy / n) + 0.25 * und
    b = (0.5 + 0.5 * np.sin(2 * np.pi * bands * np.sqrt(np.clip(r, 0, None)))) ** 3
    return _normalize01(_warp_base(b, 16, 1, rng))


# ---------------------------------------------------------------------------
# Complex, strongly-multifractal stone (warp a STRONGLY multifractal base)
# ---------------------------------------------------------------------------
def _warp_ms(base, amps, rng, beta=3.6):
    for a in amps:
        base = _warp_base(base, a, 1, rng, beta=beta)
    return base


def convoluted(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["convoluted"](n, complexity, seed)
    """Contorted / convoluted bedding: large turbulent swirls of a strongly
    multifractal base (intricate AND strongly multifractal, Delta_alpha ~ 1.1)."""
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=2.9, multifractality=1.6, smooth=0.14,
                           distribution=2.0, calm_scale=4.0, seed=seed, raw=True)
    return _normalize01(_warp_ms(b, [72, 42, 24, 12, 6], rng))


def turbulent(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["turbulent"](n, complexity, seed)
    """Fine turbulent / flow-textured stone: very intricate, the most strongly
    multifractal surface family (Delta_alpha ~ 1.5-1.7)."""
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=2.05, multifractality=2.0, smooth=0.08,
                           distribution=2.7, calm_scale=5.5, seed=seed, raw=True)
    return _normalize01(_warp_ms(b, [30, 14, 7], rng))


# curated subset: intricate + genuinely multifractal (recommended for experiments;
# excludes the simple banded families strata / liesegang / stromatolite / mylonite /
# boudinage / concentric, which are perceptually too simple)
COMPLEX_FAMILIES = [
    "turbulent", "convoluted", "agate", "marble", "weathered",
    "granite", "breccia", "vesicular",
]
# gneiss reads ~monofractal (Da~0.2) by MFDFA despite looking intricate -> use as a
# complex-looking MONOFRACTAL CONTROL, not a multifractal condition.


# ---------------------------------------------------------------------------
# Directional / flow + waxy + vein families (selected additions)
# ---------------------------------------------------------------------------
def _awarp(base, ax, ay, rng, beta=3.6):
    """Anisotropic domain warp (different x / y amplitude) -> directional flow."""
    n = base.shape[0]; yy, xx = np.mgrid[0:n, 0:n].astype(float)
    wx = _fractional_field(n, beta=beta, seed=int(rng.integers(0, 2**31 - 1)))
    wy = _fractional_field(n, beta=beta, seed=int(rng.integers(0, 2**31 - 1)))
    return map_coordinates(base, [(yy + ay * wy) % n, (xx + ax * wx) % n],
                           order=1, mode="grid-wrap")


def flow_banded(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["flow_banded"](n, complexity, seed)
    """Volcanic-glass flow: strongly anisotropic warp -> stretched flow streaks."""
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=2.5, multifractality=1.4, smooth=0.15,
                           seed=seed, raw=True)
    for _ in range(3):
        b = _awarp(b, 48, 9, rng)
    return _normalize01(b)


def schist(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["schist"](n, complexity, seed)
    """Foliated mica schist: directional foliation with bright specular flecks."""
    rng = np.random.default_rng(seed)
    b = _normalize01(_awarp(multifractal_cloud(n, granularity=2.1, multifractality=1.3,
                            smooth=0.1, seed=seed, raw=True), 26, 6, rng))
    flecks = gaussian_filter((_normalize01(_fractional_field(n, 2.2,
                             seed=(None if seed is None else seed + 3))) > 0.82).astype(float), 0.6)
    return _normalize01(0.82 * b + 0.45 * flecks)


def chert(n=512, seed=None):
    """Waxy cryptocrystalline chert: smooth high-key body with faint conchoidal
    fracture lines."""
    rng = np.random.default_rng(seed)
    b = _normalize01(_warp_base(multifractal_cloud(n, granularity=2.9, multifractality=1.0,
                                smooth=0.4, seed=seed, raw=True), 20, 1, rng))
    v = _normalize01(_warp_base(multifractal_cloud(n, granularity=2.7, multifractality=1.2,
                                seed=(None if seed is None else seed + 2), raw=True), 42, 1, rng))
    gy, gx = np.gradient(v)
    return _normalize01(b * 0.9 - 0.45 * (_normalize01(np.hypot(gx, gy)) > 0.62))


def serpentinite(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["serpentinite"](n, complexity, seed)
    """Sheared serpentinite: anisotropically sheared mottled body with dark veinlets."""
    rng = np.random.default_rng(seed)
    b = _normalize01(_awarp(multifractal_cloud(n, granularity=2.3, multifractality=1.7,
                            smooth=0.2, seed=seed, raw=True), 34, 10, rng))
    v = _normalize01(_warp_base(multifractal_cloud(n, granularity=2.6, multifractality=1.2,
                                seed=(None if seed is None else seed + 4), raw=True), 38, 1, rng))
    veinlet = gaussian_filter((np.abs(v - 0.5) < 0.02).astype(float), 0.5)
    return _normalize01(b * (1 - 0.4 * veinlet))


def veined(n=512, seed=None, complexity=None):
    if complexity is not None:
        return _CX_RECIPES["veined"](n, complexity, seed)
    """Stockwork veining: bright crosscutting vein network over a dark matrix."""
    rng = np.random.default_rng(seed)
    matrix = 0.16 + 0.14 * _normalize01(multifractal_cloud(n, granularity=2.4,
                          multifractality=1.3, seed=seed, raw=True))
    veins = np.zeros((n, n))
    for k in range(3):
        v = _normalize01(_warp_base(multifractal_cloud(n, granularity=2.6, multifractality=1.1,
                                    seed=(None if seed is None else seed + k), raw=True), 50, 1, rng))
        veins = np.maximum(veins, (np.abs(v - 0.5) < 0.012).astype(float))
    veins = gaussian_filter(veins, 0.5)
    return _normalize01(matrix * (1 - veins) + 0.92 * veins)



# ===========================================================================
# Native within-family complexity recipes for the remaining 12 families.
# complexity in [0,1] (clamped extensions allowed) co-varies grain, base
# multifractality and structural depth from simple -> intricate.
# ===========================================================================
def _cxL(a, b, c):
    return a + (b - a) * float(c)


def _cxM(x):
    return max(0.0, x)


def _cx_vor(n, cells, rng):
    pts = rng.uniform(0, n, (max(2, int(round(cells))) ** 2, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    P = np.c_[xx.ravel(), yy.ravel()]
    d, idx = cKDTree(pts, boxsize=n).query(P, k=2)
    return (d[:, 0].reshape(n, n), d[:, 1].reshape(n, n),
            idx[:, 0].reshape(n, n), len(pts))


def _sd(seed, k):
    return None if seed is None else seed + k


def _cx_weathered(n, c, seed):
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=_cxL(3.4, 2.2, c), multifractality=_cxM(_cxL(0.1, 1.9, c)),
                           smooth=_cxL(0.4, 0.18, c), distribution=2.4, calm_scale=3.2, seed=seed, raw=True)
    stain = _normalize01(_fractional_field(n, 4.2, seed=_sd(seed, 7)))
    return _normalize01(_normalize01(_warp_ms(b, [_cxM(_cxL(0, 34, c)), _cxM(_cxL(0, 14, c))], rng)) * (0.45 + 0.55 * stain))


def _cx_cellular_stone(n, c, seed):
    rng = np.random.default_rng(seed)
    F1, F2, cid, _ = _cx_vor(n, _cxL(8, 40, c), rng)
    cracks = 1 - np.exp(-(F2 - F1) / _cxL(5, 2, c))
    mod = _normalize01(_fractional_field(n, 3.0, seed=_sd(seed, 3)))
    return _normalize01(_normalize01(0.6 * cracks + 0.4 * _normalize01(F1)) * (0.6 + 0.4 * mod))


def _cx_gneiss(n, c, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    prof = 0.5 + 0.5 * np.sin(2 * np.pi * _cxL(4, 13, c) * yy / n)
    band = _normalize01(prof + 0.3 * _normalize01(multifractal_cloud(n, granularity=_cxL(3.0, 2.1, c),
                        multifractality=_cxM(_cxL(0.05, 1.2, c)), seed=seed, raw=True)))
    return _normalize01(_warp_base(band, _cxL(22, 60, c), 1, rng, beta=3.4))


def _cx_concentric(n, c, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    und = _normalize01(multifractal_cloud(n, granularity=2.6, multifractality=_cxM(_cxL(0.3, 1.4, c)), seed=seed, raw=True))
    acc = np.zeros((n, n))
    for _ in range(int(round(_cxL(2, 6, c)))):
        cx, cy = rng.uniform(0.1 * n, 0.9 * n, 2)
        acc += 0.5 + 0.5 * np.sin(np.hypot(xx - cx, yy - cy) / (n * _cxL(0.032, 0.013, c)) + _cxL(2, 7, c) * und)
    return _normalize01(_warp_base(acc, _cxL(6, 26, c), 1, rng))


def _cx_breccia(n, c, seed):
    rng = np.random.default_rng(seed)
    F1, F2, cid, npts = _cx_vor(n, _cxL(8, 30, c), rng)
    shade = rng.uniform(0.2, 0.9, npts)[cid]
    tex = _normalize01(_fractional_field(n, _cxL(3.4, 2.4, c), seed=_sd(seed, 2)))
    return _normalize01((0.6 * shade + 0.4 * tex) * (1 - 0.7 * np.exp(-(F2 - F1) / 2.0)))


def _cx_vesicular(n, c, seed):
    rng = np.random.default_rng(seed)
    matrix = _normalize01(multifractal_cloud(n, granularity=_cxL(2.7, 2.0, c), multifractality=_cxM(_cxL(0.2, 1.6, c)),
                          smooth=0.15, seed=seed, raw=True))
    pf = _normalize01(multifractal_cloud(n, granularity=_cxL(2.9, 2.3, c), multifractality=1.0, smooth=0.2, seed=_sd(seed, 5), raw=True))
    pits = 1.0 / (1.0 + np.exp((pf - _cxL(0.45, 0.22, c)) * 30))
    return _normalize01(matrix * (1 - 0.8 * pits))


def _cx_convoluted(n, c, seed):
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=_cxL(3.4, 2.6, c), multifractality=_cxM(_cxL(0.1, 1.7, c)),
                           smooth=0.14, distribution=2.0, calm_scale=4.0, seed=seed, raw=True)
    return _normalize01(_warp_ms(b, [_cxM(_cxL(0, 72, c)), _cxM(_cxL(0, 42, c)), _cxM(_cxL(0, 24, c)),
                                     _cxM(_cxL(0, 12, c)), _cxM(_cxL(0, 6, c))], rng))


def _cx_turbulent(n, c, seed):
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=_cxL(2.6, 1.9, c), multifractality=_cxM(_cxL(0.1, 2.0, c)),
                           smooth=0.08, distribution=2.7, calm_scale=5.5, seed=seed, raw=True)
    return _normalize01(_warp_ms(b, [_cxM(_cxL(0, 30, c)), _cxM(_cxL(0, 14, c)), _cxM(_cxL(0, 7, c))], rng))


def _cx_flow_banded(n, c, seed):
    rng = np.random.default_rng(seed)
    b = multifractal_cloud(n, granularity=_cxL(3.0, 2.3, c), multifractality=_cxM(_cxL(0.1, 1.8, c)), smooth=0.15, seed=seed, raw=True)
    for _ in range(3):
        b = _awarp(b, _cxL(12, 52, c), _cxL(3, 10, c), rng)
    return _normalize01(b)


def _cx_schist(n, c, seed):
    rng = np.random.default_rng(seed)
    b = _normalize01(_awarp(multifractal_cloud(n, granularity=_cxL(2.8, 2.0, c), multifractality=_cxM(_cxL(0.1, 1.6, c)),
                            smooth=0.1, seed=seed, raw=True), _cxL(12, 30, c), 6, rng))
    flecks = gaussian_filter((_normalize01(_fractional_field(n, 2.2, seed=_sd(seed, 3))) > _cxL(0.93, 0.78, c)).astype(float), 0.6)
    return _normalize01(0.82 * b + 0.45 * flecks)


def _cx_serpentinite(n, c, seed):
    rng = np.random.default_rng(seed)
    b = _normalize01(_awarp(multifractal_cloud(n, granularity=_cxL(2.9, 2.2, c), multifractality=_cxM(_cxL(0.2, 1.8, c)),
                            smooth=0.2, seed=seed, raw=True), _cxL(14, 38, c), 10, rng))
    v = _normalize01(_warp_base(multifractal_cloud(n, granularity=2.6, multifractality=1.2, seed=_sd(seed, 4), raw=True), 38, 1, rng))
    veinlet = gaussian_filter((np.abs(v - 0.5) < _cxL(0.008, 0.028, c)).astype(float), 0.5)
    return _normalize01(b * (1 - 0.4 * veinlet))


def _cx_veined(n, c, seed):
    rng = np.random.default_rng(seed)
    matrix = 0.16 + 0.14 * _normalize01(multifractal_cloud(n, granularity=2.4, multifractality=_cxM(_cxL(0.3, 1.3, c)), seed=seed, raw=True))
    veins = np.zeros((n, n))
    for k in range(int(round(_cxL(1, 4, c)))):
        v = _normalize01(_warp_base(multifractal_cloud(n, granularity=2.6, multifractality=1.1, seed=_sd(seed, k), raw=True), 50, 1, rng))
        veins = np.maximum(veins, (np.abs(v - 0.5) < _cxL(0.006, 0.014, c)).astype(float))
    veins = gaussian_filter(veins, 0.5)
    return _normalize01(matrix * (1 - veins) + 0.92 * veins)


_CX_RECIPES = {
    "weathered": _cx_weathered, "cellular_stone": _cx_cellular_stone, "gneiss": _cx_gneiss,
    "concentric": _cx_concentric, "breccia": _cx_breccia, "vesicular": _cx_vesicular,
    "convoluted": _cx_convoluted, "turbulent": _cx_turbulent, "flow_banded": _cx_flow_banded,
    "schist": _cx_schist, "serpentinite": _cx_serpentinite, "veined": _cx_veined,
}
