"""
metal.py -- corrosion / oxidation textures.

Three families covering distinct rust morphologies:

  rust_bloom    -- radial oxidation blooms spreading from nucleation sites;
                   Voronoi cells with multifractal cascade interiors.
                   cx grows the bloom count.
  rust_streaks  -- vertical drip patterns from oxidation runoff. Strongly
                   anisotropic cascade with column-wise persistence; lateral
                   wisps via horizontal warp. cx narrows streaks / adds detail.
  rust_pitted   -- granular pitting on a corroded surface. Sparse-bright
                   cascade with crisp tanh contrast; multi-octave (large
                   patches + small pits). cx grows pit density and crispness.

All return [0, 1] floats; display with mfractal.flagships.punch and colorize
with the FAMILY_PALETTES entries (rust_oxide / patina_drip / corrosion_grit).
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.spatial import cKDTree

from .generators import _normalize01

__all__ = ["rust_bloom", "rust_streaks", "rust_pitted", "METAL_FAMILIES"]

METAL_FAMILIES = ("rust_bloom", "rust_streaks", "rust_pitted")


def _cascade(n, depth, sigma, seed):
    """Centered log-normal multiplicative cascade. exp() of accumulated log."""
    rng = default_rng(seed)
    logM = np.zeros((n, n))
    for j in range(1, depth + 1):
        s = 1 << j
        up = zoom(rng.normal(-sigma ** 2 / 2, sigma, (s, s)), n / s, order=3)[:n, :n]
        if up.shape != (n, n):
            t = np.zeros((n, n)); t[:up.shape[0], :up.shape[1]] = up; up = t
        logM += np.roll(up,
                        (int(rng.integers(0, n)), int(rng.integers(0, n))),
                        axis=(0, 1))
    logM -= logM.max()
    return np.exp(logM)


def _aniso_cascade(n, depth, sigma, seed, anisotropy=4.0):
    """Cascade with column-wise persistence: blocks elongated vertically so
    drips run top-to-bottom while remaining laterally narrow."""
    rng = default_rng(seed)
    logM = np.zeros((n, n))
    for j in range(1, depth + 1):
        s = 1 << j
        # narrower in x, taller in y -> drip-like
        sx = max(1, s // int(round(anisotropy)))
        block = rng.normal(-sigma ** 2 / 2, sigma, (s, sx))
        up = zoom(block, (n / s, n / sx), order=3)[:n, :n]
        if up.shape != (n, n):
            t = np.zeros((n, n)); t[:up.shape[0], :up.shape[1]] = up; up = t
        # only roll vertically; horizontal roll would smear the drip alignment
        logM += np.roll(up, int(rng.integers(0, n)), axis=0)
    logM -= logM.max()
    return np.exp(logM)


def _contrast(x, gain):
    x = _normalize01(x)
    return _normalize01(0.5 + np.tanh(gain * (x - 0.5)) / (2 * np.tanh(gain * 0.5)))


def _bright_target(x, target=0.5):
    """Monotonic gamma to hold mean brightness near target."""
    x = _normalize01(x)
    lo, hi = 0.12, 1.6
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if (x ** g).mean() < target:
            hi = g
        else:
            lo = g
    return x ** (0.5 * (lo + hi))


def rust_bloom(n=512, seed=None, complexity=0.5):
    """Irregular oxidation blooms spreading across a metal surface.

    A low-frequency cascade defines *where* rust is concentrated (the bloom
    layout); a higher-frequency cascade gives the *texture* of the rust
    (crusty fine detail); a soft threshold + multifractal-modulated edge
    produces the crumbly transition between rusted and intact regions.

    complexity in [0, 1]:
      cx=0  -> few isolated bloom patches on mostly-intact metal
      cx=1  -> rust covers most of the surface, blooms merging with rich
              crusty texture
    """
    cx = float(np.clip(complexity, 0, 1))
    s = seed if seed is not None else 0

    # Low-frequency "where is rust" envelope.
    env_seed = s
    env = _cascade(n, 5, 0.5, env_seed)
    env = _normalize01(np.log1p(env))
    env = gaussian_filter(env, n * 0.04)
    env = _normalize01(env)

    # High-frequency rust texture (crusty interior).
    sigma_tex = 0.55 + 0.5 * cx
    tex = _cascade(n, 7, sigma_tex, s + 101)
    tex = _normalize01(np.log1p(tex))

    # Crumbly edge: perturb the threshold spatially by mid-frequency cascade.
    edge_noise = _cascade(n, 6, 0.45, s + 211)
    edge_noise = _normalize01(np.log1p(edge_noise))
    edge_noise = gaussian_filter(edge_noise, n * 0.008)

    # cx routes the threshold: low cx -> rust mostly absent (high threshold),
    # high cx -> rust everywhere (low threshold). The threshold is locally
    # modulated by edge_noise to give crumbly bloom boundaries.
    thr = 0.62 - 0.45 * cx + 0.18 * (edge_noise - 0.5)
    mask = 1.0 / (1.0 + np.exp(-(env - thr) * (12.0 + 18.0 * cx)))

    # Combine: rust regions = mask * texture, intact regions = darker bare
    # metal with subtle reflectivity from a different texture seed.
    bare = 0.20 + 0.18 * _normalize01(_cascade(n, 6, 0.30, s + 313))
    bare = gaussian_filter(bare, 1.2)
    rusted = 0.30 + 0.65 * tex
    field = (1 - mask) * bare + mask * rusted
    field = _bright_target(field, 0.45)
    return _contrast(field, 1.4 + 1.6 * cx)


def rust_streaks(n=512, seed=None, complexity=0.5):
    """Vertical drip patterns from oxidation runoff.

    Strongly anisotropic cascade with column-wise persistence so brightness
    streaks vertically. Per-column rust mass (random drip-heights from seeds
    on the top edge) modulates streak density. Lateral wisps via mild
    horizontal warp.

    complexity in [0, 1]:
      cx=0  -> few wide smooth streaks (early oxidation runoff)
      cx=1  -> many narrow crisp drip lines with intricate side wisps
    """
    cx = float(np.clip(complexity, 0, 1))
    rng = default_rng(seed)

    # Anisotropic cascade: column-correlated, brightness streaks vertically.
    aniso = 3.5 + 2.5 * cx
    sigma = 0.40 + 0.55 * cx
    streak_base = _aniso_cascade(n, 7, sigma, seed, anisotropy=aniso)
    streak_base = _normalize01(np.log1p(streak_base))

    # Per-column rust-mass envelope: random number of drip seeds along the top
    # edge, each falling some random distance.
    yyN, xxN = np.mgrid[0:n, 0:n].astype(float) / n
    n_drips = int(8 + 36 * cx)
    drip_centers = rng.uniform(0, 1, n_drips)
    drip_widths = rng.uniform(0.015, 0.05, n_drips)
    drip_lengths = rng.uniform(0.30, 0.95, n_drips)
    drip_env = np.zeros((n, n))
    for c, w, L in zip(drip_centers, drip_widths, drip_lengths):
        lateral = np.exp(-((xxN - c) ** 2) / (2 * w ** 2))
        vertical = np.clip(yyN / L, 0, 1) * np.exp(-np.clip((yyN - L) / 0.05, 0, None))
        drip_env += lateral * vertical
    drip_env = _normalize01(drip_env)

    # Mild horizontal warp -> lateral wisps without breaking the vertical
    # persistence of the streak base.
    yyp, xxp = np.mgrid[0:n, 0:n].astype(float)
    warp_scale = n / (12.0 + 18.0 * cx)
    warp_amp = 2.0 + 6.0 * cx
    fx = gaussian_filter(rng.normal(0, 1, (n, n)), warp_scale) * warp_amp
    warped = map_coordinates(streak_base, [yyp, (xxp + fx) % n],
                             order=1, mode="grid-wrap")

    # Combine: dark background + drips + streaks.
    field = 0.18 + 0.55 * drip_env + 0.50 * drip_env * warped
    field = _bright_target(field, 0.42)
    return _contrast(field, 2.0 + 2.5 * cx)


def rust_pitted(n=512, seed=None, complexity=0.5):
    """Granular pitting and corrosion grit on a metallic surface.

    The "intact" surface is a moderate-brightness cascade; the pits are
    dark singularities punched through it. Multi-octave: large dark blotches
    (deep crater regions) + small dark pits (granular detail). Background
    is held mid-gray so colorization reads as intact ferric crust with
    dark cavities, not glittering metal flakes.

    complexity in [0, 1]:
      cx=0  -> smooth ochre crust with sparse subtle pits
      cx=1  -> heavy granular pitting at all scales
    """
    cx = float(np.clip(complexity, 0, 1))
    s = seed if seed is not None else 0

    # Intact-crust base: moderate cascade, mid-gray with soft variation.
    base = _cascade(n, 6, 0.30 + 0.20 * cx, s)
    base = _normalize01(np.log1p(base))
    base = gaussian_filter(base, 1.0)
    base = 0.45 + 0.35 * base

    # Pit mask: take a sparse high-frequency cascade and INVERT it so the
    # bright multifractal singularities become dark pits. Mid + small layers.
    pit_small = _cascade(n, 7, 0.55 + 0.55 * cx, s + 101)
    pit_small = _normalize01(np.log1p(pit_small))
    pit_small = _contrast(pit_small, 2.5 + 4.0 * cx)
    # Bright singularities -> dark pits.
    pit_small = 1.0 - pit_small

    pit_large = _cascade(n, 6, 0.45 + 0.30 * cx, s + 211)
    pit_large = _normalize01(np.log1p(pit_large))
    pit_large = gaussian_filter(pit_large, 1.5)
    pit_large = _contrast(pit_large, 1.8 + 2.0 * cx)
    pit_large = 1.0 - pit_large

    # Combine: multiply base by pit masks so each pit darkens the local
    # base value. Small pits dominate (granular detail); large pits give
    # the broad pitted patches.
    pit_strength_small = 0.45 + 0.40 * cx
    pit_strength_large = 0.30 + 0.30 * cx
    pits = (1 - pit_strength_small * (1 - pit_small)) * \
           (1 - pit_strength_large * (1 - pit_large))
    field = base * pits
    field = _bright_target(field, 0.42)
    return _contrast(field, 1.4 + 1.6 * cx)
