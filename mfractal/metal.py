"""
metal.py -- corrosion / oxidation textures.

Three families covering distinct rust morphologies:

  rust_bloom    -- irregular oxidation blooms spreading across a metal surface.
                   Low-frequency cascade envelope picks bloom location; higher
                   cascade fills with crusty texture; a soft, multifractally-
                   perturbed threshold gives crumbly bloom boundaries. cx grows
                   bloom coverage and texture richness.
  rust_pitted   -- granular pitting on a corroded surface. Bright multifractal
                   singularities are inverted to dark pits punched through a
                   mid-gray ferric crust; multi-octave (large patches + small
                   pits). cx grows pit density and crispness.
  rust_dewy     -- bright ferric beads (small wet rust droplets / crystallized
                   ferric deposits) sprinkled across a darker oxidized base.
                   Inverse polarity of rust_pitted: bright concentrations on
                   dark, not dark pits on bright. cx grows bead density.

All return [0, 1] floats; display with mfractal.flagships.punch and colorize
with the FAMILY_PALETTES entries (rust_oxide / corrosion_grit / ferric_dew).
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter, zoom

from .generators import _normalize01

__all__ = ["rust_bloom", "rust_pitted", "rust_dewy", "METAL_FAMILIES"]

METAL_FAMILIES = ("rust_bloom", "rust_pitted", "rust_dewy")


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

    env_seed = s
    env = _cascade(n, 5, 0.5, env_seed)
    env = _normalize01(np.log1p(env))
    env = gaussian_filter(env, n * 0.04)
    env = _normalize01(env)

    sigma_tex = 0.55 + 0.5 * cx
    tex = _cascade(n, 7, sigma_tex, s + 101)
    tex = _normalize01(np.log1p(tex))

    edge_noise = _cascade(n, 6, 0.45, s + 211)
    edge_noise = _normalize01(np.log1p(edge_noise))
    edge_noise = gaussian_filter(edge_noise, n * 0.008)

    thr = 0.62 - 0.45 * cx + 0.18 * (edge_noise - 0.5)
    mask = 1.0 / (1.0 + np.exp(-(env - thr) * (12.0 + 18.0 * cx)))

    bare = 0.20 + 0.18 * _normalize01(_cascade(n, 6, 0.30, s + 313))
    bare = gaussian_filter(bare, 1.2)
    rusted = 0.30 + 0.65 * tex
    field = (1 - mask) * bare + mask * rusted
    field = _bright_target(field, 0.45)
    return _contrast(field, 1.4 + 1.6 * cx)


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

    base = _cascade(n, 6, 0.30 + 0.20 * cx, s)
    base = _normalize01(np.log1p(base))
    base = gaussian_filter(base, 1.0)
    base = 0.45 + 0.35 * base

    # Bright multifractal singularities -> dark pits via inversion.
    pit_small = _cascade(n, 7, 0.55 + 0.55 * cx, s + 101)
    pit_small = _normalize01(np.log1p(pit_small))
    pit_small = _contrast(pit_small, 2.5 + 4.0 * cx)
    pit_small = 1.0 - pit_small

    pit_large = _cascade(n, 6, 0.45 + 0.30 * cx, s + 211)
    pit_large = _normalize01(np.log1p(pit_large))
    pit_large = gaussian_filter(pit_large, 1.5)
    pit_large = _contrast(pit_large, 1.8 + 2.0 * cx)
    pit_large = 1.0 - pit_large

    pit_strength_small = 0.45 + 0.40 * cx
    pit_strength_large = 0.30 + 0.30 * cx
    pits = (1 - pit_strength_small * (1 - pit_small)) * \
           (1 - pit_strength_large * (1 - pit_large))
    field = base * pits
    field = _bright_target(field, 0.42)
    return _contrast(field, 1.4 + 1.6 * cx)


def rust_dewy(n=512, seed=None, complexity=0.5):
    """Dense ferric beads scattered across a darker corroded base.

    Inverse character of rust_pitted: instead of dark pits punched through
    a bright crust, this generates bright ferric concentrations (small wet
    rust droplets / crystallized ferric deposits) sprinkled across a
    darker oxidized base. Multi-scale -- larger blooms underlay small
    sharp-bright droplets.

    complexity in [0, 1]:
      cx=0  -> few large bright concentrations on darker base
      cx=1  -> dense fine ferric beading at all scales
    """
    cx = float(np.clip(complexity, 0, 1))
    s = seed if seed is not None else 0

    base = _cascade(n, 6, 0.30 + 0.20 * cx, s)
    base = _normalize01(np.log1p(base))
    base = gaussian_filter(base, 1.2)
    base = 0.16 + 0.22 * base

    # Sparse bright multifractal singularities kept bright (no inversion).
    beads_small = _cascade(n, 7, 0.55 + 0.55 * cx, s + 101)
    beads_small = _normalize01(np.log1p(beads_small))
    beads_small = _contrast(beads_small, 2.8 + 4.0 * cx)

    beads_mid = _cascade(n, 7, 0.50 + 0.30 * cx, s + 211)
    beads_mid = _normalize01(np.log1p(beads_mid))
    beads_mid = gaussian_filter(beads_mid, 0.8)
    beads_mid = _contrast(beads_mid, 2.0 + 2.5 * cx)

    bead_strength_small = 0.55 + 0.30 * cx
    bead_strength_mid = 0.35 + 0.25 * cx
    field = base + bead_strength_small * beads_small + bead_strength_mid * beads_mid * 0.6
    field = np.clip(field, 0, 1)
    field = _bright_target(field, 0.42)
    return _contrast(field, 1.6 + 1.8 * cx)
