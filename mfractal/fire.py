"""
fire.py -- bright fire / molten / inferno textures.

Three families with drastically different spatial silhouettes:

  firestorm        -- compound-warped fire across the frame; cx grows from
                      broad smooth lobes to tight chaotic vortices.
  lava_pool        -- Voronoi cellular pools of brightness separated by darker
                      cooled crust; cx grows the cell count.
  volcanic_fissure -- meandering vertical cracks pouring fire upward through
                      a dark mass; cx grows the fissure count.

All return [0,1] floats; display with mfractal.flagships.punch.

Bounzaï-validated (june 2026) out of {inferno_field, firestorm, wildfire_ridge,
plasma_corona, lava_pool, volcanic_fissure}.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.spatial import cKDTree

from .generators import _normalize01

__all__ = ["firestorm", "lava_pool", "volcanic_fissure", "FIRE_FAMILIES"]

FIRE_FAMILIES = ("firestorm", "lava_pool", "volcanic_fissure")


def _bright_mf(n, seed, sigma=0.9, mean_target=0.55):
    """Smoothly-upsampled tamed lognormal cascade with mean renormalized to
    `mean_target`. Internal helper -- kept here so fire.py is self-contained."""
    rng = default_rng(seed)
    f = np.ones((n, n))
    levels = 7
    for l in range(levels):
        scale = 2 ** l
        sz = max(2, n // scale)
        w = np.exp(rng.normal(0, sigma, (sz, sz)))
        w_up = zoom(w, n / sz, order=1)[:n, :n] if sz != n else w
        f *= w_up
    log_f = np.log(np.clip(f, 1e-3, None))
    lo, hi = np.percentile(log_f, [5, 95])
    g = np.clip((log_f - lo) / max(hi - lo, 1e-6), 0, 1)
    m = float(g.mean())
    if m > 1e-6:
        g = np.clip(g * (mean_target / m), 0, 1)
    return g


def firestorm(n=512, seed=None, complexity=0.5):
    """Compound-warped fire. cx=0 = broad calm lobes, cx=1 = tight chaotic
    vortices. The base smoothness also scales with cx (heavy blur at cx=0 so
    the field reads as solid lobes; sharp at cx=1 so warps bite into fine
    structure)."""
    cx = float(np.clip(complexity, 0, 1))
    base = _bright_mf(n, seed if seed is not None else 0, sigma=0.85)
    base = gaussian_filter(base, 4.0 - 3.2 * cx)
    rng = default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    scale1 = n / (3.0 + 18.0 * cx)
    amp1 = 4.0 + 95.0 * cx
    wy = gaussian_filter(rng.normal(0, 1, (n, n)), scale1) * amp1
    wx = gaussian_filter(rng.normal(0, 1, (n, n)), scale1) * amp1
    warped = map_coordinates(base, [(yy + wy) % n, (xx + wx) % n],
                             order=1, mode="grid-wrap")

    if cx > 0.3:
        c2 = (cx - 0.3) / 0.7
        scale2 = n / (8.0 + 28.0 * c2)
        amp2 = 30.0 * c2
        wy2 = gaussian_filter(rng.normal(0, 1, (n, n)), scale2) * amp2
        wx2 = gaussian_filter(rng.normal(0, 1, (n, n)), scale2) * amp2
        warped = map_coordinates(warped, [(yy + wy2) % n, (xx + wx2) % n],
                                 order=1, mode="grid-wrap")

    warped = gaussian_filter(warped, 0.5 + 0.6 * (1 - cx))
    return _normalize01(0.20 + 0.75 * warped)


def lava_pool(n=512, seed=None, complexity=0.5):
    """Molten pools separated by darker cooled crust. complexity in [0,1]
    grows the cell count from ~64 (sqrt 8) to ~1024 (sqrt 32)."""
    cx = float(np.clip(complexity, 0, 1))
    rng = default_rng(seed)
    n_cells = int(8 + 24 * cx)
    pts = rng.uniform(0, n, size=(n_cells * n_cells, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    P = np.c_[xx.ravel(), yy.ravel()]
    d, _ = cKDTree(pts, boxsize=n).query(P, k=2)
    F1 = d[:, 0].reshape(n, n); F2 = d[:, 1].reshape(n, n)
    crust = 1 - np.exp(-(F2 - F1) / 6.0)
    pool = np.exp(-F1 / (n * 0.045))
    base = _bright_mf(n, (seed if seed is not None else 0) + 3, sigma=0.95)
    base = gaussian_filter(base, 1.8)
    field = (1 - 0.6 * crust) * (0.35 + 0.65 * pool) * (0.55 + 0.45 * base)
    return _normalize01(0.10 + 0.90 * field)


def volcanic_fissure(n=512, seed=None, complexity=0.5):
    """Meandering vertical cracks pouring fire upward through a dark mass.
    complexity in [0,1] grows the fissure count from 2 to 5."""
    cx = float(np.clip(complexity, 0, 1))
    base = _bright_mf(n, seed if seed is not None else 0, sigma=0.95)
    base = gaussian_filter(base, 1.0)
    rng = default_rng((seed if seed is not None else 0) + 11)
    n_fissures = int(2 + 3 * cx)
    yy, xx = np.mgrid[0:n, 0:n].astype(float) / n
    env = np.zeros((n, n))
    for _ in range(n_fissures):
        x0 = rng.uniform(0.15, 0.85)
        amp = rng.uniform(0.03, 0.10)
        freq = rng.uniform(3, 6)
        phi = rng.uniform(0, 2 * np.pi)
        width = rng.uniform(0.025, 0.05)
        center_x = x0 + amp * np.sin(freq * np.pi * yy + phi)
        rise = 0.30 + 0.85 * (1 - yy)
        env += rise * np.exp(-((xx - center_x) ** 2) / (2 * width ** 2))
    env = np.clip(env, 0, 1.4)
    field = 0.06 + 0.90 * env * (0.45 + 0.55 * base)
    return _normalize01(field)
