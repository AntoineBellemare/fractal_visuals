"""
attractors.py -- strange-attractor density family.

Different *kind* of base process: structure emerges from iterating a chaotic
2D map and accumulating the orbit as a 2D histogram. No stochasticity in the
dynamics -- the field's complexity comes from sensitivity to initial
conditions and parameter geometry, not from a random process. Visually:
filamentary wisps, fern-like or cosmic-cloud-like structures.

  de_jong_attractor:
      x_{n+1} = sin(a y_n) - cos(b x_n)
      y_{n+1} = sin(c x_n) - cos(d y_n)

seed picks (a, b, c, d) from a wide parameter range so different seeds give
visually distinct attractors. cx routes iteration count.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter

from .generators import _normalize01

__all__ = ["de_jong_attractor", "ATTRACTOR_FAMILIES"]

ATTRACTOR_FAMILIES = ("de_jong_attractor",)


def _iterate_de_jong(a, b, c, d, n_iter, x0=0.1, y0=0.1):
    """Iterate the de Jong attractor map and return the (x, y) trajectory."""
    xs = np.empty(n_iter, dtype=np.float64)
    ys = np.empty(n_iter, dtype=np.float64)
    x, y = x0, y0
    for i in range(n_iter):
        x_new = np.sin(a * y) - np.cos(b * x)
        y_new = np.sin(c * x) - np.cos(d * y)
        x, y = x_new, y_new
        xs[i] = x; ys[i] = y
    return xs, ys


def _orbit_to_image(xs, ys, n, smooth=0.7):
    """Bin a trajectory into a (n, n) histogram, log-compressed and smoothed."""
    lo_x, hi_x = float(xs.min()), float(xs.max())
    lo_y, hi_y = float(ys.min()), float(ys.max())
    px = 0.05 * (hi_x - lo_x); py = 0.05 * (hi_y - lo_y)
    H, _, _ = np.histogram2d(
        ys, xs,
        bins=(n, n),
        range=[[lo_y - py, hi_y + py], [lo_x - px, hi_x + px]],
    )
    H = np.flipud(H)
    if smooth > 0:
        H = gaussian_filter(H, smooth)
    H = np.log1p(H)
    return _normalize01(H)


def _orbit_quality(xs, ys, min_bbox=0.6, min_occupancy=0.04, probe_bins=96):
    """True if the orbit is a non-trivial strange attractor: finite, bbox
    >= min_bbox per axis, histogram occupancy >= min_occupancy. The
    occupancy check catches short cycles that pass the bbox check but
    visit too few cells."""
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        return False
    bx = float(xs.max() - xs.min())
    by = float(ys.max() - ys.min())
    if bx < min_bbox or by < min_bbox:
        return False
    H, _, _ = np.histogram2d(ys, xs, bins=probe_bins)
    occupancy = float((H > 0).mean())
    return occupancy >= min_occupancy


def _sample_attractor(iterate_fn, n_iter, rng, low=-2.5, high=2.5, tries=40):
    """Sample (a, b, c, d) from a wide range; retry if the orbit isn't a
    non-trivial strange attractor. Returns (xs, ys, params). Uses an 8k
    probe iter to cheaply discard degenerate sets; only accepted sets are
    iterated for the full n_iter."""
    probe_iters = 8000
    for _ in range(tries):
        params = rng.uniform(low, high, size=4)
        xs, ys = iterate_fn(*params, n_iter=probe_iters)
        xs = xs[400:]; ys = ys[400:]
        if _orbit_quality(xs, ys):
            xs, ys = iterate_fn(*params, n_iter=n_iter)
            return xs[400:], ys[400:], params
    xs, ys = iterate_fn(*params, n_iter=n_iter)
    return xs[400:], ys[400:], params


def de_jong_attractor(n=384, seed=None, complexity=0.5):
    """de Jong attractor density. seed picks (a, b, c, d) from a wide
    parameter range so different seeds give visually distinct attractors.
    cx routes iteration count."""
    cx = float(np.clip(complexity, 0.0, 1.0))
    rng = default_rng(seed)
    n_iter = int(150_000 + 850_000 * cx)
    xs, ys, _ = _sample_attractor(_iterate_de_jong, n_iter, rng,
                                   low=-2.7, high=2.7)
    return _orbit_to_image(xs, ys, n, smooth=max(0.4, 1.2 - 0.8 * cx))
