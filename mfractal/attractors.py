"""
attractors.py -- strange-attractor density families.

Different *kind* of base process: structure emerges from iterating a chaotic
2D map and accumulating the orbit as a 2D histogram. No stochasticity in the
dynamics -- the field's complexity comes from sensitivity to initial
conditions and parameter geometry, not from a random process. Visually:
filamentary wisps, fern-like or cosmic-cloud-like structures.

Two flagship maps:
  clifford_attractor:
      x_{n+1} = sin(a y_n) + c cos(a x_n)
      y_{n+1} = sin(b x_n) + d cos(b y_n)
  de_jong_attractor:
      x_{n+1} = sin(a y_n) - cos(b x_n)
      y_{n+1} = sin(c x_n) - cos(d y_n)

complexity in [0,1] routes iteration count (sparseness -> density) and a
small per-axis parameter perturbation so different seeds reveal different
slices of the (a,b,c,d) parameter geometry.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter

from .generators import _normalize01

__all__ = [
    "clifford_attractor", "de_jong_attractor", "ATTRACTOR_FAMILIES",
]

ATTRACTOR_FAMILIES = ("clifford_attractor", "de_jong_attractor")


def _iterate_clifford(a, b, c, d, n_iter, x0=0.1, y0=0.1):
    """Iterate the Clifford attractor map and return the (x, y) trajectory."""
    xs = np.empty(n_iter, dtype=np.float64)
    ys = np.empty(n_iter, dtype=np.float64)
    x, y = x0, y0
    for i in range(n_iter):
        x_new = np.sin(a * y) + c * np.cos(a * x)
        y_new = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_new, y_new
        xs[i] = x; ys[i] = y
    return xs, ys


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
    # both maps live in ~[-2.5, 2.5] x [-2.5, 2.5]; auto-fit to the orbit
    lo_x, hi_x = float(xs.min()), float(xs.max())
    lo_y, hi_y = float(ys.min()), float(ys.max())
    # pad 5%
    px = 0.05 * (hi_x - lo_x); py = 0.05 * (hi_y - lo_y)
    H, _, _ = np.histogram2d(
        ys, xs,  # rows = y so row 0 is the bottom; we'll flip later
        bins=(n, n),
        range=[[lo_y - py, hi_y + py], [lo_x - px, hi_x + px]],
    )
    H = np.flipud(H)
    if smooth > 0:
        H = gaussian_filter(H, smooth)
    # log compress to make sparse outliers visible
    H = np.log1p(H)
    return _normalize01(H)


def clifford_attractor(n=384, seed=None, complexity=0.5):
    """Clifford attractor density. cx routes iteration count + parameter
    perturbation; seed picks the parameter neighborhood."""
    cx = float(np.clip(complexity, 0.0, 1.0))
    rng = default_rng(seed)
    n_iter = int(150_000 + 850_000 * cx)
    # base params known to give rich orbits; perturb mildly with seed and cx
    base = np.array([-1.4, 1.6, 1.0, 0.7])
    jitter = 0.35 * (rng.random(4) - 0.5)
    perturb = 0.15 * cx * rng.standard_normal(4)
    a, b, c, d = base + jitter + perturb
    xs, ys = _iterate_clifford(a, b, c, d, n_iter)
    # discard transient
    xs = xs[200:]; ys = ys[200:]
    return _orbit_to_image(xs, ys, n, smooth=max(0.4, 1.2 - 0.8 * cx))


def de_jong_attractor(n=384, seed=None, complexity=0.5):
    """de Jong attractor density. cx routes iteration count + parameter
    perturbation; seed picks the parameter neighborhood."""
    cx = float(np.clip(complexity, 0.0, 1.0))
    rng = default_rng(seed)
    n_iter = int(150_000 + 850_000 * cx)
    base = np.array([1.4, -2.3, 2.4, -2.1])
    jitter = 0.30 * (rng.random(4) - 0.5)
    perturb = 0.15 * cx * rng.standard_normal(4)
    a, b, c, d = base + jitter + perturb
    xs, ys = _iterate_de_jong(a, b, c, d, n_iter)
    xs = xs[200:]; ys = ys[200:]
    return _orbit_to_image(xs, ys, n, smooth=max(0.4, 1.2 - 0.8 * cx))
