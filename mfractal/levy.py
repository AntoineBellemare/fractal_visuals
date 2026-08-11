"""
levy.py -- universal multifractal cascades a la Schertzer & Lovejoy.

A discrete pyramid multiplicative cascade where the per-level log-multipliers
are drawn from an alpha-stable Levy distribution. This generalizes the MRW /
lognormal cascade (recovered at alpha=2) along an extra "multifractality
index" axis alpha in (0, 2]:

  alpha = 2.0  -> Gaussian log-multipliers (smooth, MRW-like)
  alpha < 2.0  -> heavy-tailed log-multipliers, sparse and clustered
                  singularities (bright veins, dark sinks)

Two intermittency knobs:
  alpha (multifractality index)  -- distribution shape
  sigma (intermittency strength) -- log-multiplier scale

Family:
  universal_cascade -- bare log-Levy cascade, brightness renormalized

Reference: Schertzer & Lovejoy (1987), "Physical modeling and analysis of rain
and clouds by anisotropic scaling multiplicative processes", J. Geophys. Res.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import zoom
from scipy.stats import levy_stable

from .generators import _normalize01

__all__ = ["universal_cascade", "LEVY_FAMILIES"]

LEVY_FAMILIES = ("universal_cascade",)


def _log_levy_cascade(n, alpha, sigma, seed, depth=7):
    """Centered log-Levy multiplicative cascade. Mirrors the structure of
    clouds._cascade (centered log-normal cascade) but with alpha-stable
    log-multipliers. Returns the exp() of the accumulated log-field.

    The per-level log-cascade is centered so the exp() has unit expectation
    (in the Gaussian limit alpha=2, the centering is -sigma^2/2; for alpha<2
    we approximate the same role with -sigma * E[S+] where S~stable). This
    keeps the cascade well-behaved across the alpha axis.
    """
    alpha = float(np.clip(alpha, 1.05, 2.0))
    rng = default_rng(seed)
    log_M = np.zeros((n, n), dtype=np.float64)
    for j in range(1, depth + 1):
        s = 1 << j
        if alpha >= 2.0 - 1e-6:
            block = rng.normal(-sigma ** 2 / 2.0, sigma, size=(s, s))
        else:
            seed_l = int(rng.integers(0, 2**31 - 1))
            raw = levy_stable.rvs(alpha=alpha, beta=-1.0, loc=0.0, scale=1.0,
                                  size=(s, s), random_state=seed_l)
            raw = np.clip(raw, -25.0, 25.0)
            # empirical mean of beta=-1 stable is negative; sigma*scale +
            # subtract block mean to enforce zero mean log per level
            block = sigma * raw
            block -= block.mean()
            # nudge by -0.5*sigma^2 to match lognormal centering at alpha=2
            block -= 0.5 * sigma * sigma
        block_up = zoom(block, n / s, order=3)[:n, :n]
        if block_up.shape != (n, n):
            t = np.zeros((n, n)); t[:block_up.shape[0], :block_up.shape[1]] = block_up
            block_up = t
        log_M += np.roll(block_up,
                         (int(rng.integers(0, n)), int(rng.integers(0, n))),
                         axis=(0, 1))
    log_M -= log_M.max()
    return np.exp(log_M)


def _fracint(field, H):
    """Spectral fractional integration: F(k) -> F(k) / |k|^H. Smooths the
    cascade into a correlated multifractal field (matches clouds._fracint)."""
    n = field.shape[0]
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2); k[0, 0] = 1.0
    F = np.fft.fft2(field) / (k ** H); F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F))


def _auto_gamma(x, target=0.5):
    """Monotonic gamma adjustment to hold mean brightness at `target`. Mirrors
    clouds._cloud_bright -- structure-preserving, unlike clip-based brightness
    fixes which destroy the c2 signature."""
    x = _normalize01(x)
    lo, hi = 0.12, 1.6
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if (x ** g).mean() < target:
            hi = g
        else:
            lo = g
    return x ** (0.5 * (lo + hi))


def universal_cascade(n=512, seed=None, complexity=0.5,
                      alpha=None, sigma=None, H=None):
    """Bare log-Levy multiplicative cascade with fractional integration.

    complexity in [0, 1] routes (alpha, sigma, H) along an arc:
      cx=0.0 -> alpha=2.0, sigma=0.20, H=0.72 (smooth Gaussian / MRW-like)
      cx=1.0 -> alpha=1.30, sigma=0.95, H=0.42 (sparse heavy-tailed, clustered)

    Override any of (alpha, sigma, H) for direct control.
      alpha   -- multifractality index in (1.05, 2.0]; 2.0 = Gaussian limit
      sigma   -- intermittency strength (~ analog of C1 axis)
      H       -- fractional integration exponent (correlates the field)
    """
    cx = float(np.clip(complexity, 0.0, 1.0))
    a = (2.0 - 0.7 * cx) if alpha is None else float(alpha)
    s = (0.20 + 0.75 * cx) if sigma is None else float(sigma)
    h = (0.72 - 0.30 * cx) if H is None else float(H)
    cascade = _log_levy_cascade(n, alpha=a, sigma=s, seed=seed, depth=7)
    field = _fracint(cascade, h)
    return _auto_gamma(field)
