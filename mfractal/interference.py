"""
interference.py -- random plane-wave interference patterns.

Different *kind* of base process: the field is a finite sum of plane waves
with random angles, frequencies, and phases. Statistical character is
distinct from cascade- and process-based bases:
  - Bounded power spectrum (not 1/f^beta), set by the frequency-range knob.
  - Smooth analytic structure (no singularities at all -> c2 ~ 0, c1 ~ 1).
  - Moire / wave-front / fabric reads: standing-wave nodes and antinodes.

Useful as a *monofractal anchor* with completely different visual character
than fBM or warped_fbm -- and as a stimulus that breaks the implicit
assumption that "natural-looking texture" implies power-law spectrum.

Family:
  wave_interference(n, seed, complexity)
      cx routes wave count + frequency range. Low cx -> few broad waves
      (moire ripples); high cx -> many tight waves (fabric weave).
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng

from .generators import _normalize01

__all__ = ["wave_interference", "INTERFERENCE_FAMILIES"]

INTERFERENCE_FAMILIES = ("wave_interference",)


def wave_interference(n=384, seed=None, complexity=0.5, crispness=0.0):
    """Sum of random plane waves: field(x,y) = sum_i cos(k_i (cos theta_i x +
    sin theta_i y) + phi_i). Normalized to [0, 1].

    Parameters
    ----------
    complexity : float in [0, 1]
        routes wave count (N = 8..64) and frequency range
        ([2..12] -> [8..92]).
    crispness : float in [0, 1]
        post-process contrast push toward binary level-curves. 0 = smooth
        sums-of-cosines (default; original behavior); 1 = near-binary
        wave-front nodes (tanh saturation at high gain). Useful when you
        want the interference fringes to read as crisp lines rather than
        soft gradients.
    """
    cx = float(np.clip(complexity, 0.0, 1.0))
    cr = float(np.clip(crispness, 0.0, 1.0))
    rng = default_rng(seed)
    N = int(8 + 56 * cx)
    k_lo = 2.0 + 6.0 * cx
    k_hi = 12.0 + 80.0 * cx
    thetas = rng.uniform(0, 2 * np.pi, size=N)
    ks = rng.uniform(k_lo, k_hi, size=N)
    phis = rng.uniform(0, 2 * np.pi, size=N)
    amps = 1.0 / np.sqrt(1.0 + (ks / k_lo) ** 1.5)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    xx /= n; yy /= n
    field = np.zeros((n, n), dtype=np.float64)
    for k, th, ph, a in zip(ks, thetas, phis, amps):
        field += a * np.cos(k * (np.cos(th) * xx + np.sin(th) * yy) + ph)
    field = _normalize01(field)
    if cr > 0:
        gain = 1.0 + 18.0 * cr
        field = 0.5 + 0.5 * np.tanh(gain * (field - 0.5))
        field = _normalize01(field)
    return field
