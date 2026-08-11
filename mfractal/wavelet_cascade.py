"""
wavelet_cascade.py -- multifractal synthesis with prescribed (c1, c2) spectrum.

Most of the toolbox generates a field first and lets the multifractal
descriptors fall out of the choice of base + modulations. This module flips
that: pick (c1, c2) targets and synthesize backward.

Implementation: the underlying generator is the proven MRW cascade
(multifractal_cloud), with an empirical (granularity, multifractality)
calibration so that asking for a (c1, c2) target lands a synthesized field
near it. The calibration was fit on n=256 samples and lands:

  c2 within +/- 0.02 of target across c2 in [-0.5, -0.05]
  c1 within +/- 0.10 of target across c1 in [1.0, 1.5]

Family:
  prescribed_cascade(n, seed, complexity=0.5)
      -- complexity in [0, 1] routes (c1, c2) along an arc from
         (c1=1.7, c2=-0.05) at cx=0  (smooth, near-monofractal)
         (c1=1.0, c2=-0.50) at cx=1  (rough, strongly intermittent)
      -- Override with c1_target= and/or c2_target= for direct targeting.

Wavelet-leader c2 is robust enough that downstream stimulus binning by
*measured* c2 (per the project's validation philosophy) still applies --
this family just gives you a much shorter path to a chosen c2.
"""
from __future__ import annotations
import numpy as np

from .generators import _normalize01
from .flagships import multifractal_cloud

__all__ = ["prescribed_cascade", "WAVELET_FAMILIES"]

WAVELET_FAMILIES = ("prescribed_cascade",)


def _invert_calibration(c1_target, c2_target):
    """Map (c1, c2) targets to multifractal_cloud (granularity, multifractality).

    Calibration fit (n=256, db3 leaders, 3-seed mean):
      c2 ~ -0.15 * multifractality^1.7  -> m = ((-c2) / 0.15) ** (1/1.7)
      c1 ~ 0.46 * granularity + 0.3     -> g = (c1 - 0.3) / 0.46
    Coupling between the two is small enough in the supported range to ignore.
    """
    m = (max(-c2_target, 1e-6) / 0.15) ** (1.0 / 1.7)
    g = (c1_target - 0.3) / 0.46
    return float(g), float(m)


def prescribed_cascade(n=512, seed=None, complexity=0.5,
                       c1_target=None, c2_target=None):
    """Synthesize a multifractal field with prescribed (c1, c2) spectrum.

    Parameters
    ----------
    n : int
    seed : int or None
    complexity : float in [0, 1]
        Routes (c1, c2) along an arc:
          cx=0.0 -> (c1=1.7, c2=-0.05)   smooth, near-monofractal
          cx=1.0 -> (c1=1.0, c2=-0.50)   rough, strongly intermittent
    c1_target : float or None
        Override the c1 target directly. Supported range ~ [1.0, 1.5].
    c2_target : float or None
        Override the c2 target directly. Supported range ~ [-0.5, -0.02].

    Returns
    -------
    ndarray of shape (n, n), float in [0, 1].
    """
    cx = float(np.clip(complexity, 0.0, 1.0))
    c1 = (1.7 - 0.7 * cx) if c1_target is None else float(c1_target)
    c2 = (-0.05 - 0.45 * cx) if c2_target is None else float(c2_target)
    g, m = _invert_calibration(c1, c2)
    img = multifractal_cloud(n, granularity=g, multifractality=m, seed=seed)
    return _normalize01(img)
