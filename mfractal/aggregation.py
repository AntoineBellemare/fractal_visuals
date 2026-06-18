"""
aggregation.py -- diffusion-limited aggregation (DLA) as a stimulus family.

Different *kind* of base process from anything else in the toolbox:
the structure is built by random walkers sticking to a growing cluster,
not by spectral or cascade synthesis. The cluster's coarse occupancy is a
harmonic-measure multifractal -- a *physical* multifractal in the sense
that it arises from a microscopic stochastic process rather than a
stipulated singularity spectrum.

Pareidolically: dendritic forks read as neurons, lightning, frost on glass,
coral. Single-center DLA in particular triggers strong "branched living
thing" recognition.

Family:
  dla_dendrite(n, seed, complexity)
      -- single-center walker DLA, blurred and normalized.
      -- complexity in [0,1] routes particle count (cluster size) + blur:
         cx=0.0 -> 800 particles, blur 1.8  (small soft blob)
         cx=1.0 -> 5000 particles, blur 0.7 (large crisp dendrite)
"""
from __future__ import annotations
import numpy as np

from .generators import _normalize01
from .organic import dla as _dla_walker

__all__ = ["dla_dendrite", "AGGREGATION_FAMILIES"]

AGGREGATION_FAMILIES = ("dla_dendrite",)


def dla_dendrite(n=384, seed=None, complexity=0.5):
    """Single-center diffusion-limited aggregation rendered as a blurred field."""
    cx = float(np.clip(complexity, 0.0, 1.0))
    n_particles = int(800 + 4200 * cx)
    blur = 1.8 - 1.1 * cx
    # walker DLA assumes odd grid size; pad to nearest odd value
    n_grid = n if n % 2 == 1 else n + 1
    field = _dla_walker(n=n_grid, n_particles=n_particles,
                        stick=1.0, blur=blur, seed=seed)
    if field.shape != (n, n):
        field = field[:n, :n]
    return _normalize01(field)
