"""
reaction_diffusion.py -- emergent pattern families from the Gray-Scott PDE.

Gray-Scott is a two-species reaction-diffusion system:
  du/dt = Du * laplacian(u) - u v^2 + F (1 - u)
  dv/dt = Dv * laplacian(v) + u v^2 - (F + k) v

For appropriate (F, k) in the "Pearson zoo" the v field organizes into
spots, stripes, labyrinths, solitons, etc. Pareidolically rich -- distinct
from cascade-based textures because the structure is *biological* rather
than turbulent. Multifractal signature is mild but the spatial-statistics
distribution is unlike anything else in the toolbox.

Families:
  rd_coral      -- coral / brain-coral labyrinth.       (F=0.060, k=0.062)
  rd_labyrinth  -- pure stripe-labyrinth.               (F=0.039, k=0.058)
  rd_solitons   -- discrete spot pattern.               (F=0.030, k=0.062)

complexity in [0, 1] routes simulation time (fewer steps -> still-emerging
seed mosaic; more steps -> fully developed pattern) and the initial-condition
seed density (lower density -> sparser features; higher -> denser).
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng

from .generators import _normalize01

__all__ = [
    "rd_coral", "rd_labyrinth", "rd_solitons", "REACTION_DIFFUSION_FAMILIES",
]

REACTION_DIFFUSION_FAMILIES = ("rd_coral", "rd_labyrinth", "rd_solitons")


def _laplacian(field):
    """5-point Laplacian with periodic boundaries (np.roll-based, vectorized)."""
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
            - 4.0 * field)


def _gray_scott(n, F, k, steps, seed=None, Du=0.16, Dv=0.08, dt=1.0,
                seed_density=0.05, seed_size=4):
    """Run a Gray-Scott simulation. Returns the v field in [0, 1].

    Seed density: fraction of cells initially perturbed with v=1 patches.
    Seed size: pixel half-width of each perturbation patch.
    """
    rng = default_rng(seed)
    u = np.ones((n, n), dtype=np.float64)
    v = np.zeros((n, n), dtype=np.float64)
    # scatter seed patches
    n_seeds = max(1, int(seed_density * n * n / (2 * seed_size + 1) ** 2))
    cy = rng.integers(seed_size, n - seed_size, size=n_seeds)
    cx = rng.integers(seed_size, n - seed_size, size=n_seeds)
    for y, x in zip(cy, cx):
        u[y - seed_size:y + seed_size, x - seed_size:x + seed_size] = 0.5
        v[y - seed_size:y + seed_size, x - seed_size:x + seed_size] = 0.25
    # add a noisy nudge so each seed evolves slightly differently
    u += 0.01 * rng.normal(0, 1, (n, n))
    v += 0.01 * rng.normal(0, 1, (n, n))
    np.clip(u, 0.0, 1.0, out=u)
    np.clip(v, 0.0, 1.0, out=v)

    for _ in range(steps):
        Lu = _laplacian(u)
        Lv = _laplacian(v)
        uvv = u * v * v
        u += dt * (Du * Lu - uvv + F * (1.0 - u))
        v += dt * (Dv * Lv + uvv - (F + k) * v)
    return _normalize01(v)


def rd_coral(n=384, seed=None, complexity=0.5):
    """Coral / brain-coral labyrinth at (F=0.060, k=0.062). complexity in [0,1]
    routes simulation steps (1500 -> 6000) and seed density (sparse->dense)."""
    cx = float(np.clip(complexity, 0.0, 1.0))
    steps = int(1500 + 4500 * cx)
    sd = 0.02 + 0.06 * cx
    return _gray_scott(n, F=0.060, k=0.062, steps=steps,
                       seed=seed, seed_density=sd)


def rd_labyrinth(n=384, seed=None, complexity=0.5):
    """Pure stripe-labyrinth at (F=0.039, k=0.058). complexity routes
    simulation steps and seed density."""
    cx = float(np.clip(complexity, 0.0, 1.0))
    steps = int(2000 + 5000 * cx)
    sd = 0.03 + 0.05 * cx
    return _gray_scott(n, F=0.039, k=0.058, steps=steps,
                       seed=seed, seed_density=sd)


def rd_solitons(n=384, seed=None, complexity=0.5):
    """Discrete spot solitons at (F=0.030, k=0.062). complexity routes
    simulation steps and seed density (fewer / sparser -> isolated; more /
    denser -> tightly packed)."""
    cx = float(np.clip(complexity, 0.0, 1.0))
    steps = int(2000 + 4000 * cx)
    sd = 0.015 + 0.05 * cx
    return _gray_scott(n, F=0.030, k=0.062, steps=steps,
                       seed=seed, seed_density=sd)
