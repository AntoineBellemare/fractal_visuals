"""
smoke.py -- soft rising-plume textures.

Two families: bright opaque smoke fields with a plume envelope (narrow column /
broad rising mass). complexity in [0,1] is routed to the fBM exponent of the
underlying texture, so cx=0 reads as smooth haze and cx=1 as turbulent wisps;
brightness stays stable since the field is renormalized.

  chimney_plume -- narrow rising column on a dim ground.
  billow_smoke  -- broad rising mass that fills most of the frame.

Both return [0,1] floats; display with mfractal.flagships.punch.

Bounzaï-validated (june 2026). The cx knob is a *visual-structure* knob (smooth
vs turbulent fBM); rho(c2, cx) is mildly positive and small -- bin experimental
stimuli by measured c2 per the project's validation philosophy.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter, map_coordinates

from .generators import _fractional_field, _normalize01

__all__ = ["chimney_plume", "billow_smoke", "SMOKE_FAMILIES"]

SMOKE_FAMILIES = ("chimney_plume", "billow_smoke")


def chimney_plume(n=512, seed=None, complexity=0.5):
    """Narrow rising smoke column on a dim ground.

    complexity in [0,1]: fBM beta sweeps from 4.0 (smooth haze) down to 2.0
    (turbulent wisps). A swirl warp on top makes the column meander more as
    cx grows.
    """
    cx = float(np.clip(complexity, 0, 1))
    yy, xx = np.mgrid[0:n, 0:n].astype(float) / n
    col_x = 0.5 + 0.05 * np.sin(5 * yy)
    col_w = 0.16 + 0.20 * yy
    rise = np.clip(1.2 * (1 - yy) + 0.3, 0.3, 1.0)
    env = rise * np.exp(-((xx - col_x) ** 2) / (2 * col_w ** 2))
    env = gaussian_filter(env, 2.0)

    beta = 4.0 - 2.0 * cx
    base = _normalize01(_fractional_field(n, beta=beta, seed=seed))

    rng = default_rng(seed)
    yyp, xxp = np.mgrid[0:n, 0:n].astype(float)
    swirl_amp = 18.0 * cx
    swirl_scale = n / (8.0 + 18.0 * cx)
    fy = gaussian_filter(rng.normal(0, 1, (n, n)), swirl_scale) * swirl_amp
    fx = gaussian_filter(rng.normal(0, 1, (n, n)), swirl_scale) * swirl_amp

    field0 = 0.25 + 0.55 * env + 0.40 * base * env
    warped = map_coordinates(field0, [(yyp + fy) % n, (xxp + fx) % n],
                             order=1, mode="grid-wrap")
    return _normalize01(warped)


def billow_smoke(n=512, seed=None, complexity=0.5):
    """Broad rising billowing smoke that fills most of the frame.

    complexity in [0,1]: same fBM-beta routing as chimney_plume.
    """
    cx = float(np.clip(complexity, 0, 1))
    yy, _ = np.mgrid[0:n, 0:n].astype(float) / n
    rise = 0.55 + 0.50 * (1 - yy)
    rise = gaussian_filter(rise, 1.0)

    beta = 4.0 - 2.0 * cx
    base = _normalize01(_fractional_field(n, beta=beta, seed=seed))

    rng = default_rng(seed)
    yyp, xxp = np.mgrid[0:n, 0:n].astype(float)
    swirl_amp = 22.0 * cx
    swirl_scale = n / (6.0 + 18.0 * cx)
    fy = gaussian_filter(rng.normal(0, 1, (n, n)), swirl_scale) * swirl_amp
    fx = gaussian_filter(rng.normal(0, 1, (n, n)), swirl_scale) * swirl_amp

    field0 = 0.30 * rise + 0.65 * base * rise
    warped = map_coordinates(field0, [(yyp + fy) % n, (xxp + fx) % n],
                             order=1, mode="grid-wrap")
    return _normalize01(warped)
