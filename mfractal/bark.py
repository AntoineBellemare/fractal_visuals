"""
bark.py -- intricate-knot wood surface textures.

Two families: domain-warped cosine grain studded with elliptical knots that have
multiple concentric growth rings and radial cracks split through them. The grain
field bends radially around each knot, so the wood fibres read as flowing around
the dead-knot scars (as in real timber).

  burled_oak -- dense overlapping knot population on a heavily warped grain.
  riven_oak  -- a few maximally detailed anchor knots (many rings + cracks)
                in a more open grain field.

Both accept complexity in [0,1]: more complexity -> more anchor knots, more
rings/cracks per anchor, and a stronger multifractal modulation of the wood
value. Both return [0,1] floats; display with mfractal.flagships.punch.

Bounzaï-validated (june 2026): chosen out of {gnarled_oak, burled_oak, riven_oak}
for visual intricacy. NOTE: their rho(c2, complexity) is positive (more knots ->
more spatial uniformity -> less intermittency) -- the complexity knob is a
*visual-intricacy* knob here, not a multifractality knob. Bin experimental
stimuli by *measured* c2, per the validation philosophy in README.md.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter

__all__ = ["burled_oak", "riven_oak", "BARK_FAMILIES"]

BARK_FAMILIES = ("burled_oak", "riven_oak")


# ---------- internals ----------

def _normalize01(f):
    lo, hi = float(f.min()), float(f.max())
    if hi - lo < 1e-9:
        return np.zeros_like(f)
    return (f - lo) / (hi - lo)


def _cascade_log(n, levels, sigma, seed):
    """Lognormal multiplicative cascade -> tamed brightness field in [0,1].

    Raw lognormal cascades have heavy-tail blowups (the 'plume' failure mode); we
    log + percentile-clip 5/95 to keep the field bright and well-conditioned.
    """
    rng = default_rng(seed)
    f = np.ones((n, n))
    for l in range(levels):
        scale = 2 ** l
        sz = max(1, n // scale)
        w = np.exp(rng.normal(0, sigma, (sz, sz)))
        w_up = np.repeat(np.repeat(w, scale, axis=0), scale, axis=1)[:n, :n]
        f *= w_up
    log_f = np.log(np.clip(f, 1e-3, None))
    lo, hi = np.percentile(log_f, [5, 95])
    return np.clip((log_f - lo) / max(hi - lo, 1e-6), 0, 1)


def _perlin_like(n, seed, octaves=5, persistence=0.55):
    """Cheap multi-octave smooth noise via gaussian-blurred white noise stacks."""
    rng = default_rng(seed)
    out = np.zeros((n, n))
    amp = 1.0
    norm = 0.0
    for o in range(octaves):
        sigma = max(1.5, n / (4.0 * (2 ** o)))
        layer = gaussian_filter(rng.normal(0, 1, (n, n)), sigma)
        layer = _normalize01(layer) * 2 - 1
        out += amp * layer
        norm += amp
        amp *= persistence
    return out / norm


def _grain_field(n, seed, knot_centers, knot_strengths,
                 grain_freq, knot_warp_radius, knot_warp_amp):
    """Cosine ridges in y, warped by a smooth global flow + per-knot radial warp."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float) / n
    fy = 0.06 * _perlin_like(n, seed * 7 + 1, octaves=4)
    fx = 0.04 * _perlin_like(n, seed * 7 + 2, octaves=4)
    for (kx, ky), ks in zip(knot_centers, knot_strengths):
        r2 = (xx - kx) ** 2 + (yy - ky) ** 2
        mask = np.exp(-r2 / (2 * knot_warp_radius ** 2))
        fy += ks * knot_warp_amp * mask * (yy - ky)
    phase = grain_freq * (yy + fy) + 0.6 * np.sin(grain_freq * (xx + fx))
    ridges = 0.5 * (1 + np.cos(2 * np.pi * phase))
    return 0.55 + 0.35 * ridges


def _intricate_knot(field, n, cx_px, cy_px, base_r_px, rng,
                    n_rings, n_cracks, ring_decay, core_dark, rim_dark):
    """Paint one intricate knot in-place (elliptical core + rings + cracks)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    theta = rng.uniform(0, np.pi)
    a = base_r_px * rng.uniform(0.8, 1.0)
    b = base_r_px * rng.uniform(0.45, 0.75)
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xx - cx_px) * ct + (yy - cy_px) * st
    yr = -(xx - cx_px) * st + (yy - cy_px) * ct
    rr = np.sqrt((xr / max(a, 1)) ** 2 + (yr / max(b, 1)) ** 2)
    ang = np.arctan2(yr, xr)

    core_mask = np.exp(-(rr / 0.6) ** 2)
    field -= core_dark * core_mask

    amp = 1.0
    for k in range(n_rings):
        r0 = 0.7 + 0.55 * k
        ring = np.exp(-((rr - r0) / 0.16) ** 2)
        sign = -1.0 if k % 2 == 0 else +0.45
        field += sign * rim_dark * amp * ring
        amp *= ring_decay

    if n_cracks > 0:
        crack_outer = 0.7 + 0.55 * n_rings + 0.4
        crack_R = np.clip(crack_outer - rr, 0, None) / max(crack_outer, 1e-3)
        crack_env = (rr > 0.5).astype(float) * crack_R
        for a0 in rng.uniform(0, 2 * np.pi, n_cracks):
            d = np.angle(np.exp(1j * (ang - a0)))
            crack = np.exp(-(d / 0.05) ** 2)
            field -= 0.22 * crack_env * crack
    return field


def _place_knots(rng, n_anchor, n_satellite,
                 anchor_r, sat_r, anchor_min_sep):
    """Return list of (cx_norm, cy_norm, r_norm, strength)."""
    anchors = []
    tries = 0
    while len(anchors) < n_anchor and tries < 500:
        x = rng.uniform(0.12, 0.88)
        y = rng.uniform(0.12, 0.88)
        if all(np.hypot(x - ax, y - ay) > anchor_min_sep for ax, ay, *_ in anchors):
            anchors.append((x, y, rng.uniform(*anchor_r), 1.0))
        tries += 1
    satellites = []
    for ax, ay, ar, _ in anchors:
        for _ in range(int(rng.integers(3, 7))):
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(ar * 2.0, ar * 4.0)
            sx = float(np.clip(ax + dist * np.cos(ang), 0.05, 0.95))
            sy = float(np.clip(ay + dist * np.sin(ang), 0.05, 0.95))
            satellites.append((sx, sy, rng.uniform(*sat_r),
                               float(rng.uniform(0.35, 0.7))))
    for _ in range(max(0, n_satellite - len(satellites))):
        satellites.append((float(rng.uniform(0.05, 0.95)),
                           float(rng.uniform(0.05, 0.95)),
                           float(rng.uniform(*sat_r)),
                           float(rng.uniform(0.25, 0.55))))
    return anchors + satellites


# ---------- public families ----------

def burled_oak(n=512, seed=None, complexity=0.5):
    """Dense burl: many overlapping anchor knots on a heavily warped grain.

    complexity in [0,1] grows anchor count (6..14), satellite count (25..50),
    and the cascade modulation sigma (0.75..1.25). Per-anchor ring count
    (4..7) and crack count (6..10).
    """
    rng = default_rng(seed)
    cx = float(np.clip(complexity, 0, 1))
    n_anchor = int(6 + 8 * cx)
    knots = _place_knots(rng, n_anchor=n_anchor,
                         n_satellite=int(25 + 25 * cx),
                         anchor_r=(0.05, 0.085),
                         sat_r=(0.014, 0.032),
                         anchor_min_sep=0.12)
    centers = [(kx, ky) for kx, ky, *_ in knots]
    strengths = [s for *_, s in knots]
    grain = _grain_field(n, seed, centers, strengths,
                         grain_freq=18.0, knot_warp_radius=0.18, knot_warp_amp=2.2)
    mod = _cascade_log(n, 7, 0.75 + 0.5 * cx,
                       (None if seed is None else seed + 11))
    field = 0.72 * grain + 0.28 * mod
    for (kx, ky, r, strength) in knots:
        is_anchor = strength >= 0.9
        n_rings = int(rng.integers(4, 7) if is_anchor else rng.integers(2, 4))
        n_cracks = int(rng.integers(6, 10) if is_anchor else rng.integers(0, 3))
        core_dark = (0.24 if is_anchor else 0.16) * (0.5 + 0.5 * cx)
        rim_dark = (0.34 if is_anchor else 0.24) * (0.5 + 0.5 * cx)
        field = _intricate_knot(field, n, kx * n, ky * n, r * n, rng,
                                n_rings=n_rings, n_cracks=n_cracks,
                                ring_decay=0.55, core_dark=core_dark,
                                rim_dark=rim_dark)
    return _normalize01(np.clip(field, 0, 1))


def riven_oak(n=512, seed=None, complexity=0.5):
    """Sparse but maximally detailed: a few anchor knots, each with many rings
    and many cracks split out through the surrounding grain.

    complexity in [0,1] grows anchor count (2..5), satellite count (8..20),
    and per-anchor detail (5..8 rings, 8..12 cracks).
    """
    rng = default_rng(seed)
    cx = float(np.clip(complexity, 0, 1))
    n_anchor = int(2 + 3 * cx)
    knots = _place_knots(rng, n_anchor=n_anchor,
                         n_satellite=int(8 + 12 * cx),
                         anchor_r=(0.06, 0.10),
                         sat_r=(0.015, 0.03),
                         anchor_min_sep=0.30)
    centers = [(kx, ky) for kx, ky, *_ in knots]
    strengths = [s for *_, s in knots]
    grain = _grain_field(n, seed, centers, strengths,
                         grain_freq=20.0, knot_warp_radius=0.22, knot_warp_amp=2.5)
    mod = _cascade_log(n, 7, 0.6 + 0.4 * cx,
                       (None if seed is None else seed + 11))
    field = 0.75 * grain + 0.25 * mod
    for (kx, ky, r, strength) in knots:
        is_anchor = strength >= 0.9
        n_rings = int(rng.integers(5, 8) if is_anchor else rng.integers(2, 4))
        n_cracks = int(rng.integers(8, 12) if is_anchor else rng.integers(0, 2))
        core_dark = (0.28 if is_anchor else 0.15) * (0.5 + 0.5 * cx)
        rim_dark = (0.38 if is_anchor else 0.22) * (0.5 + 0.5 * cx)
        field = _intricate_knot(field, n, kx * n, ky * n, r * n, rng,
                                n_rings=n_rings, n_cracks=n_cracks,
                                ring_decay=0.68, core_dark=core_dark,
                                rim_dark=rim_dark)
    return _normalize01(np.clip(field, 0, 1))
