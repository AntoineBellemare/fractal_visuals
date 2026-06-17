"""
flagships.py -- the two finalized stimulus generators for the pareidolia work,
with the recipes converged on through visual iteration.

FLAGSHIP 1 -- multifractional() : within-image varying fractal dimension.
    Full brightness. Heterogeneous local roughness. Controls the *local-FD
    spectrum* (a region's smoothness varies across the image). NOT multifractal
    in the moment sense (by design); quantify with local_fd_map / heterogeneity.

FLAGSHIP 2 -- multifractal_cloud() : full-brightness multifractal field.
    Volatility-modulated fractional field (2D multifractal random walk) with
    the gravest modes high-passed (no dominant gray patch) and singularities
    spatially correlated (connected black structures, not scattered dots).
    Genuinely multifractal in the moment sense; quantify with mfdfa_2d.

Display with punch() for legible black/white. Use complexity() for a single
simple->complex macro-dial, or sample_battery() to draw a varied stimulus set.

Design rules discovered along the way (see README):
  * multifractality lives in amplitude -> linear contrast OK, equalize/threshold
    destroys it; mBm heterogeneity is geometric -> survives thresholding better.
  * strong multifractality necessarily concentrates activity (dark/calm regions)
    -> the moderate range is the perceptual sweet spot.
"""
from __future__ import annotations
import numpy as np

from .generators import (_fractional_field, _power_law_filter, _normalize01,
                         _radial_freq, mbm_blend)
from .quantify import mfdfa_2d, spectrum_summary


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _highpass(field, kcut_cycles):
    """Suppress modes below kcut_cycles (cycles across the image)."""
    if kcut_cycles <= 0:
        return field
    k = _radial_freq(field.shape[0]); kc = kcut_cycles / field.shape[0]
    h = np.where(k > 0, k ** 2 / (k ** 2 + kc ** 2), 0.0)
    f = np.fft.ifft2(np.fft.fft2(field) * h).real
    return (f - f.mean()) / (f.std() + 1e-9)


def _volatility(n, calm_scale, seed):
    """High-passed log-correlated volatility field (unit variance)."""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((n, n)))
    k = _radial_freq(n); kc = calm_scale / n
    filt = np.zeros_like(k); nz = k > 0
    filt[nz] = k[nz] / (k[nz] ** 2 + kc ** 2)
    om = np.fft.ifft2(F * filt).real; om -= om.mean()
    return om / (om.std() + 1e-9)


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------
def punch(field, low=5, high=95, strength=2.5):
    """
    Display transform: linear percentile stretch + mild S-curve. Adds black/
    white legibility while preserving multifractality (monotonic, mild). Apply
    the SAME settings to every stimulus; quantify on the raw field, not this.
    """
    a, b = np.percentile(field, low), np.percentile(field, high)
    x = np.clip((field - a) / (b - a + 1e-12), 0, 1)
    if strength <= 0:
        return x
    return (np.tanh(strength * (x - 0.5)) / np.tanh(strength * 0.5)) * 0.5 + 0.5


# ---------------------------------------------------------------------------
# FLAGSHIP 1 : multifractional (within-image varying FD)
# ---------------------------------------------------------------------------
def multifractional(n=512, fd_center=2.5, fd_range=1.4, scale=5.0,
                    seed=None, return_map=False):
    """
    Within-image varying fractal dimension (full brightness).

    fd_center : mean local FD (2.0 coarse/smooth .. 3.0 fine/rough)
    fd_range  : how much local FD varies across the image (0 = uniform)
    scale     : spatial scale of the variation (larger = bigger patches)
    """
    blo = max(2.0, 8 - 2 * (fd_center + fd_range / 2))   # FD->beta = 8-2*FD
    bhi = min(4.6, 8 - 2 * (fd_center - fd_range / 2))
    return mbm_blend(n, beta_lo=blo, beta_hi=bhi, map_beta=scale, seed=seed,
                     return_map=return_map)


# ---------------------------------------------------------------------------
# FLAGSHIP 2 : multifractal cloud (non-darkening, connected structures)
# ---------------------------------------------------------------------------
def multifractal_cloud(n=512, granularity=2.9, multifractality=1.2,
                       smooth=0.3, distribution=1.3, calm_scale=3.0,
                       seed=None, raw=False):
    """
    Full-brightness multifractal cloud with connected structures.

    granularity     : base slope beta. 2.0 fine/high-FD .. 4.5 coarse/low-FD.
    multifractality : sigma. 0 = monofractal; ~1.0-1.6 sweet spot; up to ~2.5.
    smooth          : ~0.3 connects singularities into filaments (vs dot speckle).
                      0 = sharp dots; higher = smoother (erodes multifractality).
    distribution    : base high-pass (cycles). 0 = one big calm patch;
                      ~1-2 mixed; >=4 fully distributed.
    calm_scale      : volatility high-pass (cycles). low = big calm chains;
                      high = broken/small calm regions.

    Returns signed field (raw=True) or normalized [0,1]. For display call punch().
    """
    rng = np.random.default_rng(seed)
    G = _highpass(_fractional_field(n, beta=granularity,
                                    seed=int(rng.integers(0, 2 ** 31 - 1))),
                  distribution)
    om = _volatility(n, calm_scale, int(rng.integers(0, 2 ** 31 - 1)))
    f = G * np.exp(multifractality * om)
    if smooth > 0:
        f = _power_law_filter(f, exponent=smooth)
    return f if raw else _normalize01(f)


# ---------------------------------------------------------------------------
# complexity macro-dial (bundles the structural dials)
# ---------------------------------------------------------------------------
def complexity(c, n=512, seed=None):
    """
    Single simple->complex dial, c in [0,1], for flagship 2. Co-varies all
    structural dials. NOTE: bundles granularity with multifractality, so the
    raw Delta_alpha at the simple end mostly reflects the (granularity-set)
    monofractal floor; genuine multifractality appears from c~0.4 up. Use the
    individual dials of multifractal_cloud() if you need to isolate one factor.
    """
    lerp = lambda a, b: a + (b - a) * c
    return multifractal_cloud(
        n=n, granularity=lerp(4.5, 2.4), multifractality=lerp(0.05, 2.2),
        smooth=lerp(0.7, 0.3), distribution=lerp(0.0, 1.8),
        calm_scale=lerp(2.0, 3.5), seed=seed)


# ---------------------------------------------------------------------------
# battery sampler
# ---------------------------------------------------------------------------
def sample_battery(n_images=12, n=512, kind="multifractal_cloud",
                   measure=False, seed0=0):
    """
    Draw a varied stimulus set with random parameter combinations across the
    sensible ranges (so the set naturally mixes big-calm-patch and distributed,
    fine and coarse, etc.). Returns a list of dicts:
      {image (punched, [0,1]), field (raw), params, delta_alpha (if measure)}.
    """
    rng = np.random.default_rng(seed0)
    out = []
    for i in range(n_images):
        s = seed0 + 1000 + i
        if kind == "multifractal_cloud":
            p = dict(granularity=float(rng.uniform(2.4, 3.6)),
                     multifractality=float(rng.uniform(0.6, 1.8)),
                     smooth=0.3,
                     distribution=float(rng.uniform(0.5, 3.0)),
                     calm_scale=float(rng.uniform(2.5, 5.0)),
                     seed=s)
            field = multifractal_cloud(n=n, raw=True, **p)
        elif kind == "multifractional":
            p = dict(fd_center=float(rng.uniform(2.2, 2.8)),
                     fd_range=float(rng.uniform(0.4, 1.6)),
                     scale=float(rng.uniform(3.0, 7.0)), seed=s)
            field = multifractional(n=n, **p)
            field = field - field.mean()
        else:
            raise ValueError(kind)
        rec = dict(image=punch(field), field=field, params=p)
        if measure:
            rec["delta_alpha"] = spectrum_summary(
                np.linspace(-3, 3, 15),
                mfdfa_2d(_normalize01(field), q_list=np.linspace(-3, 3, 15))["tau"]
            )["delta_alpha"]
        out.append(rec)
    return out
