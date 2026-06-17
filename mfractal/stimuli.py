"""
stimuli.py
==========
Turn a raw field into a pareidolia stimulus, and build experimental batteries.

post-processing
  set_rms_contrast()  : match RMS contrast (your 2022 contrast manipulation)
  threshold()         : binarize at a percentile (your high-contrast B/W trick)
  mirror_symmetrize() : impose vertical mirror symmetry (the symmetry lever)

battery design
  make_battery()      : a grid that crosses FD (slope/H) with DELTA_ALPHA
                        (multifractality), the natural successor to your
                        FD x contrast design.
"""

from __future__ import annotations
import numpy as np
from . import generators as G


def hist_equalize(img):
    """
    Map a field to a uniform luminance histogram via its rank (CDF) transform.

    Multifractal flux fields are extremely spiky (rare intense concentrations),
    so a raw linear display reads as almost black. Equalizing makes the spatial
    structure visible AND gives a controlled, flat luminance distribution --
    which is what you want for a stimulus anyway. It is a monotonic per-pixel
    remap, so it preserves the *spatial* arrangement of singularities (and thus
    the support geometry) while standardising first-order luminance statistics.
    """
    x = np.asarray(img, float).ravel()
    order = np.argsort(np.argsort(x))
    eq = order / (len(x) - 1)
    return eq.reshape(np.asarray(img).shape)


def set_rms_contrast(img, rms=0.2):
    """Rescale to mean 0.5 and a target RMS contrast (std/mean of luminance)."""
    x = np.asarray(img, float)
    x = (x - x.mean())
    s = x.std()
    if s > 0:
        x = x / s * (rms)
    return np.clip(x + 0.5, 0, 1)


def threshold(img, black_fraction=0.5):
    """Binarize so that `black_fraction` of pixels are black (percentile cut)."""
    x = np.asarray(img, float)
    cut = np.quantile(x, black_fraction)
    return (x <= cut).astype(float)   # 1 = black


def mirror_symmetrize(img, axis="vertical"):
    """Impose mirror symmetry by averaging the field with its reflection."""
    x = np.asarray(img, float)
    if axis == "vertical":
        return 0.5 * (x + x[:, ::-1])
    return 0.5 * (x + x[::-1, :])


def make_battery(n=512, H_values=(0.2, 0.5, 0.8),
                 sigma_values=(0.0, 0.8, 1.6), seed0=0):
    """
    Crossed FD x MULTIFRACTALITY battery using the log-normal MRM.

    H (rows)     ~ controls global slope -> fractal dimension (FD axis)
    sigma (cols) ~ controls intermittency -> Delta_alpha (multifractality axis)
    sigma = 0 reduces to the monofractal (your 2022) case at each FD.

    Returns a dict keyed by (H, sigma) -> grayscale image, plus the flux
    fields (key ('flux', H, sigma)) for measure-based quantification.
    """
    out = {}
    s = seed0
    for H in H_values:
        for sg in sigma_values:
            img, flux = G.lognormal_mrm(n=n, sigma=sg, H=H, seed=s,
                                        return_flux=True)
            out[(round(H, 3), round(sg, 3))] = img
            out[("flux", round(H, 3), round(sg, 3))] = flux
            s += 1
    return out
