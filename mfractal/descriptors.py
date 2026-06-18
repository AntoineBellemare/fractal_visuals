"""
descriptors.py
==============
Derived single-number texture descriptors beyond c1/c2/Delta_alpha/alpha0.

Three families, all 2D-image -> scalar (or scalar dict):

  ANISOTROPY -- is structure oriented or isotropic?
    directional_hurst(image)   -> H(theta), one Hurst per orientation
    anisotropy_index(image)    -> (Hmax - Hmin) / Hmean from H(theta)
    subband_anisotropy(image)  -> fast wavelet-subband proxy (H/V/D)

  SPATIAL COHERENCE -- how far does the field stay correlated with itself?
    autocorr_radial(image)     -> (r, R(r))   radial autocorrelation
    coherence_length(image)    -> scalar, first r where R(r) <= 1/e
    integral_length(image)     -> scalar, integral of R(r) over r

  WITHIN-IMAGE HETEROGENEITY -- is multifractality spatially homogeneous?
    local_c2_map(image)        -> 2D array, windowed wavelet-leader c2
    heterogeneity_full(image)  -> dict (std, range, entropy, skew, kurt)

And a bundler:

  feature_vector(image)        -> dict with c1, c2, Delta_alpha, alpha0,
                                  anisotropy, coherence_length,
                                  heterogeneity_std, ...

The feature_vector is what you'd hand a downstream model (ControlNet
condition, pareidolia regressor) when c2 alone isn't enough.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates

from .quantify import wavelet_leaders_2d, mfdfa_2d, spectrum_summary, local_fd_map


# ===========================================================================
# Anisotropy
# ===========================================================================

def directional_hurst(image, n_angles=8, lags=(1, 2, 4, 8, 16)):
    """
    Hurst exponent per orientation, via the directional second-order structure
    function. For each angle theta, SF(r,theta) = <|X(p) - X(p + r*u_theta)|^2>
    scales as r^{2H(theta)}; the slope gives the local Hurst.

    A perfectly isotropic field has H(theta) = const. Strongly oriented fields
    (cosine ridges, fibred surfaces) have H varying strongly with theta.

    Parameters
    ----------
    image    : 2D array.
    n_angles : number of orientations in [0, pi).
    lags     : pixel separations to use for the structure function.

    Returns
    -------
    angles : (n_angles,) array of theta in radians.
    H      : (n_angles,) array of Hurst exponents.
    """
    x = np.asarray(image, dtype=float)
    M, N = x.shape
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    yy, xx = np.mgrid[0:M, 0:N].astype(float)
    logr = np.log(np.asarray(lags, float))
    H = np.full(n_angles, np.nan)
    for ai, theta in enumerate(angles):
        # unit vector at angle theta (image-coords: row = -y, col = x)
        dy, dx = -np.sin(theta), np.cos(theta)
        sf = []
        for r in lags:
            shifted = map_coordinates(x, [yy + r * dy, xx + r * dx],
                                      order=1, mode="reflect")
            sf.append(float(np.mean((x - shifted) ** 2)))
        sf = np.asarray(sf)
        ok = sf > 0
        if ok.sum() >= 2:
            slope = np.polyfit(logr[ok], np.log(sf[ok]), 1)[0]
            H[ai] = slope / 2.0
    return angles, H


def anisotropy_index(image, n_angles=8, lags=(1, 2, 4, 8, 16)):
    """
    Scalar anisotropy index = (Hmax - Hmin) / Hmean from directional Hurst.

    0 = perfectly isotropic; higher values = more oriented structure.
    """
    _, H = directional_hurst(image, n_angles=n_angles, lags=lags)
    H = H[np.isfinite(H)]
    if H.size < 2 or H.mean() == 0:
        return float("nan")
    return float((H.max() - H.min()) / np.abs(H.mean()))


def subband_anisotropy(image, wavelet="db3"):
    """
    Fast wavelet-subband proxy for anisotropy. Compares the variance of the
    horizontal, vertical, and diagonal detail coefficients at the first scale.

    Returns a dict with:
      ratio_HV : log2(var_H / var_V)  (positive = horizontal-dominant)
      ratio_HD : log2(var_H / var_D)
      asymmetry : (max - min) / mean across H, V, D variances. 0 = isotropic.
    """
    import pywt
    img = np.asarray(image, dtype=float)
    coeffs = pywt.wavedec2(img, wavelet, level=1, mode="periodization")
    _, (cH, cV, cD) = coeffs
    vH, vV, vD = float(cH.var()), float(cV.var()), float(cD.var())
    eps = 1e-12
    vmax, vmin, vmean = max(vH, vV, vD), min(vH, vV, vD), (vH + vV + vD) / 3.0
    return dict(
        ratio_HV=float(np.log2((vH + eps) / (vV + eps))),
        ratio_HD=float(np.log2((vH + eps) / (vD + eps))),
        asymmetry=float((vmax - vmin) / (vmean + eps)),
    )


# ===========================================================================
# Spatial coherence
# ===========================================================================

def autocorr_radial(image, n_bins=64):
    """
    Radially-averaged autocorrelation R(r), normalized to R(0)=1.

    Computed via the Wiener-Khinchin theorem on the mean-subtracted image
    (R = ifft(|fft|^2)), then radially binned.

    Returns
    -------
    r  : (n_bins,) array of bin centers in pixels.
    R  : (n_bins,) array of mean autocorrelation in each bin.
    """
    x = np.asarray(image, dtype=float)
    x = x - x.mean()
    F = np.fft.rfft2(x)
    P = (F.conj() * F).real
    R2 = np.fft.irfft2(P, s=x.shape)
    R2 = np.fft.fftshift(R2)
    R2 = R2 / R2.max()           # normalize R(0)=1 (peak at center after shift)
    M, N = R2.shape
    cy, cx = M // 2, N // 2
    yy, xx = np.mgrid[0:M, 0:N]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    bins = np.linspace(0, rmax, n_bins + 1)
    idx = np.digitize(rr.ravel(), bins) - 1
    R_avg = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = idx == k
        if sel.any():
            R_avg[k] = float(R2.ravel()[sel].mean())
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    return r_centers, R_avg


def coherence_length(image, threshold=np.exp(-1), n_bins=64):
    """
    Spatial coherence length = first r at which R(r) drops to `threshold`.

    Default threshold is 1/e (the classic e-folding length). Returns pixels.
    NaN if R never crosses the threshold within the radial window.
    """
    r, R = autocorr_radial(image, n_bins=n_bins)
    ok = np.isfinite(R)
    r, R = r[ok], R[ok]
    if R.size < 3:
        return float("nan")
    below = np.where(R <= threshold)[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(r[0])
    # linear interp between (r[i-1], R[i-1]) and (r[i], R[i])
    R0, R1 = R[i - 1], R[i]
    r0, r1 = r[i - 1], r[i]
    if R0 == R1:
        return float(r1)
    t = (threshold - R0) / (R1 - R0)
    return float(r0 + t * (r1 - r0))


def integral_length(image, rmax_frac=0.5, n_bins=64):
    """
    Integral length scale = trapezoidal integral of R(r) from 0 to rmax.
    A scale-aware coherence summary; weights all lags rather than just the
    1/e crossing.

    rmax_frac : fraction of the image's half-side used as upper limit.
    """
    r, R = autocorr_radial(image, n_bins=n_bins)
    ok = np.isfinite(R)
    r, R = r[ok], R[ok]
    rmax = rmax_frac * r.max()
    sel = r <= rmax
    if sel.sum() < 3:
        return float("nan")
    return float(np.trapezoid(R[sel], r[sel]))


# ===========================================================================
# Within-image heterogeneity
# ===========================================================================

def local_c2_map(image, window=128, stride=64, jmax=None):
    """
    Sliding-window wavelet-leader c2. Each window gets its own intermittency
    estimate, so the resulting map shows where in the image the multifractality
    sits. Expensive: O((M*N/stride^2) * window^2 * log(window)).

    Returns
    -------
    2D array of c2 values (NaN where the wavelet fit fails).
    """
    x = np.asarray(image, dtype=float)
    M, N = x.shape
    iy = list(range(0, M - window + 1, stride))
    ix = list(range(0, N - window + 1, stride))
    out = np.full((len(iy), len(ix)), np.nan)
    for a, i in enumerate(iy):
        for b, j in enumerate(ix):
            w = x[i:i + window, j:j + window]
            try:
                out[a, b] = float(wavelet_leaders_2d(w, jmax=jmax)["c2"])
            except Exception:
                pass
    return out


def heterogeneity_full(image, window=48, stride=16, lags=(1, 2, 4, 8)):
    """
    Within-image heterogeneity of the local fractal dimension map.

    Returns a dict:
      fd_mean   : mean local FD (~ rotation-averaged FD)
      fd_std    : std of local FD (the original 'heterogeneity' index)
      fd_range  : range = max - min
      fd_skew   : skewness of FD distribution
      fd_kurt   : excess kurtosis
      fd_entropy: Shannon entropy of binned FD distribution (bits)
    """
    fd = local_fd_map(image, window=window, stride=stride, lags=lags)
    flat = fd[np.isfinite(fd)]
    if flat.size < 8:
        return dict(fd_mean=np.nan, fd_std=np.nan, fd_range=np.nan,
                    fd_skew=np.nan, fd_kurt=np.nan, fd_entropy=np.nan)
    mu = float(flat.mean()); sd = float(flat.std())
    skew = float(np.mean(((flat - mu) / (sd + 1e-12)) ** 3))
    kurt = float(np.mean(((flat - mu) / (sd + 1e-12)) ** 4)) - 3.0
    # Shannon entropy on 24-bin histogram
    hist, _ = np.histogram(flat, bins=24, density=False)
    p = hist / hist.sum()
    p = p[p > 0]
    ent = float(-np.sum(p * np.log2(p)))
    return dict(fd_mean=mu, fd_std=sd, fd_range=float(flat.max() - flat.min()),
                fd_skew=skew, fd_kurt=kurt, fd_entropy=ent)


# ===========================================================================
# Bundled feature vector
# ===========================================================================

def feature_vector(image, q_for_spectrum=None):
    """
    Bundle the headline descriptors as one flat dict. Suitable for stuffing
    into a manifest CSV or feeding a regressor.

    Fields:
      c1, c2                  : wavelet-leader log-cumulants (Hurst-like + intermittency)
      delta_alpha, alpha0     : singularity-spectrum width + mode (MFDFA)
      asymmetry               : f(alpha) left/right asymmetry
      anisotropy              : directional-Hurst (Hmax - Hmin) / Hmean
      subband_asymmetry       : fast wavelet-subband proxy for anisotropy
      coherence_length        : 1/e crossing of radial autocorrelation (pixels)
      integral_length         : integral of R(r) up to half-image
      fd_std, fd_range, fd_entropy : local-FD heterogeneity facets
    """
    out = {}
    img = np.asarray(image, dtype=float)

    # wavelet leaders -- c1, c2
    try:
        wl = wavelet_leaders_2d(img)
        out["c1"] = float(wl["c1"]); out["c2"] = float(wl["c2"])
    except Exception:
        out["c1"] = out["c2"] = float("nan")

    # MFDFA spectrum
    try:
        if q_for_spectrum is None:
            q_for_spectrum = np.linspace(-3, 3, 13)
        mf = mfdfa_2d(img, q_list=q_for_spectrum)
        sp = spectrum_summary(mf["q"], mf["tau"])
        out["delta_alpha"] = float(sp["delta_alpha"])
        out["alpha0"] = float(sp["alpha0"])
        out["asymmetry"] = float(sp["asymmetry"])
    except Exception:
        out["delta_alpha"] = out["alpha0"] = out["asymmetry"] = float("nan")

    # anisotropy
    try:
        out["anisotropy"] = anisotropy_index(img)
    except Exception:
        out["anisotropy"] = float("nan")
    try:
        out["subband_asymmetry"] = subband_anisotropy(img)["asymmetry"]
    except Exception:
        out["subband_asymmetry"] = float("nan")

    # coherence
    try:
        out["coherence_length"] = coherence_length(img)
        out["integral_length"] = integral_length(img)
    except Exception:
        out["coherence_length"] = out["integral_length"] = float("nan")

    # heterogeneity
    try:
        het = heterogeneity_full(img)
        out["fd_std"] = het["fd_std"]
        out["fd_range"] = het["fd_range"]
        out["fd_entropy"] = het["fd_entropy"]
    except Exception:
        out["fd_std"] = out["fd_range"] = out["fd_entropy"] = float("nan")

    return out


__all__ = [
    "directional_hurst", "anisotropy_index", "subband_anisotropy",
    "autocorr_radial", "coherence_length", "integral_length",
    "local_c2_map", "heterogeneity_full",
    "feature_vector",
]
