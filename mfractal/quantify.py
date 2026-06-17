"""
quantify.py
===========
Estimators of multifractality for 2D fields, plus the single-number
descriptors you would correlate with pareidolia behaviour.

Two complementary estimators (use the one matched to the object):

  mfdfa_2d(field)        -> generalized Hurst h(q): best for SURFACES
                            (fBm, value noise, the final smoothed image).
  moments_2d(measure)    -> mass exponent tau(q) by partition function:
                            best for positive MEASURES (cascades, the flux eps).

Both feed the same downstream machinery:

  legendre(q, tau)       -> singularity spectrum (alpha, f(alpha))
  spectrum_summary(...)  -> Delta_alpha (WIDTH = degree of multifractality),
                            asymmetry, alpha0, and generalized dims D0/D1/D2.

Also:
  box_count_support()    -> fractal dimension of a BINARY set (for thresholded
                            images), with the loud caveat that binarizing a
                            field destroys most of its multifractal content.

Conventions
-----------
2D embedding dimension d = 2.
MFDFA:   F_q(s) ~ s^{h(q)};   tau(q) = q*h(q) - d
Moments: Z(q,s) = <mu^q> scaling ~ s^{tau(q)} (box size s); D(q)=tau(q)/(q-1)
Legendre: alpha = d tau / dq ;  f(alpha) = q*alpha - tau(q)
A perfect MONOFRACTAL has h(q) = const -> tau linear -> f(alpha) a single
point (Delta_alpha ~ 0). A MULTIFRACTAL has curved tau -> broad f(alpha).
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 2D MFDFA  (Gu & Zhou 2006)
# ---------------------------------------------------------------------------
def mfdfa_2d(field, q_list=None, scales=None, order=1):
    """
    2D Multifractal Detrended Fluctuation Analysis.

    Parameters
    ----------
    field  : 2D array (a surface/signal, not necessarily positive).
    q_list : moment orders. Default linspace(-5,5,21) excluding ~0 handled.
    scales : window sizes s (pixels). Default ~ log-spaced 6..N/4.
    order  : detrending polynomial order (1 = plane fit).

    Returns
    -------
    dict with q, scales, Fq (len(q) x len(scales)), h(q), tau(q).
    """
    field = np.asarray(field, float)
    M, N = field.shape
    if q_list is None:
        q_list = np.linspace(-5, 5, 21)
    q_list = np.asarray(q_list, float)
    if scales is None:
        smax = min(M, N) // 4
        scales = np.unique(np.round(
            np.logspace(np.log10(6), np.log10(smax), 12)).astype(int))
    scales = np.asarray([s for s in scales if s >= 4], int)

    # 2D cumulative profile (double integration), mean removed
    prof = np.cumsum(np.cumsum(field - field.mean(), axis=0), axis=1)

    # design matrix for a polynomial surface in a window (built per scale)
    def detrend_var(block):
        s0, s1 = block.shape
        yy, xx = np.mgrid[0:s0, 0:s1]
        if order == 1:
            A = np.column_stack([np.ones(s0 * s1), xx.ravel(), yy.ravel()])
        else:
            A = np.column_stack([np.ones(s0 * s1), xx.ravel(), yy.ravel(),
                                 (xx * yy).ravel(), (xx ** 2).ravel(),
                                 (yy ** 2).ravel()])
        coef, *_ = np.linalg.lstsq(A, block.ravel(), rcond=None)
        resid = block.ravel() - A @ coef
        return np.mean(resid ** 2)

    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, s in enumerate(scales):
        nblk_i = M // s
        nblk_j = N // s
        if nblk_i == 0 or nblk_j == 0:
            continue
        f2 = []
        for bi in range(nblk_i):
            for bj in range(nblk_j):
                blk = prof[bi * s:(bi + 1) * s, bj * s:(bj + 1) * s]
                f2.append(detrend_var(blk))
        f2 = np.asarray(f2)
        f2 = f2[f2 > 0]
        if f2.size == 0:
            continue
        for qi, q in enumerate(q_list):
            if abs(q) < 1e-6:
                Fq[qi, si] = np.exp(0.25 * np.mean(np.log(f2)))  # q->0 limit
            else:
                Fq[qi, si] = np.mean(f2 ** (q / 2.0)) ** (1.0 / q)

    # h(q): slope of log Fq vs log s
    logs = np.log(scales)
    hq = np.full(len(q_list), np.nan)
    for qi in range(len(q_list)):
        y = np.log(Fq[qi])
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            hq[qi] = np.polyfit(logs[ok], y[ok], 1)[0]
    tau = q_list * hq - 2.0      # d = 2
    return dict(q=q_list, scales=scales, Fq=Fq, h=hq, tau=tau, method="mfdfa")


# ---------------------------------------------------------------------------
# Partition-function method of moments (for positive measures)
# ---------------------------------------------------------------------------
def moments_2d(measure, q_list=None, box_sizes=None):
    """
    Box-counting partition function for a positive MEASURE.

    Coarse-grain the (normalized) measure into boxes of side s, form
    Z(q,s) = sum_i mu_i(s)^q, and read tau(q) from Z(q,s) ~ s^{tau(q)}.

    Parameters
    ----------
    measure : 2D array, will be made non-negative and normalized to sum 1.
    """
    mu = np.asarray(measure, float)
    mu = mu - mu.min() if mu.min() < 0 else mu
    total = mu.sum()
    if total <= 0:
        raise ValueError("measure must have positive total mass")
    mu = mu / total
    M, N = mu.shape
    if q_list is None:
        q_list = np.linspace(-5, 5, 21)
    q_list = np.asarray(q_list, float)
    if box_sizes is None:
        kmax = int(np.log2(min(M, N))) - 1
        box_sizes = [2 ** k for k in range(1, kmax + 1)]
    box_sizes = np.asarray(box_sizes, int)

    eps = np.asarray(box_sizes, float) / min(M, N)   # normalized box size
    logeps = np.log(eps)
    Z = np.full((len(q_list), len(box_sizes)), np.nan)
    for bi, s in enumerate(box_sizes):
        ni, nj = M // s, N // s
        if ni == 0 or nj == 0:
            continue
        # block-sum the measure
        coarse = mu[:ni * s, :nj * s].reshape(ni, s, nj, s).sum(axis=(1, 3))
        c = coarse[coarse > 0].ravel()
        for qi, q in enumerate(q_list):
            if abs(q) < 1e-6:
                # tau(0) = -D0; use number of occupied boxes ~ eps^{-D0}
                Z[qi, bi] = c.size
            else:
                Z[qi, bi] = np.sum(c ** q)

    tau = np.full(len(q_list), np.nan)
    for qi, q in enumerate(q_list):
        y = np.log(Z[qi])
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            # Z(q, eps) ~ eps^{tau(q)} for all q (including q=0, where
            # Z = number of occupied boxes ~ eps^{-D0} so tau(0) = -D0).
            tau[qi] = np.polyfit(logeps[ok], y[ok], 1)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        Dq = np.where(np.abs(q_list - 1) < 1e-6, np.nan, tau / (q_list - 1))
    return dict(q=q_list, box_sizes=box_sizes, Z=Z, tau=tau, Dq=Dq,
                method="moments")


# ---------------------------------------------------------------------------
# Legendre transform  ->  singularity spectrum
# ---------------------------------------------------------------------------
def legendre(q, tau):
    """alpha = dtau/dq ; f(alpha) = q*alpha - tau(q)."""
    q = np.asarray(q, float); tau = np.asarray(tau, float)
    ok = np.isfinite(tau)
    q, tau = q[ok], tau[ok]
    order = np.argsort(q); q, tau = q[order], tau[order]
    alpha = np.gradient(tau, q)
    f = q * alpha - tau
    return alpha, f


def spectrum_summary(q, tau):
    """
    Single-number descriptors of the singularity spectrum.

    Returns dict:
      delta_alpha : WIDTH alpha_max - alpha_min  (degree of multifractality)
      alpha0      : alpha at the maximum of f (most probable singularity)
      asymmetry   : (alpha0 - alpha_min) / (alpha_max - alpha0); >1 left-skew
      D0, D1, D2  : capacity, information, correlation dimensions
                    (only meaningful for the moments/measure estimator)
    """
    alpha, f = legendre(q, tau)
    q = np.asarray(q, float)[np.isfinite(tau)]
    finite = np.isfinite(alpha) & np.isfinite(f)
    alpha, f = alpha[finite], f[finite]
    if alpha.size < 3:
        return dict(delta_alpha=np.nan, alpha0=np.nan, asymmetry=np.nan,
                    D0=np.nan, D1=np.nan, D2=np.nan)
    a_min, a_max = alpha.min(), alpha.max()
    a0 = alpha[np.argmax(f)]
    left = a0 - a_min
    right = a_max - a0
    asym = left / right if right > 1e-9 else np.nan

    def D_at(qq):
        tt = np.asarray(tau, float)
        qs = np.asarray(q, float)
        i = np.argmin(np.abs(qs - qq))
        if abs(qq - 1) < 1e-6:   # information dimension D1 = lim tau/(q-1)
            # use derivative of tau at q=1
            dtau = np.gradient(tt, qs)
            return dtau[i]
        return tt[i] / (qq - 1)

    return dict(delta_alpha=a_max - a_min, alpha0=a0, asymmetry=asym,
                D0=D_at(0.0), D1=D_at(1.0), D2=D_at(2.0))


# ---------------------------------------------------------------------------
# Binary / thresholded images
# ---------------------------------------------------------------------------
def box_count_support(binary, box_sizes=None):
    """
    Box-counting fractal dimension of the SUPPORT of a binary image
    (the set of 'on' pixels). This is the right thing to measure once you
    threshold -- but note it returns only D0 (one number), NOT a multifractal
    spectrum, because a 0/1 set has no measure variation to be multifractal in.

    To keep multifractal structure, quantify the GRAYSCALE field BEFORE
    thresholding, or define a density measure (e.g. local fraction of black
    in a window) and run moments_2d on that.
    """
    b = np.asarray(binary).astype(bool)
    M, N = b.shape
    if box_sizes is None:
        kmax = int(np.log2(min(M, N))) - 1
        box_sizes = [2 ** k for k in range(1, kmax + 1)]
    counts, sizes = [], []
    for s in box_sizes:
        ni, nj = M // s, N // s
        if ni == 0 or nj == 0:
            continue
        block = b[:ni * s, :nj * s].reshape(ni, s, nj, s).any(axis=(1, 3))
        counts.append(block.sum())
        sizes.append(s)
    counts = np.asarray(counts, float)
    sizes = np.asarray(sizes, float)
    ok = counts > 0
    D0 = -np.polyfit(np.log(sizes[ok]), np.log(counts[ok]), 1)[0]
    return dict(D0=D0, sizes=sizes, counts=counts)


# ---------------------------------------------------------------------------
# Local fractal-dimension map  (the right ruler for multifractional fields)
# ---------------------------------------------------------------------------
def local_fd_map(image, window=48, stride=16, lags=(1, 2, 4, 8)):
    """
    Windowed local fractal dimension via the structure function.

    In each sliding window, the second-order structure function
    SF(r)=<|X(p+r)-X(p)|^2> scales as r^{2H}; the slope gives a local Hurst
    exponent H and local surface dimension FD = 3 - H. The map's SPREAD
    (std / range) is the natural 'texture heterogeneity' metric for
    multifractional fields, where global moment methods (MFDFA) are blind.
    """
    x = np.asarray(image, float)
    M, N = x.shape
    logr = np.log(np.asarray(lags, float))
    iy = list(range(0, M - window + 1, stride))
    ix = list(range(0, N - window + 1, stride))
    fd = np.full((len(iy), len(ix)), np.nan)
    for a, i in enumerate(iy):
        for b, j in enumerate(ix):
            w = x[i:i + window, j:j + window]
            sf = []
            for r in lags:
                d1 = w[r:, :] - w[:-r, :]
                d2 = w[:, r:] - w[:, :-r]
                sf.append(0.5 * (np.mean(d1 ** 2) + np.mean(d2 ** 2)))
            sf = np.asarray(sf)
            ok = sf > 0
            if ok.sum() >= 2:
                H = np.polyfit(logr[ok], np.log(sf[ok]), 1)[0] / 2.0
                fd[a, b] = 3.0 - H
    return fd


def heterogeneity(image, **kw):
    """Scalar texture-heterogeneity index = std of the local FD map."""
    fd = local_fd_map(image, **kw)
    return float(np.nanstd(fd)), float(np.nanmax(fd) - np.nanmin(fd))


# ---------------------------------------------------------------------------
# 2D wavelet-leader multifractal log-cumulants (robust alternative to MFDFA).
# Requires PyWavelets (pywt). c1 ~ dominant Holder exponent (Hurst-like),
# c2 ~ intermittency / multifractality (|c2| larger => more multifractal;
# c2 ~ 0 for a monofractal field). These are the Wendt/Abry estimators.
# ---------------------------------------------------------------------------
def wavelet_leaders_2d(image, wavelet="db3", jmax=None, j_fit=None):
    import pywt
    from scipy.ndimage import maximum_filter
    img = np.asarray(image, dtype=float)
    n = min(img.shape)
    J = (int(np.log2(n)) - 3) if jmax is None else jmax
    J = max(3, J)
    coeffs = pywt.wavedec2(img, wavelet, level=J, mode="periodization")
    # coeffs = [cA_J, (cH_J,cV_J,cD_J), ... , (cH_1,cV_1,cD_1)]  (coarsest -> finest)
    dmag = []
    for k in range(1, len(coeffs)):
        cH, cV, cD = coeffs[k]
        dmag.append(np.maximum(np.maximum(np.abs(cH), np.abs(cV)), np.abs(cD)))
    dmag = dmag[::-1]  # finest (j=1) -> coarsest
    leaders = []
    prev = None
    for d in dmag:
        ell = maximum_filter(d, size=3, mode="nearest")
        if prev is not None:
            ph, pw = (prev.shape[0] // 2) * 2, (prev.shape[1] // 2) * 2
            ds = np.maximum.reduce([prev[0:ph:2, 0:pw:2], prev[1:ph:2, 0:pw:2],
                                    prev[0:ph:2, 1:pw:2], prev[1:ph:2, 1:pw:2]])
            h = min(ell.shape[0], ds.shape[0]); w = min(ell.shape[1], ds.shape[1])
            ell[:h, :w] = np.maximum(ell[:h, :w], ds[:h, :w])
        leaders.append(ell)
        prev = ell
    js = np.arange(1, len(leaders) + 1)
    C1 = np.array([np.log(e[e > 0] + 1e-12).mean() for e in leaders])
    C2 = np.array([np.log(e[e > 0] + 1e-12).var() for e in leaders])
    if j_fit is None:
        j_fit = (2, max(3, len(leaders) - 1))
    lo, hi = j_fit
    sel = slice(lo - 1, hi)
    c1 = np.polyfit(js[sel], C1[sel], 1)[0] / np.log(2)
    c2 = np.polyfit(js[sel], C2[sel], 1)[0] / np.log(2)
    return {"c1": float(c1), "c2": float(c2), "C1": C1, "C2": C2, "js": js}
