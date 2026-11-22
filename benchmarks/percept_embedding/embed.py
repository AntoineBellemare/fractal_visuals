"""Percept embedding in controlled-multifractality textures — the two routes
that survived prototyping (see README.md for the full menu + dead ends).

Both take a base multifractal field ``F`` (e.g. ``mf.prescribed_cascade(...)``)
and a percept energy map ``P`` (any (n,n) array in [0,1]; higher = more percept)
and return an embedded field in [0,1].

  luminance_embed(F, P, eps)        -- S1, FIRST-ORDER. Percept rides a gentle
                                       low-frequency brightness modulation.
                                       Trivially visible; c1/c2 essentially
                                       untouched. The percept is a *luminance*
                                       cue, not a texture cue.

  wavelet_embed(F, P, beta)         -- P3, SECOND-ORDER. Per coarse/mid wavelet
  + lock_c1(...)                      subband, the fractal field's coefficient
                                       MAGNITUDES are reordered spatially to
                                       follow the percept's energy (preserving
                                       each subband's magnitude distribution ->
                                       the leader stats, hence c2). Finest
                                       ``keep_fine`` levels stay pure fractal.
                                       The percept is *texture-structural* and
                                       survives in the multifractal signature.
                                       lock_c1 restores the base c1 afterwards
                                       via an isotropic spectral tilt.

  embed_percept(F, P, beta)         -- convenience: wavelet_embed + lock_c1.

Recommended default for "visible AND multifractal": ``embed_percept`` at
beta~0.25. Visibility scales with the base field's complexity (a richer cascade
sculpts a sharper percept at the same beta).
"""
import sys
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mfractal as mf  # noqa: E402

LEVELS = 5
KEEP_FINE = 2
WAVELET = "db2"


def _norm(x):
    return (x - x.min()) / (np.ptp(x) + 1e-12)


def measure(img):
    """(c1, c2) via 2D wavelet leaders."""
    o = mf.wavelet_leaders_2d(img)
    return o["c1"], o["c2"]


# --------------------------------------------------------------------------- #
# S1 — luminance (first-order)
# --------------------------------------------------------------------------- #
def luminance_embed(F, P, eps=0.25):
    """Modulate brightness by the (mean-removed) percept. eps = presence depth.
    c1/c2 are essentially preserved because this is a smooth, low-amplitude
    multiplicative tilt; the percept is a luminance cue."""
    return np.clip(F * (1.0 + eps * (P - P.mean())), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# P3 — wavelet-subband (second-order)
# --------------------------------------------------------------------------- #
def _reorder_magnitudes(fsub, psub):
    """Place the fractal subband's sorted magnitudes onto positions ranked by
    the percept's local energy; keep the fractal signs. The magnitude
    distribution is preserved exactly; only the spatial layout follows P."""
    mags = np.sort(np.abs(fsub).ravel())
    order = np.argsort(np.abs(psub).ravel())
    out = np.empty(fsub.size)
    out[order] = mags
    return out.reshape(fsub.shape) * np.sign(fsub)


def wavelet_embed(F, P, beta=0.25, levels=LEVELS, keep_fine=KEEP_FINE,
                  wavelet=WAVELET):
    """Percept-guided magnitude reorder on coarse/mid detail subbands.
    beta in [0,1] = presence (0 = pure fractal). Finest ``keep_fine`` levels and
    the approximation stay pure fractal to protect the c2 estimate."""
    cF = pywt.wavedec2(F, wavelet, level=levels)
    cP = pywt.wavedec2(P, wavelet, level=levels)
    out = [cF[0]]                                   # approximation = pure fractal
    ndetail = len(cF) - 1
    for di in range(1, len(cF)):
        is_fine = di > ndetail - keep_fine
        if is_fine or beta == 0:
            out.append(cF[di])
            continue
        new = []
        for fsub, psub in zip(cF[di], cP[di]):
            r = _reorder_magnitudes(fsub, psub)
            new.append((1 - beta) * fsub + beta * r)
        out.append(tuple(new))
    G = pywt.waverec2(out, wavelet)[:F.shape[0], :F.shape[1]]
    return _norm(G)


# --------------------------------------------------------------------------- #
# c1-lock — isotropic spectral tilt
# --------------------------------------------------------------------------- #
def _radial_k(n):
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(fy ** 2 + fx ** 2)
    k[0, 0] = 1.0
    return k


def radial_tilt(img, delta, k=None):
    """Multiply the spectrum by k^{-delta} (DC untouched). delta>0 steepens the
    spectrum -> raises c1. Isotropic, so the percept's spatial form is intact;
    a deterministic per-scale offset, so c2 (log-leader *variance*) is ~invariant."""
    if k is None:
        k = _radial_k(img.shape[0])
    H = k ** (-delta)
    H[0, 0] = 1.0
    G = np.fft.ifft2(np.fft.fft2(img) * H).real
    return _norm(G)


def lock_c1(G, target_c1, k=None, n_iter=4, tol=0.005):
    """Secant-solve the tilt delta so measured c1(G) == target_c1.
    Returns (locked_image, delta, c1, c2)."""
    if k is None:
        k = _radial_k(G.shape[0])
    d0 = 0.0
    c0, c2_0 = measure(G)
    if abs(c0 - target_c1) < tol:
        return G, 0.0, c0, c2_0
    d1 = 0.2
    g1 = radial_tilt(G, d1, k)
    c1m, c2m = measure(g1)
    for _ in range(n_iter):
        if abs(c1m - target_c1) < tol or c1m == c0:
            break
        d2 = d1 + (target_c1 - c1m) * (d1 - d0) / (c1m - c0)
        d2 = float(np.clip(d2, -0.9, 0.9))
        g1 = radial_tilt(G, d2, k)
        cnew, c2m = measure(g1)
        d0, c0, d1, c1m = d1, c1m, d2, cnew
    return g1, d1, c1m, c2m


def embed_percept(F, P, beta=0.25, lock=True, **kw):
    """Recommended entry point: wavelet-subband embed (visible, second-order) +
    c1-lock back to the base field's c1. Returns the embedded image in [0,1]."""
    target_c1 = measure(F)[0]
    G = wavelet_embed(F, P, beta=beta, **kw)
    if not lock:
        return G
    return lock_c1(G, target_c1)[0]
