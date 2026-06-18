"""Cloud-like multifractal stimulus families, each with a native `complexity`
knob in [0, 1] (simple -> intricate). Brightness is held roughly constant across
complexity so intricacy reads as structure, not darkness.

Families and what their complexity axis does:
  cascade_lognormal     log-normal multiplicative cascade (FIF). MULTIFRACTAL;
                        complexity raises cascade sigma -> c2 more negative.
  stratified            anisotropic cascade -> layered sky. MULTIFRACTAL.
  billow                cascade advected by a multi-scale vortex flow. MULTIFRACTAL;
                        complexity adds finer turbulent swirls.
  cirrus                cascade sheared into crisp fibers. mildly MULTIFRACTAL;
                        complexity adds finer, crisper fibers.
  ridged                multi-octave crisp ridge folds. near-MONOFRACTAL (c2~0 at
                        the detailed end); complexity = more / finer crisp folds.
  warped_fbm            domain-warped fBm. MONOFRACTAL control (c2~0); complexity is
                        a pure fractal-dimension / roughness axis.
  cloud_mrw             flagship multifractal_cloud (MRW) wrapped with complexity.
  cloud_multifractional flagship multifractional (within-image varying FD), wrapped.

Reliable multifractality readout is the wavelet-leader c2
(mfractal.quantify.wavelet_leaders_2d); bin experimental stimuli by MEASURED c2.
"""
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from .generators import _normalize01, _fractional_field
from .flagships import multifractal_cloud, multifractional
from .fluids import _vortex_flow, _advect

__all__ = [
    "cascade_lognormal", "warped_fbm", "stratified", "billow", "cirrus", "ridged",
    "cloud_mrw", "cloud_multifractional", "billow_smoke", "CLOUD_FAMILIES",
]

CLOUD_FAMILIES = [
    "cascade_lognormal", "stratified", "billow", "cirrus", "ridged",
    "warped_fbm", "cloud_mrw", "cloud_multifractional", "billow_smoke",
]


def _L(a, b, c):
    return a + (b - a) * float(c)


def _cloud_bright(x, target=0.5):
    """Auto-gamma (monotonic, structure-preserving) to hold mean brightness."""
    x = _normalize01(x)
    lo, hi = 0.12, 1.6
    for _ in range(20):
        g = 0.5 * (lo + hi)
        if (x ** g).mean() < target:
            hi = g
        else:
            lo = g
    return x ** (0.5 * (lo + hi))


def _cloud_contrast(x, gain):
    x = _normalize01(x)
    if gain <= 0:
        return x
    return _normalize01(0.5 + np.tanh(gain * (x - 0.5)) / (2 * np.tanh(gain * 0.5)))


def _fracint(field, H):
    n = field.shape[0]
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2); k[0, 0] = 1
    F = np.fft.fft2(field) / (k ** H); F[0, 0] = 0
    return np.real(np.fft.ifft2(F))


def _cascade(n, depth, sigma, seed):
    rng = np.random.default_rng(seed)
    logM = np.zeros((n, n))
    for j in range(1, depth + 1):
        s = 1 << j
        up = zoom(rng.normal(-sigma ** 2 / 2, sigma, (s, s)), n / s, order=3)[:n, :n]
        if up.shape != (n, n):
            t = np.zeros((n, n)); t[:up.shape[0], :up.shape[1]] = up; up = t
        logM += np.roll(up, (int(rng.integers(0, n)), int(rng.integers(0, n))), axis=(0, 1))
    logM -= logM.max()
    return np.exp(logM)


def cascade_lognormal(n=512, seed=None, complexity=0.5):
    c = complexity
    return _cloud_bright(_fracint(_cascade(n, 7, _L(0.18, 0.9, c), seed), _L(0.72, 0.42, c)))


def warped_fbm(n=512, seed=None, complexity=0.5):
    c = complexity
    b = _normalize01(_fractional_field(n, beta=_L(3.5, 2.3, c), seed=seed))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    w = _L(5, 28, c)
    sa = (None if seed is None else seed + 1); sb = (None if seed is None else seed + 2)
    wx = _fractional_field(n, beta=3.2, seed=sa); wy = _fractional_field(n, beta=3.2, seed=sb)
    return _normalize01(map_coordinates(
        b, [(yy + w * _normalize01(wy)) % n, (xx + w * _normalize01(wx)) % n],
        order=1, mode="grid-wrap"))


def stratified(n=512, seed=None, complexity=0.5):
    c = complexity
    m = _cascade(n, 7, _L(0.2, 0.85, c), seed)
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt((kx * 4) ** 2 + ky ** 2); k[0, 0] = 1
    F = np.fft.fft2(m) / (k ** _L(0.65, 0.42, c)); F[0, 0] = 0
    return _cloud_bright(gaussian_filter(np.real(np.fft.ifft2(F)), (0.4, 2.5)))


def billow(n=512, seed=None, complexity=0.5):
    c = complexity
    rng = np.random.default_rng(seed)
    f = _normalize01(_fracint(_cascade(n, 7, _L(0.4, 0.9, c), seed), _L(0.55, 0.42, c)))
    v1 = _vortex_flow(n, rng, int(_L(25, 60, c)), smin=24, smax=70)
    v2 = _vortex_flow(n, rng, int(_L(90, 180, c)), smin=8, smax=28)
    v3 = _vortex_flow(n, rng, int(_L(220, 440, c)), smin=2, smax=_L(14, 7, c))
    vx = v1[0] + 0.8 * v2[0] + 0.7 * v3[0]; vy = v1[1] + 0.8 * v2[1] + 0.7 * v3[1]
    return _cloud_bright(_advect(f, vx, vy, int(_L(18, 34, c)), 1.3))


def cirrus(n=512, seed=None, complexity=0.5):
    c = complexity
    base = _normalize01(_fracint(_cascade(n, 7, _L(0.3, 0.7, c), seed), 0.6))
    streak = gaussian_filter(base, (_L(1.6, 0.6, c), _L(10, 4, c)))
    gy, gx = np.gradient(streak)
    fib = _cloud_contrast(_normalize01(np.abs(gx)), _L(2.6, 4.2, c))
    return _cloud_contrast(_cloud_bright(0.6 * _normalize01(streak) + 0.6 * fib, 0.55), _L(2.2, 3.0, c))


def ridged(n=512, seed=None, complexity=0.5):
    c = complexity
    rng = np.random.default_rng(seed)
    octaves = int(round(_L(2, 6, c)))
    out = np.zeros((n, n)); amp = 1.0; tot = 0.0; weight = np.ones((n, n))
    for o in range(octaves):
        sc = max(0.7, n / (2.0 ** (o + 2)) * _L(1.0, 0.7, c))
        band = _normalize01(gaussian_filter(rng.normal(size=(n, n)), sc))
        fold = 1.0 - np.abs(2 * band - 1.0)
        out += amp * fold * weight; tot += amp
        weight = np.clip(fold * 1.8, 0.25, 1.0); amp *= 0.6
    return _cloud_contrast(_cloud_bright(_normalize01(out / tot), 0.55), _L(2.0, 3.2, c))


def cloud_mrw(n=512, seed=None, complexity=0.5):
    """Flagship multifractal_cloud (MRW) wrapped as a complexity-controlled family."""
    c = complexity
    f = multifractal_cloud(n, granularity=_L(3.2, 2.0, c), multifractality=_L(0.1, 2.2, c),
                           smooth=_L(0.3, 0.1, c), seed=seed, raw=True)
    return _cloud_bright(f)


def cloud_multifractional(n=512, seed=None, complexity=0.5):
    """Flagship multifractional (within-image varying FD) wrapped with complexity."""
    c = complexity
    f = multifractional(n, fd_center=_L(2.2, 2.7, c), fd_range=_L(0.3, 1.7, c),
                        scale=5.0, seed=seed)
    return _cloud_bright(_normalize01(f))


def billow_smoke(n=512, seed=None, complexity=0.5):
    """Broad rising billowing smoke cloud filling most of the frame.

    complexity in [0,1]: fBM beta sweeps from 4.0 (smooth haze) down to 2.0
    (turbulent wisps). The rise envelope gets a seed-dependent slant and
    lateral pinch so different seeds show distinct billow shapes, not just
    different internal textures.
    """
    cx = float(np.clip(complexity, 0, 1))
    rng = default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float) / n

    horizon = 0.05 + 0.20 * rng.random()
    slant = 0.25 * (rng.random() - 0.5) * 2.0
    pinch_x = 0.3 + 0.4 * rng.random()
    pinch_w = 0.25 + 0.20 * rng.random()
    pinch_str = 0.20 + 0.20 * rng.random()
    rise_y = (1 - yy) + slant * (xx - 0.5)
    pinch = 1.0 - pinch_str * np.exp(-((xx - pinch_x) ** 2) / (2 * pinch_w ** 2))
    rise = 0.40 + 0.55 * np.clip(rise_y - horizon, 0, 1) * pinch
    rise = gaussian_filter(rise, 1.5)

    beta = 4.0 - 2.0 * cx
    base = _normalize01(_fractional_field(n, beta=beta, seed=seed))

    yyp, xxp = np.mgrid[0:n, 0:n].astype(float)
    swirl_amp = 22.0 * cx
    swirl_scale = n / (6.0 + 18.0 * cx)
    fy = gaussian_filter(rng.normal(0, 1, (n, n)), swirl_scale) * swirl_amp
    fx = gaussian_filter(rng.normal(0, 1, (n, n)), swirl_scale) * swirl_amp

    field0 = 0.30 * rise + 0.65 * base * rise
    warped = map_coordinates(field0, [(yyp + fy) % n, (xxp + fx) % n],
                             order=1, mode="grid-wrap")
    return _normalize01(warped)
