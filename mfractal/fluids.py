"""Liquid / turbulence patterns for pareidolia.

A velocity field is built from a superposition of power-law-sized vortices via the
stream function (psi = inverse-Laplacian of the vorticity, velocity = curl psi), and
a multifractal dye is advected through it semi-Lagrangianly. The flow is fluid-like
and the dye carries genuine multifractality; stretching and folding by the flow is
exactly what makes turbulent scalar fields multifractal.

All generators take (n, seed) and return a float array in [0, 1]; pair with punch()
for display and colorize_natural() for colour.
"""
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, zoom

from .generators import _normalize01, _fractional_field
from .flagships import multifractal_cloud

__all__ = ["eddies", "vorticity", "plume", "curl_weave", "dye_diffusion",
           "rheoscopic", "choppy", "FLUID_FAMILIES", "CX_FLUIDS"]


def _vel_from_w(w):
    """Divergence-free velocity (vx, vy) from a vorticity field via FFT Poisson."""
    n = w.shape[0]
    k = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2
    k[0, 0] = 1.0
    psi = np.real(np.fft.ifft2(np.fft.fft2(w) / k))
    gy, gx = np.gradient(psi)
    vx, vy = gy, -gx
    sd = np.hypot(vx, vy).std() + 1e-9
    return vx / sd, vy / sd


def _vortex_flow(n, rng, n_vort=150, smin=4, smax=40):
    """Turbulent flow from a superposition of power-law-sized signed vortices."""
    w = np.zeros((n, n))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    for _ in range(n_vort):
        cx, cy = rng.uniform(0, n, 2)
        s = rng.uniform(smin, smax) * rng.uniform(0.3, 1.0)
        w += rng.choice([-1.0, 1.0]) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    return _vel_from_w(w)


def _advect(dye, vx, vy, steps, dt, vbias=0.0):
    """Semi-Lagrangian back-trace advection with periodic wrap."""
    n = dye.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    for _ in range(steps):
        dye = map_coordinates(dye, [(yy - dt * (vy + vbias)) % n, (xx - dt * vx) % n],
                              order=1, mode="grid-wrap")
    return dye


def _ns2d(n, steps, seed, dt=0.0015, k0=6.0, urms=0.7, buoyancy=0.0, blobs=0):
    """Pseudo-spectral 2D incompressible Navier-Stokes in vorticity form with a
    passive (optionally buoyant) scalar. McWilliams random initial vorticity, 2/3
    dealiasing, hyperviscosity, Heun time-stepping. Returns (vorticity, scalar)."""
    rng = np.random.default_rng(seed)
    nu = 3e-5 * (256.0 / n) ** 4                      # keep cutoff damping ~constant
    k1 = np.fft.fftfreq(n) * n
    KY = k1[:, None] + 0 * k1[None, :]
    KX = 0 * k1[:, None] + k1[None, :]
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    iK2 = 1.0 / K2
    iK2[0, 0] = 0.0
    kmax = n / 2
    dealias = ((np.abs(KX) < (2 / 3) * kmax) & (np.abs(KY) < (2 / 3) * kmax)).astype(float)
    kk = np.sqrt(K2)
    amp = np.sqrt(kk ** (-1) * (1 + (kk / k0) ** 4) ** (-1))
    amp[0, 0] = 0.0
    amp = amp.astype(np.float32)
    psih = np.fft.fft2(np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n))))).astype(np.complex64))
    wh = K2 * psih
    u = np.real(np.fft.ifft2(1j * KY * psih))
    v = np.real(np.fft.ifft2(-1j * KX * psih))
    wh *= urms / np.sqrt(np.mean(u ** 2 + v ** 2) + 1e-12)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    if blobs > 0:
        th = np.zeros((n, n))
        for _ in range(blobs):
            cx = rng.uniform(0, n); cy = rng.uniform(0.65 * n, 0.95 * n)
            r = rng.uniform(0.05, 0.1) * n
            th += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    else:
        th = _normalize01(np.sin(5 * 2 * np.pi * xx / n) + np.sin(4 * 2 * np.pi * yy / n) +
                          0.5 * np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n))))))
    thh = np.fft.fft2(th)
    visc = np.exp(-nu * K2 ** 2 * dt).astype(np.complex64)
    viscT = np.exp(-0.5 * nu * K2 ** 2 * dt).astype(np.complex64)
    KXc = (1j * KX).astype(np.complex64); KYc = (1j * KY).astype(np.complex64)
    iK2 = iK2.astype(np.float32); dealias = dealias.astype(np.complex64)
    wh = wh.astype(np.complex64); thh = thh.astype(np.complex64)

    def rhs(wh, thh):
        psih = wh * iK2
        u = np.real(np.fft.ifft2(KYc * psih)); v = np.real(np.fft.ifft2(-KXc * psih))
        wx = np.real(np.fft.ifft2(KXc * wh)); wy = np.real(np.fft.ifft2(KYc * wh))
        tx = np.real(np.fft.ifft2(KXc * thh)); ty = np.real(np.fft.ifft2(KYc * thh))
        dwh = -np.fft.fft2((u * wx + v * wy).astype(np.complex64)) * dealias
        if buoyancy:
            dwh = dwh + buoyancy * (KXc * thh)
        return dwh, -np.fft.fft2((u * tx + v * ty).astype(np.complex64)) * dealias

    for _ in range(steps):
        a, b = rhs(wh, thh)
        c, d = rhs(wh + dt * a, thh + dt * b)
        wh = visc * (wh + 0.5 * dt * (a + c))
        thh = viscT * (thh + 0.5 * dt * (b + d))
    return np.real(np.fft.ifft2(wh)), np.real(np.fft.ifft2(thh))


def eddies(n=256, seed=None, steps=900):
    """Decaying 2D turbulence (real Navier-Stokes): a passive scalar advected by the
    evolving flow develops naturalistic multifractal filaments and tendrils."""
    _, scalar = _ns2d(n, steps, seed)
    return _normalize01(scalar)


def vorticity(n=256, seed=None, steps=900):
    """The vorticity field of decaying 2D turbulence: coherent vortices (light/dark
    cores) wrapped by filaments -- the classic 2D-turbulence / soap-film look."""
    w, _ = _ns2d(n, steps, seed)
    return _normalize01(w)


def plume(n=256, seed=None, steps=800):
    """Buoyant turbulent plume (Boussinesq): dye injected low, rising and rolling
    into billows as it stirs its own vorticity."""
    _, scalar = _ns2d(n, steps, seed, buoyancy=10.0, blobs=4)
    return _normalize01(scalar)


def _L(a, b, c):
    return a + (b - a) * float(c)


def _fl_fracint(field, H):
    n = field.shape[0]
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2); k[0, 0] = 1
    F = np.fft.fft2(field) / (k ** H); F[0, 0] = 0
    return np.real(np.fft.ifft2(F))


def _fl_cascade(n, depth, sigma, seed):
    rng = np.random.default_rng(seed); logM = np.zeros((n, n))
    for jj in range(1, depth + 1):
        sps = 1 << jj
        up = zoom(rng.normal(-sigma ** 2 / 2, sigma, (sps, sps)), n / sps, order=3)[:n, :n]
        if up.shape != (n, n):
            t = np.zeros((n, n)); t[:up.shape[0], :up.shape[1]] = up; up = t
        logM += np.roll(up, (int(rng.integers(0, n)), int(rng.integers(0, n))), axis=(0, 1))
    logM -= logM.max()
    return np.exp(logM)


def _fl_contrast(x, gain):
    x = _normalize01(x)
    return _normalize01(0.5 + np.tanh(gain * (x - 0.5)) / (2 * np.tanh(gain * 0.5)))


def _fl_bright(x, target=0.5):
    x = _normalize01(x); lo, hi = 0.12, 1.6
    for _ in range(18):
        g = 0.5 * (lo + hi)
        if (x ** g).mean() < target:
            hi = g
        else:
            lo = g
    return x ** (0.5 * (lo + hi))


def _msflow(n, rng, c):
    """Multi-scale vortex flow: a large-swirl field always present, with mid and
    fine scales faded in by complexity so low c is broad/laminar, high c intricate."""
    v1 = _vortex_flow(n, rng, int(_L(16, 36, c)), smin=28, smax=75)
    v2 = _vortex_flow(n, rng, int(_L(40, 180, c)), smin=8, smax=28)
    v3 = _vortex_flow(n, rng, int(_L(60, 420, c)), smin=2, smax=_L(18, 7, c))
    return (v1[0] + (0.25 + 0.75 * c) * v2[0] + (0.08 + 0.92 * c) * v3[0],
            v1[1] + (0.25 + 0.75 * c) * v2[1] + (0.08 + 0.92 * c) * v3[1])


def curl_weave(n=512, seed=None, complexity=0.5):
    """Continuous dye advected by a multi-scale vortex flow into woven marbled
    filaments: broad swirls at low complexity, fine interlacing at high. MULTIFRACTAL."""
    c = complexity; rng = np.random.default_rng(seed)
    vx, vy = _msflow(n, rng, c)
    dye = _normalize01(_fractional_field(n, beta=_L(3.4, 2.0, c), seed=seed))
    return _fl_contrast(_normalize01(_advect(dye, vx, vy, int(_L(10, 40, c)), 1.1)), _L(1.4, 2.4, c))


def dye_diffusion(n=512, seed=None, complexity=0.5):
    """Ink in water: a few large dye masses at low complexity (full, smooth) dispersed
    into many fine tendrils at high complexity. Brightness held up so it never goes black."""
    c = complexity; rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:n, 0:n].astype(float)
    dye = np.zeros((n, n))
    for _ in range(int(_L(4, 18, c))):
        cx, cy = rng.uniform(0, n, 2); r = _L(0.16, 0.05, c) * n
        dye += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    dye = _normalize01(dye); vx, vy = _msflow(n, rng, c)
    for _ in range(int(_L(8, 20, c))):
        dye = _advect(dye, vx, vy, 1, 1.2); dye = gaussian_filter(dye, _L(0.7, 0.35, c))
    return _fl_bright(dye, 0.42)


def rheoscopic(n=512, seed=None, complexity=0.5):
    """Kalliroscope silk via line-integral convolution of band-limited noise along a
    curl-noise flow. Low complexity = large smooth silk swirls (heavily smoothed flow,
    coarse noise); high complexity = fine busy silk. ~MONOFRACTAL flow-viz texture."""
    c = complexity; rng = np.random.default_rng(seed); psi = np.zeros((n, n))
    octaves = 1 + int(round(2 * c))
    for o in range(octaves):
        psi += (0.6 ** o) * _normalize01(_fractional_field(n, beta=3.2, seed=(None if seed is None else seed + o)))
    gy, gx = np.gradient(gaussian_filter(_normalize01(psi), _L(7.0, 1.5, c))); sp = np.hypot(gx, gy) + 1e-6
    vx = gy / sp * (n * 0.011); vy = -gx / sp * (n * 0.011)
    noise = gaussian_filter(rng.normal(size=(n, n)), _L(2.6, 0.5, c))
    steps = int(_L(42, 18, c)); acc = noise.copy(); cur = noise.copy()
    for _ in range(steps):
        cur = _advect(cur, vx, vy, 1, 1.0); acc = acc + cur
    cur = noise.copy()
    for _ in range(steps):
        cur = _advect(cur, -vx, -vy, 1, 1.0); acc = acc + cur
    return _normalize01(gaussian_filter(_normalize01(acc / (2 * steps + 1)), 0.4) * (0.45 + 0.55 * _normalize01(sp)))


def choppy(n=512, seed=None, complexity=0.5):
    """Water surface as a single multifractal cascade: smooth and calm at low
    complexity, rough with dense crisp glints (figure-ground) at high complexity.
    Light relief + strong contrast. MULTIFRACTAL (c2 ~ -0.08 .. -0.40); complexity
    raises both visual intricacy AND multifractality in the same direction."""
    c = complexity
    base = _normalize01(_fl_fracint(_fl_cascade(n, 7, _L(0.32, 0.98, c), seed), _L(1.15, 0.40, c)))
    hs = gaussian_filter(base, 0.6); hy, hx = np.gradient(hs)
    lx, ly, lz = -0.6, -0.5, 0.72; ln = np.sqrt(lx * lx + ly * ly + lz * lz); lx, ly, lz = lx / ln, ly / ln, lz / ln
    nrm = 1.0 / np.sqrt(hx ** 2 + hy ** 2 + 1); diffuse = np.clip((-hx * lx - hy * ly + lz) * nrm, 0, None)
    ao = _normalize01(hs - gaussian_filter(hs, 9)); shaded = _normalize01(0.32 + 0.5 * diffuse + 0.3 * ao)
    return _fl_contrast(_fl_bright(_normalize01(0.78 * base + 0.22 * shaded), 0.5), _L(2.8, 4.6, c))


FLUID_FAMILIES = ["eddies", "vorticity", "plume", "curl_weave", "dye_diffusion", "rheoscopic", "choppy"]
# fluid families that accept a `complexity` argument (the NS sims do not)
CX_FLUIDS = ["curl_weave", "dye_diffusion", "rheoscopic", "choppy"]
