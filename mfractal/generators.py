"""
generators.py
=============
Generators for 2D scalar fields with controlled fractal / multifractal
structure, aimed at building ambiguous (cloud-like) pareidolia stimuli.

Three families are provided, ordered from "what you already used" to the
most principled multifractal generator:

1. MONOFRACTAL baselines (single power-law spectrum -> single Hurst exponent):
     - fbm_spectral()       : fractional Brownian surface via spectral synthesis
     - stacked_value_noise(): octave-summed value noise (the "stacked Perlin"
                              family). Approximately monofractal.

2. MULTIPLICATIVE CASCADE (the textbook multifractal *measure*):
     - multiplicative_cascade()

3. UNIVERSAL / CONTINUOUS MULTIFRACTALS:
     - lognormal_mrm()       : log-normal multifractal random measure via
                               Gaussian multiplicative chaos (alpha = 2 case).
                               Tunable intermittency (sigma) and smoothness (H).
     - universal_multifractal(): general Levy index alpha in (0,2] (advanced).

Design note
-----------
For a stimulus battery you want to vary FRACTAL DIMENSION (set by the global
spectral slope / H) and DEGREE OF MULTIFRACTALITY (spectrum width, set by the
intermittency parameters) as independently as possible. The log-normal MRM
exposes exactly those two knobs: H (-> slope/FD) and sigma (-> multifractality),
which are close to orthogonal. See demo for an empirical check.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _radial_freq(n, m=None):
    """Radial spatial frequency grid |k| for an (n x m) FFT, DC at [0,0]."""
    m = n if m is None else m
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(m)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    return k


def _normalize01(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _power_law_filter(field, exponent, rng=None):
    """Multiply the FFT amplitude of `field` by |k|^(-exponent), DC removed."""
    F = np.fft.fft2(field)
    k = _radial_freq(*field.shape)
    filt = np.zeros_like(k)
    nz = k > 0
    filt[nz] = k[nz] ** (-exponent)
    out = np.fft.ifft2(F * filt).real
    return out


# ---------------------------------------------------------------------------
# 1. MONOFRACTAL baselines
# ---------------------------------------------------------------------------
def fbm_spectral(n=512, beta=3.0, seed=None):
    """
    Fractional Brownian surface via Fourier (spectral) synthesis.

    Power spectral density S(k) ~ |k|^(-beta). This is what your 2022
    1/f^beta stimuli were. It is MONOFRACTAL: a single slope -> a single
    Hurst exponent H = (beta - 2) / 2 (2D convention) -> the multifractal
    spectrum f(alpha) collapses to (almost) a single point.

    Common slope -> dimension mappings (conventions differ; pick one and
    state it): for a 2D fBm surface, D_surface = 4 - beta/2  (range 2..3),
    and the iso-set / image "fractal dimension" people often quote is
    D = (8 - beta) / 2.

    Parameters
    ----------
    n     : image size (n x n)
    beta  : spectral slope (larger = smoother = lower FD)
    seed  : RNG seed
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    F = np.fft.fft2(white)
    k = _radial_freq(n)
    amp = np.zeros_like(k)
    nz = k > 0
    amp[nz] = k[nz] ** (-beta / 2.0)
    field = np.fft.ifft2(F * amp).real
    return _normalize01(field)


def stacked_value_noise(n=512, octaves=7, persistence=0.5, lacunarity=2.0,
                        seed=None):
    """
    Octave-summed smooth value noise -- the "stacked Perlin" family.

    Summing band-limited noise octaves with a fixed amplitude ratio
    (persistence) yields a field whose spectrum is approximately a single
    power law -> approximately MONOFRACTAL. Included to show that this
    popular trick does NOT, by itself, buy you multifractality.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp = 1.0
    freq = 1.0
    total_amp = 0.0
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(octaves):
        cells = max(2, int(round(freq)) + 1)
        grid = rng.standard_normal((cells + 1, cells + 1))
        # bilinear upsample of the coarse grid with smoothstep weights
        gy = yy / n * cells
        gx = xx / n * cells
        y0 = np.floor(gy).astype(int); x0 = np.floor(gx).astype(int)
        y0 = np.clip(y0, 0, cells - 1); x0 = np.clip(x0, 0, cells - 1)
        ty = gy - y0; tx = gx - x0
        sy = ty * ty * (3 - 2 * ty); sx = tx * tx * (3 - 2 * tx)
        v00 = grid[y0, x0]; v10 = grid[y0 + 1, x0]
        v01 = grid[y0, x0 + 1]; v11 = grid[y0 + 1, x0 + 1]
        top = v00 * (1 - sx) + v01 * sx
        bot = v10 * (1 - sx) + v11 * sx
        layer = top * (1 - sy) + bot * sy
        out += amp * layer
        total_amp += amp
        amp *= persistence
        freq *= lacunarity
    return _normalize01(out / total_amp)


# ---------------------------------------------------------------------------
# 2. MULTIPLICATIVE CASCADE (textbook multifractal measure)
# ---------------------------------------------------------------------------
def multiplicative_cascade(levels=9, weights=(0.6, 0.8, 1.2, 1.4),
                           randomize=True, microcanonical=True, seed=None):
    """
    2D multiplicative (random) cascade -> a positive multifractal MEASURE.

    Start from a uniform square. Recursively split every cell into 2x2 and
    multiply each child by a random weight drawn from `weights`. After
    `levels` subdivisions the image is 2**levels on a side.

    - microcanonical=True : the 4 weights in each parent are a random
      permutation of the 4 values in `weights` -> measure exactly conserved
      at every step (sum preserved). Sharpest, most reproducible spectrum.
    - microcanonical=False: each child weight drawn i.i.d. (canonical),
      mean is preserved only on average -> heavier tails, wider spectrum.

    The SPREAD of `weights` sets the degree of multifractality:
    equal weights -> monofractal (flat measure); wide spread -> broad f(alpha).
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, float)
    field = np.ones((1, 1))
    for _ in range(levels):
        n = field.shape[0]
        new = np.empty((2 * n, 2 * n))
        for (di, dj) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            if microcanonical:
                w = np.ones((n, n))  # filled below per-parent
            new[di::2, dj::2] = field  # placeholder, weights applied next
        # apply weights per 2x2 block
        if microcanonical:
            # for each parent cell assign a permutation of the 4 weights
            perm_idx = np.argsort(rng.random((n, n, 4)), axis=2)
            wsel = weights[perm_idx % len(weights)]  # (n,n,4)
            # normalize each block so the 4 weights average to 1 (conserve)
            wsel = wsel * (4.0 / wsel.sum(axis=2, keepdims=True))
            for q, (di, dj) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                new[di::2, dj::2] = field * wsel[:, :, q]
        else:
            for (di, dj) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                w = rng.choice(weights, size=(n, n)) if randomize \
                    else weights[(di * 2 + dj) % len(weights)]
                new[di::2, dj::2] = field * w
        field = new
    # return both the raw measure and a [0,1] image view
    return field


# ---------------------------------------------------------------------------
# 3. UNIVERSAL / CONTINUOUS MULTIFRACTALS
# ---------------------------------------------------------------------------
def _log_correlated_gaussian(n, seed=None):
    """
    2D log-correlated Gaussian field (discrete GFF): filter white Gaussian
    noise by |k|^(-1) so PSD ~ |k|^(-2). Returned with unit variance.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    g = _power_law_filter(white, exponent=1.0)
    g -= g.mean()
    s = g.std()
    if s > 0:
        g /= s
    return g


def lognormal_mrm(n=512, sigma=1.2, H=0.0, seed=None, return_flux=False):
    """
    Log-normal Multifractal Random Measure (alpha = 2 universal multifractal)
    via Gaussian multiplicative chaos.

    eps = exp(sigma * g) / mean(...)      with g log-correlated, unit variance
    -> a conservative (mean-1) multifractal FLUX. Optionally fractionally
    integrate by H to set the global spectral slope (-> FD) of the final image.

    Knobs
    -----
    sigma : intermittency. 0 -> monofractal; larger -> WIDER f(alpha).
            (Controls multifractality, ~ proportional to sqrt of C1.)
    H     : fractional-integration exponent applied AFTER exponentiation.
            Larger H -> smoother image -> steeper spectrum -> lower FD.
            H mainly moves the slope/FD and only weakly affects spectrum width,
            so (sigma, H) act as a near-orthogonal (multifractality, FD) basis.

    return_flux : if True, also return the bare flux measure `eps` (best
                  analysed with the moment / partition-function method).
    """
    g = _log_correlated_gaussian(n, seed=seed)
    eps = np.exp(sigma * g)
    eps /= eps.mean()                      # canonical normalization (mean 1)
    if H > 0:
        img = _power_law_filter(eps, exponent=H)
    else:
        img = eps.copy()
    image = _normalize01(img)
    if return_flux:
        return image, eps
    return image


def _levy_stable(alpha, size, rng, beta=1.0):
    """
    Maximally-skewed (extremal) alpha-stable variates via Chambers-Mallows-
    Stuck. Used as the subgenerator for general universal multifractals.
    """
    if abs(alpha - 1.0) < 1e-6:
        alpha = 1.0 - 1e-4
    U = (rng.random(size) - 0.5) * np.pi          # uniform(-pi/2, pi/2)
    W = -np.log(rng.random(size))                  # exp(1)
    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    xi = np.arctan(-zeta) / alpha
    X = ((1 + zeta ** 2) ** (1 / (2 * alpha))
         * np.sin(alpha * (U + xi)) / (np.cos(U) ** (1 / alpha))
         * (np.cos(U - alpha * (U + xi)) / W) ** ((1 - alpha) / alpha))
    return X


def universal_multifractal(n=512, alpha=1.8, C1=0.15, H=0.0, seed=None,
                           return_flux=False):
    """
    General universal multifractal (Schertzer-Lovejoy) FIF construction.

    Levy subgenerator -> fractional integration by |k|^(-d/alpha) (d=2)
    -> exponentiate -> canonical (mean-1) normalization -> optional
    fractional integration by H.

    alpha : multifractality index in (0,2]. 2 == log-normal (use lognormal_mrm
            for the stable version). Smaller alpha -> more "all-or-nothing"
            intermittency.
    C1    : codimension of the mean singularity (sparseness of the support
            of the dominant activity). Larger C1 -> sparser, spikier, WIDER.
    H     : final fractional integration (slope / FD knob), as in lognormal_mrm.

    Note: heavy Levy tails make high-order empirical moments diverge; analyse
    with a bounded q-range. lognormal_mrm is the robust default.
    """
    rng = np.random.default_rng(seed)
    L = _levy_stable(alpha, (n, n), rng, beta=1.0)
    # scale so intermittency ~ C1 (approximate, empirical normalization follows)
    scale = (C1 / abs(alpha - 1.0)) ** (1.0 / alpha)
    gamma = _power_law_filter(L, exponent=2.0 / alpha)  # d/alpha, d=2
    gamma -= gamma.mean()
    s = gamma.std()
    if s > 0:
        gamma = gamma / s * scale
    gamma = np.clip(gamma, None, 20.0)        # guard against overflow
    eps = np.exp(gamma)
    eps /= eps.mean()
    if H > 0:
        img = _power_law_filter(eps, exponent=H)
    else:
        img = eps.copy()
    image = _normalize01(img)
    if return_flux:
        return image, eps
    return image


# ---------------------------------------------------------------------------
# 4. COMBINING MONOFRACTALS
# ---------------------------------------------------------------------------
def additive_fbm_sum(n=512, betas=(2.0, 3.0, 4.0), weights=None, seed=None):
    """
    Independent fBm surfaces, SUMMED.

    This is the naive "combine monofractals" idea -- and it does NOT give a true
    multifractal. At small scales the sum is dominated by the ROUGHEST component
    (smallest beta), so the local Holder exponent is ~constant -> still
    monofractal. Included precisely to demonstrate that additive mixing fails.
    """
    rng = np.random.default_rng(seed)
    betas = np.asarray(betas, float)
    weights = np.ones(len(betas)) if weights is None else np.asarray(weights, float)
    out = np.zeros((n, n))
    for b, wt in zip(betas, weights):
        s = int(rng.integers(0, 2 ** 31 - 1))
        out += wt * fbm_spectral(n, beta=b, seed=s)     # independent fields
    return _normalize01(out)


def fbm_bank(n, betas, seed=None):
    """
    A bank of fBm fields that share the SAME random phases (one white-noise
    realization, different spectral slopes). Shared phases keep the large-scale
    structure aligned across the bank, so per-pixel blending between them is
    seamless. Each field is returned zero-mean, unit-variance.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    F = np.fft.fft2(white)
    k = _radial_freq(n)
    bank = []
    for beta in betas:
        amp = np.zeros_like(k); nz = k > 0
        amp[nz] = k[nz] ** (-beta / 2.0)
        f = np.fft.ifft2(F * amp).real
        f -= f.mean(); s = f.std()
        if s > 0:
            f /= s
        bank.append(f)
    return np.asarray(bank)                              # (K, n, n)


def _smooth_control_map(n, beta=4.0, seed=None):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    f = _power_law_filter(white, exponent=beta / 2.0)
    return _normalize01(f)


def mbm_blend(n=512, beta_lo=2.2, beta_hi=4.2, K=8, map_beta=4.5,
              seed=None, return_map=False):
    """
    Multifractional field: roughness (local Hurst exponent) varies SMOOTHLY in
    space. Built the right way to "combine monofractals": blend a phase-locked
    bank of fBm fields per-pixel according to a smooth control map H(x,y).

    Because it is an additive Gaussian-type field, it keeps FULL DYNAMIC RANGE
    everywhere -> it does NOT go dark or sparse, unlike multiplicative cascades.
    Yet local regularity ranges over [H_lo, H_hi] -> a broad distribution of
    Holder exponents -> a broad f(alpha) (detectable by MFDFA).

    beta_lo/beta_hi : slope range of the bank (-> roughness range; H=(beta-2)/2)
    map_beta        : smoothness of the spatial roughness map (larger = smoother
                      patches of constant roughness)
    """
    betas = np.linspace(beta_lo, beta_hi, K)
    bank = fbm_bank(n, betas, seed=seed)
    m = _smooth_control_map(n, beta=map_beta,
                            seed=None if seed is None else seed + 1)
    idx = m * (K - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, K - 1)
    w = idx - lo
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    field = (1 - w) * bank[lo, ii, jj] + w * bank[hi, ii, jj]
    image = _normalize01(field)
    if return_map:
        return image, m
    return image


def modulated_fbm(n=512, beta=3.0, mod_beta=5.0, depth=1.0, seed=None):
    """
    A monofractal fBm whose local AMPLITUDE is modulated by a smooth positive
    field (a 'multiplier'). amplitude-modulation is a mild multiplicative move,
    so it injects local-variance fluctuations (a weak, controllable
    multifractality) while staying full-range and cloud-like.

    depth : 0 -> pure monofractal; larger -> stronger local-variance variation.
    """
    rng = np.random.default_rng(seed)
    base = fbm_spectral(n, beta=beta, seed=int(rng.integers(0, 2 ** 31 - 1)))
    base = base - base.mean()
    env = _smooth_control_map(n, beta=mod_beta,
                              seed=int(rng.integers(0, 2 ** 31 - 1)))
    multiplier = np.exp(depth * (env - env.mean()))
    return _normalize01(base * multiplier)


# ---------------------------------------------------------------------------
# 5. TRUE MULTIFRACTAL WITHOUT DARKENING -- candidate techniques
# ---------------------------------------------------------------------------
def _fractional_field(n, beta=3.0, seed=None):
    """Signed, zero-mean, unit-variance fractional field (fBm-like surface)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    F = np.fft.fft2(white)
    k = _radial_freq(n)
    amp = np.zeros_like(k); nz = k > 0
    amp[nz] = k[nz] ** (-beta / 2.0)
    f = np.fft.ifft2(F * amp).real
    f -= f.mean(); s = f.std()
    return f / s if s > 0 else f


def mrw_field(n=512, beta=3.0, sigma=0.5, seed=None, return_parts=False):
    """
    2D Multifractal Random Walk (volatility-modulated fractional field).

    field = G(x,y) * exp(sigma * omega(x,y))
      G     : signed fractional field (sets the spatial slope / mean FD)
      omega : log-correlated Gaussian field (the multiplier / 'volatility')
      sigma : intermittency -> moment-multifractality width

    Because G is SIGNED and full-range, the field uses the whole gray axis
    (dark..mid..bright); the multifractality lives in the LOCAL VARIANCE, not
    the mean intensity, so it does NOT collapse to a sparse bright set. This is
    the canonical Bacry-Muzy-Delour multifractal model (turbulence/finance).
    """
    rng = np.random.default_rng(seed)
    G = _fractional_field(n, beta=beta, seed=int(rng.integers(0, 2 ** 31 - 1)))
    omega = _log_correlated_gaussian(n, seed=int(rng.integers(0, 2 ** 31 - 1)))
    field = G * np.exp(sigma * omega)
    if return_parts:
        return field, G, omega
    return field


def floored_mrm(n=512, sigma=1.2, H=0.4, floor=0.3, seed=None):
    """
    Multiplicative MRM with an additive FLOOR before normalization, to lift the
    dark voids into visibility. floor in [0,1] as a fraction of the mean.
    Trades a little multifractality for far less darkening.
    """
    img, eps = lognormal_mrm(n, sigma=sigma, H=H, seed=seed, return_flux=True)
    lifted = eps + floor * eps.mean()
    if H > 0:
        lifted = _power_law_filter(lifted, exponent=H)
    return _normalize01(lifted)


def compressed_mrm(n=512, sigma=1.2, H=0.4, power=0.4, seed=None):
    """
    Multiplicative MRM passed through a mild power compression x**power
    (power<1) to lift midtones. Less aggressive than log/equalization, so some
    multifractality survives. (Any monotone compression erodes the moment
    spectrum -- this just does it gently.)
    """
    img = lognormal_mrm(n, sigma=sigma, H=H, seed=seed)
    return _normalize01(np.power(img + 1e-6, power))


def mrw_cloud(n=512, beta=3.2, sigma=0.7, vol_scale=0.8, smooth=0.6,
              seed=None, return_parts=False):
    """
    FLAGSHIP non-darkening multifractal: a smoothed 2D multifractal random walk.

    A signed fractional field G is modulated by a coherent log-correlated
    volatility field, then mildly fractionally integrated to turn high-volatility
    bursts into formable cloud structure. Genuinely multifractal in the MOMENT
    sense (wide f(alpha)) yet full gray-range (no darkening), because the
    multifractality lives in local variance, not mean intensity.

    sigma     : intermittency -> moment-multifractality (Delta_alpha). Main knob.
    beta      : base slope -> mean roughness / FD.
    vol_scale : extra smoothing of the volatility field -> size of 'active'
                vs 'calm' patches (larger = bigger, more coherent regions).
    smooth    : post fractional integration -> cloudiness. Keep mild (~0.4-0.7);
                too much erodes the multifractality.
    """
    rng = np.random.default_rng(seed)
    G = _fractional_field(n, beta=beta, seed=int(rng.integers(0, 2 ** 31 - 1)))
    omega = _log_correlated_gaussian(n, seed=int(rng.integers(0, 2 ** 31 - 1)))
    if vol_scale > 0:
        omega = _power_law_filter(omega, exponent=vol_scale)
        omega = (omega - omega.mean()) / (omega.std() + 1e-9)
    field = G * np.exp(sigma * omega)
    if smooth > 0:
        field = _power_law_filter(field, exponent=smooth)
    if return_parts:
        return _normalize01(field), G, omega
    return _normalize01(field)
