"""Photorealistic cloudscape renderer (COLOR output, HxWx3 in [0, 1]).

NOTE: this is an aesthetic renderer, NOT a validated multifractal stimulus family.
Clouds are built as stacked 3D spheres (metaballs) to give a real height surface
with rounded cauliflower lobes, then lit (diffuse + ambient occlusion) and composited
over a sky gradient. Use it for pretty/naturalistic cumulus imagery; use the abstract
families in `clouds.py` for controlled multifractal pareidolia stimuli.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .generators import _normalize01
from .clouds import _cascade, _fracint

__all__ = ["cloudscape", "SKY_PALETTES"]

# sky_top, sky_bottom (RGB), cloud_white (lit), cloud_shadow (cool/warm), diffuse_gain
SKY_PALETTES = {
    "day":      (np.array([0.20, 0.42, 0.73]), np.array([0.52, 0.74, 0.92]),
                 np.array([1.00, 1.00, 1.00]), np.array([0.52, 0.59, 0.73]), 0.70),
    "golden":   (np.array([0.28, 0.34, 0.54]), np.array([0.98, 0.78, 0.52]),
                 np.array([1.00, 0.96, 0.86]), np.array([0.52, 0.50, 0.62]), 0.78),
    "overcast": (np.array([0.60, 0.64, 0.69]), np.array([0.80, 0.83, 0.86]),
                 np.array([0.94, 0.95, 0.97]), np.array([0.58, 0.60, 0.65]), 0.42),
}


def cloudscape(n=512, seed=None, coverage=0.45, puff=1.0, n_clusters=None,
               sky="day", sun=(-0.6, -0.5, 0.6)):
    """Render a naturalistic cumulus cloudscape.

    n          output side length (returns an (n, n, 3) float array in [0, 1])
    seed       RNG seed
    coverage   roughly how much sky is filled (0.2 sparse .. 0.7 crowded)
    puff       lobe size / billowing (0.7 small fair-weather .. 1.5 big towers)
    n_clusters number of cloud masses (default scales with coverage)
    sky        palette key in SKY_PALETTES ('day', 'golden', 'overcast')
    sun        3-vector light direction (x, y up is negative, z toward viewer)
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((n, n))
    if n_clusters is None:
        n_clusters = max(2, int(round(coverage * 10)))

    def add_sphere(cx, cy, r):
        x0 = max(0, int(cx - r)); x1 = min(n, int(cx + r) + 1)
        y0 = max(0, int(cy - r)); y1 = min(n, int(cy + r) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]; d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        h = np.sqrt(np.clip(r * r - d2, 0, None))
        H[y0:y1, x0:x1] = np.maximum(H[y0:y1, x0:x1], h)

    for _ in range(n_clusters):
        ccx, ccy = rng.uniform(0.18, 0.82, 2) * n
        spread = rng.uniform(0.07, 0.13) * n * puff
        base_r = rng.uniform(0.05, 0.085) * n * puff
        nsp = int(rng.integers(70, 130))
        for _ in range(nsp):
            ang = rng.uniform(0, 2 * np.pi); rad = abs(rng.normal(0, spread))
            if rad > 2.0 * spread:          # cull stray far bubbles
                continue
            sx = ccx + rad * np.cos(ang); sy = ccy + rad * np.sin(ang) * 0.75 - rad * 0.12
            r = base_r * rng.uniform(0.6, 1.25) * np.exp(-rad / (1.5 * spread))
            add_sphere(sx, sy, max(4, r))

    # fine cauliflower surface texture so lobes are not glassy
    tex = _normalize01(_fracint(_cascade(n, 7, 0.8, (None if seed is None else seed + 1)), 0.5))
    H = H * (0.72 + 0.45 * tex)
    H = gaussian_filter(H, 2.4)

    d = gaussian_filter(np.clip(H / (0.035 * n), 0, 1), 1.2)
    support = gaussian_filter(d, 6)
    d = np.where(support < 0.08, 0.0, d)     # remove isolated specks

    gy, gx = np.gradient(H); nrm = 1.0 / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    Lx, Ly, Lz = sun; Ln = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz); Lx, Ly, Lz = Lx / Ln, Ly / Ln, Lz / Ln
    diffuse = np.clip((-gx * Lx - gy * Ly + Lz) * nrm, 0, 1)
    ao = np.clip(1.0 - 1.3 * np.clip(gaussian_filter(H, 10) - H, 0, None) / (0.02 * n + 1e-6), 0.4, 1.0)

    sky_top, sky_bot, white, shadowc, dgain = SKY_PALETTES.get(sky, SKY_PALETTES["day"])
    bright = np.clip(0.36 + dgain * diffuse, 0, 1.1) * ao
    t = np.clip(bright, 0, 1)[..., None]
    cloud = shadowc * (1 - t) + white * t

    yy = (np.arange(n)[:, None] / (n - 1)) * np.ones((1, n))
    sky_rgb = sky_top[None, None, :] * (1 - yy[..., None]) + sky_bot[None, None, :] * yy[..., None]
    a = np.clip(d, 0, 1)[..., None]
    return np.clip(sky_rgb * (1 - a) + cloud * a, 0, 1)
