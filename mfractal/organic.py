"""
organic.py -- process-based multifractal generators that look organic, not
procedural. Each enacts a multiplicative cascade through a physical process
(aggregation, chaotic mixing, branching transport) rather than spectral filtering.

  dla()              : diffusion-limited aggregation. Harmonic-measure
                       multifractal. Coral / frost / neuron dendrites.
  stir_and_fold()    : chaotic-mixing (Lagrangian) cascade. Striated
                       multifractal scalar. Marbling / agate / wood grain.
  branching_network():recursive stochastic branching with multiplicative flux
                       splits. Cascade on a tree. Venation / vessels / lightning.

Quantify dla/branching (positive measures) with quantify.moments_2d; quantify
stir_and_fold (a signed surface) with quantify.mfdfa_2d. Display with
flagships.punch.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree

from .generators import _normalize01


# ---------------------------------------------------------------------------
# 1. Diffusion-limited aggregation (walker DLA)
# ---------------------------------------------------------------------------
def dla(n=201, n_particles=2500, stick=1.0, blur=1.2, seed=None):
    """
    Diffusion-limited aggregation by random walkers. The cluster's coarse
    occupancy is a multifractal measure; rendered blurred it reads as coral /
    frost / a neuron. `stick` < 1 makes denser, less feathery clusters.
    """
    rng = np.random.default_rng(seed)
    occ = np.zeros((n, n), bool); c = n // 2; occ[c, c] = True
    R = 1.0; size = 1
    nb = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while size < n_particles:
        rad = min(R + 3, n // 2 - 2); a = rng.uniform(0, 2 * np.pi)
        x = int(c + rad * np.cos(a)); y = int(c + rad * np.sin(a))
        kill = R * 2 + 15
        for _ in range(8000):
            adj = False
            for dx, dy in nb:
                xx, yy = x + dx, y + dy
                if 0 <= xx < n and 0 <= yy < n and occ[yy, xx]:
                    adj = True; break
            if adj and rng.random() < stick:
                occ[y, x] = True; size += 1
                R = max(R, np.hypot(x - c, y - c)); break
            dx, dy = nb[rng.integers(4)]; x += dx; y += dy
            if (x - c) ** 2 + (y - c) ** 2 > kill * kill or not (0 <= x < n and 0 <= y < n):
                rad = min(R + 3, n // 2 - 2); a = rng.uniform(0, 2 * np.pi)
                x = int(c + rad * np.cos(a)); y = int(c + rad * np.sin(a))
    field = gaussian_filter(occ.astype(float), blur)
    return _normalize01(field)


# ---------------------------------------------------------------------------
# 2. Chaotic stir-and-fold (Lagrangian cascade)
# ---------------------------------------------------------------------------
def stir_and_fold(n=512, steps=70, amp=4.5, kscale=1.6, diffusion=0.4,
                  inject=0.02, mode="dissipation", seed=None):
    """
    Advect a smooth scalar by random-phase alternating sine flows (a chaotic
    mixing map), with mild diffusion and a sustaining large-scale source. Repeated
    stretch-and-fold concentrates structure onto striated filaments -> a genuine
    multifractal "strange eigenmode". Marbling / agate / wood grain / cytoplasm.

    mode='dissipation' (default) renders the scalar-gradient magnitude, where the
    multifractality lives (strongly multifractal, vivid filaments); mode='scalar'
    returns the smooth marbled scalar itself (weakly multifractal).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    source = gaussian_filter(rng.standard_normal((n, n)), n / 12.0, mode="wrap")
    source = (source - source.mean()) / (source.std() + 1e-9)
    s = source.copy()
    for _ in range(steps):
        ph = rng.uniform(0, 2 * np.pi)
        dx = amp * np.sin(2 * np.pi * kscale * yy / n + ph)
        s = map_coordinates(s, [yy % n, (xx + dx) % n], order=1, mode="grid-wrap")
        ph2 = rng.uniform(0, 2 * np.pi)
        dy = amp * np.sin(2 * np.pi * kscale * xx / n + ph2)
        s = map_coordinates(s, [(yy + dy) % n, xx % n], order=1, mode="grid-wrap")
        if diffusion > 0:
            s = gaussian_filter(s, diffusion, mode="wrap")
        if inject > 0:
            s = (1 - inject) * s + inject * source
    if mode == "dissipation":
        gy, gx = np.gradient(s)
        s = np.hypot(gx, gy)
    return _normalize01(s)


# ---------------------------------------------------------------------------
# 3. Branching network: space-colonization + fractal-texture fill
# ---------------------------------------------------------------------------
def _disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def branching_network(n=512, n_attractors=4200, influence=50, kill=7, step=3.4,
                      width_gain=1.6, texture=0.8, membrane=0.35, rough=0.6,
                      tex_granularity=2.2, tex_multifractality=1.4,
                      max_iter=600, seed=None):
    """
    Space-colonization branching network filled with multifractal texture.

    Grows toward a clustered cloud of attractor points, so the network is
    complex and space-filling (rich recursive venation, not a few bolts). Flux
    summed bottom-up sets Murray-law widths. The veins are then modulated by a
    multifractal texture field (mottled / rough) and the inter-vein membrane is
    filled with the same texture fading from the veins. Venation / vessels /
    coral / lung / lichen. Strongly multifractal (quantify with moments_2d).

    n_attractors        : structural complexity (more = denser network)
    influence/kill/step : growth geometry (smaller kill & step = finer twigs)
    texture / membrane  : strength of the fractal fill on veins / in the tissue
    rough               : how strongly the veins are mottled by the texture
    tex_granularity/tex_multifractality : the fill texture's grain and Delta_alpha
    """
    from .flagships import multifractal_cloud, punch
    rng = np.random.default_rng(seed)

    # 1. grow a space-colonization tree toward a clustered attractor cloud
    cks = rng.uniform(0.12 * n, 0.88 * n, size=(7, 2))
    pts = [rng.normal([cx, cy], 0.15 * n, size=(n_attractors // 7, 2))
           for cx, cy in cks]
    A = np.clip(np.vstack(pts), 2, n - 3)
    nodes = [np.array([n * 0.5, n * 0.9])]; parent = [-1]; it = 0
    while len(A) > 0 and it < max_iter:
        it += 1
        d, idx = cKDTree(np.asarray(nodes)).query(A, distance_upper_bound=influence)
        v = np.isfinite(d)
        if not v.any():
            break
        gm = {}
        for ai in np.where(v)[0]:
            gm.setdefault(int(idx[ai]), []).append(A[ai])
        grew = False
        for ni, at in gm.items():
            at = np.asarray(at); dd = at - nodes[ni]
            nn = np.linalg.norm(dd, axis=1, keepdims=True) + 1e-9
            dv = (dd / nn).mean(0); m = np.linalg.norm(dv)
            if m < 1e-9:
                continue
            nodes.append(nodes[ni] + step * (dv / m)); parent.append(ni); grew = True
        dk, _ = cKDTree(np.asarray(nodes)).query(A, distance_upper_bound=kill)
        A = A[~np.isfinite(dk)]
        if not grew:
            break
    nodes = np.asarray(nodes); parent = np.asarray(parent)

    # 2. conserved flux (bottom-up) -> Murray-law widths -> render
    N = len(nodes); flux = np.ones(N)
    for i in range(N - 1, 0, -1):
        if parent[i] >= 0:
            flux[parent[i]] += flux[i]
    flux /= flux.max()
    net = np.zeros((n, n)); D = {r: _disk(r) for r in range(0, 11)}
    for i in range(1, N):
        p = parent[i]
        if p < 0:
            continue
        r = int(np.clip(round(width_gain * (flux[i] ** (1 / 3)) * 8), 0, 10))
        val = flux[i] ** (1 / 3)
        x0, y0 = nodes[p]; x1, y1 = nodes[i]
        L = int(np.hypot(x1 - x0, y1 - y0)) + 1
        for t in np.linspace(0, 1, L + 1):
            ix = int(round(x0 + (x1 - x0) * t)); iy = int(round(y0 + (y1 - y0) * t))
            if ix - r < 0 or iy - r < 0 or ix + r + 1 > n or iy + r + 1 > n:
                continue
            sl = net[iy - r:iy + r + 1, ix - r:ix + r + 1]
            np.maximum(sl, val * D[r], out=sl)

    # 3. fractal-texture fill: mottled veins + fractal membrane near the veins
    tex = _normalize01(multifractal_cloud(
        n, granularity=tex_granularity, multifractality=tex_multifractality,
        smooth=0.18, distribution=2.5, calm_scale=5.0,
        seed=(None if seed is None else seed + 101)))
    tex = punch(tex, low=3, high=97, strength=1.5)
    prox = np.exp(-distance_transform_edt(net < 0.02) / 22.0)
    veins = net * (1 - rough + rough * tex)
    return _normalize01(veins * (0.6 + 0.4 * texture) + tex * prox * membrane)
