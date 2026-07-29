"""
local_cumulants.py — spatially-resolved (c1, c2) maps, and the MF-ControlNet conditioning image.

The conditioning for Phase B is a MEASUREMENT OF THE TARGET IMAGE, which is what makes
"conditioning cumulants == target cumulants" an identity rather than an approximation
(and so kills the img2img fade trap).

Method: run the SAME wavelet-leader transform as mfractal.quantify.wavelet_leaders_2d
(db3, periodization, 3x3 max filter + child-max propagation), but replace the GLOBAL
mean/var of log-leaders at each scale with a BOX-FILTERED LOCAL mean/var, then regress
across scales per cell using the SAME j_fit. Cost: one DWT + a few box filters.

Do NOT instead call wavelet_leaders_2d on independent crops: it auto-picks its fit band
from the input size, so the octaves being fitted move with the window (measured: c1 bias
+0.26 and c2 within-image sd ~2.0 at W=128). The shared pyramid keeps the fit band fixed.

Conditioning image (uint8 RGB, GRID x GRID cells upsampled to `res`):
    R = c1 map  -> 255*(v - C1_LO)/(C1_HI - C1_LO)
    G = c2 map  -> 255*(v - C2_LO)/(C2_HI - C2_LO)
    B = validity mask (255 = cell specified, 0 = free / don't care)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pywt
from PIL import Image
from scipy.ndimage import maximum_filter, uniform_filter, zoom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# --- canonical conventions (must match build_corpus / train_stage1) --------------------
CANON_RES = 512            # cumulants are defined on the 512 grayscale field
J_FIT = (2, 5)             # == wavelet_leaders_2d default at n=512 (J=6 -> (2, J-1))
WIN_PX = 256               # local window in pixels of the 512 field
GRID = 16                  # conditioning cells per side
C1_LO, C1_HI = 0.5, 2.7
C2_LO, C2_HI = -1.8, 0.4


def leaders(img, wavelet="db3", jmax=None):
    """Wavelet leaders, finest (j=1) first — identical to mfractal.quantify."""
    img = np.asarray(img, float)
    n = min(img.shape)
    J = max(3, (int(np.log2(n)) - 3) if jmax is None else jmax)
    coeffs = pywt.wavedec2(img, wavelet, level=J, mode="periodization")
    dmag = [np.maximum(np.maximum(np.abs(cH), np.abs(cV)), np.abs(cD))
            for (cH, cV, cD) in coeffs[1:]][::-1]
    out, prev = [], None
    for d in dmag:
        ell = maximum_filter(d, size=3, mode="nearest")
        if prev is not None:
            ph, pw = (prev.shape[0] // 2) * 2, (prev.shape[1] // 2) * 2
            ds = np.maximum.reduce([prev[0:ph:2, 0:pw:2], prev[1:ph:2, 0:pw:2],
                                    prev[0:ph:2, 1:pw:2], prev[1:ph:2, 1:pw:2]])
            h = min(ell.shape[0], ds.shape[0]); w = min(ell.shape[1], ds.shape[1])
            ell[:h, :w] = np.maximum(ell[:h, :w], ds[:h, :w])
        out.append(ell); prev = ell
    return out


def local_cumulant_map(field, win=WIN_PX, j_fit=J_FIT, grid=GRID):
    """(c1_map, c2_map) on a grid x grid lattice, from one shared wavelet pyramid.

    `field` must be the canonical grayscale [0,1] field (see build_corpus.to_canonical_gray).
    """
    L = leaders(field)
    lo, hi = j_fit
    js, M, V = [], [], []
    for j in range(lo, hi + 1):
        if j - 1 >= len(L):
            break
        x = np.log(np.maximum(L[j - 1], 1e-12))
        k = max(3, int(round(win / (2 ** j))))       # taps on this scale's grid
        m1 = uniform_filter(x, size=k, mode="nearest")
        v = np.maximum(uniform_filter(x * x, size=k, mode="nearest") - m1 * m1, 0.0)
        js.append(j); M.append(m1); V.append(v)
    up = lambda a: zoom(a, (grid / a.shape[0], grid / a.shape[1]), order=1)
    A = np.stack([up(m) for m in M])
    B = np.stack([up(v) for v in V])
    jc = np.asarray(js, float) - np.mean(js)
    denom = (jc ** 2).sum() * np.log(2.0)            # closed-form LSQ slope
    c1 = (jc[:, None, None] * A).sum(0) / denom
    c2 = (jc[:, None, None] * B).sum(0) / denom
    return c1, c2


# --- conditioning image encode / decode ------------------------------------------------
def encode(c1_map, c2_map, mask=None, res=1024):
    """(c1,c2) maps -> uint8 RGB PIL conditioning image at `res`."""
    r = np.clip((c1_map - C1_LO) / (C1_HI - C1_LO), 0, 1)
    g = np.clip((c2_map - C2_LO) / (C2_HI - C2_LO), 0, 1)
    b = np.ones_like(r) if mask is None else np.clip(mask, 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((res, res), Image.NEAREST)


def decode(pil, grid=GRID):
    """conditioning image -> (c1_map, c2_map, mask) on the cell grid (for debug/inference)."""
    a = np.asarray(pil.convert("RGB").resize((grid, grid), Image.BILINEAR), float) / 255.0
    return (a[..., 0] * (C1_HI - C1_LO) + C1_LO,
            a[..., 1] * (C2_HI - C2_LO) + C2_LO,
            a[..., 2])


# --- EXACT-LABEL maps (what training actually uses) -------------------------------------
# MEASURED FINDING (see PHASE_B.md): the per-cell local map above is NOISE-DOMINATED.
# On photographic images the map's spatial sd is at or below the sd produced by a
# statistically HOMOGENEOUS field (signal fraction 0-47% across win in {256,384,512} x
# grid in {8,16}); only near-global windows are stable, and those are degenerate (constant).
# Its MEAN is excellent (corr 0.993/0.998 with the global label), so we keep the map for
# ANALYSIS, but training conditioning uses EXACT labels only:
#   * uniform maps  -> the image's own measured (c1,c2): exact by definition
#   * mosaic maps   -> tiles from different images, each region carrying its own exact label,
#                      which is what forces the ControlNet to read the map LOCALLY.
# Inference maps (uniform / gradient / shape) are then in-distribution: because the CN is
# fully convolutional and these maps are smooth, every neighbourhood looks like training.

def mosaic(fields_labels, grid=GRID, tiles=2, res=CANON_RES, rng=None):
    """Stitch `tiles`x`tiles` crops with DIFFERENT statistics into one field + its exact map.

    fields_labels: list of (field(HxW float), c1, c2). Returns (field, c1_map, c2_map).
    Each tile region of the map carries that tile's own measured label — exact, not estimated.
    """
    rng = rng or np.random.default_rng(0)
    step = res // tiles
    out = np.zeros((res, res), float)
    c1m = np.zeros((grid, grid), float)
    c2m = np.zeros((grid, grid), float)
    gstep = grid // tiles
    for ti in range(tiles):
        for tj in range(tiles):
            f, c1, c2 = fields_labels[(ti * tiles + tj) % len(fields_labels)]
            f = np.asarray(f, float)
            # random crop of the source field, resized to the tile
            n = min(f.shape); s = max(step, int(n * rng.uniform(0.5, 1.0)))
            s = min(s, n)
            y = int(rng.integers(0, n - s + 1)); x = int(rng.integers(0, n - s + 1))
            crop = f[y:y + s, x:x + s]
            tile = np.asarray(Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
                              .resize((step, step), Image.BICUBIC), float) / 255.0
            out[ti * step:(ti + 1) * step, tj * step:(tj + 1) * step] = tile
            c1m[ti * gstep:(ti + 1) * gstep, tj * gstep:(tj + 1) * gstep] = c1
            c2m[ti * gstep:(ti + 1) * gstep, tj * gstep:(tj + 1) * gstep] = c2
    return out, c1m, c2m


# --- inference-time map builders -------------------------------------------------------
def uniform_map(c1, c2, grid=GRID):
    return np.full((grid, grid), float(c1)), np.full((grid, grid), float(c2))


def gradient_map(c1_lo, c1_hi, c2_lo, c2_hi, direction="h", grid=GRID):
    """Spatially varying target: ramps c1 and/or c2 across the image."""
    y, x = np.mgrid[0:grid, 0:grid] / max(grid - 1, 1)
    t = {"h": x, "v": y, "d": (x + y) / 2}.get(
        direction, np.clip(np.hypot(x - .5, y - .5) / .707, 0, 1))
    return c1_lo + (c1_hi - c1_lo) * t, c2_lo + (c2_hi - c2_lo) * t


def shape_map(c1, c2_bg, c2_shape, shape_mask, grid=GRID):
    """Hidden shape: same c1 everywhere, a different c2 inside the shape."""
    m = np.asarray(Image.fromarray((np.asarray(shape_mask, float) * 255).astype(np.uint8))
                   .resize((grid, grid), Image.BILINEAR), float) / 255.0
    return np.full((grid, grid), float(c1)), c2_bg + (c2_shape - c2_bg) * m


if __name__ == "__main__":   # quick self-check on one corpus image
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_corpus import load_field, measure_c2_field
    import pandas as pd
    d = pd.read_csv(Path(__file__).resolve().parent / "corpus" / "manifest.csv").sample(1, random_state=0)
    p = d.iloc[0]["path"]
    f = load_field(p)
    c1m, c2m = local_cumulant_map(f)
    g1, g2 = measure_c2_field(f)
    print(f"{p}\n  global  (c1={g1:+.3f}, c2={g2:+.3f})")
    print(f"  map mean(c1={c1m.mean():+.3f}, c2={c2m.mean():+.3f})  "
          f"map sd(c1={c1m.std():.3f}, c2={c2m.std():.3f})")
