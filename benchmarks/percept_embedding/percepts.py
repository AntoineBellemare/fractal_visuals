"""Procedural percept generators (the 'form' to embed).

Each returns an (n, n) float array in [0, 1] where higher = more percept
energy. Embedding routines (see ``embed.py``) only use the *spatial energy
layout* of these, so any grayscale image works as a percept; these are just
reproducible stand-ins for prototyping. Swap in a real B&W photo (face,
animal, figure) by loading it to an (n, n) [0,1] array.
"""
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter


def face_mask(n):
    """Soft frontal schematic face in [0,1] (1 = face region, eyes/mouth carved
    out so the *shape* reads as a face rather than a blob)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx, cy = n * 0.5, n * 0.52
    m = np.zeros((n, n))
    head = (((xx - cx) / (n * 0.26)) ** 2 + ((yy - cy) / (n * 0.34)) ** 2) < 1.0
    m[head] = 1.0

    def blob(px, py, rx, ry):
        return (((xx - px) / rx) ** 2 + ((yy - py) / ry) ** 2) < 1.0

    eyes = blob(cx - n * 0.11, cy - n * 0.07, n * 0.055, n * 0.035) | \
        blob(cx + n * 0.11, cy - n * 0.07, n * 0.055, n * 0.035)
    mouth = blob(cx, cy + n * 0.16, n * 0.12, n * 0.04)
    m[eyes | mouth] = 0.0
    return gaussian_filter(m, n * 0.012)


def tree_image(n, seed=0):
    """A procedural B&W tree (recursive branches + canopy) in [0,1]; 1 = tree."""
    rng = default_rng(seed)
    canvas = np.zeros((n, n))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    def stamp(x, y, r):
        if r < 0.6:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < n and 0 <= iy < n:
                canvas[iy, ix] = 1.0
            return
        sub = (((xx - x) ** 2 + (yy - y) ** 2) < r * r)
        canvas[sub] = 1.0

    def branch(x, y, ang, length, width, depth):
        steps = max(2, int(length / 3))
        for _ in range(steps):
            ang += rng.normal(0, 0.05)
            x += 3 * np.cos(ang); y += 3 * np.sin(ang)
            stamp(x, y, width)
        if depth <= 0:
            for _ in range(int(rng.uniform(3, 7))):
                lx = x + rng.normal(0, length * 0.35)
                ly = y + rng.normal(0, length * 0.35)
                stamp(lx, ly, rng.uniform(width * 1.5, width * 3.5))
            return
        nb = rng.integers(2, 4)
        for _ in range(nb):
            da = rng.uniform(0.25, 0.7) * rng.choice([-1, 1])
            na = ang + da
            na = na + 0.25 * (-np.pi / 2 - na) * 0.4
            branch(x, y, na, length * rng.uniform(0.62, 0.78),
                   width * 0.70, depth - 1)

    branch(n * 0.5, n * 0.96, -np.pi / 2, n * 0.22, n * 0.018, 6)
    return gaussian_filter(canvas, n * 0.004)
