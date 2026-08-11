"""
Modulations -- post-process operators that take a [0,1] field and produce
another [0,1] field. Orthogonal to the base process: any modulation can be
applied to any family output.

Currently implements:
  symmetrize -- mirror an image about an axis and blend with the original at
                variable strength. A cognitive-axis knob for pareidolic
                stimulus design (mirror symmetry is a strong driver of
                face-like and creature-like reads).
"""
from __future__ import annotations
import numpy as np

__all__ = ["symmetrize"]


def symmetrize(image, axis="vertical", strength=1.0):
    """Blend `image` with its mirror about `axis` at a controllable strength.

    Parameters
    ----------
    image : ndarray (H, W) or (H, W, C), float in [0, 1]
    axis : {"vertical", "horizontal", "both", "diagonal", "antidiagonal"}
        "vertical"     -- left-right mirror (mirror about a vertical midline)
        "horizontal"   -- top-bottom mirror
        "both"         -- average of vertical and horizontal mirrors
        "diagonal"     -- mirror about the main diagonal (requires square)
        "antidiagonal" -- mirror about the anti-diagonal (requires square)
    strength : float in [0, 1]
        0.0 returns the original; 1.0 returns the fully symmetrized field
        (a half-strength blend preserves more of the original texture).

    Returns
    -------
    ndarray of the same shape and dtype as `image`, clipped to [0, 1].
    """
    img = np.asarray(image)
    s = float(np.clip(strength, 0.0, 1.0))

    if axis == "vertical":
        mirrored = img[:, ::-1]
    elif axis == "horizontal":
        mirrored = img[::-1, :]
    elif axis == "both":
        mirrored = 0.5 * (img[:, ::-1] + img[::-1, :])
    elif axis == "diagonal":
        if img.shape[0] != img.shape[1]:
            raise ValueError("diagonal symmetry requires a square image")
        mirrored = np.swapaxes(img, 0, 1)
    elif axis == "antidiagonal":
        if img.shape[0] != img.shape[1]:
            raise ValueError("antidiagonal symmetry requires a square image")
        mirrored = np.swapaxes(img[::-1, ::-1], 0, 1)
    else:
        raise ValueError(f"unknown axis {axis!r}; expected one of "
                         "vertical/horizontal/both/diagonal/antidiagonal")

    out = (1.0 - s) * img + s * mirrored
    return np.clip(out, 0.0, 1.0)
