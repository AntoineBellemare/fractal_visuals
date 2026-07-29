"""
mf_creative.py — two things prompts can't do, via spatially-varying MF-ControlNet fields:

  GRADIENTS    : a control field whose multifractality varies across space (calm->turbulent),
                 so one photoreal texture is smooth on one side and intermittent on the other.
  HIDDEN SHAPES: a letter/shape painted into the field as a region of DIFFERENT c2 inside an
                 otherwise-uniform texture -> the shape has no luminance edge, only a texture-
                 statistics anomaly (pareidolia bait / statistical steganography).

Same (c1,c2) substrate via prescribed_cascade, blended by a spatial mask, fed to the tile
ControlNet. Both fields share a seed so the transition/shape is a pure multifractality change,
not a pattern discontinuity. Outputs assembled into two montages.

Usage:  python mf_creative.py            # 10 gradients + 10 hidden shapes
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import mfractal as mf                                              # noqa: E402
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN  # noqa: E402

N = 1024
CN_SCALE = 0.55             # LOW so the SCENE (prompt) dominates and the field only modulates
GUIDANCE_END = 0.8          # release late steps -> photoreal scene detail returns
STEPS = 35


def hist_match(src, ref):
    """Remap src to ref's luminance histogram (same N) -> identical brightness distribution,
    so a blended region differs ONLY in texture statistics, not brightness (truly hidden)."""
    ranks = src.ravel().argsort().argsort()
    return np.sort(ref.ravel())[ranks].reshape(src.shape)


def cascade(c1, c2, seed):
    return np.asarray(mf.prescribed_cascade(n=N, seed=int(seed), c1_target=float(c1),
                                            c2_target=float(c2)), float)


def grad_mask(direction):
    y, x = np.mgrid[0:N, 0:N] / (N - 1)
    if direction == "h":      m = x
    elif direction == "v":    m = y
    elif direction == "d":    m = (x + y) / 2
    else:                     m = np.clip(np.hypot(x - 0.5, y - 0.5) / 0.707, 0, 1)  # radial
    return m


def gradient_field(c1, c2_lo, c2_hi, direction, seed):
    flo, fhi = cascade(c1, c2_lo, seed), cascade(c1, c2_hi, seed)
    m = grad_mask(direction)
    return flo * (1 - m) + fhi * m


def shape_mask(spec):
    img = Image.new("L", (N, N), 0); d = ImageDraw.Draw(img)
    k = spec["kind"]
    if k == "text":
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", int(N * spec.get("size", 0.6)))
        except Exception:
            font = ImageFont.truetype("arialbd.ttf", int(N * 0.6)) if False else ImageFont.load_default()
        t = spec["text"]; bb = d.textbbox((0, 0), t, font=font)
        d.text(((N - (bb[2] - bb[0])) / 2 - bb[0], (N - (bb[3] - bb[1])) / 2 - bb[1]), t, 255, font=font)
    elif k == "circle":
        r = int(N * 0.32); d.ellipse([N/2-r, N/2-r, N/2+r, N/2+r], fill=255)
    elif k == "triangle":
        d.polygon([(N/2, N*0.13), (N*0.13, N*0.87), (N*0.87, N*0.87)], fill=255)
    elif k == "cross":
        w = int(N * 0.15)
        d.rectangle([N/2-w, N*0.13, N/2+w, N*0.87], fill=255); d.rectangle([N*0.13, N/2-w, N*0.87, N/2+w], fill=255)
    elif k == "heart":
        r = N * 0.2
        d.ellipse([N/2-r*1.3, N*0.28, N/2+0.05*N, N*0.28+2*r], fill=255)
        d.ellipse([N/2-0.05*N, N*0.28, N/2+r*1.3, N*0.28+2*r], fill=255)
        d.polygon([(N*0.2, N*0.5), (N*0.8, N*0.5), (N/2, N*0.85)], fill=255)
    elif k == "face":                                              # simple smiley silhouette
        d.ellipse([N*0.18, N*0.18, N*0.82, N*0.82], fill=255)
    m = np.clip(gaussian_filter(np.array(img, float) / 255, sigma=N * 0.008), 0, 1)
    return m


def shape_field(c1, c2_bg, c2_sh, spec, seed):
    fbg = cascade(c1, c2_bg, seed)
    fsh = hist_match(cascade(c1, c2_sh, seed + 100), fbg)   # different texture, SAME luminance
    m = shape_mask(spec)
    return fbg * (1 - m) + fsh * m, m


GRADIENTS = [   # SCENES (prompt) + a spatial multifractality gradient (c1, c2_lo, c2_hi, dir, seed)
    ("dense green forest canopy seen from above",   1.3, -0.20, -0.9, "h", 0),
    ("autumn forest treetops, aerial view",         1.3, -0.20, -0.9, "v", 1),
    ("vibrant coral reef underwater",               1.3, -0.20, -0.85, "radial", 2),
    ("rugged rocky mountain landscape",             1.4, -0.15, -0.8, "h", 3),
    ("meadow full of wildflowers",                  1.2, -0.20, -0.85, "d", 4),
    ("aerial view of a branching river delta",      1.3, -0.20, -0.9, "h", 5),
    ("rippled desert sand dunes",                   1.3, -0.15, -0.8, "v", 6),
    ("dense crowd of people seen from above",       1.3, -0.20, -0.85, "h", 7),
    ("tangled tropical jungle vegetation",          1.3, -0.20, -0.9, "radial", 8),
    ("frost-covered pine forest, aerial",           1.3, -0.20, -0.85, "d", 9),
]
SHAPES = [      # SCENE + a hidden shape (c2 anomaly, luminance-matched) (c1, c2_bg, c2_sh, spec, seed)
    ("dense green forest canopy seen from above",   1.3, -0.35, -0.85, {"kind":"text","text":"A"}, 0),
    ("vibrant coral reef underwater",               1.3, -0.35, -0.85, {"kind":"circle"}, 1),
    ("autumn forest treetops, aerial view",         1.3, -0.35, -0.85, {"kind":"heart"}, 2),
    ("meadow full of wildflowers",                  1.2, -0.35, -0.80, {"kind":"face"}, 3),
    ("rugged rocky mountain terrain",               1.4, -0.30, -0.80, {"kind":"text","text":"X"}, 4),
    ("tangled tropical jungle vegetation",          1.3, -0.35, -0.90, {"kind":"text","text":"HI","size":0.42}, 5),
    ("rippled desert sand dunes",                   1.3, -0.30, -0.80, {"kind":"triangle"}, 6),
    ("aerial view of a branching river delta",      1.3, -0.35, -0.85, {"kind":"text","text":"O"}, 7),
    ("frost-covered pine forest, aerial",           1.3, -0.35, -0.85, {"kind":"cross"}, 8),
    ("dense crowd of people seen from above",       1.3, -0.35, -0.85, {"kind":"text","text":"8"}, 9),
]


def montage(tiles, labels, title, fp, masks=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = 2, 5
    fig, ax = plt.subplots(nr, nc, figsize=(nc*2.7, nr*2.9))
    ax = ax.ravel()
    for i, (img, lab) in enumerate(zip(tiles, labels)):
        ax[i].imshow(img.resize((340, 340))); ax[i].set_xticks([]); ax[i].set_yticks([])
        ax[i].set_title(lab, fontsize=8, pad=2)
        if masks is not None:
            ax[i].contour(np.array(Image.fromarray((masks[i]*255).astype(np.uint8)).resize((340,340)))/255,
                          levels=[0.5], colors="#00e5ff", linewidths=1.1, alpha=0.85)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("->", fp, flush=True)


def main():
    out = HERE / "creative"; (out / "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)

    print("=== GRADIENTS ===", flush=True)
    g_tiles, g_labels = [], []
    for sub, c1, lo, hi, d, s in GRADIENTS:
        ctrl = field_to_control(gradient_field(c1, lo, hi, d, s))
        img = generate(pipe, sub, ctrl, CN_SCALE, guidance_end=GUIDANCE_END, steps=STEPS, seed=s, device=device)
        img.save(out / "img" / f"grad_{s}_{sub[:12].replace(' ','_')}.png")
        g_tiles.append(img); g_labels.append(f"{sub[:16]}\n{d}: calm→turbulent")
        print(f"  grad {sub[:16]:16s} {d}", flush=True)
    montage(g_tiles, g_labels, "Complexity gradients in one image (multifractality varies across space)",
            out / "gradients.png")

    print("=== HIDDEN SHAPES ===", flush=True)
    s_tiles, s_labels, s_masks = [], [], []
    for sub, c1, bg, sh, spec, s in SHAPES:
        field, mask = shape_field(c1, bg, sh, spec, s)
        ctrl = field_to_control(field)
        img = generate(pipe, sub, ctrl, CN_SCALE, guidance_end=GUIDANCE_END, steps=STEPS, seed=s, device=device)
        img.save(out / "img" / f"shape_{s}_{sub[:12].replace(' ','_')}.png")
        s_tiles.append(img); s_masks.append(mask)
        tag = spec.get("text", spec["kind"])
        s_labels.append(f"{sub[:16]}\nhidden: '{tag}'")
        print(f"  shape {sub[:16]:16s} {tag}", flush=True)
    montage(s_tiles, s_labels, "Hidden multifractal shapes (a c2 anomaly, no luminance edge — outline overlaid)",
            out / "hidden_shapes.png", masks=s_masks)
    print("done ->", out)


if __name__ == "__main__":
    main()
