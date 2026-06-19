"""
Render every family at cx=0.5, seed=0 with its recommended palette via
colorize_for_family() and lay them out by domain.

Output: benchmarks/figures/12_color_montage.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

DOMAIN = {
    "rock":  ["marble","agate","weathered","granite","vesicular","turbulent",
              "schist","serpentinite","flow_banded","convoluted","gneiss"],
    "cloud": ["cascade_lognormal","billow","cirrus",
              "cloud_multifractional","warped_fbm","ridged","billow_smoke",
              "universal_cascade","prescribed_cascade","de_jong_attractor"],
    "fluid": ["curl_weave","eddies","vorticity","dye_diffusion",
              "rheoscopic","wave_interference"],
    "bark":  ["burled_oak","riven_oak"],
    "fire":  ["firestorm","lava_pool","volcanic_fissure"],
}

NCOL = 6
N = 384
SEED = 0


def render():
    ordered = []
    for d, fs in DOMAIN.items():
        for f in fs:
            ordered.append((d, f))
    nrow = (len(ordered) + NCOL - 1) // NCOL
    fig, axes = plt.subplots(nrow, NCOL, figsize=(2.4 * NCOL, 2.5 * nrow),
                             squeeze=False)
    for idx, (dom, fam) in enumerate(ordered):
        img = mf.generate(fam, n=N, complexity=0.5, seed=SEED)
        disp = mf.punch(img)
        rgb = mf.colorize_for_family(disp, fam, seed=SEED + 7, saturation=0.7,
                                     sharpness=1.6, warp=28)
        ax = axes[idx // NCOL][idx % NCOL]
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        palette = mf.FAMILY_PALETTES.get(fam, "?")
        ax.set_title(f"{fam}\n[{dom}] · {palette}", fontsize=8)
    for i in range(len(ordered), nrow * NCOL):
        axes[i // NCOL][i % NCOL].axis("off")
    fig.suptitle("32 families × recommended earthy palettes (cx=0.5, seed=0)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "12_color_montage.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def palette_swatches():
    """Render swatches of every palette so you can see the raw color codes."""
    names = list(mf.NATURAL_PALETTES.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 0.7 * len(names)))
    for ax, name in zip(axes, names):
        pal = np.asarray(mf.NATURAL_PALETTES[name])
        ax.imshow(pal[None, :, :], aspect="auto")
        ax.set_yticks([0]); ax.set_yticklabels([name], fontsize=10)
        ax.set_xticks([])
    fig.suptitle("NATURAL_PALETTES — anchor swatches", fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "13_palette_swatches.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    render()
    palette_swatches()
