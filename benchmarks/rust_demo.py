"""
rust_demo.py -- quick visual catalog of the metal/rust families.

Outputs:
  benchmarks/figures/35_rust_cx_sweep.png       cx sweep for each family
  benchmarks/figures/36_rust_seed_sweep.png     seed sweep at cx=0.5
  benchmarks/figures/37_rust_color_montage.png  colorized via FAMILY_PALETTES
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

FIG = ROOT / "benchmarks" / "figures"
FAMS = list(mf.METAL_FAMILIES)
CXS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]
N = 384


def cx_sweep():
    fig, axes = plt.subplots(len(FAMS), len(CXS),
                             figsize=(2.5 * len(CXS), 2.5 * len(FAMS)),
                             squeeze=False)
    for ri, fam in enumerate(FAMS):
        for ci, cx in enumerate(CXS):
            img = mf.generate(fam, n=N, complexity=cx, seed=0)
            ax = axes[ri][ci]
            ax.imshow(mf.punch(img), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"cx = {cx:.2f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("Rust families — cx sweep (seed=0)", fontsize=12, y=1.005)
    fig.tight_layout()
    out = FIG / "35_rust_cx_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def seed_sweep():
    fig, axes = plt.subplots(len(FAMS), len(SEEDS),
                             figsize=(2.5 * len(SEEDS), 2.5 * len(FAMS)),
                             squeeze=False)
    for ri, fam in enumerate(FAMS):
        for ci, s in enumerate(SEEDS):
            img = mf.generate(fam, n=N, complexity=0.5, seed=s)
            ax = axes[ri][ci]
            ax.imshow(mf.punch(img), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"seed = {s}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("Rust families — seed sweep (cx=0.5)", fontsize=12, y=1.005)
    fig.tight_layout()
    out = FIG / "36_rust_seed_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def color_montage():
    fig, axes = plt.subplots(len(FAMS), len(CXS),
                             figsize=(2.5 * len(CXS), 2.5 * len(FAMS)),
                             squeeze=False)
    for ri, fam in enumerate(FAMS):
        for ci, cx in enumerate(CXS):
            img = mf.generate(fam, n=N, complexity=cx, seed=0)
            disp = mf.punch(img)
            rgb = mf.colorize_for_family(disp, fam)
            ax = axes[ri][ci]
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"cx = {cx:.2f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("Rust families — cx sweep, colorized via FAMILY_PALETTES "
                 "(rust_oxide / patina_drip / corrosion_grit)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = FIG / "37_rust_color_montage.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cx_sweep()
    seed_sweep()
    color_montage()
