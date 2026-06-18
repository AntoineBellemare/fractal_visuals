"""
Quick-look figures for the two new toolbox pieces:
  - mfractal.levy: universal (Levy/Schertzer-Lovejoy) multifractal cascade
  - mfractal.symmetrize: symmetry-imposition modulation

Outputs:
  benchmarks/figures/26_levy_cx_sweep.png       cx sweep, both Levy families
  benchmarks/figures/27_levy_alpha_sigma_grid.png  (alpha, sigma) plane for
                                                  universal_cascade
  benchmarks/figures/28_symmetry_demo.png       symmetry modulation applied to
                                                  a handful of existing families
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

N = 384
CXS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def levy_cx_sweep():
    fams = ["universal_cascade", "levy_marble"]
    fig, axes = plt.subplots(len(fams), len(CXS),
                             figsize=(2.4 * len(CXS), 2.5 * len(fams)),
                             squeeze=False)
    for ri, fam in enumerate(fams):
        for ci, cx in enumerate(CXS):
            img = mf.generate(fam, n=N, complexity=cx, seed=0)
            disp = mf.punch(img)
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"cx = {cx:.2f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("Universal (Levy) multifractal cascade — cx sweep "
                 "(alpha 2.0->1.3, sigma 0.2->0.95, H 0.72->0.42)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "26_levy_cx_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def levy_alpha_sigma_grid():
    """Direct (alpha, sigma) plane: rows alpha=2.0, 1.7, 1.4, 1.1;
    cols sigma=0.2, 0.5, 0.8, 1.1. H=0.55 fixed."""
    alphas = [2.0, 1.7, 1.4, 1.1]
    sigmas = [0.2, 0.5, 0.8, 1.1]
    fig, axes = plt.subplots(len(alphas), len(sigmas),
                             figsize=(2.4 * len(sigmas), 2.5 * len(alphas)),
                             squeeze=False)
    for ri, a in enumerate(alphas):
        for ci, s in enumerate(sigmas):
            img = mf.universal_cascade(n=N, seed=0, alpha=a, sigma=s, H=0.55)
            disp = mf.punch(img)
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"sigma = {s:.1f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"alpha = {a:.1f}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("universal_cascade — (alpha, sigma) plane  (H=0.55, seed=0)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "27_levy_alpha_sigma_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def symmetry_demo():
    """Apply symmetrize at a few strengths to a sample of families."""
    fams = ["cascade_lognormal", "marble", "universal_cascade",
            "burled_oak", "firestorm", "billow_smoke"]
    strengths = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(len(fams), len(strengths) + 1,
                             figsize=(2.4 * (len(strengths) + 1), 2.5 * len(fams)),
                             squeeze=False)
    for ri, fam in enumerate(fams):
        img = mf.generate(fam, n=N, complexity=0.5, seed=0)
        disp = mf.punch(img)
        for ci, s in enumerate(strengths):
            sym = mf.symmetrize(disp, axis="vertical", strength=s)
            ax = axes[ri][ci]
            ax.imshow(sym, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"vert sym = {s:.1f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
        sym_both = mf.symmetrize(disp, axis="both", strength=1.0)
        ax = axes[ri][-1]
        ax.imshow(sym_both, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("vert+horiz sym", fontsize=10)
    fig.suptitle("Symmetry-imposition modulation (mf.symmetrize)  "
                 "(cx=0.5, seed=0)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "28_symmetry_demo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    levy_cx_sweep()
    levy_alpha_sigma_grid()
    symmetry_demo()
