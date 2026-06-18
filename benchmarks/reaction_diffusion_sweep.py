"""
Quick-look figures for the reaction-diffusion families.

Outputs:
  benchmarks/figures/29_rd_cx_sweep.png    cx sweep, all 3 RD families
  benchmarks/figures/30_rd_seed_sweep.png  5 seeds at cx=0.5
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

N = 320
CXS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]
FAMS = ["rd_coral", "rd_labyrinth", "rd_solitons"]


def cx_sweep():
    fig, axes = plt.subplots(len(FAMS), len(CXS),
                             figsize=(2.4 * len(CXS), 2.5 * len(FAMS)),
                             squeeze=False)
    for ri, fam in enumerate(FAMS):
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
    fig.suptitle("Reaction-diffusion (Gray-Scott) — cx sweep "
                 f"(n={N}, seed=0; cx routes steps + seed density)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "29_rd_cx_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def seed_sweep():
    fig, axes = plt.subplots(len(FAMS), len(SEEDS),
                             figsize=(2.4 * len(SEEDS), 2.5 * len(FAMS)),
                             squeeze=False)
    for ri, fam in enumerate(FAMS):
        for ci, s in enumerate(SEEDS):
            img = mf.generate(fam, n=N, complexity=0.5, seed=s)
            disp = mf.punch(img)
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"seed = {s}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("Reaction-diffusion (Gray-Scott) — seed sweep at cx=0.5",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "30_rd_seed_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cx_sweep()
    seed_sweep()
