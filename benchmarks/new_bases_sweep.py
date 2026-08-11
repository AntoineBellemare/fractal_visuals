"""
Sweeps for the three new base processes added after Bounzaï's "more bases of
different kinds" request:

  - aggregation:     dla_dendrite           (walker DLA)
  - attractors:      clifford_attractor,    (chaotic 2D maps)
                     de_jong_attractor
  - interference:    wave_interference      (sum of random plane waves)

Outputs:
  benchmarks/figures/30_new_bases_cx_sweep.png    cx sweep, 4 families
  benchmarks/figures/31_new_bases_seed_sweep.png  5 seeds at cx=0.5
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

# DLA is the slow one; keep n moderate and seeds count modest for it
N = 320
CXS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]
FAMS = ["dla_dendrite", "clifford_attractor", "de_jong_attractor",
        "wave_interference"]


def cx_sweep():
    fig, axes = plt.subplots(len(FAMS), len(CXS),
                             figsize=(2.4 * len(CXS), 2.5 * len(FAMS)),
                             squeeze=False)
    t0 = time.time()
    for ri, fam in enumerate(FAMS):
        for ci, cx in enumerate(CXS):
            tt = time.time()
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
            print(f"  cx_sweep [{fam}, cx={cx:.2f}]  dt={time.time()-tt:.1f}s  "
                  f"total={time.time()-t0:.0f}s")
    fig.suptitle(f"New base processes — cx sweep  (n={N}, seed=0)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "30_new_bases_cx_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def seed_sweep():
    fig, axes = plt.subplots(len(FAMS), len(SEEDS),
                             figsize=(2.4 * len(SEEDS), 2.5 * len(FAMS)),
                             squeeze=False)
    t0 = time.time()
    for ri, fam in enumerate(FAMS):
        for ci, s in enumerate(SEEDS):
            tt = time.time()
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
            print(f"  seed_sweep [{fam}, s={s}]  dt={time.time()-tt:.1f}s  "
                  f"total={time.time()-t0:.0f}s")
    fig.suptitle("New base processes — seed sweep at cx=0.5",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "31_new_bases_seed_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cx_sweep()
    seed_sweep()
