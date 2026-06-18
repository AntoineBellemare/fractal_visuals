"""
Re-render after the post-feedback tweaks:
  - Attractor seed sampling widened (now spans the (a,b,c,d) parameter
    geometry, with a quality filter that rejects degenerate orbits).
  - wave_interference grew a `crispness` knob.

Outputs:
  benchmarks/figures/30_new_bases_cx_sweep.png        (regenerated)
  benchmarks/figures/31_new_bases_seed_sweep.png      (regenerated, now
                                                       with wider attractor
                                                       seed variability)
  benchmarks/figures/32_wave_crispness_demo.png       crispness sweep at
                                                       cx=0.3, 0.6, 0.9
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

N = 320
CXS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]
FAMS = ["de_jong_attractor", "wave_interference"]


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
    fig.suptitle("New base processes — seed sweep at cx=0.5  "
                 "(attractors now span the full param geometry)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "31_new_bases_seed_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def crispness_demo():
    """Three rows: cx=0.3, 0.6, 0.9; six columns: crispness 0..1."""
    cxs = [0.3, 0.6, 0.9]
    crispnesses = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axes = plt.subplots(len(cxs), len(crispnesses),
                             figsize=(2.4 * len(crispnesses), 2.5 * len(cxs)),
                             squeeze=False)
    for ri, cx in enumerate(cxs):
        for ci, cr in enumerate(crispnesses):
            img = mf.wave_interference(n=N, complexity=cx, seed=0, crispness=cr)
            disp = mf.punch(img)
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"crispness = {cr:.1f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"cx = {cx:.1f}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("wave_interference — crispness knob "
                 "(tanh saturation on the level-curves)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "32_wave_crispness_demo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    t0 = time.time()
    cx_sweep()
    print(f"cx_sweep total: {time.time()-t0:.0f}s")
    t0 = time.time()
    seed_sweep()
    print(f"seed_sweep total: {time.time()-t0:.0f}s")
    crispness_demo()
