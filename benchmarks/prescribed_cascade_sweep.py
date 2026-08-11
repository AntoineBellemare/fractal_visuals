"""
Demonstrate the prescribed_cascade family.

Two figures:
  31_prescribed_cx_sweep.png  -- default cx sweep, 3 seeds. Title labels show
                                 the (c1*, c2*) targets and the measured
                                 (c1, c2) below each cell.
  32_prescribed_c1_c2_grid.png -- direct (c1_target, c2_target) plane to show
                                  the spectrum-targeting interface.
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


def cx_sweep():
    cxs = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = [0, 1, 2]
    fig, axes = plt.subplots(len(seeds), len(cxs),
                             figsize=(2.6 * len(cxs), 2.7 * len(seeds)),
                             squeeze=False)
    for ci, cx in enumerate(cxs):
        c1t = 1.7 - 0.7 * cx
        c2t = -0.05 - 0.45 * cx
        for ri, s in enumerate(seeds):
            img = mf.prescribed_cascade(n=N, complexity=cx, seed=s)
            disp = mf.punch(img)
            r = mf.wavelet_leaders_2d(img)
            c1m, c2m = r["c1"], r["c2"]
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"cx={cx:.2f}\nc1*={c1t:+.2f}  c2*={c2t:+.2f}",
                             fontsize=9)
            ax.set_xlabel(f"measured: c1={c1m:+.2f}  c2={c2m:+.2f}",
                          fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"seed = {s}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle("prescribed_cascade — cx sweep with target vs measured (c1, c2)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "31_prescribed_cx_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def c1_c2_grid():
    c1_targets = [1.0, 1.2, 1.4]
    c2_targets = [-0.05, -0.15, -0.30, -0.45]
    fig, axes = plt.subplots(len(c1_targets), len(c2_targets),
                             figsize=(2.6 * len(c2_targets), 2.7 * len(c1_targets)),
                             squeeze=False)
    for ri, c1t in enumerate(c1_targets):
        for ci, c2t in enumerate(c2_targets):
            img = mf.prescribed_cascade(n=N, seed=0,
                                        c1_target=c1t, c2_target=c2t)
            disp = mf.punch(img)
            r = mf.wavelet_leaders_2d(img)
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"c2* = {c2t:+.2f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"c1* = {c1t:.1f}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
            ax.set_xlabel(f"meas: c1={r['c1']:+.2f}  c2={r['c2']:+.2f}",
                          fontsize=7)
    fig.suptitle("prescribed_cascade — (c1*, c2*) plane (seed=0)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "figures" / "32_prescribed_c1_c2_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cx_sweep()
    c1_c2_grid()
