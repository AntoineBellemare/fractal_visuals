"""
procedural_concept_demo.py — the visual analog of what classifier-guided diffusion
should produce.

Uses mfractal.prescribed_cascade which has a built-in (c1, c2) targeting
mechanism. For each "prompt" (here, a fixed seed standing in for prompt+seed
in the diffusion case), we sweep c2_target across the supported range and
generate one image per cell. c1_target is held fixed so c2 is the only knob.

The reader sees:
  - down the rows: different seeds (different "compositions")
  - across the columns: smoothly varying c2 at fixed composition
  - the seed anchors large-scale layout; the c2 knob shapes texture intermittency

Outputs:
  benchmarks/regressor/procedural_concept_demo.png
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mfractal as mf

SEEDS = [3, 17, 42, 91, 137]                          # rows -- 5 "compositions"
C2_TARGETS = [-0.03, -0.10, -0.20, -0.35, -0.55]      # cols -- smooth -> strong
C1_FIXED = 1.30                                       # hold c1 constant
N = 384


def main():
    fig, axes = plt.subplots(len(SEEDS), len(C2_TARGETS),
                             figsize=(2.5 * len(C2_TARGETS), 2.5 * len(SEEDS)),
                             squeeze=False)
    t0 = time.time()
    for ri, seed in enumerate(SEEDS):
        for ci, c2t in enumerate(C2_TARGETS):
            img = mf.prescribed_cascade(n=N, seed=seed,
                                        c1_target=C1_FIXED, c2_target=c2t)
            disp = mf.punch(img)
            # Measure actual c2 to annotate truthfulness of the targeting.
            try:
                wl = mf.wavelet_leaders_2d(img)
                c2m = float(wl["c2"])
            except Exception:
                c2m = float("nan")
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"target c2 = {c2t:+.2f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"seed = {seed}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
            ax.text(0.97, 0.04, f"measured\nc2 = {c2m:+.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2,
                              edgecolor="none"))
        print(f"  row {ri+1}/{len(SEEDS)}  seed={seed}  t={time.time()-t0:5.0f}s")

    fig.suptitle(
        "Procedural concept demo — same seed × c2 sweep "
        f"(c1_target = {C1_FIXED}, prescribed_cascade)\n"
        "Down each row: the seed anchors the composition. "
        "Across each row: c2_target shapes texture intermittency.",
        fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "regressor" / "procedural_concept_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
