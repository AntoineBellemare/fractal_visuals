"""
demo.py -- regenerate a showcase montage from the finalized flagships.
Run:  python -m mfractal.demo
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mfractal as mf

OUT = os.environ.get("MFRACTAL_OUT", "/home/claude/out")
os.makedirs(OUT, exist_ok=True)
N = 512


def main():
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.18, wspace=0.05)

    # row 0: flagship 1 (multifractional) -- varying FD heterogeneity
    for j, fr in enumerate([0.2, 0.7, 1.1, 1.5, 1.9]):
        ax = fig.add_subplot(gs[0, j])
        img = mf.multifractional(N, fd_center=2.5, fd_range=fr, seed=3)
        ax.imshow(mf.punch(img - img.mean(), strength=1.8), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"FD range {fr}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Flagship 1\nmultifractional", fontsize=10)

    # row 1: flagship 2 (multifractal cloud) -- multifractality span
    for j, sg in enumerate([0.3, 0.8, 1.2, 1.6, 2.2]):
        ax = fig.add_subplot(gs[1, j])
        f = mf.multifractal_cloud(N, multifractality=sg, seed=3, raw=True)
        ax.imshow(mf.punch(f), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"\u03c3={sg}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Flagship 2\nmultifractal cloud", fontsize=10)

    # row 2: a sampled battery (varied parameter combinations)
    bat = mf.sample_battery(5, n=N, seed0=7)
    for j, rec in enumerate(bat):
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(rec["image"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"battery {j+1}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("sample_battery()", fontsize=10)

    fig.suptitle("mfractal showcase  |  Flagship 1 (within-image FD) \u00b7 "
                 "Flagship 2 (multifractal cloud) \u00b7 sampled battery",
                 fontsize=14)
    path = f"{OUT}/fig_showcase.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
