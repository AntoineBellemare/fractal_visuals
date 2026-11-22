"""Demo / reproduction for the embedding routes in ``embed.py``.

Produces two figures (written to .attachments/percept_embedding/, gitignored):
  routes.png       -- S1 luminance vs P3 wavelet+c1-lock, face & tree, with c1/c2
  complexity.png   -- P3 @ beta=0.25, c1-locked, across 5 complexity levels,
                      showing embedded (c1,c2) follows the base coordinates.

Run:  python benchmarks/percept_embedding/demo.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import mfractal as mf                                                # noqa: E402
from mfractal import punch                                          # noqa: E402
from embed import (luminance_embed, wavelet_embed, lock_c1,         # noqa: E402
                   embed_percept, measure, _radial_k)
from percepts import face_mask, tree_image                          # noqa: E402

OUT = ROOT / ".attachments" / "percept_embedding"
OUT.mkdir(parents=True, exist_ok=True)
N = 512


def routes_figure():
    """S1 luminance vs P3 wavelet+lock, for face & tree."""
    F = mf.prescribed_cascade(n=N, seed=0, complexity=0.5)
    c1F, c2F = measure(F)
    percepts = [("FACE", face_mask(N)), ("TREE", tree_image(N, 0))]
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for ri, (name, P) in enumerate(percepts):
        s1 = luminance_embed(F, P, eps=0.25)
        p3 = embed_percept(F, P, beta=0.25)
        for ci, (lbl, img) in enumerate([
                ("base (no percept)", F), ("S1 luminance eps=.25", s1),
                ("P3 wavelet+lock beta=.25", p3)]):
            c1, c2 = measure(img)
            ax[ri][ci].imshow(punch(img), cmap="gray", vmin=0, vmax=1)
            ax[ri][ci].set_title(f"{lbl}\nc1={c1:.2f} c2={c2:+.3f}", fontsize=9)
            ax[ri][ci].set_xticks([]); ax[ri][ci].set_yticks([])
        ax[ri][0].set_ylabel(name, fontsize=12, rotation=0, ha="right",
                             va="center", labelpad=18)
    fig.suptitle(f"Confirmed routes — base c1={c1F:.2f} c2={c2F:+.3f}", fontsize=12)
    fig.tight_layout()
    fp = OUT / "routes.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {fp}")


def complexity_figure(betas=(0.25,), complexity=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """P3 @ fixed beta, c1-locked, across complexity levels."""
    k = _radial_k(N)
    beta = betas[0]
    percepts = [("FACE", face_mask(N)), ("TREE", tree_image(N, 0))]
    nrows = 1 + len(percepts)
    fig, ax = plt.subplots(nrows, len(complexity),
                           figsize=(3.2 * len(complexity), 3.25 * nrows))
    print(f"\nP3 @ beta={beta}, c1-locked; complexity {list(complexity)}")
    for ci, cx in enumerate(complexity):
        F = mf.prescribed_cascade(n=N, seed=0, complexity=cx)
        c1b, c2b = measure(F)
        ax[0][ci].imshow(punch(F), cmap="gray", vmin=0, vmax=1)
        ax[0][ci].set_title(f"cx={cx:g}\nbase c1={c1b:.2f} c2={c2b:+.3f}", fontsize=9)
        ax[0][ci].set_xticks([]); ax[0][ci].set_yticks([])
        print(f"cx={cx:g} base c1={c1b:.3f} c2={c2b:+.3f}")
        for ri, (name, P) in enumerate(percepts, start=1):
            G = wavelet_embed(F, P, beta)
            Gl, delta, c1e, c2e = lock_c1(G, c1b, k)
            ax[ri][ci].imshow(punch(Gl), cmap="gray", vmin=0, vmax=1)
            ax[ri][ci].set_title(f"{name}  c1={c1e:.2f} c2={c2e:+.3f}\n"
                                 f"base {c1b:.2f}/{c2b:+.2f} (δ={delta:+.2f})",
                                 fontsize=8)
            ax[ri][ci].set_xticks([]); ax[ri][ci].set_yticks([])
            print(f"   {name}: c1={c1e:.3f} c2={c2e:+.3f} (base {c1b:.3f}/{c2b:+.3f})")
    ax[0][0].set_ylabel("BASE", fontsize=11, rotation=0, ha="right",
                        va="center", labelpad=18)
    for ri, (name, _) in enumerate(percepts, start=1):
        ax[ri][0].set_ylabel(name, fontsize=11, rotation=0, ha="right",
                             va="center", labelpad=18)
    fig.suptitle(f"P3 wavelet embedding @ beta={beta}, c1-locked — across 5 "
                 "complexity levels", fontsize=12, y=1.0)
    fig.tight_layout()
    fp = OUT / "complexity.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {fp}")


if __name__ == "__main__":
    routes_figure()
    complexity_figure()
