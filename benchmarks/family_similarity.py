"""
Pairwise family similarity matrix from ACTUAL PIXELS at matched seed+cx.

For each family generate 3 cx values x 3 seeds = 9 images at n=256.
For each (cx, seed) cell, compute the 36 x 36 Pearson correlation across
all families (= cosine similarity of zero-mean unit-norm flattened images).
The "main" matrix is the average of the 9 per-cell matrices.

This is a much sharper redundancy probe than feature-vector cosine: two
families only land near +1 if their actual pixels co-vary at the same seed.
Different generative paradigms with similar multifractal stats can score 0.

Families are ordered by domain (rock, cloud, fluid, bark, fire) so
within-domain blocks read along the diagonal.

Outputs:
  benchmarks/figures/33_similarity_matrix.png    main 36x36 heatmap
  benchmarks/figures/34_similarity_per_cx.png    one matrix per cx
  benchmarks/results/similarity_redundancy.csv   pairs above threshold
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

OUT_FIG_MAIN = ROOT / "benchmarks" / "figures" / "33_similarity_matrix.png"
OUT_FIG_PERCX = ROOT / "benchmarks" / "figures" / "34_similarity_per_cx.png"
OUT_TBL = ROOT / "benchmarks" / "results" / "similarity_redundancy.csv"

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
    "metal": ["rust_bloom","rust_pitted","rust_dewy"],
}
FAM2DOM = {f: d for d, fs in DOMAIN.items() for f in fs}
DOMAIN_COLORS = {"rock":"#8c564b","cloud":"#1f77b4","fluid":"#17becf",
                 "bark":"#7f5d2f","fire":"#d62728","metal":"#a14b1f"}
ALL_FAMS = [f for d in DOMAIN for f in DOMAIN[d]]

CXS = [0.25, 0.5, 0.75]
SEEDS = [0, 1, 2]
N = 256
REDUNDANCY_THRESHOLD = 0.70


def _standardize(img):
    v = img.astype(np.float32).ravel()
    v = v - v.mean()
    nrm = np.linalg.norm(v)
    if nrm < 1e-9:
        return np.zeros_like(v)
    return v / nrm


def generate_all():
    """Returns dict[(cx, seed)] -> (n_fams, H*W) row-standardized matrix."""
    out = {}
    t0 = time.time()
    for ci, cx in enumerate(CXS):
        for si, s in enumerate(SEEDS):
            rows = np.zeros((len(ALL_FAMS), N * N), dtype=np.float32)
            for fi, fam in enumerate(ALL_FAMS):
                try:
                    img = mf.generate(fam, n=N, complexity=cx, seed=s)
                    rows[fi] = _standardize(img)
                except Exception as e:
                    print(f"    {fam} cx={cx} s={s} failed: {e}")
            out[(cx, s)] = rows
            done = ci * len(SEEDS) + si + 1
            tot = len(CXS) * len(SEEDS)
            print(f"  [{done}/{tot}] cx={cx} seed={s}  t={time.time()-t0:5.0f}s")
    return out


def per_cell_similarity(mats):
    """Returns dict[(cx, seed)] -> (n_fams, n_fams) cosine similarity matrix."""
    out = {}
    for key, M in mats.items():
        out[key] = M @ M.T   # row-standardized -> Pearson r
    return out


def per_cx_average(cell_sims):
    """Average per-seed matrices within each cx."""
    out = {}
    for cx in CXS:
        S = np.mean([cell_sims[(cx, s)] for s in SEEDS], axis=0)
        out[cx] = S
    return out


def grand_average(cell_sims):
    return np.mean(list(cell_sims.values()), axis=0)


def _plot_heatmap(ax, S, title, annotate_blocks=True, ticklabel_size=7):
    n = len(ALL_FAMS)
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(ALL_FAMS, rotation=90, fontsize=ticklabel_size)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ALL_FAMS, fontsize=ticklabel_size)
    for t, fam in zip(ax.get_xticklabels(), ALL_FAMS):
        t.set_color(DOMAIN_COLORS[FAM2DOM[fam]])
    for t, fam in zip(ax.get_yticklabels(), ALL_FAMS):
        t.set_color(DOMAIN_COLORS[FAM2DOM[fam]])
    if annotate_blocks:
        cursor = 0
        for dom, fams in DOMAIN.items():
            ax.add_patch(plt.Rectangle(
                (cursor - 0.5, cursor - 0.5), len(fams), len(fams),
                fill=False, edgecolor="black", linewidth=1.8))
            cursor += len(fams)
    ax.set_title(title, fontsize=10)
    return im


def plot_main(S):
    fig, ax = plt.subplots(figsize=(14, 13))
    im = _plot_heatmap(
        ax, S,
        "All families — pixel cosine similarity at matched (cx, seed)  "
        "(avg over 3 cx x 3 seeds, n=256)",
        annotate_blocks=True, ticklabel_size=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="pixel correlation")
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.6)
               for c in DOMAIN_COLORS.values()]
    ax.legend(handles, list(DOMAIN_COLORS.keys()),
              loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=10)
    fig.tight_layout()
    OUT_FIG_MAIN.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_MAIN, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG_MAIN.relative_to(ROOT)}")


def plot_per_cx(per_cx):
    fig, axes = plt.subplots(1, len(CXS), figsize=(7 * len(CXS), 7))
    for ax, cx in zip(axes, CXS):
        _plot_heatmap(ax, per_cx[cx], f"cx = {cx:.2f}",
                      annotate_blocks=True, ticklabel_size=6)
    fig.suptitle("Per-cx pixel similarity matrices  "
                 "(avg over 3 seeds; pixel correlation at matched seed)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_FIG_PERCX, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG_PERCX.relative_to(ROOT)}")


def redundancy_table(S, threshold=REDUNDANCY_THRESHOLD):
    n = len(ALL_FAMS)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                pairs.append(dict(
                    family_a=ALL_FAMS[i], domain_a=FAM2DOM[ALL_FAMS[i]],
                    family_b=ALL_FAMS[j], domain_b=FAM2DOM[ALL_FAMS[j]],
                    similarity=round(float(S[i, j]), 3),
                    same_domain=FAM2DOM[ALL_FAMS[i]] == FAM2DOM[ALL_FAMS[j]],
                ))
    pairs.sort(key=lambda p: -p["similarity"])
    df_pairs = pd.DataFrame(pairs)
    OUT_TBL.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(OUT_TBL, index=False)
    return df_pairs


if __name__ == "__main__":
    print(f"Generating {len(ALL_FAMS)} families x {len(CXS)} cx x {len(SEEDS)} seeds at n={N}...")
    mats = generate_all()
    cell_sims = per_cell_similarity(mats)
    S = grand_average(cell_sims)
    per_cx = per_cx_average(cell_sims)
    plot_main(S)
    plot_per_cx(per_cx)
    pairs = redundancy_table(S, threshold=REDUNDANCY_THRESHOLD)
    same = pairs[pairs.same_domain]
    cross = pairs[~pairs.same_domain]
    print(f"\n{len(pairs)} pairs above similarity {REDUNDANCY_THRESHOLD}:")
    print(f"  within-domain (redundancy candidates):  {len(same)}")
    print(f"  cross-domain (looks like wrong domain): {len(cross)}")
    print(f"\nTop 25 within-domain pairs (redundancy candidates):")
    if len(same):
        print(same.head(25).to_string(index=False))
    print(f"\nTop 15 cross-domain pairs (taxonomy challenges):")
    if len(cross):
        print(cross.head(15).to_string(index=False))
