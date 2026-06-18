"""
Sweep all 32 families x 5 seeds at cx=0.5, n=512, compute the 12-element
feature_vector, and build plots of the new (post-c2) descriptors.

Outputs:
  benchmarks/results/descriptors_manifest.csv      raw per-stim measurements
  benchmarks/figures/09_new_descriptors_per_family.png   violin grid
  benchmarks/figures/10_descriptor_correlations.png      |corr| heatmap
  benchmarks/figures/11_descriptors_per_domain.png       per-domain means
"""
from __future__ import annotations
import csv
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

OUT_CSV = ROOT / "benchmarks" / "results" / "descriptors_manifest.csv"
FIG_DIR = ROOT / "benchmarks" / "figures"

DOMAIN = {
    "rock":  ["marble","agate","weathered","granite","vesicular","turbulent",
              "schist","serpentinite","flow_banded","convoluted","gneiss"],
    "cloud": ["cascade_lognormal","stratified","billow","cirrus","cloud_mrw",
              "cloud_multifractional","warped_fbm","ridged"],
    "fluid": ["curl_weave","choppy","eddies","vorticity","dye_diffusion","rheoscopic"],
    "bark":  ["burled_oak","riven_oak"],
    "smoke": ["chimney_plume","billow_smoke"],
    "fire":  ["firestorm","lava_pool","volcanic_fissure"],
}
FAM2DOM = {f: d for d, fs in DOMAIN.items() for f in fs}
DOMAIN_COLORS = {"rock":"#8c564b","cloud":"#1f77b4","fluid":"#17becf",
                 "bark":"#7f5d2f","smoke":"#7f7f7f","fire":"#d62728"}
ALL_FAMS = [f for d in DOMAIN for f in DOMAIN[d]]

# The new descriptors (skip the c1/c2/delta_alpha/alpha0 that have prior figures).
NEW_METRICS = [
    "anisotropy", "subband_asymmetry",
    "coherence_length", "integral_length",
    "fd_std", "fd_range", "fd_entropy",
]
ALL_METRICS = [
    "c1","c2","delta_alpha","alpha0","asymmetry",
    "anisotropy","subband_asymmetry",
    "coherence_length","integral_length",
    "fd_std","fd_range","fd_entropy",
]

PRETTY = {
    "c1":"c1 (Hurst-like)", "c2":"c2 (intermittency)",
    "delta_alpha":"Δα (spectrum width)", "alpha0":"α₀ (spectrum mode)",
    "asymmetry":"f(α) asymmetry",
    "anisotropy":"anisotropy (directional Hurst)",
    "subband_asymmetry":"wavelet subband asymmetry",
    "coherence_length":"coherence length (px)",
    "integral_length":"integral length (px)",
    "fd_std":"local-FD std (heterogeneity)",
    "fd_range":"local-FD range",
    "fd_entropy":"local-FD entropy (bits)",
}


def sweep(n=512, seeds=range(5), cx=0.5):
    rows = []
    started = time.time()
    for fam_idx, fam in enumerate(ALL_FAMS, 1):
        for s in seeds:
            t0 = time.time()
            try:
                img = mf.generate(fam, n=n, complexity=cx, seed=int(s))
            except Exception as e:
                print(f"  [{fam}] gen-fail seed={s}: {e}")
                continue
            try:
                fv = mf.feature_vector(img)
            except Exception as e:
                print(f"  [{fam}] feat-fail seed={s}: {e}")
                continue
            row = dict(family=fam, domain=FAM2DOM[fam], seed=int(s), n=n, cx=cx,
                       gen_s=time.time() - t0)
            row.update(fv)
            rows.append(row)
        print(f"  [{fam_idx:2d}/{len(ALL_FAMS)}] {fam:24s}  "
              f"t={time.time()-started:5.0f}s")
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}  ({len(df)} rows)")
    return df


def violin_grid(df: pd.DataFrame, out: Path):
    """7-panel violin grid -- one panel per new metric, violin per family."""
    fams = ALL_FAMS
    nm = len(NEW_METRICS)
    fig, axes = plt.subplots(nm, 1, figsize=(16, 2.6 * nm), sharex=True)
    positions = list(range(len(fams)))
    for ax, metric in zip(axes, NEW_METRICS):
        data = [df.loc[df.family == f, metric].dropna().values for f in fams]
        parts = ax.violinplot(data, positions=positions, showmeans=True,
                              showextrema=False, widths=0.85)
        for i, f in enumerate(fams):
            c = DOMAIN_COLORS[FAM2DOM[f]]
            parts['bodies'][i].set_facecolor(c)
            parts['bodies'][i].set_alpha(0.55)
            parts['bodies'][i].set_edgecolor("k")
        parts['cmeans'].set_color("k")
        ax.set_ylabel(PRETTY[metric], fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(fams, rotation=70, ha="right", fontsize=8)
    # domain legend on top panel
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55)
               for c in DOMAIN_COLORS.values()]
    axes[0].legend(handles, list(DOMAIN_COLORS.keys()),
                   loc="upper right", fontsize=9, ncol=3)
    fig.suptitle("New descriptors: per-family violins  (32 families × 5 seeds, cx=0.5, n=512)",
                 fontsize=13, y=0.999)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def corr_heatmap(df: pd.DataFrame, out: Path):
    """Pearson correlation matrix on the 12 descriptors (family-level means)."""
    fam_means = df.groupby("family")[ALL_METRICS].mean()
    C = fam_means.corr().values
    n = len(ALL_METRICS)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ALL_METRICS, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(ALL_METRICS, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i,j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(C[i,j]) > 0.5 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Cross-descriptor Pearson r (family-level means, n=32 families)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def per_domain_means(df: pd.DataFrame, out: Path):
    """Bar chart per domain: domain-level mean of each new descriptor."""
    nm = len(NEW_METRICS)
    fig, axes = plt.subplots(1, nm, figsize=(3.0 * nm, 5))
    doms = list(DOMAIN)
    for ax, metric in zip(axes, NEW_METRICS):
        means = [df.loc[df.domain == d, metric].mean() for d in doms]
        stds = [df.loc[df.domain == d, metric].std() for d in doms]
        colors = [DOMAIN_COLORS[d] for d in doms]
        ax.bar(doms, means, yerr=stds, color=colors, edgecolor="k", alpha=0.85,
               capsize=3)
        ax.set_title(PRETTY[metric], fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("New descriptors: domain-level means (procedural, cx=0.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    if OUT_CSV.exists():
        print(f"Loading existing manifest from {OUT_CSV.relative_to(ROOT)}")
        df = pd.read_csv(OUT_CSV)
    else:
        df = sweep()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    violin_grid(df, FIG_DIR / "09_new_descriptors_per_family.png")
    corr_heatmap(df, FIG_DIR / "10_descriptor_correlations.png")
    per_domain_means(df, FIG_DIR / "11_descriptors_per_domain.png")
