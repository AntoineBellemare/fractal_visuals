"""
Benchmark figures from the validation manifest.

Produces:
  benchmarks/_out/figures/
    01_family_catalog_<domain>.png       — visual catalog per domain (one row per family, columns per complexity, seed=0)
    02_c2_vs_complexity_all.png          — c2 mean±sd vs complexity for every family (small multiples)
    03_c2_vs_complexity_curated.png      — same, curated subset only
    04_cross_family_summary.png          — c2 vs Delta_alpha scatter, color by domain, monofractal anchors highlighted
    05_complexity_ranks.png              — Spearman rho(c2, complexity) bar chart, ordered, with controls marked
    06_curated_panel.png                 — single-page visual panel of the curated set at cx=0.5, seed=0

Reads benchmarks/_out/validation_manifest.csv. Writes PNGs at 150 dpi.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

RESULTS = ROOT / "benchmarks" / "results"
FIG = ROOT / "benchmarks" / "figures"
MANIFEST = RESULTS / "validation_manifest.csv"


# Curated subsets — chosen from the validation sweep based on:
#   (a) clean monotonic |rho(c2, complexity)| or designated control,
#   (b) no estimator artifacts at extremes,
#   (c) visually compelling for pareidolia stimuli.
CURATED = {
    "rock": [
        "marble", "agate", "weathered", "granite", "vesicular", "turbulent",
        "schist", "serpentinite", "flow_banded", "convoluted",
        "gneiss",  # monofractal control
    ],
    "cloud": [
        "cascade_lognormal", "stratified", "billow", "cirrus",
        "cloud_mrw", "cloud_multifractional",
        "warped_fbm",  # monofractal control
        "ridged",      # structural-detail axis; bin by measured c2 (artifact at smooth low-cx)
    ],
    "fluid": [
        "curl_weave", "choppy",        # complexity-controlled multifractal
        "eddies", "vorticity",         # NS sims, moderate multifractal
        "dye_diffusion",               # ink tendrils; degenerate at low cx, bin by measured c2
        "rheoscopic",                  # near-monofractal flow-viz silk
    ],
}
CONTROLS = {"gneiss", "warped_fbm"}  # explicit monofractal anchors

DROPPED = {
    "plume": "estimator artifact on sparse bright structure (c2 ~ -8)",
    "cellular_stone": "inverted rho (more cells = more uniform fragmentation)",
    "concentric": "inverted rho (more rings = more uniform spacing)",
    "breccia": "inverted rho (more fragments = more uniform tiling)",
    "veined": "high-variance per seed, weak monotonicity",
}


def takes_cx(family):
    if family in mf.ROCK_FAMILIES: return True
    if family in mf.CLOUD_FAMILIES: return True
    if family in mf.CX_FLUIDS: return True
    return False


def fig_family_catalog(df, domain, families, complexities, n=256, seed=0):
    F = len(families); C = len(complexities)
    fig, axes = plt.subplots(F, C, figsize=(2.0 * C, 2.0 * F), squeeze=False)
    for i, fam in enumerate(families):
        for j, cx in enumerate(complexities):
            ax = axes[i][j]
            try:
                img = (mf.generate(fam, n=n, complexity=float(cx), seed=seed)
                       if takes_cx(fam) else mf.generate(fam, n=n, seed=seed))
                disp = mf.punch(img)
            except Exception as e:
                ax.text(0.5, 0.5, str(e)[:30], ha="center", va="center", fontsize=6)
                ax.set_axis_off()
                continue
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"cx={cx:.2f}" if takes_cx(fam) else "(NS sim)", fontsize=8)
            if j == 0:
                label = fam + (" (ctl)" if fam in CONTROLS else "")
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=20)
    fig.suptitle(f"family catalog — {domain} (seed={seed}, n={n}, punch display)", y=1.0, fontsize=10)
    fig.tight_layout()
    out = FIG / f"01_family_catalog_{domain}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig_c2_vs_complexity(df, families, out_name, title):
    F = len(families)
    cols = 4
    rows = (F + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for k, fam in enumerate(families):
        ax = axes[k // cols][k % cols]
        sub = df[(df.family == fam) & (df.status == "ok")].dropna(subset=["c2"])
        if sub.complexity.notna().any() and len(sub):
            g = sub.groupby("complexity")["c2"]
            cx = np.array(sorted(sub.complexity.dropna().unique()))
            mu = g.mean().reindex(cx).values
            sd = g.std().reindex(cx).values
            ax.errorbar(cx, mu, yerr=sd, fmt="o-", color="C0", lw=1.5, ms=4, capsize=2)
            rho, _ = spearmanr(sub.complexity, sub.c2)
            ax.set_title(f"{fam}  rho={rho:+.2f}", fontsize=9)
        elif len(sub):
            ax.bar(range(len(sub)), sub.c2.values, color="C1")
            ax.set_title(f"{fam} (NS sim)", fontsize=9)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        if fam in CONTROLS:
            ax.set_facecolor("#fff5e6")
        ax.set_xlabel("complexity", fontsize=8); ax.set_ylabel("c2", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(F, rows * cols):
        axes[k // cols][k % cols].set_axis_off()
    fig.suptitle(title, y=1.005, fontsize=11)
    fig.tight_layout()
    out = FIG / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig_cross_family(df):
    df_ok = df[df.status == "ok"].dropna(subset=["c2", "delta_alpha"]).copy()
    agg = df_ok.groupby(["domain", "family"]).agg(c2=("c2", "mean"),
                                                  c2_sd=("c2", "std"),
                                                  da=("delta_alpha", "mean")).reset_index()
    color = {"rock": "#8c564b", "cloud": "#1f77b4", "fluid": "#17becf"}
    fig, ax = plt.subplots(figsize=(11, 7))
    for dom in ["rock", "cloud", "fluid"]:
        sub = agg[agg.domain == dom]
        ax.scatter(sub.da, sub.c2, s=60, color=color[dom], label=dom, alpha=0.85, edgecolor="k", lw=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r.family, (r.da, r.c2), fontsize=7, xytext=(3, 3), textcoords="offset points",
                        color=color[dom])
    # highlight controls
    ctl = agg[agg.family.isin(CONTROLS)]
    ax.scatter(ctl.da, ctl.c2, s=140, facecolors="none", edgecolors="orange", linewidths=2.0,
               label="monofractal controls")
    drop = agg[agg.family.isin(DROPPED.keys())]
    ax.scatter(drop.da, drop.c2, s=140, facecolors="none", edgecolors="crimson", linewidths=2.0,
               label="dropped (artifacts / inverted)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("Delta_alpha (mean, 2D MFDFA)")
    ax.set_ylabel("c2 (mean, 2D wavelet leaders)")
    ax.set_title("Cross-family summary — multifractal descriptor space\n(c2≈0 = monofractal; lower = more intermittent)")
    ax.legend(loc="upper right", fontsize=9)
    # Clip the y-axis to keep the panel readable; flag outliers in the title
    extremes = agg[agg.c2 < -2]
    if len(extremes):
        ax.set_ylim(min(-1.2, agg.c2[agg.c2 > -2].min() - 0.1), 0.05)
        note = "(off-scale: " + ", ".join(f"{r.family} c2={r.c2:.1f}" for _, r in extremes.iterrows()) + ")"
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", fontsize=8, color="crimson")
    fig.tight_layout()
    out = FIG / "04_cross_family_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig_complexity_ranks(df):
    rows = []
    for (dom, fam), g in df.groupby(["domain", "family"]):
        g_ok = g[g.status == "ok"].dropna(subset=["c2"])
        if g_ok.complexity.notna().any() and len(g_ok):
            rho, _ = spearmanr(g_ok.complexity, g_ok.c2)
            rows.append(dict(domain=dom, family=fam, rho=rho))
    R = pd.DataFrame(rows).sort_values("rho")
    color_for = {"rock": "#8c564b", "cloud": "#1f77b4", "fluid": "#17becf"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ys = np.arange(len(R))
    bar_colors = [color_for[d] for d in R.domain]
    ax.barh(ys, R.rho, color=bar_colors, edgecolor="k", lw=0.4)
    for y, fam in zip(ys, R.family):
        suffix = ""
        if fam in CONTROLS: suffix = "  *control*"
        elif fam in DROPPED: suffix = "  *dropped*"
        ax.text(0.02 if R.iloc[y].rho < 0 else -0.02,
                y,
                fam + suffix,
                va="center",
                ha="left" if R.iloc[y].rho < 0 else "right",
                fontsize=8)
    ax.set_yticks([]); ax.axvline(0, color="k", lw=0.6)
    ax.axvline(-0.6, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Spearman rho(c2, complexity)  — more negative = cleaner monotonic multifractality control")
    ax.set_title("Complexity knob fidelity per family (lower is better; positive = inverted)")
    ax.set_xlim(-1.05, 1.05)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="k") for c in color_for.values()]
    ax.legend(handles, color_for.keys(), loc="lower right")
    fig.tight_layout()
    out = FIG / "05_complexity_ranks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig_curated_panel(n=256, seed=0):
    fams = []
    for d in ["rock", "cloud", "fluid"]:
        fams += [(d, f) for f in CURATED[d]]
    cols = 6
    rows = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False)
    for k, (dom, fam) in enumerate(fams):
        ax = axes[k // cols][k % cols]
        try:
            img = (mf.generate(fam, n=n, complexity=0.5, seed=seed)
                   if takes_cx(fam) else mf.generate(fam, n=n, seed=seed))
            disp = mf.punch(img)
        except Exception as e:
            ax.text(0.5, 0.5, str(e)[:30], ha="center", va="center", fontsize=6)
            ax.set_axis_off(); continue
        ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        tag = fam + ("  (control)" if fam in CONTROLS else "")
        ax.set_title(f"{dom}: {tag}", fontsize=8)
    for k in range(len(fams), rows * cols):
        axes[k // cols][k % cols].set_axis_off()
    fig.suptitle(f"curated subset ({sum(len(v) for v in CURATED.values())} families) — cx=0.5, seed={seed}, n={n}", y=1.005, fontsize=11)
    fig.tight_layout()
    out = FIG / "06_curated_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST)
    df["complexity"] = pd.to_numeric(df["complexity"], errors="coerce")

    # Catalog per domain
    print("Family catalogs:")
    cx_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for dom, label in [("rock", "rocks"), ("cloud", "clouds"), ("fluid", "fluids")]:
        fams = [f for f in (mf.ROCK_FAMILIES if dom == "rock" else
                            mf.CLOUD_FAMILIES if dom == "cloud" else mf.FLUID_FAMILIES)]
        # NS sims (no cx) get an empty-cx column treatment in the catalog
        fig_family_catalog(df, label, fams, cx_levels)

    print("c2 vs complexity (all):")
    all_fams = list(mf.ROCK_FAMILIES) + list(mf.CLOUD_FAMILIES) + list(mf.FLUID_FAMILIES)
    fig_c2_vs_complexity(df, all_fams, "02_c2_vs_complexity_all.png",
                         "c2 vs complexity — all 30 families (highlight = monofractal control)")

    print("c2 vs complexity (curated):")
    curated_flat = sum(CURATED.values(), [])
    fig_c2_vs_complexity(df, curated_flat, "03_c2_vs_complexity_curated.png",
                         "c2 vs complexity — curated 22-family subset")

    print("Cross-family summary:")
    fig_cross_family(df)

    print("Complexity-knob ranks:")
    fig_complexity_ranks(df)

    print("Curated panel:")
    fig_curated_panel()

    print(f"\nAll figures saved under {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
