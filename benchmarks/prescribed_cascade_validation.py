"""
Validation sweep for mf.prescribed_cascade.

Procedure
---------
Sweep a 4 x 5 grid of (c1*, c2*) targets, generate 10 seeds at each target
(n=384), measure (c1, c2) via wavelet leaders, and report:
  - target vs measured scatter (with mean +/- std error bars)
  - per-target residual distribution
  - overall RMSE for c1 and c2

Outputs
-------
  benchmarks/results/prescribed_validation.csv   raw per-sample measurements
  benchmarks/figures/29_prescribed_validation.png  scatter + residual panels
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

OUT_CSV = ROOT / "benchmarks" / "results" / "prescribed_validation.csv"
OUT_FIG = ROOT / "benchmarks" / "figures" / "29_prescribed_validation.png"

C1_TARGETS = [1.00, 1.20, 1.35, 1.50]
C2_TARGETS = [-0.05, -0.15, -0.25, -0.35, -0.50]
N = 384
SEEDS = list(range(10))


def sweep():
    rows = []
    t0 = time.time()
    total = len(C1_TARGETS) * len(C2_TARGETS) * len(SEEDS)
    idx = 0
    for c1t in C1_TARGETS:
        for c2t in C2_TARGETS:
            for s in SEEDS:
                idx += 1
                img = mf.prescribed_cascade(n=N, seed=s,
                                            c1_target=c1t, c2_target=c2t)
                r = mf.wavelet_leaders_2d(img)
                rows.append(dict(c1_target=c1t, c2_target=c2t, seed=s, n=N,
                                 c1_measured=r["c1"], c2_measured=r["c2"]))
            print(f"  [{idx:3d}/{total}] c1*={c1t:.2f} c2*={c2t:+.2f}  "
                  f"t={time.time()-t0:5.0f}s")
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}  ({len(df)} rows)")
    return df


def summarize(df):
    """Aggregate per-target means and RMSE."""
    agg = (df.groupby(["c1_target", "c2_target"])
             .agg(c1_mean=("c1_measured", "mean"),
                  c1_std=("c1_measured", "std"),
                  c2_mean=("c2_measured", "mean"),
                  c2_std=("c2_measured", "std"))
             .reset_index())
    c1_rmse = float(np.sqrt(((df["c1_measured"] - df["c1_target"]) ** 2).mean()))
    c2_rmse = float(np.sqrt(((df["c2_measured"] - df["c2_target"]) ** 2).mean()))
    c1_bias = float((df["c1_measured"] - df["c1_target"]).mean())
    c2_bias = float((df["c2_measured"] - df["c2_target"]).mean())
    return agg, dict(c1_rmse=c1_rmse, c1_bias=c1_bias,
                     c2_rmse=c2_rmse, c2_bias=c2_bias)


def make_figure(df, agg, stats):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # --- c1 scatter ---
    ax = axes[0][0]
    for c2t in C2_TARGETS:
        sub = agg[agg["c2_target"] == c2t]
        ax.errorbar(sub["c1_target"], sub["c1_mean"], yerr=sub["c1_std"],
                    marker="o", capsize=3, label=f"c2*={c2t:+.2f}",
                    linestyle="None", markersize=6)
    lim = [min(C1_TARGETS) - 0.1, max(C1_TARGETS) + 0.2]
    ax.plot(lim, lim, "k--", alpha=0.4, label="identity")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("c1 target", fontsize=11)
    ax.set_ylabel("c1 measured (mean +/- std over 10 seeds)", fontsize=11)
    ax.set_title(f"c1 targeting   RMSE = {stats['c1_rmse']:.3f}   "
                 f"bias = {stats['c1_bias']:+.3f}", fontsize=12)
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    # --- c2 scatter ---
    ax = axes[0][1]
    for c1t in C1_TARGETS:
        sub = agg[agg["c1_target"] == c1t]
        ax.errorbar(sub["c2_target"], sub["c2_mean"], yerr=sub["c2_std"],
                    marker="o", capsize=3, label=f"c1*={c1t:.2f}",
                    linestyle="None", markersize=6)
    lim = [min(C2_TARGETS) - 0.05, max(C2_TARGETS) + 0.05]
    ax.plot(lim, lim, "k--", alpha=0.4, label="identity")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("c2 target", fontsize=11)
    ax.set_ylabel("c2 measured (mean +/- std over 10 seeds)", fontsize=11)
    ax.set_title(f"c2 targeting   RMSE = {stats['c2_rmse']:.3f}   "
                 f"bias = {stats['c2_bias']:+.3f}", fontsize=12)
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    # --- c1 residual violin per c1 target ---
    ax = axes[1][0]
    df["c1_resid"] = df["c1_measured"] - df["c1_target"]
    data = [df.loc[df["c1_target"] == c, "c1_resid"].values for c in C1_TARGETS]
    parts = ax.violinplot(data, positions=range(len(C1_TARGETS)),
                          showmeans=True, showextrema=False, widths=0.7)
    for b in parts["bodies"]:
        b.set_facecolor("#1f77b4"); b.set_alpha(0.6); b.set_edgecolor("k")
    parts["cmeans"].set_color("k")
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(C1_TARGETS)))
    ax.set_xticklabels([f"{c:.2f}" for c in C1_TARGETS])
    ax.set_xlabel("c1 target", fontsize=11)
    ax.set_ylabel("c1 residual (measured - target)", fontsize=11)
    ax.set_title("c1 residuals per target  (across all c2 targets, 10 seeds)",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # --- c2 residual violin per c2 target ---
    ax = axes[1][1]
    df["c2_resid"] = df["c2_measured"] - df["c2_target"]
    data = [df.loc[df["c2_target"] == c, "c2_resid"].values for c in C2_TARGETS]
    parts = ax.violinplot(data, positions=range(len(C2_TARGETS)),
                          showmeans=True, showextrema=False, widths=0.7)
    for b in parts["bodies"]:
        b.set_facecolor("#d62728"); b.set_alpha(0.6); b.set_edgecolor("k")
    parts["cmeans"].set_color("k")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(C2_TARGETS)))
    ax.set_xticklabels([f"{c:+.2f}" for c in C2_TARGETS])
    ax.set_xlabel("c2 target", fontsize=11)
    ax.set_ylabel("c2 residual (measured - target)", fontsize=11)
    ax.set_title("c2 residuals per target  (across all c1 targets, 10 seeds)",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("prescribed_cascade validation  "
                 f"(n={N}, 4 c1 x 5 c2 targets, 10 seeds each)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    if OUT_CSV.exists():
        print(f"Loading existing manifest from {OUT_CSV.relative_to(ROOT)}")
        df = pd.read_csv(OUT_CSV)
    else:
        df = sweep()
    agg, stats = summarize(df)
    print(f"\nSummary:")
    print(f"  c1 RMSE = {stats['c1_rmse']:.3f}  bias = {stats['c1_bias']:+.3f}")
    print(f"  c2 RMSE = {stats['c2_rmse']:.3f}  bias = {stats['c2_bias']:+.3f}")
    make_figure(df, agg, stats)
