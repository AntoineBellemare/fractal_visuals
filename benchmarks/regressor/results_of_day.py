"""
results_of_day.py — one-page summary of the 2026-07-31 pass.

Six panels, each a measurement committed in this repo:
  A  cascade amplitude collapses as intermittency rises      -> why the old blend was broken
  B  prescribed_cascade does not deliver independent (c1,c2) -> the calibration finding
  C  the conditioning encoder, decomposed                    -> the CLIP costs 5.5x the bit depth
  D  field-construction ablation                             -> both fixes needed, different jobs
  E  end-to-end paired-polarity result                       -> inverted -> 27 %
  F  "c1 is fractal dimension", finally sourced              -> r = -0.85 on real photographs

A and B are recomputed live so the figure is self-contained; C-F read committed CSVs.

Usage:  python results_of_day.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import mfractal as mf                                                   # noqa: E402
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "benchmarks" / "figures" / "60_results_of_day.png"

C2S = [-0.20, -0.45, -0.90, -1.20]
BLUE, RED, GREY, GREEN = "#1f77b4", "#c0392b", "#b0b0b0", "seagreen"


def measure(f):
    z = np.asarray(f, float); z = (z - z.min()) / (np.ptp(z) + 1e-12)
    r = mf.wavelet_leaders_2d(z)
    return float(r["c1"]), float(r["c2"])


def panel_A(ax):
    sds = []
    for c2 in C2S:
        s = [np.asarray(mf.prescribed_cascade(n=512, seed=k, c1_target=1.3, c2_target=c2),
                        float).std() for k in range(3)]
        sds.append(np.mean(s))
    ax.bar([f"{c:+.2f}" for c in C2S], sds, color=[BLUE] + [GREY] * 3, width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(min(sds) * 0.55, max(sds) * 1.9)          # headroom so labels clear the title
    for i, v in enumerate(sds):
        ax.text(i, v * 0.88, f"{v:.5f}", ha="center", va="top", fontsize=8.5)
    ax.set_xlabel("c2 requested (at c1_target 1.3, n = 512)")
    ax.set_ylabel("field standard deviation (log)")
    ax.set_title("A · Cascade amplitude COLLAPSES as intermittency rises\n"
                 f"{sds[0]/sds[2]:.1f}× weaker at c2 −0.90 (8.3× at n = 1024), so a linear\n"
                 "blend is dominated by the LOW-c2 field", fontsize=10, loc="left")


def panel_B(ax):
    for c1t, c in zip((0.8, 1.3, 1.8), (BLUE, RED, GREEN)):
        got = []
        for c2 in C2S:
            m = [measure(mf.prescribed_cascade(n=512, seed=k, c1_target=c1t, c2_target=c2))[0]
                 for k in range(3)]
            got.append(np.mean(m))
        ax.plot(C2S, got, "o-", color=c, lw=1.8, ms=5, label=f"c1 requested {c1t}")
        ax.axhline(c1t, color=c, ls=":", lw=1)
    ax.set_xlabel("c2 requested")
    ax.set_ylabel("c1 actually measured")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("B · The synthesiser does NOT deliver independent cumulants\n"
                 "dotted = what was asked for; asking for strong c2 drags roughness up",
                 fontsize=10, loc="left")


def panel_C(ax):
    d = pd.read_csv(HERE / "cond_encoding" / "e1_summary.csv").set_index("enc")
    steps = [("pure affine\n(no clip, no quant)", d.loc["CTRL_minmax_noquant", "c2_slope"], GREEN),
             ("+ 1/99 percentile\nCLIP (still float)", d.loc["CTRL_robust_1_99_noquant", "c2_slope"], RED),
             ("+ 8-bit\nquantisation", d.loc["robust_unit_1_99", "c2_slope"], GREY)]
    ax.bar([s[0] for s in steps], [s[1] for s in steps], color=[s[2] for s in steps], width=0.6)
    for i, s in enumerate(steps):
        ax.text(i, s[1] + 0.03, f"{s[1]:.3f}", ha="center", fontsize=9, fontweight="bold")
    clip = steps[0][1] - steps[1][1]; quant = steps[1][1] - steps[2][1]
    ax.set_ylim(0, 1.18); ax.set_ylabel("c2 transmission slope")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("C · It was never the bit depth — it is the CLIP\n"
                 f"clip costs {clip:.3f}, quantisation only {quant:.3f}  →  {clip/quant:.1f} : 1",
                 fontsize=10, loc="left")


def panel_D(ax):
    a = pd.read_csv(RES / "spatial_c2_field_ablation.csv")
    y = np.arange(len(a))
    col = [BLUE if (r.amp_match and r.calibrated) else GREY for r in a.itertuples()]
    ax.barh(y, a["dc2"], xerr=a["dc2_sd"], color=col, height=0.6,
            error_kw=dict(lw=1, capsize=3, ecolor="#444"))
    ax.set_yticks(y); ax.set_yticklabels(a["construction"], fontsize=8.5); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(a["requested"].iloc[0], color=RED, ls="--", lw=1.1)
    for i, r in enumerate(a.itertuples()):
        ax.text(0.015, i, f"  Δc1 {r.dc1:+.3f}", fontsize=7.5, va="center")
    ax.set_xlabel("Δc2 delivered by the conditioning field")
    ax.set_title("D · Both fixes needed, and they do different jobs\n"
                 "amplitude-matching creates the gradient; calibration kills the c1 confound",
                 fontsize=10, loc="left")


def panel_E(ax):
    names = [("baseline", "baseline\n(old construction)", RED),
             ("fixed", "fixed,\nlinear ramp", "#9ecae1"),
             ("fixed_sharp", "fixed +\nsmoothstep ramp", BLUE)]
    means, ses, labs, cols = [], [], [], []
    for key, lab, c in names:
        e = pd.read_csv(RES / f"spatial_c2_{key}.csv")["effect_dc2"]
        means.append(e.mean()); ses.append(e.std(ddof=1) / np.sqrt(len(e)))
        labs.append(lab); cols.append(c)
    ax.bar(labs, means, yerr=ses, color=cols, width=0.55,
           error_kw=dict(lw=1.2, capsize=4, ecolor="#333"))
    ax.axhline(0, color="k", lw=0.9)
    for i, (m, s) in enumerate(zip(means, ses)):
        ax.text(i, m + (0.035 if m > 0 else -0.055), f"{m:+.3f}", ha="center",
                fontsize=9.5, fontweight="bold")
    ax.set_ylabel("paired effect  ½·[Δc2(fwd) − Δc2(rev)]")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title("E · End-to-end on the rendered image: inverted → 27 % transmission\n"
                 "paired vs matched baseline: t = −3.34, p = 0.0086 (Wilcoxon 0.0039, n = 10)",
                 fontsize=10, loc="left")


def panel_F(ax):
    d = pd.read_csv(RES / "fd_validation.csv")
    m = d[d["source"] == "macro"]
    ax.scatter(m["c1"], m["dbc_fd"], s=9, alpha=0.55, color=BLUE, edgecolors="none")
    xs = np.linspace(m["c1"].min(), m["c1"].max(), 50)
    s, i = np.polyfit(m["c1"], m["dbc_fd"], 1)
    r = np.corrcoef(m["c1"], m["dbc_fd"])[0, 1]
    ax.plot(xs, i + s * xs, color=RED, lw=2, label=f"fit  FD = {i:.2f} {s:+.2f}·c1   r = {r:.2f}")
    ax.plot(xs, 3 - xs, "k--", lw=1.2, label="idealised FD = 3 − c1")
    ax.set_xlabel("c1"); ax.set_ylabel("box-counting FD")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"F · \"c1 IS fractal dimension\" — finally sourced (n = {len(m)} real photos)\n"
                 "the claim had no artifact in the repo; DBC sanity-checked at r = −1.000 on fBm",
                 fontsize=10, loc="left")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.2))
    for fn, ax in zip((panel_A, panel_B, panel_C, panel_D, panel_E, panel_F), axes.ravel()):
        fn(ax)
    fig.suptitle("Controllable multifractality — results of 2026-07-31.   "
                 "Spatial c2 was never impossible: the conditioning field was broken, "
                 "and the fix also removes the seam artifact.",
                 fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=3.0, w_pad=2.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
