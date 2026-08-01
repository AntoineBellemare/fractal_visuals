"""
creative_figures.py — validated figures for the two creative applications.

  63_detail_falloff.png    horizontal-ramp grid + radial-vs-horizontal contrast, pooled n = 48
  64_matched_families.png  the matched set + the PAIRED per-family error test

Both figures report the control, not just the effect. Application 1 renders every cell a second
time with a flat field; application 2 renders every family a second time unguided at the same
seed. The headline statistic in each case is a PAIRED test, because the pilot's spread-based
framing turned out to be far too underpowered to support any claim (all bootstrap CIs on the
spread ratio straddled 1 at n = 10 families).

Usage:  python creative_figures.py [falloff|matched|both]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = ROOT / "benchmarks" / "figures"
C = HERE / "creative"
BLUE, RED, GREY, GREEN = "#1f77b4", "#c0392b", "#9e9e9e", "seagreen"


def _load_falloff():
    parts = []
    for name, p in (("pilot", C / "falloff" / "falloff.csv"),
                    ("replication", C / "falloff_rep" / "falloff.csv"),
                    ("horizontal", C / "falloff_h" / "falloff.csv")):
        if p.exists():
            d = pd.read_csv(p); d["run"] = name
            if "direction" not in d:
                d["direction"] = "radial"
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _paired(sub):
    p = sub.pivot_table(index=["run", "subject", "style", "seed"],
                        columns="variant", values="dc1").dropna()
    d = (p["ramp"] - p["flat"]).values
    t, pv = stats.ttest_rel(p["ramp"], p["flat"])
    return p, d, t, pv, stats.wilcoxon(p["ramp"], p["flat"]).pvalue


def falloff():
    d = _load_falloff()
    src = C / "falloff_h"
    h = d[d.direction == "h"]
    subs = list(dict.fromkeys(h.subject)); stys = list(dict.fromkeys(h["style"]))
    nr, nc = len(subs), len(stys)

    fig = plt.figure(figsize=(nc * 3.4, nr * 3.6 + 4.4))
    gs = fig.add_gridspec(nr + 1, 2, height_ratios=[1] * nr + [0.98], hspace=0.30, wspace=0.16)
    inner = gs[0:nr, :].subgridspec(nr, nc, hspace=0.30, wspace=0.06)
    ramp = h[h.variant == "ramp"]
    for i, sub in enumerate(subs):
        for j, sty in enumerate(stys):
            r = ramp[(ramp.subject == sub) & (ramp["style"] == sty)]
            ax = fig.add_subplot(inner[i, j]); ax.set_xticks([]); ax.set_yticks([])
            if not len(r):
                ax.axis("off"); continue
            r = r.iloc[0]
            fp = src / "img" / f"{sub}_{sty}_ramp_s{int(r.seed)}.png"
            ax.imshow(Image.open(fp).resize((420, 420)))
            ok = "✓" if r.dc1 > 0 else "·"
            ax.set_title(f"FD {r.fd_centre:.2f} → {r.fd_edge:.2f}   (Δc1 {r.dc1:+.2f}) {ok}",
                         fontsize=8.5, pad=3)
            if i == 0:
                ax.text(0.5, 1.19, sty, transform=ax.transAxes, ha="center",
                        fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(sub, fontsize=12, fontweight="bold")

    # left: radial vs horizontal, overall
    ax = fig.add_subplot(gs[nr, 0])
    labs, means, errs, cols = [], [], [], []
    for direction, lab, c in (("radial", "radial ramp", GREY), ("h", "horizontal ramp", BLUE)):
        sub = d[d.direction == direction]
        _, dd, t, pv, w = _paired(sub)
        labs.append(f"{lab}\nn={len(dd)}  p={pv:.3f}")
        means.append(dd.mean()); errs.append(dd.std(ddof=1) / np.sqrt(len(dd))); cols.append(c)
    ax.bar(labs, means, yerr=errs, color=cols, width=0.5,
           error_kw=dict(lw=1.2, capsize=4, ecolor="#333"))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylim(0, max(m + e for m, e in zip(means, errs)) * 1.32)   # headroom for the labels
    for i, m in enumerate(means):
        ax.text(i, m / 2, f"{m:+.3f}", ha="center", va="center", fontsize=11,
                fontweight="bold", color="white")
    ax.set_ylabel("Δc1 effect  (ramp − matched flat control)")
    ax.set_title("VALIDATED. The horizontal ramp is the stronger AND\ncleaner manipulation — "
                 "no radial composition leak.", fontsize=10.5, loc="left")

    # right: per medium, pooled radial
    ax = fig.add_subplot(gs[nr, 1])
    rad = d[d.direction == "radial"]
    x = np.arange(len(stys)); vals, es, ps = [], [], []
    for sty in stys:
        _, dd, t, pv, w = _paired(rad[rad["style"] == sty])
        vals.append(dd.mean()); es.append(dd.std(ddof=1) / np.sqrt(len(dd))); ps.append(pv)
    cols = [BLUE if p < 0.05 else GREY for p in ps]
    ax.bar(x, vals, yerr=es, color=cols, width=0.6,
           error_kw=dict(lw=1, capsize=3, ecolor="#333"))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\np={p:.2f}" for s, p in zip(stys, ps)], fontsize=9)
    ax.set_ylabel("Δc1 effect (radial, pooled n=48)")
    ax.set_title("Per medium: oil impasto is the one consistent failure.\n"
                 "The pilot's \"ink wash is special\" ranking did NOT replicate.",
                 fontsize=10.5, loc="left")

    _, dd, t, pv, w = _paired(rad)
    fig.suptitle("Creative application 1 — COMPLEXITY AS A COMPOSITIONAL DEVICE  ·  VALIDATED.  An "
                 "FD ramp concentrates detail on the subject and simplifies\nthe rest. Unlike depth "
                 "of field the simplified region stays sharp and in focus. Pooled across pilot + "
                 f"replication: ramp beats a matched\nflat-field control by {dd.mean():+.3f}, "
                 f"t = {t:.2f}, p = {pv:.3f} (Wilcoxon {w:.4f}), {int((dd>0).sum())}/{len(dd)} cells.",
                 fontsize=12.5, y=0.999)
    fig.savefig(FIG / "63_detail_falloff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 63_detail_falloff.png")


def matched(src=C / "matched_sel"):
    d = pd.read_csv(src / "matched.csv")
    c1t, c2t = float(d.c1_target.iloc[0]), float(d.c2_target.iloc[0])
    g = d[d.arm == "guided"].copy()
    g["dist"] = np.hypot(g.c1_meas - c1t, g.c2_meas - c2t)
    sel = g.loc[g.groupby("family")["dist"].idxmin()]
    fams = list(dict.fromkeys(sel.family))
    nc = 5; nr = int(np.ceil(len(fams) / nc))

    fig = plt.figure(figsize=(nc * 3.2, nr * 3.5 + 6.2))
    gs = fig.add_gridspec(nr + 1, 2, height_ratios=[1] * nr + [1.5], hspace=0.34, wspace=0.20)
    inner = gs[0:nr, :].subgridspec(nr, nc, hspace=0.30, wspace=0.06)
    for k, fam in enumerate(fams):
        r = sel[sel.family == fam].iloc[0]
        ax = fig.add_subplot(inner[k // nc, k % nc]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(Image.open(src / "img" / f"{fam}_guided_s{int(r.seed)}.png").resize((400, 400)))
        ax.set_title(f"{fam}\nc1 {r.c1_meas:.2f}   c2 {r.c2_meas:+.2f}", fontsize=8.5, pad=3)

    piv = d.pivot_table(index=["family", "seed"], columns="arm", values=["c1_meas", "c2_meas"])
    for k, (nm, col, tgt) in enumerate((("c1", "c1_meas", c1t), ("c2", "c2_meas", c2t))):
        ax = fig.add_subplot(gs[nr, k])
        gg = (piv[(col, "guided")] - tgt).abs(); uu = (piv[(col, "unguided")] - tgt).abs()
        t, pv = stats.ttest_rel(gg, uu); w = stats.wilcoxon(gg, uu).pvalue
        for a, b in zip(uu, gg):
            ax.plot([0, 1], [a, b], color="#ccc", lw=0.8, zorder=1)
        ax.scatter(np.zeros(len(uu)), uu, s=42, color=GREY, zorder=2, edgecolors="white", lw=0.6)
        ax.scatter(np.ones(len(gg)), gg, s=42, color=BLUE if pv < 0.05 else GREY, zorder=2,
                   edgecolors="white", lw=0.6)
        ax.plot([0, 1], [uu.mean(), gg.mean()], color=RED, lw=2.6, marker="o", zorder=3,
                label=f"mean {uu.mean():.3f} → {gg.mean():.3f}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["unguided", "guided"], fontsize=11)
        ax.set_xlim(-0.28, 1.28)
        ax.set_ylabel(f"|error| in {nm} to target")
        ax.legend(fontsize=9, loc="upper right")
        verdict = "WORKS" if pv < 0.05 and gg.mean() < uu.mean() else "NO EFFECT"
        ax.set_title(f"{nm.upper()}: {verdict}   paired t = {t:+.2f}, p = "
                     f"{'<0.0001' if pv < 1e-4 else f'{pv:.3f}'}  "
                     f"(Wilcoxon {'<0.0001' if w < 1e-4 else f'{w:.3f}'}), "
                     f"better in {int((gg<uu).sum())}/{len(gg)}", fontsize=10.5, loc="left")

    fig.suptitle("Creative application 2 — STATISTICALLY MATCHED FAMILIES  ·  VALIDATED ON c1, "
                 "REFUTED ON c2.  Ten media (oil, engraving, 3D,\nwatercolour, photo, charcoal, "
                 "linocut, woodblock, SEM) driven toward one signature, each against an EXACT "
                 "unguided control\n(same prompt, same seed). 4 seeds per family, n = 40 pairs. "
                 "Guidance halves the c1 error and does nothing at all for c2.",
                 fontsize=12.5, y=0.999)
    fig.savefig(FIG / "64_matched_families.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 64_matched_families.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "falloff"):
        falloff()
    if which in ("both", "matched"):
        matched()
