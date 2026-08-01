"""
creative_figures.py — figures for the two creative applications.

  63_detail_falloff.png    subject x style grid + ramp-vs-matched-flat contrast per style
  64_matched_families.png  the matched set + guided-vs-unguided spread, at TWO targets

Both figures report the control, not just the effect: application 1 renders every cell a second
time with a flat field, and application 2 renders every family a second time unguided at the same
seed. Without those the numbers would be uninterpretable.

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
BLUE, RED, GREY = "#1f77b4", "#c0392b", "#9e9e9e"


def falloff(src=HERE / "creative" / "falloff"):
    d = pd.read_csv(src / "falloff.csv")
    ramp = d[d.variant == "ramp"]
    subs = list(dict.fromkeys(ramp.subject)); stys = list(dict.fromkeys(ramp["style"]))
    nr, nc = len(subs), len(stys)

    fig = plt.figure(figsize=(nc * 3.4, nr * 3.6 + 4.2))
    gs = fig.add_gridspec(nr + 1, nc, height_ratios=[1] * nr + [0.92], hspace=0.30, wspace=0.06)
    for i, sub in enumerate(subs):
        for j, sty in enumerate(stys):
            r = ramp[(ramp.subject == sub) & (ramp["style"] == sty)]
            ax = fig.add_subplot(gs[i, j]); ax.set_xticks([]); ax.set_yticks([])
            if not len(r):
                ax.axis("off"); continue
            r = r.iloc[0]
            ax.imshow(Image.open(src / "img" / f"{sub}_{sty}_ramp.png").resize((420, 420)))
            ok = "✓" if r.dc1 > 0 else "·"
            ax.set_title(f"FD {r.fd_centre:.2f} → {r.fd_edge:.2f}   (Δc1 {r.dc1:+.2f}) {ok}",
                         fontsize=8.5, pad=3)
            if i == 0:
                ax.text(0.5, 1.19, sty, transform=ax.transAxes, ha="center",
                        fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(sub, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[nr, :])
    x = np.arange(len(stys)); w = 0.36
    gm = [d[(d["style"] == s) & (d.variant == "ramp")].dc1.mean() for s in stys]
    fm = [d[(d["style"] == s) & (d.variant == "flat")].dc1.mean() for s in stys]
    ge = [d[(d["style"] == s) & (d.variant == "ramp")].dc1.sem() for s in stys]
    fe = [d[(d["style"] == s) & (d.variant == "flat")].dc1.sem() for s in stys]
    ax.bar(x - w / 2, gm, yerr=ge, width=w, color=BLUE, label="radial FD ramp",
           error_kw=dict(lw=1, capsize=3, ecolor="#333"))
    ax.bar(x + w / 2, fm, yerr=fe, width=w, color=GREY, label="flat field (matched control)",
           error_kw=dict(lw=1, capsize=3, ecolor="#333"))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(stys, fontsize=12)
    ax.set_ylabel("Δc1  (edge − centre)\npositive = simplified periphery")
    ax.legend(fontsize=9.5, loc="upper left")
    p = d.pivot_table(index=["subject", "style"], columns="variant", values="dc1").dropna()
    t, pv = stats.ttest_rel(p["ramp"], p["flat"])
    wil = stats.wilcoxon(p["ramp"], p["flat"]).pvalue
    ax.set_title("The medium behaves as a SUBSTRATE: ink wash carries the ramp (+0.251, 4/4), "
                 "oil impasto and engraving do not.\n"
                 f"Paired ramp vs flat across all 16 cells: mean +{(p['ramp']-p['flat']).mean():.3f}, "
                 f"t = {t:.2f}, p = {pv:.2f} (Wilcoxon {wil:.3f}), ramp > flat in "
                 f"{int((p['ramp']>p['flat']).sum())}/16 — suggestive, not significant.\n"
                 "CAVEAT: the radial field also imposes radial COMPOSITION — the cathedral became a "
                 "dome, the dragon a spiral. That is structure\ntransfer, not a pure statistics "
                 "manipulation, and where content leaks hardest it can invert the measurement "
                 "(cathedral/photo, Δc1 −0.60).",
                 fontsize=10.5, loc="left")

    fig.suptitle("Creative application 1 — COMPLEXITY AS A COMPOSITIONAL DEVICE.  A radial "
                 "fractal-dimension ramp concentrates detail on the subject and simplifies the\n"
                 "periphery. Unlike depth of field the edge stays sharp and in focus — it just "
                 "becomes statistically simpler. Figurative content, four media,\n"
                 "every value measured centre-vs-corner against a matched flat-field control.",
                 fontsize=12.5, y=0.998)
    fig.savefig(FIG / "63_detail_falloff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 63_detail_falloff.png")


def _scatter(ax, d, title):
    g = d[d.arm == "guided"]; u = d[d.arm == "unguided"]
    c1t, c2t = float(g.c1_target.iloc[0]), float(g.c2_target.iloc[0])
    for _, r in g.iterrows():
        m = u[u.family == r.family]
        if len(m):
            ax.annotate("", xy=(r.c1_meas, r.c2_meas),
                        xytext=(m.c1_meas.iloc[0], m.c2_meas.iloc[0]),
                        arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.9), zorder=1)
    ax.scatter(u.c1_meas, u.c2_meas, s=70, color=GREY, label="unguided (same prompt & seed)",
               edgecolors="white", linewidths=0.7, zorder=2)
    ax.scatter(g.c1_meas, g.c2_meas, s=85, color=BLUE, label="guided", edgecolors="white",
               linewidths=0.7, zorder=3)
    ax.scatter([c1t], [c2t], s=330, marker="*", color=RED, label="target", zorder=4,
               edgecolors="white", linewidths=0.8)
    s1g, s1u = g.c1_meas.std(ddof=1), u.c1_meas.std(ddof=1)
    s2g, s2u = g.c2_meas.std(ddof=1), u.c2_meas.std(ddof=1)
    ax.set_xlabel("c1   (FD = 3 − c1)"); ax.set_ylabel("c2"); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"{title}   target ({c1t:.2f}, {c2t:+.2f})\n"
                 f"c1 spread {s1u:.3f} → {s1g:.3f}  ({(1-s1g/s1u)*100:+.0f}%)      "
                 f"c2 spread {s2u:.3f} → {s2g:.3f}  ({(1-s2g/s2u)*100:+.0f}%)",
                 fontsize=10.5, loc="left")


def matched(a=HERE / "creative" / "matched", b=HERE / "creative" / "matched_feasible"):
    da = pd.read_csv(a / "matched.csv"); db = pd.read_csv(b / "matched.csv")
    g = db[db.arm == "guided"]
    fams = list(dict.fromkeys(g.family))
    nc = 5; nr = int(np.ceil(len(fams) / nc))
    fig = plt.figure(figsize=(nc * 3.2, nr * 3.5 + 6.0))
    gs = fig.add_gridspec(nr + 1, 2, height_ratios=[1] * nr + [1.5], hspace=0.34, wspace=0.18)
    sub = [fig.add_subplot(gs[k // nc, :]) for k in range(0)]          # placeholder
    inner = gs[0:nr, :].subgridspec(nr, nc, hspace=0.30, wspace=0.06)
    for k, fam in enumerate(fams):
        r = g[g.family == fam].iloc[0]
        ax = fig.add_subplot(inner[k // nc, k % nc]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(Image.open(b / "img" / f"{fam}_guided_s{int(r.seed)}.png").resize((400, 400)))
        ax.set_title(f"{fam}\nc1 {r.c1_meas:.2f}   c2 {r.c2_meas:+.2f}", fontsize=8.5, pad=3)
    _scatter(fig.add_subplot(gs[nr, 0]), da, "A ·")
    _scatter(fig.add_subplot(gs[nr, 1]), db, "B ·")
    fig.suptitle("Creative application 2 — STATISTICALLY MATCHED FAMILIES.  Ten media (oil, "
                 "engraving, 3D, watercolour, photo, charcoal, linocut, woodblock, SEM)\ndriven "
                 "toward ONE texture signature. Replicated at two targets: guidance tightens the "
                 "set in c1 by ~1/3 — and FAILS in c2, where the\nspread gets WORSE. Uniform "
                 "mark-making media cannot reach strong c2, so the medium is a substrate with its "
                 "own feasible region.", fontsize=12.5, y=0.999)
    fig.savefig(FIG / "64_matched_families.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 64_matched_families.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "falloff"):
        falloff()
    if which in ("both", "matched"):
        matched()
