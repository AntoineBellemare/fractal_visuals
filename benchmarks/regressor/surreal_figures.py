"""
surreal_figures.py — figures for the surrealist pass.

  70_surreal_forms.png     apparitions made only of fractal dimension: the render, the local-c1
                           map beside it, and the matched flat-field null underneath
  71_surreal_breathing.png the 20-subject surreal breathing contact sheet
  72_surreal_montage.gif   3x3 loop of the widest-span surreal clips

Usage:  python surreal_figures.py [forms|breathing|both]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
FIG = ROOT / "benchmarks" / "figures"
C = HERE / "creative"
BLUE, RED, GREY = "#1f77b4", "#c0392b", "#9e9e9e"


def forms(src=C / "surreal_forms", ncols=5):
    from creative_figures2 import _c1map
    d = pd.read_csv(src / "forms.csv")
    fo = d[d.variant == "form"]; fl = d[d.variant == "flat"]
    # one strong example per form, preferring a DIFFERENT ground each time — picking purely on
    # recoverability lands every panel on lichen and hides how the ground changes the look.
    picks, used = [], set()
    for kind, g in fo.sort_values("recover_r", ascending=False).groupby("form", sort=False):
        cand = g[~g.ground.isin(used)]
        r = (cand if len(cand) else g).iloc[0]
        picks.append(r); used.add(r.ground)
    picks = sorted(picks, key=lambda r: -r.recover_r)[:ncols]

    fig = plt.figure(figsize=(ncols * 3.3, 11.4))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.2, 1.2, 0.98], hspace=0.30, wspace=0.30)
    for j, r in enumerate(picks):
        fp = src / "img" / f"{r.ground}_{r.form}_form_s{int(r.seed)}.png"
        ax = fig.add_subplot(gs[0, j]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(Image.open(fp).resize((430, 430)))
        ax.set_title(f"{r.form} in {r.ground}\nΔc1 {r.dc1:+.2f}   Δlum {r.dlum:+.3f}",
                     fontsize=9.5, pad=3)
        ax = fig.add_subplot(gs[1, j]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(_c1map(fp), cmap="magma", interpolation="bilinear")
        ax.set_title(f"its local c1 map   r = {r.recover_r:+.2f}", fontsize=9)

    ax = fig.add_subplot(gs[2, 0])
    ax.bar(["form", "flat\nnull"], [fo.dc1.mean(), fl.dc1.mean()],
           yerr=[fo.dc1.sem(), fl.dc1.sem()], color=[BLUE, GREY], width=0.55,
           error_kw=dict(lw=1.2, capsize=4, ecolor="#333"))
    ax.axhline(0, color="k", lw=0.9)
    dp = (fo.dc1.mean() - fl.dc1.mean()) / fl.dc1.std(ddof=1)
    ax.set_ylabel("Δc1  inside − outside")
    ax.set_title(f"Detectability  d′ = {dp:.1f}", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[2, 1])
    ax.hist(fl.recover_r, bins=10, color=GREY, alpha=0.85, label="flat null")
    ax.hist(fo.recover_r, bins=10, color=BLUE, alpha=0.85, label="form")
    ax.set_xlabel("corr(local c1 map, mask)"); ax.legend(fontsize=8)
    ax.set_title(f"Recoverability {fo.recover_r.mean():+.2f} vs {fl.recover_r.mean():+.2f}",
                 fontsize=10, loc="left")

    ax = fig.add_subplot(gs[2, 2])
    ax.bar(["field", "rendered\nform", "rendered\nnull"], [0.005, fo.dlum.abs().mean(),
                                                           fl.dlum.abs().mean()],
           color=["seagreen", RED, GREY], width=0.6)
    ax.set_ylabel("|Δ luminance| across the boundary")
    ax.set_title("The CONDITIONING carries no brightness cue —\nbut the renderer adds one "
                 "(≈7 % of range)", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[2, 3:])
    order = fo.groupby("ground").recover_r.mean().sort_values(ascending=False)
    ax.barh(range(len(order)), order.values,
            color=[BLUE if v > 0.3 else GREY for v in order.values])
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order.index, fontsize=10)
    ax.invert_yaxis(); ax.axvline(0.3, color=RED, ls="--", lw=1.2)
    ax.set_xlabel("mean recoverability r")
    ax.set_title("Ground matters more than the form does", fontsize=10, loc="left")

    fig.suptitle("Creative application 5 — APPARITIONS MADE OF FRACTAL DIMENSION.  Nothing is "
                 "drawn: no outline and no edge. The form is a region where the\ntexture organises "
                 f"differently, and the local c1 map recovers it — d′ = {dp:.1f}, recoverable in "
                 f"{int((fo.recover_r > 0.3).sum())}/{len(fo)}.  TWO HONEST LIMITS: it is legible, "
                 "not hidden\n(below Δc1 ≈ 1.2 there is no signal at all), and while the "
                 "CONDITIONING is luminance-matched, the renderer makes the form region ≈7 % "
                 "darker —\nso the output carries a modest brightness cue as well as a textural "
                 "one. It is a pareidolia device with prescribed statistics, not an invisible one.",
                 fontsize=12, y=0.999)
    fig.savefig(FIG / "70_surreal_forms.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 70_surreal_forms.png")


def breathing(src=C / "breathing_surreal"):
    import breathing_gallery as bg
    d, agg = bg.load(src=src)
    print(agg.to_string(index=False))
    d.to_csv(ROOT / "benchmarks" / "results" / "creative_breathing_surreal.csv", index=False)
    bg.sheet(d, agg, src=src, out="71_surreal_breathing.png",
             title="SURREAL COMPLEXITY BREATHING")
    bg.montage(agg, k=9, cell=180, colors=72, src=src, out="72_surreal_montage.gif")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "forms"):
        forms()
    if which in ("both", "breathing"):
        breathing()


def tradeoff(root=C, scales=("0.40", "0.65", "0.90"),
             subs=("melting_clocks", "moth_machine", "eye_storm")):
    """The cn_scale dilemma, measured.

    Surreal / figurative prompts need a weaker ControlNet than flat textures do, because the
    conditioning strength that buys FD control is the same strength that overwrites semantics.
    At 0.90 the scene collapses into texture; at 0.40 the scene survives but the axis does not
    move at all. This panel is the evidence for recommending 0.65.
    """
    rows = []
    for cs in scales:
        d = pd.read_csv(root / f"cnsweep_{cs}" / "anim.csv")
        for sub, g in d[d.arm == "coherent"].groupby("subject"):
            g = g.sort_values("frame")
            rows.append(dict(cn=float(cs), subject=sub,
                             span=g.c1_meas.max() - g.c1_meas.min(),
                             track=np.corrcoef(g.field_c1, g.c1_meas)[0, 1]))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "benchmarks" / "results" / "creative_cn_tradeoff.csv", index=False)

    nr, nc = len(subs), len(scales)
    fig = plt.figure(figsize=(nc * 3.4 + 7.2, nr * 3.5))
    gs = fig.add_gridspec(nr, nc + 2, wspace=0.10, hspace=0.20,
                          width_ratios=[1] * nc + [1.35, 1.35])
    for i, sub in enumerate(subs):
        for j, cs in enumerate(scales):
            ax = fig.add_subplot(gs[i, j]); ax.set_xticks([]); ax.set_yticks([])
            fp = root / f"cnsweep_{cs}" / "coherent" / sub / "f00.png"
            ax.imshow(Image.open(fp).resize((400, 400)))
            if i == 0:
                ax.set_title(f"cn_scale {cs}", fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(sub, fontsize=11, fontweight="bold")
    ax = fig.add_subplot(gs[:, nc])
    m = t.groupby("cn").span.agg(["mean", "sem"])
    ax.errorbar(m.index, m["mean"], yerr=m["sem"], fmt="o-", color=BLUE, lw=2.2, ms=8, capsize=4)
    ax.set_xlabel("cn_scale"); ax.set_ylabel("achieved FD span")
    ax.set_title("Control: rises with conditioning strength", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[:, nc + 1])
    m2 = t.groupby("cn").track.agg(["mean", "sem"])
    ax.errorbar(m2.index, m2["mean"], yerr=m2["sem"], fmt="o-", color=RED, lw=2.2, ms=8, capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("cn_scale"); ax.set_ylabel("tracking r")
    ax.set_title("Below ~0.6 the axis stops working entirely", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25)
    fig.suptitle("The cn_scale dilemma for SURREAL content — the conditioning strength that buys "
                 "FD control is the same strength that overwrites semantics.\nAt 0.40 the scene is "
                 "intact and beautiful but the axis does not move at all (span 0.05, r ≈ 0). At "
                 "0.90 the axis works fully and the clocks become dunes,\nthe eye becomes cloud. "
                 "0.65 recovers the axis (r 0.92, ~44 % of the span) but the rescue is only "
                 "partial and SUBJECT-DEPENDENT: the clockwork moth\nsurvives, the melting clocks "
                 "and the eye largely do not. TEXTURAL surrealism composes with this axis; "
                 "SCENE-level surrealism fights it.",
                 fontsize=12, y=0.999)
    fig.savefig(FIG / "73_cn_scale_tradeoff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(t.groupby("cn").agg(span=("span", "mean"), track=("track", "mean")).to_string())
    print("-> 73_cn_scale_tradeoff.png")
