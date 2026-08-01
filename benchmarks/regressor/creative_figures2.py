"""
creative_figures2.py — figures for creative applications 3 and 4.

  65_texture_shapes.png       shape rendered purely in texture statistics; recoverability maps;
                              and the detectability-vs-visibility sweep that says when it is
                              actually HIDDEN rather than merely texture-defined
  66_complexity_animation.png complexity as a motion axis: filmstrip, tracking, coherence

Usage:  python creative_figures2.py [shapes|anim|both]
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
BLUE, RED, GREY, GREEN = "#1f77b4", "#c0392b", "#9e9e9e", "seagreen"


def _c1map(path, win=192, stride=96):
    import mfractal as mf
    a = np.asarray(Image.open(path).convert("L"), float) / 255.0
    a = (a - a.min()) / (np.ptp(a) + 1e-12)
    h, w = a.shape
    ys = range(0, h - win + 1, stride); xs = range(0, w - win + 1, stride)
    return np.array([[mf.wavelet_leaders_2d(a[y:y + win, x:x + win])["c1"] for x in xs]
                     for y in ys])


def shapes(src=C / "hidden_gen", sweep=C / "hidden_sweep"):
    d = pd.read_csv(src / "hidden.csv")
    sh = d[d.variant == "shape"]; fl = d[d.variant == "flat"]

    fig = plt.figure(figsize=(17.5, 13.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.15, 1.0], hspace=0.30, wspace=0.16)

    # rows 1-2: images and their local-c1 maps
    picks = []
    for _, r in sh.sort_values("recover_r", ascending=False).iterrows():
        if not any(p[0] == r.subject for p in picks):
            picks.append((r.subject, r["shape"], int(r.seed), r.dc1, r.recover_r))
        if len(picks) == 4:
            break
    for j, (subj, shp, seed, dc1, rr) in enumerate(picks):
        fp = src / "img" / f"{subj}_{shp}_shape_s{seed}.png"
        ax = fig.add_subplot(gs[0, j]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(Image.open(fp).resize((430, 430)))
        ax.set_title(f"{subj} · {shp}\nΔc1 {dc1:+.2f}   recoverable r {rr:+.2f}",
                     fontsize=9.5, pad=3)
        ax = fig.add_subplot(gs[1, j]); ax.set_xticks([]); ax.set_yticks([])
        m = _c1map(fp)
        ax.imshow(m, cmap="magma", interpolation="bilinear")
        ax.set_title("local c1 map — the shape is IN the statistic", fontsize=9)

    # row 3a: detectability
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(["shape", "flat\n(null)"], [sh.dc1.mean(), fl.dc1.mean()],
           yerr=[sh.dc1.sem(), fl.dc1.sem()], color=[BLUE, GREY], width=0.55,
           error_kw=dict(lw=1.2, capsize=4, ecolor="#333"))
    ax.axhline(0, color="k", lw=0.9)
    dp = (sh.dc1.mean() - fl.dc1.mean()) / fl.dc1.std(ddof=1)
    ax.set_ylabel("Δc1  inside − outside the shape")
    ax.set_title(f"Detectability\nd′ = {dp:.1f} vs the flat-field null", fontsize=10, loc="left")

    # row 3b: recoverability
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(fl.recover_r, bins=12, color=GREY, alpha=0.85, label="flat (null)")
    ax.hist(sh.recover_r, bins=12, color=BLUE, alpha=0.85, label="shape")
    ax.set_xlabel("corr(local c1 map, shape mask)")
    ax.legend(fontsize=8.5)
    ax.set_title(f"Recoverability\nr = {sh.recover_r.mean():+.2f} vs null {fl.recover_r.mean():+.2f}",
                 fontsize=10, loc="left")

    # row 3c-d: the sweep — when is it HIDDEN, not just texture-defined?
    if (sweep / "hidden.csv").exists():
        s = pd.read_csv(sweep / "hidden.csv")
        s = s[s.variant == "shape"].groupby("level").agg(
            dc1=("dc1", "mean"), rr=("recover_r", "mean"),
            dlum=("dlum", "mean"), dgrad=("dgrad", "mean"),
            dcon=("dcontrast", "mean")).reset_index()
        nul = pd.read_csv(sweep / "hidden.csv")
        nul = nul[nul.variant == "flat"].recover_r
        # x-axis must be the REQUESTED level: the measured dc1 collapses onto one value for
        # every sub-threshold step, which would hide the step entirely.
        base = nul.mean()
        ax = fig.add_subplot(gs[2, 2])
        ax.plot(s.level, s.rr - base, "o-", color=BLUE, lw=2, ms=7)
        ax.axhline(2 * nul.std(ddof=1), color=RED, ls="--", lw=1.2, label="null + 2 sd")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Δc1 REQUESTED between shape and ground")
        ax.set_ylabel("recoverability above null")
        ax.legend(fontsize=8)
        ax.set_title("It is a THRESHOLD, not a gradual fade:\nnothing at all below Δc1 ≈ 1.2",
                     fontsize=10, loc="left")
        ax = fig.add_subplot(gs[2, 3])
        ax.plot(s.level, np.abs(s.dgrad), "o-", color=RED, lw=2, ms=7,
                label="|Δ local gradient| — visibility")
        ax.plot(s.level, np.abs(s.dlum), "s-", color=GREY, lw=1.6, ms=6,
                label="|Δ luminance| — stays matched")
        ax.set_xlabel("Δc1 requested"); ax.set_ylabel("difference across the boundary")
        ax.legend(fontsize=8)
        ax.set_title("Where it becomes recoverable it also becomes\nVISIBLE — so no hidden regime",
                     fontsize=10, loc="left")

    fig.suptitle("Creative application 3 — A SHAPE DRAWN IN PURE TEXTURE STATISTICS.  No outline, "
                 "no colour change, luminance matched to within 2 %: the region\ndiffers only in "
                 "fractal dimension, and is fully localisable from the image alone. The repo "
                 "recorded this as NOT ACHIEVED (|Δc2| 0.039 vs a\n~0.15 noise floor) — but that "
                 "used c2, the axis that barely transmits, and an unmatched blend. On the c1 axis "
                 "it works.  BUT it is not HIDDEN:\nrecoverability appears only above Δc1 ≈ 1.2, "
                 "and at that point the texture difference is plainly visible. No invisible-yet-"
                 "detectable window exists.",
                 fontsize=12, y=0.998)
    fig.savefig(FIG / "65_texture_shapes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 65_texture_shapes.png")


def anim(src=C / "anim"):
    d = pd.read_csv(src / "anim.csv")
    subs = list(dict.fromkeys(d.subject))
    co = d[d.arm == "coherent"]
    nshow = 8
    fig = plt.figure(figsize=(nshow * 2.3, len(subs) * 2.6 + 5.0))
    gs = fig.add_gridspec(len(subs) + 1, 2, height_ratios=[1] * len(subs) + [1.45],
                          hspace=0.35, wspace=0.20)
    for i, sub in enumerate(subs):
        s = co[co.subject == sub].sort_values("frame")
        idx = np.linspace(0, len(s) - 1, nshow).astype(int)
        inner = gs[i, :].subgridspec(1, nshow, wspace=0.04)
        for j, ii in enumerate(idx):
            r = s.iloc[ii]
            ax = fig.add_subplot(inner[0, j]); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(Image.open(src / "coherent" / sub / f"f{int(r.frame):02d}.png").resize((300, 300)))
            ax.set_title(f"FD {r.fd_meas:.2f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(sub, fontsize=11, fontweight="bold")
    ax = fig.add_subplot(gs[len(subs), 0])
    for sub, c in zip(subs, (BLUE, RED, GREEN, GREY)):
        s = co[co.subject == sub].sort_values("frame")
        rr = np.corrcoef(s.field_c1, s.c1_meas)[0, 1]
        ax.plot(s.frame, s.c1_meas, "o-", color=c, lw=2, label=f"{sub}  r = {rr:+.2f}")
    s0 = co[co.subject == subs[0]].sort_values("frame")
    ax.plot(s0.frame, s0.field_c1, "k--", lw=1.4, label="conditioning field c1")
    ax.set_xlabel("frame"); ax.set_ylabel("measured c1 of the rendered frame")
    ax.legend(fontsize=9)
    ax.set_title("The axis tracks: image complexity follows the field across the sweep",
                 fontsize=10.5, loc="left")
    ax = fig.add_subplot(gs[len(subs), 1])
    arms = list(dict.fromkeys(d.arm))
    vals = [d[d.arm == a].frame_dist.dropna().mean() for a in arms]
    errs = [d[d.arm == a].frame_dist.dropna().sem() for a in arms]
    ax.bar([{"coherent": "fixed seed\n(the method)",
             "control": "reseeded each frame\n(incoherent null)"}.get(a, a) for a in arms],
           vals, yerr=errs, color=[BLUE, GREY][:len(arms)], width=0.5,
           error_kw=dict(lw=1.2, capsize=4, ecolor="#333"))
    for i, v in enumerate(vals):
        ax.text(i, v / 2, f"{v:.4f}", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("mean frame-to-frame pixel distance")
    if len(vals) == 2:
        ax.set_title(f"Temporal coherence: {vals[1]/vals[0]:.1f}× smoother than a reseeded "
                     f"sequence", fontsize=10.5, loc="left")
    fig.suptitle("Creative application 4 — COMPLEXITY AS AN AXIS OF MOTION.  The subject, "
                 "composition and lighting hold still while the texture breathes\nbetween smooth "
                 "and intricate. Coherence comes from fixing both the cascade seed (so the field "
                 "morphs rather than being redrawn)\nand the diffusion seed. Animated GIFs are "
                 "written alongside the frames.", fontsize=12.5, y=0.999)
    fig.savefig(FIG / "66_complexity_animation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 66_complexity_animation.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "shapes"):
        shapes()
    if which in ("both", "anim"):
        anim()
