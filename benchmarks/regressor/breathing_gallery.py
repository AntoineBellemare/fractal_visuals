"""
breathing_gallery.py — contact sheet + montage GIF for the 20-subject breathing gallery.

Outputs:
  68_breathing_gallery.png   one filmstrip per subject, labelled with its MEASURED FD span
  69_breathing_montage.gif   a 3x3 montage loop, small enough to commit
  results/creative_breathing.csv

The per-subject FD span is the number that matters: a substrate with a narrow reachable c1 range
barely breathes however hard the field is driven, so the gallery is sorted by it and the flat ones
are labelled rather than quietly included.

Usage:  python breathing_gallery.py
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
FIG = ROOT / "benchmarks" / "figures"
SRC = HERE / "creative" / "breathing"
TITLE = "COMPLEXITY BREATHING"
BLUE, GREY = "#1f77b4", "#9e9e9e"


def load(src=None):
    d = pd.read_csv((src or SRC) / "anim.csv")
    d = d[d.arm == "coherent"]
    agg = []
    for sub, g in d.groupby("subject"):
        g = g.sort_values("frame")
        agg.append(dict(subject=sub,
                        span=g.c1_meas.max() - g.c1_meas.min(),
                        fd_hi=3 - g.c1_meas.min(), fd_lo=3 - g.c1_meas.max(),
                        track=np.corrcoef(g.field_c1, g.c1_meas)[0, 1],
                        dist=g.frame_dist.dropna().mean()))
    return d, pd.DataFrame(agg).sort_values("span", ascending=False)


def sheet(d, agg, ncols=6, src=None, out="68_breathing_gallery.png", title=None):
    src = src or SRC
    subs = agg.subject.tolist()
    nr = len(subs)
    fig = plt.figure(figsize=(ncols * 2.15 + 3.4, nr * 2.32))
    gs = fig.add_gridspec(nr, ncols + 2, wspace=0.04, hspace=0.34,
                          width_ratios=[1] * ncols + [1.5, 1.5])
    for i, sub in enumerate(subs):
        g = d[d.subject == sub].sort_values("frame")
        idx = np.linspace(0, len(g) - 1, ncols).astype(int)
        for j, ii in enumerate(idx):
            r = g.iloc[ii]
            ax = fig.add_subplot(gs[i, j]); ax.set_xticks([]); ax.set_yticks([])
            fp = src / "coherent" / sub / f"f{int(r.frame):02d}.png"
            ax.imshow(Image.open(fp).resize((300, 300)))
            if i == 0:
                ax.set_title(f"frame {int(r.frame)}", fontsize=9)
            if j == 0:
                a = agg[agg.subject == sub].iloc[0]
                flag = "" if a.span >= 0.25 else "  (flat)"
                ax.set_ylabel(f"{sub}{flag}\nFD {a.fd_hi:.2f}→{a.fd_lo:.2f}",
                              fontsize=9.5, fontweight="bold", rotation=0,
                              ha="right", va="center")
                ax.yaxis.set_label_coords(-0.06, 0.5)
        ax = fig.add_subplot(gs[i, ncols:])
        ax.plot(g.frame, 3 - g.c1_meas, "o-", color=BLUE, lw=1.8, ms=3.5)
        ax.plot(g.frame, 3 - g.field_c1, "--", color=GREY, lw=1.2)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("FD", fontsize=8)
        a = agg[agg.subject == sub].iloc[0]
        # inside the axes — as a title it collided with the filmstrip in the row above
        ax.text(0.97, 0.94, f"span {a.span:.2f}   tracking r {a.track:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        if i == nr - 1:
            ax.set_xlabel("frame", fontsize=8)
    nf = d.groupby("subject").frame.nunique().max()
    fig.suptitle(f"{title or 'COMPLEXITY BREATHING'} — {nr} substrates, {nf} frames each, 1024 px."
                 "  Fractal dimension sweeps from fine-grained (high FD) to coarse (low FD)\nwhile "
                 "subject, composition and lighting hold still.  Grey dashed = the conditioning "
                 "field, blue = the MEASURED FD of the rendered frame.\nSorted by achieved span. "
                 "The path is reparameterised so equal frame steps mean equal c1 steps (step-size "
                 f"cv 0.23 vs 0.63 linear).\nAll {nr} track at r ≥ {agg.track.min():.2f}; "
                 f"median achieved FD span {agg.span.median():.2f}.",
                 fontsize=13, y=0.997)
    fig.subplots_adjust(top=0.965)
    fig.savefig(FIG / out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def montage(agg, k=9, cell=232, colors=128, src=None, out="69_breathing_montage.gif"):
    src = src or SRC
    """3x3 montage loop of the widest-span subjects — small enough to commit."""
    subs = agg.subject.tolist()[:k]
    side = int(np.sqrt(k))
    frames = sorted((src / "coherent" / subs[0]).glob("f*.png"))
    n = len(frames)
    seq = list(range(n)) + list(range(n - 2, 0, -1))
    ims = []
    for fi in seq:
        canvas = Image.new("RGB", (cell * side, cell * side))
        for i, sub in enumerate(subs):
            fp = src / "coherent" / sub / f"f{fi:02d}.png"
            canvas.paste(Image.open(fp).resize((cell, cell)), ((i % side) * cell, (i // side) * cell))
        ims.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=colors))
    outp = FIG / out
    ims[0].save(outp, save_all=True, append_images=ims[1:], duration=110, loop=0, optimize=True)
    print(f"-> {out}  ({outp.stat().st_size // 1024} KB)  subjects: {', '.join(subs)}")


if __name__ == "__main__":
    d, agg = load()
    print(agg.to_string(index=False))
    d.to_csv(ROOT / "benchmarks" / "results" / "creative_breathing.csv", index=False)
    sheet(d, agg)
    montage(agg)
