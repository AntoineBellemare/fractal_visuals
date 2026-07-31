"""
spatial_c2_figure.py — summary figure for the spatial-c2 result.

Rows:
  1  the conditioning image, before and after — this is where the bug lived
  2  field-construction ablation, and ramp-shape sweep (both CPU, 8 seeds)
  3  end-to-end paired-polarity effect per substrate, all three arms

Usage:  python spatial_c2_figure.py
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
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
OUT = ROOT / "benchmarks" / "figures" / "58_spatial_c2.png"
SC = HERE / "spatial_c2"

# measured by the mask sweep (8 seeds each, n=1024, requested -0.80)
MASKS = [("linear", -0.324, 0.113), ("smoothstep", -0.471, 0.130),
         ("sigmoid k=8", -0.432, 0.151), ("sigmoid k=14", -0.639, 0.193),
         ("smoothstep_mid", -0.737, 0.198)]
STEP_CEIL = -0.768


def conditioning_pair(n=640):
    from cascade_calibration import Calibration
    from spatial_c2 import c2_gradient_field
    from mf_controlnet import field_to_control
    cal = Calibration.load()
    old = c2_gradient_field(1.3, -0.20, -1.00, "h", 2, n=n, cal=cal,
                            amp_match=False, calibrated=False, mask_shape="linear")
    new = c2_gradient_field(1.3, -0.20, -1.00, "h", 2, n=n, cal=cal,
                            amp_match=True, calibrated=True, mask_shape="smoothstep_mid")
    return [np.asarray(field_to_control(f).convert("L"), float) for f in (old, new)]


def main():
    abl = pd.read_csv(SC / "field_ablation.csv")
    arms = {}
    for name in ("baseline", "fixed", "fixed_sharp"):
        p = SC / name / "paired.csv"
        if p.exists():
            arms[name] = pd.read_csv(p)
    if not arms:
        sys.exit("no paired.csv found — run spatial_c2.py --mode gen first")

    fig = plt.figure(figsize=(15.5, 13.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 0.95, 1.0], hspace=0.30, wspace=0.24)

    # ---------- row 1: the conditioning image, before and after
    try:
        old, new = conditioning_pair()
    except Exception as e:                                  # figure still builds without it
        old = new = None
        print(f"  (skipping conditioning panel: {e})")
    if old is not None:
        for j, (img, tag) in enumerate(((old, "BEFORE — linear blend of two cascades"),
                                        (new, "AFTER — amplitude-matched + calibrated"))):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(tag, fontsize=10, pad=4)
        ax = fig.add_subplot(gs[0, 2])
        for img, lab, c in ((old, "before", "#c0392b"), (new, "after", "#1f77b4")):
            cm = img.mean(0)
            ax.plot(np.linspace(0, 1, len(cm)), cm, color=c, lw=2, label=lab)
        ax.set_ylim(0, 255); ax.set_xlabel("position across frame")
        ax.set_ylabel("mean brightness")
        ax.legend(fontsize=9)
        ax.set_title("The old field was a LUMINANCE WEDGE\n"
                     "(224 → 18), not a statistics gradient", fontsize=10, loc="left")

    # ---------- row 2A: field ablation
    ax = fig.add_subplot(gs[1, :2])
    y = np.arange(len(abl))
    col = ["#1f77b4" if (r.amp_match and r.calibrated) else "#b0b0b0" for r in abl.itertuples()]
    ax.barh(y, abl["dc2"], xerr=abl["dc2_sd"], color=col, height=0.6,
            error_kw=dict(lw=1.1, capsize=3, ecolor="#444"))
    ax.set_yticks(y); ax.set_yticklabels(abl["construction"], fontsize=9.5)
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
    ax.axvline(abl["requested"].iloc[0], color="crimson", ls="--", lw=1.2)
    ax.text(abl["requested"].iloc[0] + 0.012, 0.02, "requested", color="crimson", fontsize=8.5,
            rotation=90, va="bottom", ha="left", transform=ax.get_xaxis_transform())
    for i, r in enumerate(abl.itertuples()):
        ax.text(0.02, i, f"  Δc1 {r.dc1:+.3f}", fontsize=8, va="center", color="#222")
    ax.set_xlabel("measured Δc2 of the conditioning field (outer thirds)")
    ax.set_title("Both fixes are necessary, and they do different jobs — amplitude-matching "
                 "creates the gradient,\ncalibration removes the c1 confound (Δc1 +0.230 → +0.003)",
                 fontsize=10.5, loc="left")

    # ---------- row 2B: mask shape
    ax = fig.add_subplot(gs[1, 2])
    names = [m[0] for m in MASKS]; vals = [m[1] for m in MASKS]; sds = [m[2] for m in MASKS]
    c = ["#1f77b4" if n == "smoothstep_mid" else "#9ecae1" for n in names]
    ax.bar(range(len(names)), vals, yerr=sds, color=c, width=0.62,
           error_kw=dict(lw=1.1, capsize=3, ecolor="#444"))
    ax.axhline(STEP_CEIL, color="seagreen", ls=":", lw=1.4)
    ax.text(0.02, STEP_CEIL, " hard-step ceiling", color="seagreen", fontsize=8, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.axhline(-0.80, color="crimson", ls="--", lw=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("measured Δc2 of the field")
    ax.set_title("Ramp shape is the biggest\nsingle lever", fontsize=10.5, loc="left")

    # ---------- row 3: per-substrate paired effect
    ax = fig.add_subplot(gs[2, :])
    ref = arms.get("fixed_sharp", arms.get("fixed"))
    order = ref.sort_values("effect_dc2")["subject"].tolist()
    w = 0.8 / len(arms)
    colors = {"baseline": "#c0392b", "fixed": "#9ecae1", "fixed_sharp": "#1f77b4"}
    labels = {"baseline": "baseline (old construction)", "fixed": "fixed, linear ramp",
              "fixed_sharp": "fixed + smoothstep_mid ramp"}
    x = np.arange(len(order))
    for k, name in enumerate([n for n in ("baseline", "fixed", "fixed_sharp") if n in arms]):
        d = arms[name].set_index("subject").reindex(order)
        ax.bar(x + k * w - 0.4 + w / 2, d["effect_dc2"], width=w,
               color=colors[name], label=labels[name])
    ax.axhline(0, color="k", lw=0.8)
    req = ref["requested"].iloc[0]
    ax.axhline(req, color="crimson", ls="--", lw=1.1)
    ax.text(len(order) - 0.5, req, " requested", color="crimson", fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=22, ha="right", fontsize=9.5)
    ax.set_ylabel("paired effect  ½·[Δc2(fwd) − Δc2(rev)]")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    parts = []
    for name in [n for n in ("baseline", "fixed", "fixed_sharp") if n in arms]:
        e = arms[name]["effect_dc2"]
        se = e.std(ddof=1) / np.sqrt(len(e))
        parts.append(f"{labels[name]}: {e.mean():+.3f} ± {se:.3f} ({e.mean()/req*100:.0f}%)")
    ax.set_title("Measured on the rendered image, paired polarity so any fixed left/right "
                 "asymmetry cancels\n" + "   |   ".join(parts), fontsize=10.5, loc="left")

    fig.suptitle("Spatial control of multifractality — the blocker was the conditioning field, "
                 "not the 8-bit encoder and not the ControlNet", fontsize=14, y=0.995)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
