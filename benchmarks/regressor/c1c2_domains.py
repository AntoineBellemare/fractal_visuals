"""
c1c2_domains.py — the natural (c1, c2) structure of each texture domain.

Reads the corpus manifest and plots c1 vs c2 coloured by domain (water, ice, fire, rock,
cloud, fluid, ...), with a per-domain regression line. Shows where each texture type lives
in the cumulant plane and how strongly c1 and c2 co-vary *within* it — the backdrop for the
control experiment (c1c2_coupling.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# family -> domain (diffusion/procedural families); macro folders + sources handled below.
DOMAIN = {
    "rock": ["marble", "agate", "weathered", "granite", "vesicular", "turbulent", "schist",
             "serpentinite", "flow_banded", "convoluted", "gneiss", "levy_marble", "breccia",
             "cellular_stone", "concentric", "veined"],
    "cloud": ["cascade_lognormal", "stratified", "billow", "cirrus", "cloud_mrw",
              "cloud_multifractional", "warped_fbm", "ridged", "billow_smoke"],
    "fluid": ["curl_weave", "choppy", "eddies", "vorticity", "dye_diffusion", "rheoscopic",
              "wave_interference"],
    "bark": ["burled_oak", "riven_oak"],
    "fire": ["firestorm", "lava_pool", "volcanic_fissure"],
    "metal": ["rust_bloom", "rust_pitted", "rust_dewy"],
}
FAM2DOM = {f: d for d, fs in DOMAIN.items() for f in fs}
COLORS = {"water": "#1f77b4", "ice": "#17becf", "lichen": "#2ca02c", "fire": "#d62728",
          "rock": "#8c564b", "cloud": "#9467bd", "fluid": "#0aa6c2", "bark": "#7f5d2f",
          "metal": "#bcbd22"}


def domain_of(row):
    src, grp = row["source"], str(row["group"])
    parts = grp.split(":")
    if src == "macro":            # macro:SymphonieArctique_IceLora -> ice/water/lichen
        name = parts[-1].lower()
        for key in ("ice", "water", "lichen"):
            if key in name:
                return key
        return "macro"
    fam = parts[1] if len(parts) > 1 else ""
    return FAM2DOM.get(fam)        # None for prescribed/scaffold/unmapped


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "corpus" / "manifest.csv"))
    ap.add_argument("--sources", default="diffusion,macro",
                    help="which corpus sources to include (real-ish textures by default)")
    ap.add_argument("--out", default=str(HERE / "c1c2_domains.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["source"].isin(args.sources.split(","))].copy()
    df = df[(df.c2 > -1.6) & (df.c2 < 0.3) & (df.c1 > 0.3) & (df.c1 < 2.6)]   # drop artifacts
    df["domain"] = df.apply(domain_of, axis=1)
    df = df[df["domain"].notna()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    print(f"{'domain':8s} {'n':>4} {'c1_mean':>8} {'c2_mean':>8} {'corr':>6} {'slope dc2/dc1':>13}")
    summary = []
    for dom in sorted(df["domain"].unique(), key=lambda d: -len(df[df.domain == d])):
        sub = df[df.domain == dom]
        if len(sub) < 4:
            continue
        c = COLORS.get(dom, "#888888")
        ax.scatter(sub.c1, sub.c2, s=14, alpha=0.45, color=c, edgecolor="none")
        # per-domain regression dc2/dc1
        slope, intc = np.polyfit(sub.c1, sub.c2, 1)
        r = float(np.corrcoef(sub.c1, sub.c2)[0, 1])
        xs = np.array([sub.c1.min(), sub.c1.max()])
        ax.plot(xs, slope * xs + intc, color=c, lw=2,
                label=f"{dom} (n={len(sub)}, r={r:+.2f}, slope={slope:+.2f})")
        print(f"{dom:8s} {len(sub):4d} {sub.c1.mean():8.3f} {sub.c2.mean():8.3f} "
              f"{r:6.2f} {slope:13.2f}")
        summary.append((dom, len(sub), r, slope))
    ax.set_xlabel("c1  (mean roughness / detail density — higher = smoother)")
    ax.set_ylabel("c2  (intermittency — more negative = more clustered)")
    ax.set_title("Natural c1–c2 structure by texture domain\n(corpus: " + args.sources + ")")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {args.out}")
    print("Read: a domain with steep negative slope means making it busier (lower c1) also "
          "makes it more intermittent (lower c2) — the axes are coupled there; slope~0 means "
          "c1 and c2 are independent for that texture.")


if __name__ == "__main__":
    main()
