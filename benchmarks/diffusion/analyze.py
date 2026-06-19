"""
Measure multifractality (c1, c2) of every diffusion-generated image, build the
report figures, and emit a comparison against the procedural counterpart.

Inputs:
  out/manifest.csv           # written by generate.py
  out/<family>/*.png         # diffusion images, 1024x1024 RGB

Outputs:
  out/analysis.csv           # per-image c1, c2, std
  out/report/per_family_c2.png
  out/report/domain_c2.png
  out/report/diffusion_vs_procedural.png
  out/report/REPORT.md       # human-readable summary
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mfractal as mf

DOMAIN = {
    "rock": ["marble", "agate", "weathered", "granite", "vesicular", "turbulent",
             "schist", "serpentinite", "flow_banded", "convoluted", "gneiss"],
    "cloud": ["cascade_lognormal", "billow", "cirrus",
              "cloud_multifractional", "warped_fbm", "ridged", "billow_smoke",
              "universal_cascade", "prescribed_cascade", "de_jong_attractor"],
    "fluid": ["curl_weave", "eddies", "vorticity", "dye_diffusion",
              "rheoscopic", "wave_interference"],
    "bark":  ["burled_oak", "riven_oak"],
    "fire":  ["firestorm", "lava_pool", "volcanic_fissure"],
}
FAM2DOM = {f: d for d, fs in DOMAIN.items() for f in fs}
DOMAIN_COLORS = {"rock": "#8c564b", "cloud": "#1f77b4", "fluid": "#17becf",
                 "bark": "#7f5d2f", "fire": "#d62728"}


def load_gray(path: Path, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size and (img.width != size or img.height != size):
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main(out_dir: Path):
    out_dir = Path(out_dir)
    manifest = pd.read_csv(out_dir / "manifest.csv")
    rows = []
    for k, r in manifest.iterrows():
        p = out_dir / r["path"]
        if not p.exists():
            continue
        img = load_gray(p, size=512)   # quantify at 512 (consistent with procedural sweep)
        try:
            wl = mf.wavelet_leaders_2d(img)
            c1, c2 = float(wl["c1"]), float(wl["c2"])
        except Exception as e:
            c1, c2 = float("nan"), float("nan")
        rows.append(dict(family=r["family"], domain=FAM2DOM[r["family"]],
                         prompt_idx=r["prompt_idx"], seed=r["seed"],
                         path=r["path"], c1=c1, c2=c2,
                         mean=float(img.mean()), std=float(img.std())))
        if (k + 1) % 20 == 0:
            print(f"  measured {k+1}/{len(manifest)}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "analysis.csv", index=False)
    print(f"Wrote {out_dir/'analysis.csv'}  ({len(df)} rows)")

    report = out_dir / "report"
    report.mkdir(exist_ok=True)

    # ---- Figure 1: per-family c2 distribution ----
    fams = [f for d in DOMAIN for f in DOMAIN[d]]
    fig, ax = plt.subplots(figsize=(16, 6))
    positions = list(range(len(fams)))
    bp_data = [df.loc[df.family == f, "c2"].dropna().values for f in fams]
    parts = ax.violinplot(bp_data, positions=positions, showmeans=True,
                          showextrema=False, widths=0.85)
    for i, f in enumerate(fams):
        c = DOMAIN_COLORS[FAM2DOM[f]]
        parts['bodies'][i].set_facecolor(c)
        parts['bodies'][i].set_alpha(0.55)
        parts['bodies'][i].set_edgecolor("k")
    parts['cmeans'].set_color("k")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(fams, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("c2 (intermittency)  —  more negative = more multifractal")
    ax.set_title("Diffusion-generated natural textures: per-family c2 distribution (n=10 / family)")
    # domain legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55) for c in DOMAIN_COLORS.values()]
    ax.legend(handles, list(DOMAIN_COLORS.keys()), loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(report / "per_family_c2.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {report/'per_family_c2.png'}")

    # ---- Figure 2: domain-level c2 distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    dom_data = [df.loc[df.domain == d, "c2"].dropna().values for d in DOMAIN]
    parts = ax.violinplot(dom_data, positions=range(len(DOMAIN)),
                          showmeans=True, showextrema=False, widths=0.7)
    for i, d in enumerate(DOMAIN):
        parts['bodies'][i].set_facecolor(DOMAIN_COLORS[d])
        parts['bodies'][i].set_alpha(0.55)
        parts['bodies'][i].set_edgecolor("k")
    parts['cmeans'].set_color("k")
    ax.set_xticks(range(len(DOMAIN)))
    ax.set_xticklabels(list(DOMAIN))
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel("c2")
    ax.set_title("Diffusion textures: c2 by domain")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(report / "domain_c2.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {report/'domain_c2.png'}")

    # ---- Figure 3: diffusion vs procedural per-family means ----
    # Sample the procedural counterpart at cx=0.5, seed in [0,1,2], n=512.
    proc_means = {}
    for f in fams:
        c2s = []
        for s in (0, 1, 2):
            try:
                img = mf.generate(f, n=512, complexity=0.5, seed=s)
                wl = mf.wavelet_leaders_2d(img)
                c2s.append(float(wl["c2"]))
            except Exception:
                pass
        proc_means[f] = float(np.mean(c2s)) if c2s else float("nan")
    diff_means = {f: float(df.loc[df.family == f, "c2"].mean()) for f in fams}

    fig, ax = plt.subplots(figsize=(7, 7))
    for f in fams:
        if not (np.isfinite(proc_means[f]) and np.isfinite(diff_means[f])):
            continue
        c = DOMAIN_COLORS[FAM2DOM[f]]
        ax.scatter(proc_means[f], diff_means[f], c=c, s=70, edgecolor="k", lw=0.5)
        ax.annotate(f, (proc_means[f], diff_means[f]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points", color=c)
    lim_lo = min(min(proc_means.values()), min(diff_means.values()))
    lim_hi = max(max(proc_means.values()), max(diff_means.values()))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("c2  —  procedural (cx=0.5, seed avg)")
    ax.set_ylabel("c2  —  diffusion (10-image mean)")
    ax.set_title("Per-family c2: procedural vs diffusion")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(report / "diffusion_vs_procedural.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {report/'diffusion_vs_procedural.png'}")

    # ---- markdown report ----
    md = []
    md.append("# Diffusion natural-texture multifractality report")
    md.append("")
    md.append(f"- Model: **SG161222/RealVisXL_V4.0** (SDXL fine-tune, photorealism)")
    md.append(f"- Resolution: 1024 x 1024  (measured at 512 x 512 for c2 consistency with procedural sweep)")
    md.append(f"- 28 families x 10 images = {len(df)} stimuli")
    md.append(f"- Per family: 5 prompt variations x 2 seeds")
    md.append("")
    md.append("## c2 summary (mean across 10 images per family)")
    md.append("")
    md.append("```")
    md.append(f"{'family':<24s} {'domain':<6s} {'c2_mean':>9s} {'c2_std':>8s} {'proc_c2':>9s} {'delta':>8s}")
    for d, fs in DOMAIN.items():
        for f in fs:
            c2m = df.loc[df.family == f, "c2"].mean()
            c2s = df.loc[df.family == f, "c2"].std()
            pc = proc_means[f]
            delta = c2m - pc
            md.append(f"{f:<24s} {d:<6s} {c2m:>9.3f} {c2s:>8.3f} {pc:>9.3f} {delta:>+8.3f}")
    md.append("```")
    md.append("")
    md.append("## Figures")
    md.append("")
    md.append("- `report/per_family_c2.png` -- per-family c2 distribution (violins, colored by domain)")
    md.append("- `report/domain_c2.png` -- c2 by domain")
    md.append("- `report/diffusion_vs_procedural.png` -- diffusion mean c2 vs procedural mean c2 per family")
    md.append("")
    (report / "REPORT.md").write_text("\n".join(md))
    print(f"Wrote {report/'REPORT.md'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/diffusion/out")
    args = ap.parse_args()
    main(Path(args.out))
