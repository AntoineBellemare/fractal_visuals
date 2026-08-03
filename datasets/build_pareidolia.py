"""
Build a c2-balanced pareidolia stimulus dataset across the curated 25 families
(11 rocks + 8 clouds + 6 fluids).

Strategy
--------
1. Generate a candidate pool: curated families x complexity grid x seeds (or seeds
   only for NS sims). Measure c2 via mfractal.wavelet_leaders_2d on each candidate.
2. Bin candidates by measured c2 into 5 bins spanning monofractal -> strongly
   multifractal. Per (domain, bin), select up to N stimuli with family diversity
   (round-robin across families before allowing repeats).
3. Save PNGs at the target resolution + a manifest CSV + a montage figure.

Default config: n=512, 4 seeds, 5 complexities, ~144 stimuli (8 per bin per
domain, 5 bins, 3 domains, plus the monofractal anchors).

Outputs (committed to repo):
  datasets/pareidolia/images/<stim_id>.png
  datasets/pareidolia/manifest.csv
  datasets/pareidolia/montage.png
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf


# --- Curated set (same as benchmarks/make_figures.py) ---
CURATED = {
    "rock": ["marble", "agate", "weathered", "granite", "vesicular", "turbulent",
             "schist", "serpentinite", "flow_banded", "convoluted", "gneiss"],
    "cloud": ["cascade_lognormal", "billow", "cirrus",
              "cloud_multifractional", "warped_fbm", "ridged"],
    "fluid": ["curl_weave", "eddies", "vorticity",
              "dye_diffusion", "rheoscopic"],
}
CONTROLS = {"gneiss", "warped_fbm"}

# Fail fast and loudly if this list ever drifts onto a retired family again.
# `stratified`, `cloud_mrw` and `choppy` sat here unnoticed after commit af17c50
# pruned them, and 17 of the 106 shipped stimuli were built from names that no
# longer generate. See mfractal/retired.py.
for _dom, _fams in CURATED.items():
    mf.validate_families(_fams)

# c2 bins: monofractal-near-0 -> strong-multifractal.
# Anything below -0.5 we put in the "extreme" bin; above 0 (rare) goes into the
# monofractal bin. Bin edges are inclusive on the lower side.
C2_BIN_EDGES = [-1.5, -0.30, -0.18, -0.10, -0.04, 0.05]
C2_BIN_NAMES = ["strong", "moderate", "mild", "weak", "monofractal"]

OUT_DIR = ROOT / "datasets" / "pareidolia"
IMG_DIR = OUT_DIR / "images"


def takes_cx(family):
    return family in mf.ROCK_FAMILIES or family in mf.CLOUD_FAMILIES or family in mf.CX_FLUIDS


def assign_bin(c2: float) -> int:
    """Return the index of the c2 bin (0=strong .. 4=monofractal)."""
    if not np.isfinite(c2):
        return -1
    for k, lo in enumerate(C2_BIN_EDGES[:-1]):
        hi = C2_BIN_EDGES[k + 1]
        if c2 >= lo and c2 < hi:
            return k
    # very negative (below -1.5) -> still "strong"
    if c2 < C2_BIN_EDGES[0]:
        return 0
    return -1


def generate_pool(n: int, cx_grid: list[float], seeds: list[int]) -> list[dict]:
    """Sweep curated families and return candidate stimuli with measured c2."""
    pool = []
    started = time.time()
    for domain, families in CURATED.items():
        for fam in families:
            cx_list = cx_grid if takes_cx(fam) else [None]
            for cx in cx_list:
                for seed in seeds:
                    t0 = time.time()
                    try:
                        if cx is None:
                            img = mf.generate(fam, n=n, complexity=None, seed=seed)
                        else:
                            img = mf.generate(fam, n=n, complexity=float(cx), seed=seed)
                    except Exception as e:
                        print(f"  [gen-fail] {fam} cx={cx} seed={seed}: {e}")
                        continue
                    try:
                        wl = mf.wavelet_leaders_2d(img)
                        c1 = float(wl["c1"]); c2 = float(wl["c2"])
                    except Exception as e:
                        c1, c2 = float("nan"), float("nan")
                    bin_i = assign_bin(c2)
                    pool.append(dict(
                        domain=domain, family=fam,
                        complexity=cx, seed=seed, n=n,
                        c1=c1, c2=c2, c2_bin=bin_i,
                        bin_name=(C2_BIN_NAMES[bin_i] if 0 <= bin_i < len(C2_BIN_NAMES) else "out"),
                        mean=float(img.mean()), std=float(img.std()),
                        img=img,
                        gen_s=time.time() - t0,
                    ))
            print(f"  [{domain:5s}] {fam:22s} pool={len(pool):4d}  t={time.time()-started:5.0f}s")
    return pool


def select_balanced(pool: list[dict], per_bin_per_domain: int = 8,
                    max_per_family_per_bin: int = 2) -> list[dict]:
    """Round-robin selection from each (domain, bin) cell, prioritising family diversity."""
    chosen = []
    seen_ids = set()
    for domain in CURATED:
        for bin_i in range(len(C2_BIN_NAMES)):
            candidates = [r for r in pool if r["domain"] == domain and r["c2_bin"] == bin_i]
            if not candidates:
                continue
            by_family = {}
            for r in candidates:
                by_family.setdefault(r["family"], []).append(r)
            # sort within each family by abs distance to bin center for "cleanest"
            bin_lo, bin_hi = C2_BIN_EDGES[bin_i], C2_BIN_EDGES[bin_i + 1]
            center = 0.5 * (bin_lo + bin_hi)
            for fam in by_family:
                by_family[fam].sort(key=lambda r: abs(r["c2"] - center))
            family_iters = {fam: iter(rs) for fam, rs in by_family.items()}
            count_per_family = {fam: 0 for fam in by_family}
            took = 0
            while took < per_bin_per_domain:
                progressed = False
                # iterate families in a deterministic order
                for fam in sorted(by_family):
                    if count_per_family[fam] >= max_per_family_per_bin:
                        continue
                    try:
                        r = next(family_iters[fam])
                    except StopIteration:
                        continue
                    chosen.append(r)
                    count_per_family[fam] += 1
                    took += 1
                    progressed = True
                    if took >= per_bin_per_domain:
                        break
                if not progressed:
                    break
    return chosen


def save_dataset(chosen: list[dict]):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, r in enumerate(chosen):
        disp = mf.punch(r["img"])  # 0..1
        u8 = np.clip(disp * 255, 0, 255).astype(np.uint8)
        stim_id = f"{r['domain'][0]}{k:03d}_{r['family']}"
        path = IMG_DIR / f"{stim_id}.png"
        Image.fromarray(u8, mode="L").save(path, optimize=True)
        cx = "" if r["complexity"] is None else f"{r['complexity']:.2f}"
        rows.append(dict(
            stim_id=stim_id,
            image_path=str(path.relative_to(OUT_DIR)),
            domain=r["domain"], family=r["family"], complexity=cx, seed=r["seed"], n=r["n"],
            c1=f"{r['c1']:.4f}", c2=f"{r['c2']:.4f}",
            c2_bin=r["bin_name"], control=("yes" if r["family"] in CONTROLS else "no"),
            mean=f"{r['mean']:.4f}", std=f"{r['std']:.4f}",
        ))
    manifest = OUT_DIR / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} stimuli to {IMG_DIR.relative_to(ROOT)}")
    print(f"Wrote {manifest.relative_to(ROOT)}")
    return rows


def save_montage(rows: list[dict]):
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["domain"], r["c2_bin"]), []).append(r)
    domains = ["rock", "cloud", "fluid"]
    bins = C2_BIN_NAMES  # left-to-right: strong .. monofractal
    fig, axes = plt.subplots(len(domains), len(bins),
                             figsize=(2.4 * len(bins), 2.6 * len(domains)),
                             squeeze=False)
    for di, dom in enumerate(domains):
        for bi, name in enumerate(bins):
            ax = axes[di][bi]
            cell = by_cell.get((dom, name), [])
            if not cell:
                ax.set_axis_off(); continue
            # Show the median-c2 image from this cell, with family label
            cell_sorted = sorted(cell, key=lambda r: float(r["c2"]))
            r = cell_sorted[len(cell_sorted) // 2]
            img = np.asarray(Image.open(OUT_DIR / r["image_path"]))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r['family']}\nc2={float(r['c2']):+.2f}  cx={r['complexity'] or 'NS'}", fontsize=8)
            if bi == 0:
                ax.set_ylabel(dom, fontsize=11, rotation=0, ha="right", va="center", labelpad=24)
    for bi, name in enumerate(bins):
        axes[0][bi].set_xlabel("")
        axes[0][bi].xaxis.set_label_position("top")
        bin_lo, bin_hi = C2_BIN_EDGES[bi], C2_BIN_EDGES[bi + 1]
        axes[0][bi].xaxis.set_label_text(f"{name}\n[{bin_lo:.2f}, {bin_hi:.2f})")
    fig.suptitle("pareidolia dataset montage — one sample per (domain, c2 bin) cell", y=1.005, fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "montage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--complexities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--max-per-family-per-bin", type=int, default=2)
    args = ap.parse_args()

    cx_grid = [float(x) for x in args.complexities.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    started = time.time()

    print(f"Building candidate pool at n={args.n} ...")
    pool = generate_pool(args.n, cx_grid, seeds)
    pool_with_c2 = [r for r in pool if np.isfinite(r["c2"])]
    print(f"\nPool: {len(pool_with_c2)} candidates with finite c2 "
          f"(out of {len(pool)})  t={time.time()-started:.0f}s")

    # Bin distribution
    print("\nBin distribution per (domain, bin):")
    print(f"  {'domain':6s} " + " ".join(f"{b:11s}" for b in C2_BIN_NAMES))
    for dom in CURATED:
        counts = [sum(1 for r in pool_with_c2 if r["domain"] == dom and r["c2_bin"] == bi)
                  for bi in range(len(C2_BIN_NAMES))]
        print(f"  {dom:6s} " + " ".join(f"{c:11d}" for c in counts))

    print(f"\nSelecting balanced subset ({args.per_bin}/bin/domain, max {args.max_per_family_per_bin}/fam/bin) ...")
    chosen = select_balanced(pool_with_c2, args.per_bin, args.max_per_family_per_bin)
    print(f"Chosen: {len(chosen)} stimuli")

    rows = save_dataset(chosen)
    save_montage(rows)
    print(f"\nTotal wall time: {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
