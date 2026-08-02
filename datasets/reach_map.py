"""
Map the reachable region of the (c1, c2) plane for the PROCEDURAL generators,
per domain, so a crossed c1 x c2 pareidolia design can be built on cells that
actually exist.

Why this script exists
----------------------
datasets/pareidolia/ manipulates c2 cleanly but leaves c1 -- the perceptually
dominant axis -- as a large nuisance variable. The plan is to EXTEND that set to
a crossed domain x c1 x c2 design. That is only possible in regions of the
(c1, c2) plane the generators can actually reach, and c1/c2 are known to be
negatively coupled in natural texture. This script measures the reachable region
rather than assuming it.

Design decisions (measured, not assumed -- see MEASUREMENT NOTES below)
----------------------------------------------------------------------
* The pool is generated and measured at **n=512**, not 256/384. Reason: the
  generators are not scale-invariant (the same seed produces a *different field*
  at a different n) and mf.wavelet_leaders_2d chooses its octave count from the
  image size (J = log2(n) - 3, so J=5 at n=256/384 but J=6 at n=512, with fit
  ranges j=2..4 vs j=2..5). Both effects shift the measured cumulants. Empirically
  the n=384 -> n=512 shift reaches |d_c1| = 0.31 and |d_c2| = 0.18 on single
  seeds -- larger than the cell tolerance we want to use. n=512 only costs ~1.7x
  n=384, and it is the size the existing dataset was built at, so the pool is
  measured in exactly the coordinates the final stimuli will live in. Run
  `--stage check` to reproduce that comparison.
* generate() + wavelet_leaders_2d() are deterministic: regenerating manifest rows
  reproduces their stored c1/c2 to 4 decimals. So a pool row IS the stimulus.
* Three families listed in build_pareidolia.CURATED no longer exist in mfractal
  (`stratified`, `cloud_mrw`, `choppy` -> AttributeError). They are excluded and
  reported.

Stages
------
  python datasets/reach_map.py --stage sweep     # build datasets/reach_pool.csv
  python datasets/reach_map.py --stage check     # n=256/384 vs 512 stability
  python datasets/reach_map.py --stage analyse   # grid feasibility + figure

Outputs
-------
  datasets/reach_pool.csv
  datasets/reach_size_check.csv
  benchmarks/figures/74_procedural_reachable.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

POOL_CSV = ROOT / "datasets" / "reach_pool.csv"
CHECK_CSV = ROOT / "datasets" / "reach_size_check.csv"
FIG_PATH = ROOT / "benchmarks" / "figures" / "74_procedural_reachable.png"

# --- Curated families, from datasets/build_pareidolia.py, minus the three that
#     no longer resolve in mfractal (verified by calling mf.generate on them). ---
CURATED_DECLARED = {
    "rock": ["marble", "agate", "weathered", "granite", "vesicular", "turbulent",
             "schist", "serpentinite", "flow_banded", "convoluted", "gneiss"],
    "cloud": ["cascade_lognormal", "stratified", "billow", "cirrus",
              "cloud_mrw", "cloud_multifractional", "warped_fbm", "ridged"],
    "fluid": ["curl_weave", "choppy", "eddies", "vorticity",
              "dye_diffusion", "rheoscopic"],
}
DEAD_FAMILIES = ["stratified", "cloud_mrw", "choppy"]  # AttributeError in mfractal
CURATED = {d: [f for f in fams if f not in DEAD_FAMILIES]
           for d, fams in CURATED_DECLARED.items()}

# Families whose per-image generation cost is ~100x the rest (measured: ~300 s at
# n=512 vs 1-17 s). They also take no complexity argument, so extra samples buy
# only seed jitter -- give them fewer seeds.
SLOW_FAMILIES = {"eddies", "vorticity"}


def takes_cx(family: str) -> bool:
    """Same helper as build_pareidolia.takes_cx."""
    return (family in mf.ROCK_FAMILIES or family in mf.CLOUD_FAMILIES
            or family in mf.CX_FLUIDS)


def domain_of(family: str) -> str:
    for d, fams in CURATED.items():
        if family in fams:
            return d
    return "?"


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
def _measure(job):
    """Generate one field and measure its wavelet-leader cumulants."""
    family, cx, seed, n = job
    t0 = time.time()
    try:
        img = mf.generate(family, n=n, complexity=cx, seed=seed)
        w = mf.wavelet_leaders_2d(img)
        return dict(domain=domain_of(family), family=family,
                    complexity=("" if cx is None else f"{cx:.3f}"),
                    seed=seed, n=n,
                    c1=float(w["c1"]), c2=float(w["c2"]),
                    mean=float(img.mean()), std=float(img.std()),
                    gen_s=time.time() - t0, error="")
    except Exception as e:  # noqa: BLE001
        return dict(domain=domain_of(family), family=family,
                    complexity=("" if cx is None else f"{cx:.3f}"),
                    seed=seed, n=n, c1=float("nan"), c2=float("nan"),
                    mean=float("nan"), std=float("nan"),
                    gen_s=time.time() - t0, error=f"{type(e).__name__}: {e}")


FIELDS = ["domain", "family", "complexity", "seed", "n", "c1", "c2",
          "mean", "std", "gen_s", "error"]


def _run_jobs(jobs, out_csv, workers, label):
    """Run jobs in a process pool, writing rows incrementally (slowest first)."""
    t0 = time.time()
    done = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_measure, j) for j in jobs]
            for fut in as_completed(futs):
                row = fut.result()
                w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                            for k in FIELDS})
                done += 1
                if done % 100 == 0 or done == len(jobs):
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    eta = (len(jobs) - done) / max(rate, 1e-9)
                    print(f"  [{label}] {done}/{len(jobs)}  elapsed={el:6.0f}s  eta={eta:6.0f}s",
                          flush=True)
                fh.flush()
    return time.time() - t0


# ----------------------------------------------------------------------------
# stage: sweep
# ----------------------------------------------------------------------------
def stage_sweep(n, cx_grid, seeds, slow_seeds, workers):
    jobs = []
    for domain, fams in CURATED.items():
        for fam in fams:
            fam_seeds = seeds[:slow_seeds] if fam in SLOW_FAMILIES else seeds
            if takes_cx(fam):
                for cx in cx_grid:
                    for s in fam_seeds:
                        jobs.append((fam, float(cx), int(s), n))
            else:
                for s in fam_seeds:
                    jobs.append((fam, None, int(s), n))
    # slowest families first so they do not straggle at the end
    jobs.sort(key=lambda j: (j[0] not in SLOW_FAMILIES,
                             j[0] not in {"billow", "curl_weave", "dye_diffusion", "rheoscopic"}))
    print(f"Sweep: {len(jobs)} candidates at n={n}, {workers} workers")
    print(f"  families: {sum(len(v) for v in CURATED.values())} "
          f"({', '.join(f'{d}={len(v)}' for d, v in CURATED.items())})")
    print(f"  excluded (do not exist in mfractal): {', '.join(DEAD_FAMILIES)}")
    wall = _run_jobs(jobs, POOL_CSV, workers, "sweep")
    print(f"Sweep wall time: {wall:.0f}s -> {POOL_CSV}")


# ----------------------------------------------------------------------------
# stage: check (size stability)
# ----------------------------------------------------------------------------
def stage_check(cx_grid, workers, per_family_seeds=2):
    """Re-measure the same (family, cx, seed) triples at n=256, 384, 512."""
    fams = [f for d in CURATED for f in CURATED[d] if f not in SLOW_FAMILIES]
    cxs = [cx_grid[0], cx_grid[len(cx_grid) // 2], cx_grid[-1]]
    jobs = []
    for fam in fams:
        for cx in (cxs if takes_cx(fam) else [None]):
            for s in range(per_family_seeds):
                for n in (256, 384, 512):
                    jobs.append((fam, (None if cx is None else float(cx)), s, n))
    print(f"Size check: {len(jobs)} measurements ({len(jobs)//3} triples x 3 sizes)")
    wall = _run_jobs(jobs, CHECK_CSV, workers, "check")
    print(f"Check wall time: {wall:.0f}s -> {CHECK_CSV}")
    report_check()


def report_check():
    import pandas as pd
    df = pd.read_csv(CHECK_CSV)
    df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    df["cxk"] = df["complexity"].fillna(-1).round(3)
    key = ["family", "cxk", "seed"]
    piv = df.pivot_table(index=key, columns="n", values=["c1", "c2"])
    print("\n--- SIZE STABILITY: measured (c1,c2) at n=256 / 384 vs n=512 ---")
    print("(same family+complexity+seed; the generator is NOT scale-invariant,")
    print(" and the estimator uses J=5 octaves at 256/384 but J=6 at 512)")
    for small in (256, 384):
        for v in ("c1", "c2"):
            a = piv[(v, small)].to_numpy(float)
            b = piv[(v, 512)].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            a, b = a[m], b[m]
            d = b - a
            r = np.corrcoef(a, b)[0, 1]
            print(f"  n={small} -> 512  {v}:  n={len(a)}  bias(mean d)={d.mean():+.4f}  "
                  f"sd(d)={d.std(ddof=1):.4f}  max|d|={np.abs(d).max():.4f}  r={r:.4f}")
    return piv


# ----------------------------------------------------------------------------
# stage: analyse
# ----------------------------------------------------------------------------
def load_pool():
    import pandas as pd
    df = pd.read_csv(POOL_CSV)
    bad = df[~(df["error"].isna() | (df["error"].astype(str) == ""))]
    if len(bad):
        print(f"  {len(bad)} generation errors:")
        print(bad.groupby("family")["error"].first().to_string())
    df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    df = df[np.isfinite(df["c1"]) & np.isfinite(df["c2"])].copy()
    return df


# ---------------------------------------------------------------------------
# RECOMMENDED DESIGN -- derived from the measured pool, not chosen a priori.
#
# c1 levels are DOMAIN-SPECIFIC because the domains occupy disjoint c1 bands:
# rock lives at c1 ~1.0-1.7, cloud at ~1.6-2.3. There is no c1 triple that is
# populated in all three domains at more than 2 c2 levels (measured: a common
# 3 c1 x 3 c2 grid has a min cell count of 0 at every tolerance tried).
# The c2 levels ARE shared between rock and cloud, so c2 stays crossed with
# domain; c1 is crossed within domain with a matched 0.40-wide contrast.
# ---------------------------------------------------------------------------
DESIGN = {
    "rock":  ([1.10, 1.30, 1.50], [-0.30, -0.20, -0.10]),
    "cloud": ([1.75, 1.95, 2.15], [-0.30, -0.20, -0.10]),
    "fluid": ([1.65, 1.85], [-0.16, -0.06]),
}
# rock alone additionally supports a 4th c2 level:
DESIGN_ROCK_EXTENDED = ([1.10, 1.30, 1.50], [-0.38, -0.28, -0.18, -0.08])
FAMILY_CAP = 3  # max stimuli drawn from one family per cell, for diversity


def cell_counts(df, design, tol_c1, tol_c2, family_cap=FAMILY_CAP):
    """Count candidates within tolerance of each (domain, c1, c2) cell."""
    out = []
    for dom, (c1_levels, c2_levels) in design.items():
        sub = df[df["domain"] == dom]
        for a in c1_levels:
            for b in c2_levels:
                m = (np.abs(sub["c1"] - a) <= tol_c1) & (np.abs(sub["c2"] - b) <= tol_c2)
                hit = sub[m]
                vc = hit["family"].value_counts()
                usable = int(sum(min(v, family_cap) for v in vc.values))
                out.append(dict(domain=dom, c1_level=float(a), c2_level=float(b),
                                n_candidates=int(len(hit)),
                                n_families=int(hit["family"].nunique()),
                                n_usable_cap3=usable,
                                families=",".join(sorted(hit["family"].unique()))))
    return out


def c2_gap_report(df):
    """Which c2 bands are simply unreachable in each domain -- the reason cells die."""
    print("\n--- c2 COVERAGE PER DOMAIN (0.05-wide bands; EMPTY = unreachable) ---")
    edges = np.arange(-0.90, 0.001, 0.05)
    for dom in ("rock", "cloud", "fluid"):
        s = df[df["domain"] == dom]
        gaps = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s["c2"] >= lo) & (s["c2"] < hi)
            if m.sum() == 0:
                gaps.append((lo, hi))
        print(f"  {dom:6s} c2 span [{s['c2'].min():+.3f}, {s['c2'].max():+.3f}]  "
              f"empty bands: {gaps if gaps else 'none'}")
        # single-family regions
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s["c2"] >= lo) & (s["c2"] < hi)
            if 0 < m.sum() and s[m]["family"].nunique() == 1:
                fam = s[m]["family"].iloc[0]
                print(f"         [{lo:+.2f},{hi:+.2f}) reachable by ONE family only: {fam} "
                      f"(c1 {s[m]['c1'].min():.2f}-{s[m]['c1'].max():.2f})")


def stage_analyse(c1_levels, c2_levels, tol_c1, tol_c2):
    import pandas as pd
    df = load_pool()
    print(f"\nPool: {len(df)} finite candidates at n={int(df['n'].iloc[0])}")
    print(f"Total generation CPU time: {df['gen_s'].sum()/3600:.2f} core-hours")

    print("\n--- REACHABLE REGION PER DOMAIN ---")
    for dom in ("rock", "cloud", "fluid"):
        s = df[df["domain"] == dom]
        r = np.corrcoef(s["c1"], s["c2"])[0, 1]
        print(f"  {dom:6s} n={len(s):5d}  c1 [{s['c1'].min():+.3f}, {s['c1'].max():+.3f}] "
              f"med={s['c1'].median():+.3f}   c2 [{s['c2'].min():+.3f}, {s['c2'].max():+.3f}] "
              f"med={s['c2'].median():+.3f}   corr(c1,c2)={r:+.3f}")

    print("\n--- PER-FAMILY REACH (c1 range, c2 range across cx x seeds) ---")
    g = df.groupby(["domain", "family"]).agg(
        n=("c1", "size"), c1_lo=("c1", "min"), c1_hi=("c1", "max"),
        c2_lo=("c2", "min"), c2_hi=("c2", "max"))
    print(g.round(3).to_string())

    print("\n--- SEED NOISE at fixed (family, complexity) -> sets usable tolerance ---")
    sd = df.groupby(["family", "complexity"], dropna=False).agg(
        sd_c1=("c1", "std"), sd_c2=("c2", "std"), k=("c1", "size"))
    sd = sd[sd["k"] >= 3]
    print(f"  median within-(family,cx) sd:  c1={sd['sd_c1'].median():.4f}  "
          f"c2={sd['sd_c2'].median():.4f}")
    print(f"  90th pct within-(family,cx) sd: c1={sd['sd_c1'].quantile(.9):.4f}  "
          f"c2={sd['sd_c2'].quantile(.9):.4f}")

    c2_gap_report(df)

    design = dict(DESIGN)
    cells = cell_counts(df, design, tol_c1, tol_c2)
    cdf = pd.DataFrame(cells)
    print(f"\n--- RECOMMENDED DESIGN: CELL FEASIBILITY "
          f"(tol +-{tol_c1} on c1, +-{tol_c2} on c2) ---")
    print("    cells shown as n_candidates / n_families / n_usable(max 3 per family)")
    for dom, (c1l, c2l) in design.items():
        s = cdf[cdf["domain"] == dom]
        print(f"\n  {dom}:")
        print("    c1\\c2   " + "".join(f"{b:>16.2f}" for b in c2l))
        for a in c1l:
            row = f"    {a:5.2f}   "
            for b in c2l:
                r = s[(s["c1_level"] == a) & (s["c2_level"] == b)].iloc[0]
                row += f"{r['n_candidates']:>6d}/{r['n_families']}f/{r['n_usable_cap3']:<2d}   "
            print(row)

    # rock's extra c2 level
    c1l, c2l = DESIGN_ROCK_EXTENDED
    ext = pd.DataFrame(cell_counts(df, {"rock": (c1l, c2l)}, tol_c1, tol_c2))
    print(f"\n  rock EXTENDED to 4 c2 levels {c2l}:")
    print("    c1\\c2   " + "".join(f"{b:>16.2f}" for b in c2l))
    for a in c1l:
        row = f"    {a:5.2f}   "
        for b in c2l:
            r = ext[(ext["c1_level"] == a) & (ext["c2_level"] == b)].iloc[0]
            row += f"{r['n_candidates']:>6d}/{r['n_families']}f/{r['n_usable_cap3']:<2d}   "
        print(row)

    cdf.to_csv(ROOT / "datasets" / "reach_cells.csv", index=False)
    print(f"\nWrote {ROOT / 'datasets' / 'reach_cells.csv'}")
    make_figure(df, design, tol_c1, tol_c2)
    return df, cdf


def make_figure(df, design, tol_c1, tol_c2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        ConvexHull = None

    domains = ["rock", "cloud", "fluid"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2), sharex=True, sharey=True)
    c1lo, c1hi = 0.6, 3.4
    c2lo, c2hi = -1.05, 0.10

    for ax, dom in zip(axes, domains):
        s = df[df["domain"] == dom]
        ax.hexbin(s["c1"], s["c2"], gridsize=48, cmap="Blues", mincnt=1,
                  extent=(c1lo, c1hi, c2lo, c2hi), linewidths=0.15, alpha=0.85)
        # convex hull of the reachable region
        if ConvexHull is not None and len(s) > 3:
            pts = s[["c1", "c2"]].to_numpy()
            pts = pts[(pts[:, 1] > c2lo) & (pts[:, 0] < c1hi) & (pts[:, 0] > c1lo)]
            try:
                h = ConvexHull(pts)
                poly = np.vstack([pts[h.vertices], pts[h.vertices][:1]])
                ax.plot(poly[:, 0], poly[:, 1], color="k", lw=1.3, alpha=0.6,
                        label="reachable hull")
            except Exception:
                pass
        # empty c2 bands -- the reason cells die
        edges = np.arange(-0.90, 0.001, 0.05)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if ((s["c2"] >= lo) & (s["c2"] < hi)).sum() == 0:
                ax.axhspan(lo, hi, color="tab:red", alpha=0.13, zorder=0)
        c1l, c2l = design[dom]
        for a in c1l:
            for b in c2l:
                m = (np.abs(s["c1"] - a) <= tol_c1) & (np.abs(s["c2"] - b) <= tol_c2)
                hit = s[m]
                k = int(m.sum())
                nf = hit["family"].nunique()
                col = ("tab:green" if (k >= 8 and nf >= 2)
                       else ("tab:orange" if k >= 3 else "tab:red"))
                ax.add_patch(Rectangle((a - tol_c1, b - tol_c2), 2 * tol_c1, 2 * tol_c2,
                                       fill=False, ec=col, lw=2.2, zorder=6))
                ax.text(a, b, f"{k}\n{nf}f", ha="center", va="center",
                        fontsize=7.5, color=col, zorder=7, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.82))
        r = np.corrcoef(s["c1"], s["c2"])[0, 1]
        ax.set_title(f"{dom}   n={len(s)}   corr(c1,c2)={r:+.2f}\n"
                     f"grid: {len(c1l)} c1 x {len(c2l)} c2", fontsize=11)
        ax.set_xlabel("c1   (mean Holder exponent = roughness;  FD = 3 - c1)")
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("c2   (intermittency; 0 = monofractal, more negative = more multifractal)")
    axes[0].set_xlim(c1lo, c1hi)
    axes[0].set_ylim(c2lo, c2hi)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Procedurally reachable region in the (c1, c2) plane, per domain — pool of "
        f"{len(df)} fields measured at n=512\n"
        f"boxes = recommended design cells (±{tol_c1} c1 × ±{tol_c2} c2), label = "
        "n_candidates / n_families;  green = fillable (≥8 cands, ≥2 families)\n"
        "red bands = c2 values NO family in that domain can produce — this is why a "
        "3-domain crossed grid fails outside c2 ∈ [−0.20, 0]",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="analyse", choices=["sweep", "check", "analyse"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--complexities", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--slow-seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--c1-levels", default="1.00,1.45,1.90")
    ap.add_argument("--c2-levels", default="-0.05,-0.15,-0.30,-0.55")
    ap.add_argument("--tol-c1", type=float, default=0.10)
    ap.add_argument("--tol-c2", type=float, default=0.05)
    args = ap.parse_args()

    cx_grid = [float(x) for x in args.complexities.split(",")]
    seeds = list(range(args.seeds))
    c1_levels = [float(x) for x in args.c1_levels.split(",")]
    c2_levels = [float(x) for x in args.c2_levels.split(",")]

    if args.stage == "sweep":
        stage_sweep(args.n, cx_grid, seeds, args.slow_seeds, args.workers)
    elif args.stage == "check":
        stage_check(cx_grid, args.workers)
    else:
        stage_analyse(c1_levels, c2_levels, args.tol_c1, args.tol_c2)


if __name__ == "__main__":
    main()
