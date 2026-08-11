"""
Sweep all 30 families (rocks + clouds + fluids) over complexity and seed, measure
multifractal log-cumulants c1, c2 (wavelet leaders) and Delta_alpha (2D MFDFA),
and write a manifest CSV. This is the empirical input that drives family curation,
benchmark figures, and the pareidolia dataset.

Usage:
    python benchmarks/run_validation.py            # default: n=256, 5 cx levels, 4 seeds
    python benchmarks/run_validation.py --n 256 --complexities 0,0.25,0.5,0.75,1 --seeds 4

Manifest schema:
    domain, family, complexity (or NaN for NS sims), seed, n,
    c1, c2, delta_alpha, alpha0, asymmetry, mean, std,
    gen_s, wl_s, mf_s, status, note
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mfractal as mf


OUT_DIR = Path(__file__).resolve().parent / "results"
MANIFEST = OUT_DIR / "validation_manifest.csv"


def all_families():
    yield from (("rock", f) for f in mf.ROCK_FAMILIES)
    yield from (("cloud", f) for f in mf.CLOUD_FAMILIES)
    yield from (("fluid", f) for f in mf.FLUID_FAMILIES)


def takes_complexity(family: str) -> bool:
    if family in mf.ROCK_FAMILIES:
        return True
    if family in mf.CLOUD_FAMILIES:
        return True
    if family in mf.CX_FLUIDS:
        return True
    return False  # NS sims: eddies, vorticity, plume


def measure(img: np.ndarray) -> dict:
    out = {}
    t0 = time.time()
    try:
        wl = mf.wavelet_leaders_2d(img)
        out["c1"] = float(wl["c1"])
        out["c2"] = float(wl["c2"])
    except Exception as e:
        out["c1"] = float("nan")
        out["c2"] = float("nan")
        out["wl_err"] = str(e)
    out["wl_s"] = time.time() - t0

    t0 = time.time()
    try:
        r = mf.mfdfa_2d(img)
        spec = mf.spectrum_summary(r["q"], r["tau"])
        out["delta_alpha"] = float(spec["delta_alpha"])
        out["alpha0"] = float(spec["alpha0"])
        out["asymmetry"] = float(spec["asymmetry"])
    except Exception as e:
        out["delta_alpha"] = float("nan")
        out["alpha0"] = float("nan")
        out["asymmetry"] = float("nan")
        out["mf_err"] = str(e)
    out["mf_s"] = time.time() - t0
    return out


def run(n: int, complexities: list[float], n_seeds: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    total_families = sum(1 for _ in all_families())
    fi = 0
    for domain, family in all_families():
        fi += 1
        cx_levels = complexities if takes_complexity(family) else [float("nan")]
        for cx in cx_levels:
            for seed in range(n_seeds):
                t0 = time.time()
                try:
                    if np.isnan(cx):
                        img = mf.generate(family, n=n, complexity=None, seed=seed)
                    else:
                        img = mf.generate(family, n=n, complexity=float(cx), seed=seed)
                    gen_s = time.time() - t0
                    status = "ok"
                    note = ""
                except Exception as e:
                    gen_s = time.time() - t0
                    status = "gen_fail"
                    note = repr(e)[:200]
                    rows.append({
                        "domain": domain, "family": family,
                        "complexity": "" if np.isnan(cx) else cx, "seed": seed, "n": n,
                        "c1": "", "c2": "", "delta_alpha": "", "alpha0": "", "asymmetry": "",
                        "mean": "", "std": "",
                        "gen_s": f"{gen_s:.3f}", "wl_s": "", "mf_s": "",
                        "status": status, "note": note,
                    })
                    continue
                m = measure(img)
                row = {
                    "domain": domain, "family": family,
                    "complexity": "" if np.isnan(cx) else cx, "seed": seed, "n": n,
                    "c1": f"{m['c1']:.6f}", "c2": f"{m['c2']:.6f}",
                    "delta_alpha": f"{m['delta_alpha']:.6f}",
                    "alpha0": f"{m['alpha0']:.6f}",
                    "asymmetry": f"{m['asymmetry']:.6f}",
                    "mean": f"{float(img.mean()):.6f}",
                    "std": f"{float(img.std()):.6f}",
                    "gen_s": f"{gen_s:.3f}",
                    "wl_s": f"{m['wl_s']:.3f}",
                    "mf_s": f"{m['mf_s']:.3f}",
                    "status": status, "note": note,
                }
                rows.append(row)
        elapsed = time.time() - started
        eta = elapsed / fi * (total_families - fi)
        print(f"[{fi:2d}/{total_families}] {domain:5s} {family:24s} cumulative={elapsed:6.1f}s  ETA={eta:5.0f}s")

    cols = ["domain", "family", "complexity", "seed", "n",
            "c1", "c2", "delta_alpha", "alpha0", "asymmetry",
            "mean", "std", "gen_s", "wl_s", "mf_s", "status", "note"]
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {MANIFEST.relative_to(MANIFEST.parents[2])}")
    print(f"Total wall time: {time.time() - started:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--complexities", default="0.0,0.25,0.5,0.75,1.0",
                    help="Comma-separated complexity levels in [0,1].")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    cx = [float(x) for x in args.complexities.split(",")]
    run(n=args.n, complexities=cx, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
