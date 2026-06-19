"""
Focused multifractal validation sweep for the 3 metal families.

Runs the same wavelet-leader (c1, c2) + MFDFA (Delta_alpha, alpha0, asymmetry)
measurement protocol as run_validation.py but only for rust_bloom, rust_pitted,
rust_dewy. Writes validation_manifest_metal.csv alongside the legacy
validation_manifest.csv, with the same schema so the two can be concatenated.

This is the pre-GPU step that gives the c2 distribution for metal -- needed
both to confirm rho(c2, complexity) is monotonic and to feed the c2 regressor
training data with metal examples on the next training run.

Usage:
    python benchmarks/validate_metal.py
    python benchmarks/validate_metal.py --n 256 --seeds 4
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


OUT = Path(__file__).resolve().parent / "results" / "validation_manifest_metal.csv"


def measure(img: np.ndarray) -> dict:
    out = {}
    t0 = time.time()
    try:
        wl = mf.wavelet_leaders_2d(img)
        out["c1"] = float(wl["c1"]); out["c2"] = float(wl["c2"])
    except Exception as e:
        out["c1"] = float("nan"); out["c2"] = float("nan")
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    fams = list(mf.METAL_FAMILIES)
    for fi, family in enumerate(fams, 1):
        for cx in complexities:
            for seed in range(n_seeds):
                t0 = time.time()
                try:
                    img = mf.generate(family, n=n, complexity=float(cx), seed=seed)
                    gen_s = time.time() - t0
                    status, note = "ok", ""
                except Exception as e:
                    rows.append({
                        "domain": "metal", "family": family,
                        "complexity": cx, "seed": seed, "n": n,
                        "c1": "", "c2": "", "delta_alpha": "",
                        "alpha0": "", "asymmetry": "", "mean": "", "std": "",
                        "gen_s": f"{time.time() - t0:.3f}",
                        "wl_s": "", "mf_s": "",
                        "status": "gen_fail", "note": repr(e)[:200],
                    })
                    continue
                m = measure(img)
                rows.append({
                    "domain": "metal", "family": family,
                    "complexity": cx, "seed": seed, "n": n,
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
                })
        elapsed = time.time() - started
        print(f"[{fi}/{len(fams)}] {family:14s} elapsed={elapsed:6.1f}s")

    cols = ["domain", "family", "complexity", "seed", "n",
            "c1", "c2", "delta_alpha", "alpha0", "asymmetry",
            "mean", "std", "gen_s", "wl_s", "mf_s", "status", "note"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT.relative_to(OUT.parents[2])}")
    print(f"Total wall time: {time.time() - started:.1f}s")

    # Monotonicity check.
    print("\nPer-family Spearman rho(c2, complexity):")
    for family in fams:
        rr = [r for r in rows if r["family"] == family and r["status"] == "ok"]
        if not rr:
            continue
        cs = np.array([float(r["complexity"]) for r in rr])
        c2s = np.array([float(r["c2"]) for r in rr])
        rx = np.argsort(np.argsort(cs)); ry = np.argsort(np.argsort(c2s))
        rx = rx - rx.mean(); ry = ry - ry.mean()
        rho = float((rx * ry).sum() /
                    (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12))
        print(f"  {family:14s}  rho={rho:+.3f}  "
              f"c2 mean={c2s.mean():+.3f}  range=[{c2s.min():+.3f}, {c2s.max():+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--complexities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    cx = [float(x) for x in args.complexities.split(",")]
    run(n=args.n, complexities=cx, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
