"""
build_pareidolia_rock.py — the crossed c1 x c2 confirmatory stimulus set.

WHY ROCK ONLY. A crossed design needs the two factors to be independent. Measured over 2656
procedural fields (datasets/reach_map.py):

    rock   n=1452   corr(c1, c2) = +0.256   <- decoupled, can be crossed
    cloud  n= 792   corr(c1, c2) = -0.837
    fluid  n= 412   corr(c1, c2) = -0.937   <- varying one forces the other

You cannot cross factors correlated at -0.94. Rock is the only domain where the design exists.
Fluid additionally has a structural hole — c2 in [-0.55,-0.20) holds 1 of 412 candidates, and the
only family reaching c2 <= -0.20 is dye_diffusion, at c1 >= 2.34 — so any deep fluid cell would be
single-family, i.e. family perfectly confounded with condition. Cloud and fluid belong in a
c1-varying replication strip at fixed c2, not in the crossed grid.

ORDER OF OPERATIONS MATTERS, and an earlier draft of this design got it wrong twice:
  1. generate  ->  2. NORMALISE  ->  3. MEASURE  ->  4. SELECT
Normalisation is a pointwise monotone map, so it moves the cumulants. Measuring before it would
select on values the shipped image does not have. Likewise the presentation aperture is applied
LAST and is never part of the acceptance measurement — applying it first drives measured c2 from
~-0.3 to -56, so rejection sampling would be selecting on aperture geometry rather than texture.

NORMALISATION IS A TRILEMMA, AND IT DOES NOT HAVE A CLEAN SOLUTION. Measured on rock:

    normalisation                        c2 IQR preserved      RMS equalised?
    none (raw)                              0.218              no
    affine to fixed mean/sd, clipped        0.103              NO — corr(c2, RMS) = +0.798
    rank-match to a common marginal         0.037              yes, exactly

Both options fail, in opposite directions, and for the same reason: **c2 is partly a property of
the amplitude distribution**. Force every stimulus onto one marginal and you delete most of the
manipulation (IQR 0.218 -> 0.037, a 6x compression). Leave the marginal free and contrast tracks
the manipulation. The affine map does not rescue it — clipping only 1.1 % of pixels is enough,
because heavier-tailed (more negative c2) images clip more and so lose RMS.

So the reviewer's question — "how do you know it is not just contrast?" — cannot be answered by
equating contrast. It has to be answered by a CONTROL CONDITION: a phase-scrambled twin of every
stimulus, which preserves the marginal and the power spectrum exactly while destroying the
higher-order structure that c2 measures. If the effect tracks c2 in the real stimuli and not in
the twins, it is the multifractal structure; if it tracks in both, it was the low-level statistics
all along. That is the standard move in texture psychophysics and it is the only honest one here.

This script therefore ships `--normalise {none,affine,rank}` (default `affine`, mild), RECORDS
mean / RMS / beta / anisotropy for every stimulus so they can enter the model as covariates rather
than be pretended away, and leaves the twin generation to a separate explicit step.

WHAT IS RECORDED BUT NOT SELECTED ON. Every stimulus also ships its radially-averaged power
spectrum slope (beta) and its anisotropy index. beta matters because "roughness" has two candidate
operationalisations — measured c1, or spectral beta — and they are NOT interchangeable: on the
delivered images beta ~ -0.31 + 2.19*c1 + 0.97*c2, so c2 leaks into beta at fixed c1. Shipping
both means the roughness factor can be defined either way at analysis time without regenerating.

Usage:
  python datasets/build_pareidolia_rock.py --pool-seeds 24
  python datasets/build_pareidolia_rock.py --per-cell 8 --out datasets/pareidolia_rock
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf                                                    # noqa: E402

ROCK = ["marble", "agate", "weathered", "granite", "vesicular", "turbulent",
        "schist", "serpentinite", "flow_banded", "convoluted", "gneiss"]

C1_LEVELS = [1.10, 1.30, 1.50]
C2_LEVELS = [-0.38, -0.28, -0.18, -0.08]
TOL_C1, TOL_C2 = 0.10, 0.05          # ~2.5x and ~2.2x the measured within-condition seed sd

# One reference marginal for the whole set. Gaussian, clipped to [0,1]; sd chosen so that
# clipping touches <0.5% of pixels and the 8-bit range is well used.
REF_MU, REF_SD = 0.5, 0.12


def reference_marginal(npix, mu=REF_MU, sd=REF_SD):
    q = (np.arange(npix) + 0.5) / npix
    from scipy.special import erfinv
    return np.clip(mu + sd * np.sqrt(2) * erfinv(2 * q - 1), 0, 1)


def affine_norm(a, mu=REF_MU, sd=0.16):
    """Mild normalisation: fix mean and nominal sd, clip to [0,1]. Preserves ~half the c2 range
    (IQR 0.103 vs 0.218 raw) but does NOT equalise delivered RMS — clipping is non-affine and
    heavier-tailed images clip more, leaving corr(c2, RMS) = +0.798. Record RMS, do not claim it."""
    a = np.asarray(a, float)
    return np.clip(mu + (a - a.mean()) / (a.std() + 1e-12) * sd, 0, 1)


def rank_match(field, ref_sorted):
    """Map pixel RANKS onto a fixed reference marginal -> identical histogram for every stimulus."""
    f = np.asarray(field, float).ravel()
    order = f.argsort().argsort()
    return ref_sorted[order].reshape(np.asarray(field).shape)


def spectral_beta(a):
    """Slope of the radially-averaged power spectrum, log-log. The alternative 'roughness'."""
    a = np.asarray(a, float); a = a - a.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    n = a.shape[0]; c = n // 2
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - c, y - c).astype(int)
    lo, hi = 3, n // 4
    m = (r >= lo) & (r <= hi)
    rad = np.bincount(r[m].ravel(), F[m].ravel()) / np.maximum(np.bincount(r[m].ravel()), 1)
    k = np.arange(len(rad))
    sel = (k >= lo) & (k <= hi) & (rad > 0)
    if sel.sum() < 8:
        return np.nan
    return float(-np.polyfit(np.log(k[sel]), np.log(rad[sel]), 1)[0])


def measure_all(a):
    r = mf.wavelet_leaders_2d(a)
    return dict(c1=float(r["c1"]), c2=float(r["c2"]),
                beta=spectral_beta(a),
                aniso=float(mf.anisotropy_index(a)),
                mean=float(a.mean()), rms=float(a.std()))


def normalise(raw, ref, mode):
    if mode == "rank":
        return rank_match(raw, ref)
    if mode == "affine":
        return affine_norm(raw)
    return np.clip((raw - raw.min()) / (np.ptp(raw) + 1e-12), 0, 1)


def build_pool(seeds, n, cx_grid, mode="affine", verbose=True):
    ref = reference_marginal(n * n)
    rows = []
    t0 = time.time()
    tot = len(ROCK) * len(cx_grid) * seeds
    k = 0
    for fam in ROCK:
        for cx in cx_grid:
            for s in range(seeds):
                k += 1
                try:
                    raw = np.asarray(mf.generate(fam, n=n, complexity=float(cx), seed=s), float)
                except Exception as e:                       # a family that ignores cx, etc.
                    if verbose and s == 0 and cx == cx_grid[0]:
                        print(f"    ({fam}: {e})", flush=True)
                    continue
                if not np.isfinite(raw).all() or np.ptp(raw) < 1e-9:
                    continue
                img = normalise(raw, ref, mode)              # NORMALISE, then measure
                m = measure_all(img)
                rows.append(dict(family=fam, complexity=float(cx), seed=s, n=n, **m))
                if verbose and k % 200 == 0:
                    print(f"    [{k}/{tot}] {(time.time()-t0)/60:.1f}m", flush=True)
    return rows


def auto_levels(pool, aniso_cap, n_c1=3, n_c2=4, min_fams=3, per_cell=8):
    """Choose the grid FROM THE NORMALISED POOL.

    The reachability map was built on raw fields, but rank matching is a pointwise monotone map
    and moves the cumulants — so levels taken from raw coordinates land in near-empty cells here
    (measured: 5 of 12 cells with <=1 candidate). The grid has to be derived in the same
    coordinate system the stimuli are actually delivered in.

    Searches equally-spaced c1 and c2 ladders and maximises the worst cell, subject to every cell
    having at least `min_fams` distinct families so that family cannot be confounded with cell.
    """
    ok = [r for r in pool if r["aniso"] <= aniso_cap]
    c1s = np.array([r["c1"] for r in ok]); c2s = np.array([r["c2"] for r in ok])
    best = None
    for c1lo in np.arange(np.percentile(c1s, 5), np.percentile(c1s, 45), 0.05):
        for span1 in np.arange(0.20, 0.85, 0.05):
            L1 = list(np.round(np.linspace(c1lo, c1lo + span1, n_c1), 3))
            for c2lo in np.arange(np.percentile(c2s, 5), np.percentile(c2s, 55), 0.02):
                for span2 in np.arange(0.10, 0.55, 0.02):
                    L2 = list(np.round(np.linspace(c2lo, c2lo + span2, n_c2), 3))
                    worst, worst_f = 1e9, 1e9
                    for a in L1:
                        for b in L2:
                            c = [r for r in ok if abs(r["c1"] - a) <= TOL_C1
                                 and abs(r["c2"] - b) <= TOL_C2]
                            worst = min(worst, len(c))
                            worst_f = min(worst_f, len({r["family"] for r in c}))
                            if worst < per_cell or worst_f < min_fams:
                                break
                        if worst < per_cell or worst_f < min_fams:
                            break
                    if worst >= per_cell and worst_f >= min_fams:
                        score = (worst_f, worst, span1 + span2)
                        if best is None or score > best[0]:
                            best = (score, L1, L2)
    return best


def select(pool, per_cell, max_per_family, aniso_cap):
    """Rejection-sample each cell: nearest to the cell centre, capped per family."""
    chosen, report = [], []
    for c1t in C1_LEVELS:
        for c2t in C2_LEVELS:
            cand = [r for r in pool
                    if abs(r["c1"] - c1t) <= TOL_C1 and abs(r["c2"] - c2t) <= TOL_C2
                    and r["aniso"] <= aniso_cap]
            cand.sort(key=lambda r: ((r["c1"] - c1t) / TOL_C1) ** 2 + ((r["c2"] - c2t) / TOL_C2) ** 2)
            taken, per_fam = [], {}
            for r in cand:
                if per_fam.get(r["family"], 0) >= max_per_family:
                    continue
                per_fam[r["family"]] = per_fam.get(r["family"], 0) + 1
                taken.append(dict(r, c1_level=c1t, c2_level=c2t))
                if len(taken) == per_cell:
                    break
            chosen += taken
            report.append(dict(c1_level=c1t, c2_level=c2t, n_candidates=len(cand),
                               n_taken=len(taken), n_families=len(per_fam),
                               families=",".join(sorted(per_fam))))
    return chosen, report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-seeds", type=int, default=24)
    ap.add_argument("--pool-n", type=int, default=512)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--max-per-family", type=int, default=3)
    ap.add_argument("--aniso-cap", type=float, default=0.15,
                    help="orientation anisotropy varies 0.017-0.414 across families and tracks "
                         "the c1 axis; cap it so orientation cannot carry the manipulation")
    ap.add_argument("--normalise", default="affine", choices=["none", "affine", "rank"],
                    help="see the module docstring: 'rank' equalises the marginal exactly but "
                         "compresses the c2 IQR 6x; 'affine' keeps ~half the range but leaves "
                         "corr(c2, RMS) = +0.80. Neither is clean — use phase-scrambled twins.")
    ap.add_argument("--auto-levels", action="store_true",
                    help="derive the grid from the NORMALISED pool instead of the raw reach map")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "pareidolia_rock"))
    args = ap.parse_args()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)

    mf.validate_families(ROCK)
    cx_grid = list(np.round(np.linspace(0.0, 1.0, 9), 3))
    print(f"pool: {len(ROCK)} families x {len(cx_grid)} complexities x {args.pool_seeds} seeds "
          f"at n={args.pool_n}", flush=True)
    pool = build_pool(args.pool_seeds, args.pool_n, cx_grid, mode=args.normalise)
    print(f"  {len(pool)} usable fields", flush=True)
    with (out / "pool.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pool[0].keys())); w.writeheader(); w.writerows(pool)

    global C1_LEVELS, C2_LEVELS
    if args.auto_levels:
        got = auto_levels(pool, args.aniso_cap, per_cell=args.per_cell)
        if got is None:
            print("  no grid satisfies the per-cell and family floors; relax --per-cell")
            return
        (fams, worst, _), C1_LEVELS, C2_LEVELS = got
        print(f"  auto levels: c1 {C1_LEVELS}  c2 {C2_LEVELS}  "
              f"(worst cell {worst} candidates, {fams} families)", flush=True)

    chosen, report = select(pool, args.per_cell, args.max_per_family, args.aniso_cap)
    print(f"\n{'c1':>5}{'c2':>7}{'cand':>7}{'taken':>7}{'fams':>6}  families")
    for r in report:
        print(f"{r['c1_level']:5.2f}{r['c2_level']:7.2f}{r['n_candidates']:7d}"
              f"{r['n_taken']:7d}{r['n_families']:6d}  {r['families']}")

    ref = reference_marginal(args.pool_n * args.pool_n)
    rows = []
    for i, r in enumerate(chosen):
        raw = np.asarray(mf.generate(r["family"], n=args.pool_n,
                                     complexity=r["complexity"], seed=r["seed"]), float)
        img = normalise(raw, ref, args.normalise)
        sid = f"rk{i:03d}_{r['family']}"
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
            out / "images" / f"{sid}.png", optimize=True)
        rows.append(dict(stim_id=sid, image_path=f"images/{sid}.png", domain="rock",
                         family=r["family"], complexity=r["complexity"], seed=r["seed"],
                         n=args.pool_n, c1_level=r["c1_level"], c2_level=r["c2_level"],
                         c1=round(r["c1"], 4), c2=round(r["c2"], 4),
                         beta=round(r["beta"], 4), aniso=round(r["aniso"], 4),
                         mean=round(r["mean"], 4), rms=round(r["rms"], 4)))
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys())); w.writeheader(); w.writerows(report)

    import pandas as pd
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} stimuli -> {out}")
    print("\nMANIPULATION CHECK (between-level range vs within-cell sd):")
    for nm in ("c1", "c2"):
        lv = f"{nm}_level"
        btw = d.groupby(lv)[nm].mean()
        wth = d.groupby(["c1_level", "c2_level"])[nm].std().mean()
        print(f"  {nm}: levels {btw.round(3).tolist()}  range {btw.max()-btw.min():.3f}  "
              f"within-cell sd {wth:.3f}  ratio {(btw.max()-btw.min())/wth:.1f}")
    print("\nORTHOGONALITY / CONFOUND CHECK (want |r| small):")
    for a, b in (("c1", "c2"), ("c1", "mean"), ("c1", "rms"), ("c2", "mean"),
                 ("c2", "rms"), ("c1", "aniso"), ("c2", "aniso"), ("c2", "beta")):
        if d[b].std() < 1e-9:
            print(f"  corr({a}, {b}) = n/a  ({b} is CONSTANT by construction — rank matching)")
        else:
            print(f"  corr({a}, {b}) = {np.corrcoef(d[a], d[b])[0,1]:+.3f}")
    print("\n  (beta is RECORDED, not selected on — see the module docstring on the roughness "
          "factor choice)")


if __name__ == "__main__":
    main()
