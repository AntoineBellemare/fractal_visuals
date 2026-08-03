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


def auto_levels(pool, aniso_cap, c1_span, c2_span, n_c1=3, n_c2=4,
                min_fams=3, per_cell=8):
    """Place a grid of the REQUESTED spans so that contrast is as balanced as possible.

    The spans are an input, not something to maximise, because these quantities trade against
    each other and the tradeoff has to be chosen rather than discovered. Measured frontier on
    this pool (min cell / min families / RMS spread across cells):

        c1 0.20 x c2 0.10 -> 13 / 3 / 0.0004      c1 0.30 x c2 0.20 -> 38 / 4 / 0.0182
        c1 0.40 x c2 0.15 -> 30 / 3 / 0.0104      c1 0.40 x c2 0.30 -> 14 / 3 / 0.0789
        c1 0.50 x c2 0.20 -> infeasible

    Widening c2 costs contrast balance fast: an earlier build that maximised total span reached
    c1 span 0.72 but drove corr(c1, RMS) to +0.677 and squeezed c2 to 0.079. Given c2 is the
    scarce axis and the scientifically contested one, the default trades c1 span for c2 span.

    Within the requested spans this minimises the spread of median RMS across cells, subject to
    every cell having `per_cell` candidates and `min_fams` distinct families.
    """
    ok = [r for r in pool if r["aniso"] <= aniso_cap]
    c1s = np.array([r["c1"] for r in ok]); c2s = np.array([r["c2"] for r in ok])
    best = None
    for c1lo in np.arange(np.percentile(c1s, 3), np.percentile(c1s, 70), 0.05):
        L1 = list(np.round(np.linspace(c1lo, c1lo + c1_span, n_c1), 3))
        for c2lo in np.arange(np.percentile(c2s, 3), -0.04, 0.02):
            L2 = list(np.round(np.linspace(c2lo, c2lo + c2_span, n_c2), 3))
            cells = []
            for a in L1:
                for b in L2:
                    c = [r for r in ok if abs(r["c1"] - a) <= TOL_C1
                         and abs(r["c2"] - b) <= TOL_C2]
                    cells.append((len(c), len({r["family"] for r in c}),
                                  float(np.median([r["rms"] for r in c])) if c else np.nan))
            if min(x[0] for x in cells) < per_cell or min(x[1] for x in cells) < min_fams:
                continue
            rms = [x[2] for x in cells]
            spread = max(rms) - min(rms)
            score = (-spread, min(x[1] for x in cells), min(x[0] for x in cells))
            if best is None or score > best[0]:
                best = (score, L1, L2)
    return best


def select(pool, per_cell, max_per_family, aniso_cap, rms_target=None, rms_w=1.0,
           aniso_target=None):
    """Rejection-sample each cell: nearest to the cell centre, capped per family.

    `rms_target` additionally pulls every cell toward a COMMON contrast. RMS cannot be equated
    by normalisation without destroying the c2 manipulation (see the module docstring), but it
    can be BALANCED by selection: among candidates that already satisfy the cumulant tolerances,
    prefer those whose RMS sits near the global median. Without it the first build had RMS rising
    monotonically across cells, 0.141 -> 0.159, i.e. contrast tracked both factors.
    """
    chosen, report = [], []
    for c1t in C1_LEVELS:
        for c2t in C2_LEVELS:
            cand = [r for r in pool
                    if abs(r["c1"] - c1t) <= TOL_C1 and abs(r["c2"] - c2t) <= TOL_C2
                    and r["aniso"] <= aniso_cap]
            def cost(r):
                c = ((r["c1"] - c1t) / TOL_C1) ** 2 + ((r["c2"] - c2t) / TOL_C2) ** 2
                # Balance BOTH nuisances toward a common target. Capping anisotropy is not
                # enough: with only a cap, corr(c1, aniso) came out at -0.594, because the
                # families that populate high c1 happen to be the more isotropic ones.
                if rms_target is not None:
                    c += rms_w * ((r["rms"] - rms_target) / (0.25 * rms_target)) ** 2
                if aniso_target is not None:
                    c += rms_w * ((r["aniso"] - aniso_target) / (0.5 * aniso_target)) ** 2
                return c
            cand.sort(key=cost)
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
    ap.add_argument("--c1-span", type=float, default=0.30)
    ap.add_argument("--c2-span", type=float, default=0.20)
    ap.add_argument("--min-families", type=int, default=3)
    ap.add_argument("--pool-csv", default=None,
                    help="reuse a previously measured pool instead of regenerating (25 min)")
    ap.add_argument("--no-rms-balance", action="store_true",
                    help="disable contrast balancing across cells (it cannot be normalised away)")
    ap.add_argument("--rms-weight", type=float, default=1.0)
    ap.add_argument("--auto-levels", action="store_true",
                    help="derive the grid from the NORMALISED pool instead of the raw reach map")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "pareidolia_rock"))
    args = ap.parse_args()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)

    mf.validate_families(ROCK)
    cx_grid = list(np.round(np.linspace(0.0, 1.0, 9), 3))
    print(f"pool: {len(ROCK)} families x {len(cx_grid)} complexities x {args.pool_seeds} seeds "
          f"at n={args.pool_n}", flush=True)
    if args.pool_csv and Path(args.pool_csv).exists():
        import pandas as _pd
        pool = _pd.read_csv(args.pool_csv).to_dict("records")
        print(f"  reusing pool: {len(pool)} fields from {args.pool_csv}", flush=True)
    else:
        pool = build_pool(args.pool_seeds, args.pool_n, cx_grid, mode=args.normalise)
    print(f"  {len(pool)} usable fields", flush=True)
    with (out / "pool.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pool[0].keys())); w.writeheader(); w.writerows(pool)

    global C1_LEVELS, C2_LEVELS
    if args.auto_levels:
        got = auto_levels(pool, args.aniso_cap, args.c1_span, args.c2_span,
                          per_cell=args.per_cell, min_fams=args.min_families)
        if got is None:
            print("  no grid satisfies the per-cell and family floors; relax --per-cell")
            return
        (negspread, minfam, mincell), C1_LEVELS, C2_LEVELS = got
        print(f"  auto levels: c1 {C1_LEVELS}  c2 {C2_LEVELS}  "
              f"(worst cell {mincell} candidates, {minfam} families, "
              f"RMS spread {-negspread:.4f})", flush=True)

    _ok = [r for r in pool if r["aniso"] <= args.aniso_cap]
    rms_med = float(np.median([r["rms"] for r in _ok]))
    ani_med = float(np.median([r["aniso"] for r in _ok]))
    chosen, report = select(pool, args.per_cell, args.max_per_family, args.aniso_cap,
                            rms_target=(None if args.no_rms_balance else rms_med),
                            rms_w=args.rms_weight,
                            aniso_target=(None if args.no_rms_balance else ani_med))
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
