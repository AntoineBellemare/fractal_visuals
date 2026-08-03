"""
make_twins.py — phase-scrambled control twins for the pareidolia stimulus set.

WHY THESE ARE NECESSARY, not optional. The c2 manipulation cannot be separated from low-level
image statistics by normalisation: measured on rock, forcing a common marginal preserves only
0.037 of the 0.218 c2 IQR, while leaving the marginal free lets contrast track the manipulation
(corr(c2, RMS) up to +0.80). c2 is *partly* a property of the amplitude distribution, so
"equate contrast and see if the effect survives" is not an available experiment.

The available experiment is a matched control stimulus. An IAAFT twin has, to within a small
tolerance, the SAME marginal histogram and the SAME radially-averaged power spectrum as its
parent, but randomised Fourier phase — which destroys the higher-order spatial structure that
the wavelet-leader cumulants are sensitive to. So:

    effect tracks c2 in parents but NOT in twins  ->  it is the multifractal structure
    effect tracks c2 in both                      ->  it was marginal / spectrum all along

This is the standard control in texture psychophysics, and it is the only construction that
answers the reviewer's question honestly here.

METHOD: iterative amplitude-adjusted Fourier transform (Schreiber & Schmitz). Alternate between
(a) imposing the parent's Fourier amplitude spectrum and (b) imposing the parent's exact sorted
marginal by rank. Neither constraint is exactly satisfiable simultaneously, so the loop converges
to a compromise; both residuals are measured and reported per twin rather than assumed.

Usage:
  python datasets/make_twins.py --set datasets/pareidolia_rock
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf                                                    # noqa: E402
sys.path.insert(0, str(ROOT / "datasets"))
from build_pareidolia_rock import spectral_beta, measure_all             # noqa: E402


def iaaft(img, iters=60, seed=0):
    """Phase-randomised surrogate with the parent's amplitude spectrum and exact marginal."""
    rng = np.random.default_rng(seed)
    a = np.asarray(img, float)
    target_amp = np.abs(np.fft.fft2(a))
    target_sorted = np.sort(a.ravel())

    # start from a random permutation of the parent's own pixels -> marginal already exact
    x = rng.permutation(a.ravel()).reshape(a.shape)
    for _ in range(iters):
        F = np.fft.fft2(x)
        F = target_amp * np.exp(1j * np.angle(F))       # impose amplitude spectrum
        y = np.real(np.fft.ifft2(F))
        order = y.ravel().argsort().argsort()           # impose exact marginal by rank
        x = target_sorted[order].reshape(a.shape)
    return x


def radial_spectrum(a):
    a = np.asarray(a, float) - np.mean(a)
    F = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    n = a.shape[0]; c = n // 2
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - c, y - c).astype(int)
    return np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="root", default=str(ROOT / "datasets" / "pareidolia_rock"))
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    root = Path(args.root)
    import pandas as pd
    man = pd.read_csv(root / "manifest.csv")
    outdir = root / "twins"; outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in man.iterrows():
        par = np.asarray(Image.open(root / r.image_path).convert("L"), float) / 255.0
        tw = iaaft(par, iters=args.iters, seed=i)
        sid = f"{r.stim_id}_twin"
        Image.fromarray((np.clip(tw, 0, 1) * 255).astype(np.uint8)).save(
            outdir / f"{sid}.png", optimize=True)
        m = measure_all(tw)
        sp_p, sp_t = radial_spectrum(par), radial_spectrum(tw)
        k = slice(3, len(sp_p) // 4)
        spec_err = float(np.mean(np.abs(np.log(sp_t[k] + 1e-12) - np.log(sp_p[k] + 1e-12))))
        marg_err = float(np.max(np.abs(np.sort(tw.ravel()) - np.sort(par.ravel()))))
        rows.append(dict(stim_id=sid, parent_id=r.stim_id,
                         image_path=f"twins/{sid}.png",
                         c1_level=r.c1_level, c2_level=r.c2_level,
                         c1=round(m["c1"], 4), c2=round(m["c2"], 4),
                         beta=round(m["beta"], 4), aniso=round(m["aniso"], 4),
                         mean=round(m["mean"], 4), rms=round(m["rms"], 4),
                         parent_c1=r.c1, parent_c2=r.c2, parent_beta=r.beta,
                         spec_log_err=round(spec_err, 5), marginal_max_err=round(marg_err, 6)))
        if (i + 1) % 24 == 0:
            print(f"  [{i+1}/{len(man)}]", flush=True)

    with (root / "twins.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} twins -> {outdir}")
    print("\nMATCHING CHECK (twin vs its parent):")
    print(f"  marginal    max abs diff  {d.marginal_max_err.max():.2e}   (0 = identical histogram)")
    print(f"  spectrum    mean |log ratio|  {d.spec_log_err.mean():.4f} "
          f"(max {d.spec_log_err.max():.4f})")
    print(f"  beta        parent {d.parent_beta.mean():.3f}  twin {d.beta.mean():.3f}   "
          f"diff {(d.beta - d.parent_beta).abs().mean():.4f}")
    print("\nDESTRUCTION CHECK (the twin should NOT carry the manipulation):")
    for nm in ("c1", "c2"):
        lv = f"{nm}_level"
        pr = man.groupby(lv)[nm].mean(); tw = d.groupby(lv)[nm].mean()
        print(f"  {nm}: parent range across levels {pr.max()-pr.min():.3f}   "
              f"twin range {tw.max()-tw.min():.3f}   "
              f"retained {100*(tw.max()-tw.min())/(pr.max()-pr.min()):.0f}%")


if __name__ == "__main__":
    main()
