"""
Independent cross-check of mfractal.wavelet_leaders_2d against pymultifracs
(Wendt/Abry-style log-cumulants c1, c2). Both implementations target the same
estimator family, so they should agree directionally on which families are
multifractal (c2 << 0) vs monofractal (c2 ~ 0).

We use the 1D log-cumulant from pymultifracs on row + column slices and average;
that gives an independent multifractal signature without depending on the
internal mfractal implementation. The point is not exact numerical agreement
(different wavelet, different leader construction, different j-range) but
agreement on the RANK of families by multifractality.

Writes benchmarks/results/pymultifracs_check.csv with both estimates per family,
and a small comparison plot at benchmarks/figures/07_pymultifracs_check.png.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

OUT_CSV = ROOT / "benchmarks" / "results" / "pymultifracs_check.csv"
OUT_FIG = ROOT / "benchmarks" / "figures" / "07_pymultifracs_check.png"


def pymf_c1c2_1d(signal: np.ndarray) -> tuple[float, float]:
    """Log-cumulants from pymultifracs on a 1D signal using wavelet leaders."""
    from pymultifracs.wavelet import wavelet_analysis
    from pymultifracs.mf_analysis import mfa
    sig = np.asarray(signal, dtype=float)
    sig = sig - sig.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wt = wavelet_analysis(sig, wt_name="db3")
        leaders = wt.get_leaders(p_exp=2)
        j2 = max(3, leaders.j2_eff() - 1)
        result = mfa(leaders, scaling_ranges=[(2, j2)],
                     n_cumul=2, q=np.array([2.0]))
    cm = np.asarray(result.cumulants.log_cumulants).reshape(2, -1)
    return float(cm[0].mean()), float(cm[1].mean())


def pymf_c1c2_2d_via_slices(image: np.ndarray, n_rows: int = 32) -> tuple[float, float]:
    """Average 1D log-cumulants over a sample of rows and columns of the 2D field.
    This is the simplest pymultifracs-based 2D signature: the spectrum of any 1D
    cross-section of a 2D multifractal carries the same intermittency signal.
    """
    img = np.asarray(image, dtype=float)
    H, W = img.shape
    rng = np.random.default_rng(0)
    row_idx = rng.choice(H, size=min(n_rows, H), replace=False)
    col_idx = rng.choice(W, size=min(n_rows, W), replace=False)
    c1_list, c2_list = [], []
    for i in row_idx:
        try:
            c1, c2 = pymf_c1c2_1d(img[i, :])
            if np.isfinite(c1) and np.isfinite(c2):
                c1_list.append(c1); c2_list.append(c2)
        except Exception:
            pass
    for j in col_idx:
        try:
            c1, c2 = pymf_c1c2_1d(img[:, j])
            if np.isfinite(c1) and np.isfinite(c2):
                c1_list.append(c1); c2_list.append(c2)
        except Exception:
            pass
    if not c1_list:
        return float("nan"), float("nan")
    return float(np.mean(c1_list)), float(np.mean(c2_list))


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fams = []
    fams += [("rock", f) for f in mf.ROCK_FAMILIES]
    fams += [("cloud", f) for f in mf.CLOUD_FAMILIES]
    fams += [("fluid", f) for f in mf.FLUID_FAMILIES]

    rows = []
    n = 256
    print(f"pymultifracs cross-check at n={n}, cx=0.5 (or NS sim), seed=0")
    print(f"{'domain':6s} {'family':24s} {'mf_c2':>8s} {'pymf_c2':>9s} {'pymf_c1':>9s} {'wall_s':>7s}")
    for dom, fam in fams:
        try:
            if fam in mf.CX_FLUIDS or fam in mf.ROCK_FAMILIES or fam in mf.CLOUD_FAMILIES:
                img = mf.generate(fam, n=n, complexity=0.5, seed=0)
            else:
                img = mf.generate(fam, n=n, seed=0)
            t0 = time.time()
            wl = mf.wavelet_leaders_2d(img)
            mf_c2 = float(wl["c2"])
            pymf_c1, pymf_c2 = pymf_c1c2_2d_via_slices(img, n_rows=16)
            dt = time.time() - t0
            rows.append(dict(domain=dom, family=fam, mf_c2=mf_c2,
                             pymf_c2=pymf_c2, pymf_c1=pymf_c1))
            print(f"{dom:6s} {fam:24s} {mf_c2:8.3f} {pymf_c2:9.3f} {pymf_c1:9.3f} {dt:7.1f}")
        except Exception as e:
            print(f"{dom:6s} {fam:24s} ERROR: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}")

    # Rank correlation between mfractal c2 and pymultifracs c2
    valid = df.dropna(subset=["mf_c2", "pymf_c2"])
    rho, p = spearmanr(valid.mf_c2, valid.pymf_c2)
    pearson = np.corrcoef(valid.mf_c2, valid.pymf_c2)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    color = {"rock": "#8c564b", "cloud": "#1f77b4", "fluid": "#17becf"}
    for dom in ["rock", "cloud", "fluid"]:
        sub = valid[valid.domain == dom]
        ax.scatter(sub.mf_c2, sub.pymf_c2, c=color[dom], label=dom, s=60,
                   edgecolor="k", lw=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r.family, (r.mf_c2, r.pymf_c2), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", color=color[dom])
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("c2 — mfractal.wavelet_leaders_2d (full 2D wavelet leaders)")
    ax.set_ylabel("c2 — pymultifracs 1D, averaged over slices")
    ax.set_title(f"Cross-estimator agreement on c2  (Spearman rho={rho:+.3f}, Pearson={pearson:+.3f})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG.relative_to(ROOT)}")
    print(f"Spearman rho={rho:+.3f}  p={p:.2e}  Pearson={pearson:+.3f}  n={len(valid)}")


if __name__ == "__main__":
    main()
