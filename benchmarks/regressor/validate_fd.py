"""
validate_fd.py
===============
Independent audit of the APPLICATIONS.md claim (section 1, lines 18-21):

    "c1 is fractal dimension. Validated in this repo against differential
     box-counting on 200 photographic corpus images: corr(c1, FD) = -0.91,
     FD ~ 2.91 - 0.33*c1 (the idealised relation is FD = 3 - c1; box-counting
     compresses the range but the mapping is near-deterministic)."

No script, CSV, or log in this repo produces those numbers, and
`mfractal.box_count_support` (D0 of a binary set) and `mfractal.local_fd_map`
(a structure-function estimator) are NOT differential box counting of a
greyscale surface. This script implements real DBC (Sarkar & Chaudhuri 1994),
sanity-checks it against known-answer inputs, and then measures corr(c1, FD)
on this repo's own labelled corpus.

CPU only. numpy / scipy / PIL / pandas / matplotlib.

Usage
-----
    python validate_fd.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_corpus as bc  # noqa: E402  (reuses load_field / to_canonical_gray -- the
                            # repo's own faithful-field reader, so a heavy-tailed .npy
                            # cascade field is read at full float precision and never
                            # routed through an 8-bit round-trip before we measure it)

RESULTS_DIR = ROOT / "benchmarks" / "results"
FIG_DIR = ROOT / "benchmarks" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CANON_RES = bc.CANON_RES  # 512 -- same resolution the rest of the repo measures c1/c2 at


# --------------------------------------------------------------------------- #
# Differential box counting (Sarkar & Chaudhuri, 1994)
# --------------------------------------------------------------------------- #
def dbc_fd(field: np.ndarray, box_sizes=None, return_fit: bool = False):
    """Fractal dimension of a greyscale surface by differential box counting.

    Partition the M x M field into non-overlapping s x s spatial grids. The
    grey axis is quantised into boxes of height h; with grey range G set
    equal to the spatial side M (the standard DBC choice, giving literally
    cubical boxes: h = (G/M)*s = s), for each cell
        n_r = ceil(max_grey / h) - ceil(min_grey / h) + 1
    N_r = sum of n_r over all cells. FD = slope of log N_r vs log(1/r),
    r = s/M, fit by least squares over the sampled box sizes.

    `field` may be any real-valued 2D array (not necessarily 0-255) -- the
    grey axis is min-max normalised to [0, M-1] using the array's OWN
    min/max, in float64, in one step. This is important for heavy-tailed
    generated fields: rounding to 8-bit *before* this normalisation is a
    known pathology in this repo (it collapses near-zero mass and reads
    c2 as -50 for the wavelet-leader estimator) -- we sidestep the
    equivalent risk here by never doing a separate lossy uint8 cast.
    """
    x = np.asarray(field, dtype=np.float64)
    assert x.ndim == 2 and x.shape[0] == x.shape[1], "field must be square 2D"
    M = x.shape[0]
    mn, mx = x.min(), x.max()
    g = np.zeros_like(x) if (mx - mn) < 1e-12 else (x - mn) / (mx - mn) * (M - 1)

    if box_sizes is None:
        smax = max(4, M // 4)
        box_sizes = sorted(set(int(round(s)) for s in np.geomspace(2, smax, 16)))
        box_sizes = [s for s in box_sizes if s >= 2]

    logr, logN, used = [], [], []
    for s in box_sizes:
        ns = M // s
        if ns < 2:
            continue
        h = float(s)  # G/M == 1 by construction
        xc = g[: ns * s, : ns * s]
        blocks = xc.reshape(ns, s, ns, s)
        bmin = blocks.min(axis=(1, 3))
        bmax = blocks.max(axis=(1, 3))
        nr_cell = np.ceil(bmax / h) - np.ceil(bmin / h) + 1.0
        Nr = nr_cell.sum()
        logr.append(np.log(1.0 / (s / M)))
        logN.append(np.log(Nr))
        used.append(s)

    logr = np.asarray(logr)
    logN = np.asarray(logN)
    slope, intercept = np.polyfit(logr, logN, 1)
    if return_fit:
        pred = slope * logr + intercept
        ss_res = np.sum((logN - pred) ** 2)
        ss_tot = np.sum((logN - logN.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(slope), dict(logr=logr, logN=logN, r2=float(r2),
                                   box_sizes=used, intercept=float(intercept))
    return float(slope)


# --------------------------------------------------------------------------- #
# Independent ground truth: spectral-synthesis fBm (own code, not mfractal)
# --------------------------------------------------------------------------- #
def fbm2d(n: int, H: float, seed: int | None = None) -> np.ndarray:
    """2D fractional Brownian surface via spectral synthesis. PSD ~ f^-(2H+2)
    (amplitude ~ f^-(H+1)) is the standard isotropic-fBm spectral exponent for
    a 2D index domain. Ground truth: FD of the graph = 3 - H."""
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(n).reshape(-1, 1)
    fy = np.fft.fftfreq(n).reshape(1, -1)
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    amplitude = f ** (-(H + 1.0))
    A = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * amplitude
    A[0, 0] = 0.0
    return np.fft.ifft2(A).real


def realized_hurst(field: np.ndarray) -> float:
    """Independent (non-DBC, non-mfractal) structure-function estimate of H,
    used only to check that fbm2d() actually realised close to its target H
    -- decoupled from the DBC estimator under test."""
    x = np.asarray(field, float)
    lags = [1, 2, 4, 8, 16]
    logr = np.log(lags)
    sf = []
    for r in lags:
        d1 = x[r:, :] - x[:-r, :]
        d2 = x[:, r:] - x[:, :-r]
        sf.append(0.5 * (np.mean(d1 ** 2) + np.mean(d2 ** 2)))
    slope = np.polyfit(logr, np.log(sf), 1)[0]
    return slope / 2.0


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def run_sanity(n: int = CANON_RES, seeds_per_H: int = 6, verbose: bool = True) -> dict:
    out = {}

    flat = np.full((n, n), 0.5)
    out["flat_fd"] = dbc_fd(flat)

    rng = np.random.default_rng(0)
    noise = rng.uniform(0, 1, size=(n, n))
    out["noise_fd"] = dbc_fd(noise)

    Hs = np.linspace(0.05, 0.95, 10)
    fd_means, fd_stds, hreal_means = [], [], []
    for H in Hs:
        fds, hreals = [], []
        for s in range(seeds_per_H):
            surf = fbm2d(n, float(H), seed=1000 + int(round(H * 100)) * 100 + s)
            fds.append(dbc_fd(surf))
            hreals.append(realized_hurst(surf))
        fd_means.append(float(np.mean(fds)))
        fd_stds.append(float(np.std(fds)))
        hreal_means.append(float(np.mean(hreals)))
    fd_means = np.asarray(fd_means)
    hreal_means = np.asarray(hreal_means)

    r_fbm = float(np.corrcoef(hreal_means, fd_means)[0, 1])
    slope_fbm, intercept_fbm = np.polyfit(hreal_means, fd_means, 1)

    out.update(dict(Hs=Hs, fd_means=fd_means, fd_stds=fd_stds, hreal_means=hreal_means,
                     r_fbm=r_fbm, slope_fbm=float(slope_fbm), intercept_fbm=float(intercept_fbm)))

    if verbose:
        print("=" * 70)
        print("SANITY CHECKS (n=%d)" % n)
        print("=" * 70)
        print(f"flat image:  FD = {out['flat_fd']:.4f}   (expect ~2.0)")
        print(f"white noise: FD = {out['noise_fd']:.4f}   (expect ~3.0)")
        print()
        print("fractional Brownian surfaces (spectral synthesis, independent of mfractal):")
        print(f"{'H_target':>9} {'H_realized':>11} {'FD_dbc':>10} {'3-H_real (ideal)':>18}")
        for H, hr, fd in zip(Hs, hreal_means, fd_means):
            print(f"{H:9.2f} {hr:11.3f} {fd:10.3f} {3 - hr:18.3f}")
        print()
        print(f"corr(H_realized, FD_dbc) = {r_fbm:.4f}  "
              f"(expect strongly negative, ~ -1)")
        print(f"FD_dbc ~ {intercept_fbm:.3f} + {slope_fbm:.3f} * H_realized   "
              f"(idealised: FD = 3 - H, i.e. intercept=3, slope=-1)")
        compress = -slope_fbm  # magnitude of slope vs ideal 1.0
        print(f"DBC compresses the H->FD slope by a factor {compress:.2f} "
              f"relative to the idealised 1.0 -- consistent with the well-known "
              f"downward bias of box-counting FD estimators on rough surfaces.")
        print("=" * 70)
    return out


# --------------------------------------------------------------------------- #
# Corpus pass
# --------------------------------------------------------------------------- #
def measure_corpus(manifest_path: Path, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    rows = []
    t0 = time.time()
    n = len(df)
    for i, r in enumerate(df.itertuples(index=False)):
        p = Path(r.path)
        if not p.is_absolute():
            p_try = ROOT / p
            p = p_try if p_try.exists() else p
        try:
            field = bc.load_field(p)  # float32 grayscale CANON_RES x CANON_RES,
                                       # faithful raw values for .npy, /255 8-bit for images
            fd = dbc_fd(field)
        except Exception as e:  # noqa: BLE001 -- keep a long batch alive
            if verbose:
                print(f"  ! {r.path}: {type(e).__name__}: {e}")
            continue
        rows.append(dict(path=r.path, source=r.source, c1=float(r.c1), dbc_fd=fd))
        if verbose and (i + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{n}  ({dt:.0f}s, {dt/(i+1)*1000:.0f} ms/img)")
    if verbose:
        print(f"corpus: {len(rows)}/{n} measured in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


def measure_natural_atlas(csv_path: Path, base_dir: Path, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    t0 = time.time()
    for r in df.itertuples(index=False):
        p = base_dir / r.path
        if not p.exists():
            continue
        field = bc.to_canonical_gray(Image.open(p))
        fd = dbc_fd(field)
        rows.append(dict(path=str(Path("natural_atlas") / r.path), source="natural_atlas",
                          c1=float(r.c1_meas), dbc_fd=fd))
    if verbose:
        print(f"natural_atlas: {len(rows)}/{len(df)} measured in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def fit_stats(c1: np.ndarray, fd: np.ndarray) -> dict:
    ok = np.isfinite(c1) & np.isfinite(fd)
    c1, fd = c1[ok], fd[ok]
    n = len(c1)
    if n < 3:
        return dict(n=n, r=np.nan, slope=np.nan, intercept=np.nan)
    r = float(np.corrcoef(c1, fd)[0, 1])
    slope, intercept = np.polyfit(c1, fd, 1)
    return dict(n=n, r=r, slope=float(slope), intercept=float(intercept))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(sanity: dict, corpus_df: pd.DataFrame, atlas_df: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # --- Panel A: fBm sanity ---
    ax = axes[0, 0]
    hr = sanity["hreal_means"]
    fdm = sanity["fd_means"]
    fds = sanity["fd_stds"]
    ax.errorbar(hr, fdm, yerr=fds, fmt="o-", color="tab:blue", capsize=3,
                label="DBC-FD on spectral-synthesis fBm")
    hh = np.linspace(hr.min(), hr.max(), 50)
    ax.plot(hh, 3 - hh, "k--", label="ideal FD = 3 - H")
    ax.plot(hh, sanity["intercept_fbm"] + sanity["slope_fbm"] * hh, color="tab:red",
            label=f"fit: FD={sanity['intercept_fbm']:.2f}{sanity['slope_fbm']:+.2f}·H")
    ax.axhline(sanity["flat_fd"], color="grey", ls=":", lw=1,
               label=f"flat image FD={sanity['flat_fd']:.2f}")
    ax.axhline(sanity["noise_fd"], color="green", ls=":", lw=1,
               label=f"white noise FD={sanity['noise_fd']:.2f}")
    ax.set_xlabel("H (realized, structure-function estimate)")
    ax.set_ylabel("DBC-measured FD")
    ax.set_title(f"Sanity: known-H fBm  (r={sanity['r_fbm']:.3f})")
    ax.legend(fontsize=7, loc="upper right")

    # --- Panel B: full corpus scatter, colored by source ---
    ax = axes[0, 1]
    sources = sorted(corpus_df["source"].unique())
    cmap = plt.get_cmap("tab10")
    for i, src in enumerate(sources):
        sub = corpus_df[corpus_df.source == src]
        ax.scatter(sub.c1, sub.dbc_fd, s=6, alpha=0.4, color=cmap(i % 10), label=f"{src} (n={len(sub)})")
    c1_all = corpus_df.c1.values
    xx = np.linspace(np.percentile(c1_all, 1), np.percentile(c1_all, 99), 50)
    st_all = fit_stats(corpus_df.c1.values, corpus_df.dbc_fd.values)
    ax.plot(xx, st_all["intercept"] + st_all["slope"] * xx, color="black", lw=2,
            label=f"overall fit r={st_all['r']:.2f}")
    ax.plot(xx, 3 - xx, "k--", lw=1, label="ideal FD=3-c1")
    ax.plot(xx, 2.91 - 0.33 * xx, color="magenta", lw=1.5, ls="-.",
            label="doc's claim FD=2.91-0.33c1")
    ax.set_xlabel("c1 (mean Hölder exponent)")
    ax.set_ylabel("DBC-measured FD")
    ax.set_title("Full corpus: c1 vs DBC-FD")
    ax.legend(fontsize=6, loc="best", ncol=1)

    # --- Panel C: photographic-only (macro) ---
    ax = axes[1, 0]
    macro = corpus_df[corpus_df.source == "macro"]
    st_macro = fit_stats(macro.c1.values, macro.dbc_fd.values)
    ax.scatter(macro.c1, macro.dbc_fd, s=14, alpha=0.6, color="tab:brown")
    if len(macro) >= 3:
        xx = np.linspace(macro.c1.min(), macro.c1.max(), 50)
        ax.plot(xx, st_macro["intercept"] + st_macro["slope"] * xx, color="black", lw=2,
                label=f"fit: FD={st_macro['intercept']:.2f}{st_macro['slope']:+.2f}c1, "
                      f"r={st_macro['r']:.2f}, n={st_macro['n']}")
        ax.plot(xx, 3 - xx, "k--", lw=1, label="ideal FD=3-c1")
        ax.plot(xx, 2.91 - 0.33 * xx, color="magenta", lw=1.5, ls="-.",
                label="doc's claim")
    ax.set_xlabel("c1")
    ax.set_ylabel("DBC-measured FD")
    ax.set_title("Photographic only (source='macro', real photos)")
    ax.legend(fontsize=7, loc="best")

    # --- Panel D: natural_atlas cross-check ---
    ax = axes[1, 1]
    st_atlas = fit_stats(atlas_df.c1.values, atlas_df.dbc_fd.values)
    ax.scatter(atlas_df.c1, atlas_df.dbc_fd, s=10, alpha=0.5, color="tab:purple")
    xx = np.linspace(atlas_df.c1.min(), atlas_df.c1.max(), 50)
    ax.plot(xx, st_atlas["intercept"] + st_atlas["slope"] * xx, color="black", lw=2,
            label=f"fit: FD={st_atlas['intercept']:.2f}{st_atlas['slope']:+.2f}c1, "
                  f"r={st_atlas['r']:.2f}, n={st_atlas['n']}")
    ax.plot(xx, 3 - xx, "k--", lw=1, label="ideal FD=3-c1")
    ax.set_xlabel("c1_meas")
    ax.set_ylabel("DBC-measured FD")
    ax.set_title("Independent 2nd sample: natural_atlas (n=300)")
    ax.legend(fontsize=7, loc="best")

    fig.suptitle("Validating \"c1 is fractal dimension\" against real differential box counting",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    sanity = run_sanity()

    print()
    print("=" * 70)
    print("CORPUS PASS  (benchmarks/regressor/corpus/manifest.csv)")
    print("=" * 70)
    manifest_path = ROOT / "benchmarks" / "regressor" / "corpus" / "manifest.csv"
    corpus_df = measure_corpus(manifest_path)

    print()
    print("=" * 70)
    print("NATURAL_ATLAS PASS (independent 2nd sample)")
    print("=" * 70)
    atlas_csv = ROOT / "benchmarks" / "results" / "natural_atlas.csv"
    atlas_base = ROOT / "benchmarks" / "regressor" / "natural_atlas"
    atlas_df = measure_natural_atlas(atlas_csv, atlas_base)

    # combined CSV output
    combined = pd.concat([corpus_df, atlas_df], ignore_index=True)
    out_csv = RESULTS_DIR / "fd_validation.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(combined)} rows)")

    # stats
    print()
    print("=" * 70)
    print("RESULTS: corr(c1, DBC-FD)")
    print("=" * 70)
    st_overall = fit_stats(corpus_df.c1.values, corpus_df.dbc_fd.values)
    print(f"{'ALL SOURCES':<14} n={st_overall['n']:5d}  r={st_overall['r']:+.4f}  "
          f"FD ~ {st_overall['intercept']:.3f} {st_overall['slope']:+.3f}*c1")
    for src in sorted(corpus_df.source.unique()):
        sub = corpus_df[corpus_df.source == src]
        st = fit_stats(sub.c1.values, sub.dbc_fd.values)
        tag = "  <-- claimed 'photographic' source" if src == "macro" else ""
        print(f"{src:<14} n={st['n']:5d}  r={st['r']:+.4f}  "
              f"FD ~ {st['intercept']:.3f} {st['slope']:+.3f}*c1{tag}")
    st_atlas = fit_stats(atlas_df.c1.values, atlas_df.dbc_fd.values)
    print(f"{'natural_atlas':<14} n={st_atlas['n']:5d}  r={st_atlas['r']:+.4f}  "
          f"FD ~ {st_atlas['intercept']:.3f} {st_atlas['slope']:+.3f}*c1  (independent 2nd sample)")

    print()
    print("Doc's specific claim: corr(c1,FD) = -0.91, FD ~ 2.91 - 0.33*c1, n=200 photographic")

    # figure
    fig_path = FIG_DIR / "57_fd_validation.png"
    make_figure(sanity, corpus_df, atlas_df, fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
