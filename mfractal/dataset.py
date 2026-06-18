"""Build and validate a balanced multifractal-stimulus dataset.

build_dataset() sweeps family x complexity x seed, renders each stimulus, saves a
PNG, and writes a manifest.csv with per-image multifractal measurements.
validate_dataset() aggregates the manifest, checks that the multifractal descriptor
moves monotonically with the complexity knob, and writes a summary + figure.

Rock families accept a `complexity` argument; fluid families do not (complexity is
recorded as NaN for them). The recommended descriptor is the wavelet-leader
intermittency c2 (robust, monotonic). MFDFA delta_alpha / alpha0 are also available
but noisier.
"""
import os
import csv
import math
import numpy as np

from . import (textures, fluids, clouds, bark, fire, levy, wavelet_cascade,
                attractors, interference)
from .textures import ROCK_FAMILIES
from .fluids import FLUID_FAMILIES
from .clouds import CLOUD_FAMILIES
from .fluids import CX_FLUIDS
from .bark import BARK_FAMILIES
from .fire import FIRE_FAMILIES
from .levy import LEVY_FAMILIES
from .wavelet_cascade import WAVELET_FAMILIES
from .attractors import ATTRACTOR_FAMILIES
from .interference import INTERFERENCE_FAMILIES
from .flagships import punch
from .quantify import wavelet_leaders_2d, mfdfa_2d, spectrum_summary

__all__ = ["build_dataset", "validate_dataset", "generate"]


def generate(family, n=256, complexity=None, seed=None):
    """Generate one raw [0,1] stimulus for any rock, cloud, fluid, or bark family."""
    if family in FLUID_FAMILIES:
        if family in CX_FLUIDS:
            return getattr(fluids, family)(n, seed=seed,
                                          complexity=(0.5 if complexity is None else complexity))
        return getattr(fluids, family)(n, seed=seed)
    if family in CLOUD_FAMILIES:
        return getattr(clouds, family)(n, seed=seed, complexity=(0.5 if complexity is None else complexity))
    if family in BARK_FAMILIES:
        return getattr(bark, family)(n, seed=seed, complexity=(0.5 if complexity is None else complexity))
    if family in FIRE_FAMILIES:
        return getattr(fire, family)(n, seed=seed, complexity=(0.5 if complexity is None else complexity))
    if family in LEVY_FAMILIES:
        return getattr(levy, family)(n, seed=seed, complexity=(0.5 if complexity is None else complexity))
    if family in WAVELET_FAMILIES:
        return getattr(wavelet_cascade, family)(n, seed=seed, complexity=(0.5 if complexity is None else complexity))
    if family in ATTRACTOR_FAMILIES:
        return getattr(attractors, family)(n, seed=seed, complexity=(0.5 if complexity is None else complexity))
    if family in INTERFERENCE_FAMILIES:
        return getattr(interference, family)(n, seed=seed, complexity=(0.5 if complexity is None else complexity))
    fn = getattr(textures, family)
    if complexity is None:
        return fn(n, seed=seed)
    return fn(n, seed=seed, complexity=complexity)


def _save_png(arr01, path):
    import matplotlib.pyplot as plt
    plt.imsave(path, np.clip(arr01, 0, 1), cmap="gray", vmin=0, vmax=1)


def build_dataset(out_dir, families=None, complexities=(0.0, 0.25, 0.5, 0.75, 1.0),
                  seeds=range(5), n=256, estimator="wavelet", strength=1.6,
                  save_images=True, qn=13, verbose=True):
    """Render a balanced dataset and write out_dir/manifest.csv.

    estimator: 'wavelet' (c1,c2), 'mfdfa' (delta_alpha,alpha0), 'both', or None.
    Returns the list of manifest rows.
    """
    if families is None:
        families = list(ROCK_FAMILIES)
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    if save_images:
        os.makedirs(img_dir, exist_ok=True)
    qs = np.linspace(-3, 3, qn)
    rows = []
    for family in families:
        rock = (family not in FLUID_FAMILIES) and (family not in CLOUD_FAMILIES)
        takes_complexity = (family not in FLUID_FAMILIES) or (family in CX_FLUIDS)
        for c in (complexities if takes_complexity else [None]):
            for s in seeds:
                raw = generate(family, n=n, complexity=c, seed=int(s))
                disp = punch(raw, low=2, high=98, strength=strength)
                fname = f"{family}_c{(-9 if c is None else c):.2f}_s{int(s)}.png"
                if save_images:
                    _save_png(disp, os.path.join(img_dir, fname))
                row = {"filename": os.path.join("images", fname), "family": family,
                       "type": ("fluid" if family in FLUID_FAMILIES
                                else "cloud" if family in CLOUD_FAMILIES
                                else "bark" if family in BARK_FAMILIES
                                else "fire" if family in FIRE_FAMILIES
                                else "rock"),
                       "complexity": ("" if c is None else round(c, 4)),
                       "seed": int(s), "n": n}
                if estimator in ("wavelet", "both"):
                    r = wavelet_leaders_2d(raw)
                    row["c1"] = round(r["c1"], 4)
                    row["c2"] = round(r["c2"], 4)
                if estimator in ("mfdfa", "both"):
                    sp = spectrum_summary(qs, mfdfa_2d(raw, q_list=qs)["tau"])
                    row["delta_alpha"] = round(sp["delta_alpha"], 4)
                    row["alpha0"] = round(sp["alpha0"], 4)
                rows.append(row)
        if verbose:
            print(f"  {family}: done")
    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"manifest: {os.path.join(out_dir, 'manifest.csv')}  ({len(rows)} rows)")
    return rows


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (math.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12))


def validate_dataset(manifest_csv, out_fig=None, value="c2", summary_csv=None):
    """Aggregate a manifest: per-family mean/std of `value` vs complexity, plus the
    Spearman rank correlation of `value` with complexity (monotonicity check).
    Writes a summary CSV and (optionally) a figure. Returns the summary rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(manifest_csv)))
    rocks = sorted({r["family"] for r in rows if r["complexity"] != ""})
    summary = []
    fams_plot = []
    for fam in rocks:
        rr = [r for r in rows if r["family"] == fam and r["complexity"] != ""]
        if not rr:
            continue
        cs = sorted({float(r["complexity"]) for r in rr})
        means, stds = [], []
        allc, allv = [], []
        for c in cs:
            vals = [float(r[value]) for r in rr if float(r["complexity"]) == c]
            means.append(np.mean(vals)); stds.append(np.std(vals))
            allc += [c] * len(vals); allv += vals
        rho = _spearman(allc, allv)
        summary.append({"family": fam, "value": value,
                        "low": round(means[0], 4), "high": round(means[-1], 4),
                        "span": round(means[-1] - means[0], 4),
                        "spearman_rho": round(rho, 3)})
        fams_plot.append((fam, cs, means, stds))
    if summary_csv is None:
        summary_csv = os.path.join(os.path.dirname(manifest_csv), f"validation_{value}.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    if out_fig:
        ncol = 5
        nrow = int(np.ceil(len(fams_plot) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow))
        for ax, (fam, cs, means, stds) in zip(np.atleast_1d(axes).ravel(), fams_plot):
            ax.errorbar(cs, means, yerr=stds, marker="o", lw=1.8, capsize=3)
            rho = next(s["spearman_rho"] for s in summary if s["family"] == fam)
            ax.set_title(f"{fam}  rho={rho:+.2f}", fontsize=10)
            ax.set_xlabel("complexity", fontsize=8); ax.set_ylabel(value, fontsize=8)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        for ax in np.atleast_1d(axes).ravel()[len(fams_plot):]:
            ax.axis("off")
        lab = {"c2": "wavelet-leader intermittency c2 (more negative = more multifractal)",
               "delta_alpha": "MFDFA \u0394\u03b1", "c1": "wavelet-leader c1 (~ dominant Holder exp)"}.get(value, value)
        fig.suptitle(f"Dataset validation: {lab} vs complexity", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_fig, dpi=92, bbox_inches="tight")
    return summary
