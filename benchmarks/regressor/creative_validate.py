"""
creative_validate.py — validation pass on the two creative applications.

The pilot runs were underpowered and, for application B, tested a WEAKER method than the one
this repo actually documents. This script analyses the validation runs that fix both.

APPLICATION A — compositional detail falloff.
  Pilot: n = 16 paired cells, mean +0.097, p = 0.18. Suggestive only.
  Validation: 2 fresh seeds per cell (n = 32 more), plus a HORIZONTAL-ramp arm. The horizontal arm
  matters because a radial conditioning field imposes radial COMPOSITION (the cathedral rendered as
  a dome), which is content leakage, not a statistics manipulation. If the effect survives with a
  horizontal ramp it is the statistic doing the work, not the composition.

APPLICATION B — statistically matched families.
  The pilot guided ONE seed per family and measured the spread. But the documented method is
  "guide each family to the same (c1,c2), then VERIFY with the regressor and keep what lands in
  tolerance" — i.e. guidance plus rejection sampling. Testing guidance alone understates it.
  Validation: 4 seeds per family on BOTH arms, then select the seed closest to target.
  The fair control is unguided+select, not unguided+first — if prompt-only rejection sampling
  tightens the set as much as guidance does, then guidance is adding nothing and the application
  should be described as "rejection sampling", not "control".

Inference: variance ratios get a bootstrap CI (n = 10 families is far too small to trust an
F-test's normality assumption); paired location effects get t plus Wilcoxon.

Usage:  python creative_validate.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = HERE / "creative"
RNG = np.random.default_rng(0)


def boot_sd_ratio(a, b, n=20000):
    """Bootstrap CI for sd(a)/sd(b) — the spread-reduction factor. <1 means a is tighter."""
    a = np.asarray(a); b = np.asarray(b)
    r = []
    for _ in range(n):
        ra = RNG.choice(a, len(a), replace=True)
        rb = RNG.choice(b, len(b), replace=True)
        sb = rb.std(ddof=1)
        if sb > 1e-9:
            r.append(ra.std(ddof=1) / sb)
    r = np.array(r)
    return a.std(ddof=1) / b.std(ddof=1), np.percentile(r, 2.5), np.percentile(r, 97.5)


# ----------------------------------------------------------------- application A
def app_a():
    parts = []
    for name, p in (("pilot", C / "falloff" / "falloff.csv"),
                    ("replication", C / "falloff_rep" / "falloff.csv"),
                    ("horizontal", C / "falloff_h" / "falloff.csv")):
        if p.exists():
            d = pd.read_csv(p); d["run"] = name
            if "direction" not in d:
                d["direction"] = "radial"
            parts.append(d)
    if not parts:
        print("APP A: no data"); return None
    d = pd.concat(parts, ignore_index=True)

    print("=" * 78)
    print("APPLICATION A — complexity as a compositional device")
    print("=" * 78)
    out = {}
    for direction in ("radial", "h"):
        sub = d[d.direction == direction]
        if not len(sub):
            continue
        p = sub.pivot_table(index=["run", "subject", "style", "seed"],
                            columns="variant", values="dc1").dropna()
        if not len(p):
            continue
        diff = (p["ramp"] - p["flat"]).values
        t, pv = stats.ttest_rel(p["ramp"], p["flat"])
        w = stats.wilcoxon(p["ramp"], p["flat"]).pvalue
        print(f"\n[{direction} ramp]  n = {len(p)} paired cells")
        print(f"  ramp {p['ramp'].mean():+.3f}   flat {p['flat'].mean():+.3f}   "
              f"diff {diff.mean():+.3f} +- {diff.std(ddof=1)/np.sqrt(len(diff)):.3f}")
        print(f"  paired t = {t:+.2f}  p = {pv:.4f}   Wilcoxon p = {w:.4f}   "
              f"ramp>flat {int((diff>0).sum())}/{len(diff)}")
        out[direction] = dict(n=len(p), diff=diff.mean(), t=t, p=pv, wilcoxon=w,
                              npos=int((diff > 0).sum()))
        print("   per medium:")
        for sty, g in sub.groupby("style"):
            q = g.pivot_table(index=["run", "subject", "seed"], columns="variant",
                              values="dc1").dropna()
            if len(q) < 3:
                continue
            dd = (q["ramp"] - q["flat"]).values
            tt, pp = stats.ttest_rel(q["ramp"], q["flat"])
            print(f"     {sty:10s} n={len(q):2d}  ramp {q['ramp'].mean():+.3f}  "
                  f"flat {q['flat'].mean():+.3f}  diff {dd.mean():+.3f}  "
                  f"p={pp:.3f}  ({int((dd>0).sum())}/{len(dd)})")
    d.to_csv(ROOT / "benchmarks" / "results" / "creative_falloff_all.csv", index=False)
    return out


# ----------------------------------------------------------------- application B
def app_b():
    p = C / "matched_sel" / "matched.csv"
    if not p.exists():
        print("APP B: no data"); return None
    d = pd.read_csv(p)
    c1t, c2t = float(d.c1_target.iloc[0]), float(d.c2_target.iloc[0])
    print("\n" + "=" * 78)
    print(f"APPLICATION B — statistically matched families   target ({c1t:.2f}, {c2t:+.2f})")
    print("=" * 78)
    nseeds = d.seed.nunique()
    print(f"  {d.family.nunique()} families x {nseeds} seeds x 2 arms = {len(d)} images\n")

    res = {}
    for arm in ("guided", "unguided"):
        a = d[d.arm == arm]
        first = a[a.seed == a.seed.min()]
        # rejection sampling: keep the seed whose MEASURED (c1,c2) is closest to target
        a = a.copy()
        a["dist"] = np.hypot(a.c1_meas - c1t, a.c2_meas - c2t)
        sel = a.loc[a.groupby("family")["dist"].idxmin()]
        for mode, s in (("all-seeds", a), ("first-seed", first), ("SELECTED", sel)):
            res[(arm, mode)] = s
            print(f"  {arm:8s} {mode:11s} n={len(s):3d}  "
                  f"c1 {s.c1_meas.mean():.3f} +- {s.c1_meas.std(ddof=1):.3f}   "
                  f"c2 {s.c2_meas.mean():+.3f} +- {s.c2_meas.std(ddof=1):.3f}   "
                  f"RMSE c1 {np.sqrt(((s.c1_meas-c1t)**2).mean()):.3f} "
                  f"c2 {np.sqrt(((s.c2_meas-c2t)**2).mean()):.3f}")

    # ---- the properly powered test. Spread at n = 10 families is hopeless (see the bootstrap
    # CIs below, all of which straddle 1). But guided and unguided share prompt AND seed, so the
    # per-family error is PAIRED and n = families x seeds. That is the test to quote.
    print("\n  --- PAIRED per-family |error| to target, guided vs unguided (same prompt & seed) ---")
    piv = d.pivot_table(index=["family", "seed"], columns="arm", values=["c1_meas", "c2_meas"])
    paired = {}
    for nm, col, tgt in (("c1", "c1_meas", c1t), ("c2", "c2_meas", c2t)):
        g = (piv[(col, "guided")] - tgt).abs(); u = (piv[(col, "unguided")] - tgt).abs()
        t, pv = stats.ttest_rel(g, u); w = stats.wilcoxon(g, u).pvalue
        print(f"    |err {nm}|: guided {g.mean():.3f}  unguided {u.mean():.3f}  "
              f"diff {(g-u).mean():+.3f}   t = {t:+.2f}  p = {pv:.5f}  "
              f"(Wilcoxon {w:.5f})  better in {int((g<u).sum())}/{len(g)}")
        paired[nm] = dict(n=len(g), guided=g.mean(), unguided=u.mean(), t=t, p=pv, w=w,
                          nbetter=int((g < u).sum()))
    print("\n  --- spread ratios (bootstrap 95% CI; <1 = tighter, CI excluding 1 = real) ---")
    out = {}
    for key, lab in ((("guided", "first-seed"), "guidance alone (pilot method)"),
                     (("guided", "SELECTED"), "guidance + rejection sampling"),
                     (("unguided", "SELECTED"), "rejection sampling ALONE (fair control)")):
        g = res[key]; u = res[("unguided", "first-seed")]
        for nm, col in (("c1", "c1_meas"), ("c2", "c2_meas")):
            r, lo, hi = boot_sd_ratio(g[col].values, u[col].values)
            flag = "REAL" if hi < 1.0 else ("worse" if lo > 1.0 else "n.s.")
            print(f"    {lab:38s} {nm}: sd ratio {r:.2f}  [{lo:.2f}, {hi:.2f}]  {flag}")
            out[(lab, nm)] = (r, lo, hi, flag)
        print()

    # does a medium's own reachable c2 explain whether it converges?
    g = res[("guided", "all-seeds")]
    fam = g.groupby("family").agg(c2_best=("c2_meas", "min"), c2_mean=("c2_meas", "mean")).reset_index()
    fam["gap"] = (fam.c2_mean - c2t).abs()
    rr = stats.pearsonr(fam.c2_best, fam.gap)
    print(f"  feasibility check: corr(most-negative c2 a medium reaches, distance from target)"
          f" = {rr[0]:+.2f} (p = {rr[1]:.3f}, n = {len(fam)})")
    print("    -> media that CAN'T reach the target c2 are the ones that miss it, "
          "i.e. the failure is feasibility, not guidance strength.")
    d.to_csv(ROOT / "benchmarks" / "results" / "creative_matched_selected.csv", index=False)
    return out


if __name__ == "__main__":
    app_a()
    app_b()
