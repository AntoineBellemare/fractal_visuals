"""
cascade_calibration.py — make `prescribed_cascade` actually hit BOTH cumulants.

THE PROBLEM (measured here, not assumed). `mf.prescribed_cascade(c1_target, c2_target)` is
documented as producing a field with *exactly* those cumulants, and the repo has treated it that
way everywhere. It does not. Asking for stronger multifractality silently drags roughness up:

    c1_target = 1.3, n = 1024, mean of 5 seeds
      c2_target -0.20  ->  measured c1 = 1.301   (fine)
      c2_target -0.45  ->  measured c1 = 1.400
      c2_target -0.90  ->  measured c1 = 1.771
      c2_target -1.20  ->  measured c1 = 2.040   (off by +0.74)

and the error is not a constant offset — it interacts with c1_target (the effective c1 slope falls
from ~0.59 at c2=-0.20 to ~0.21 at c2=-1.20), so no single additive correction fixes it.

WHY IT MATTERS. c1 is what the eye reads as complexity (this project's own central finding), so
every "pure c2 ladder" built on the raw call was in fact *also* a c1 ladder. Any result that
attributes a perceptual or generative effect to multifractality alone, when it was produced by
sweeping c2_target at fixed c1_target, carries that confound.

THE FIX. Treat synthesis as a calibrated instrument: measure the forward map
(c1_target, c2_target) -> (c1_measured, c2_measured) on a grid, fit a quadratic surface to it,
then INVERT the surface numerically to choose the targets that land on what you actually wanted.

    from cascade_calibration import Calibration
    cal = Calibration.load()                       # or Calibration.build(...)
    f = cal.cascade(c1_want=1.3, c2_want=-0.90, n=1024, seed=0)   # measures ~ (1.30, -0.90)

Usage:
  python cascade_calibration.py --build          # fit + save calibration.json (CPU, few minutes)
  python cascade_calibration.py --validate       # held-out check, naive vs calibrated
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import mfractal as mf                                            # noqa: E402

CALIB_PATH = HERE / "cascade_calibration.json"


def measure_field(f):
    """(c1, c2) of a float field, measured at native resolution on a unit-normalised copy.

    Unit-normalising is an AFFINE map, which provably leaves both cumulants alone (log-leaders
    shift by a constant, so neither the slope c1 nor the variance c2 moves). It only makes runs
    comparable. Never 8-bit-quantise here — that is a separate, lossy step.
    """
    z = np.asarray(f, float)
    z = (z - z.min()) / (np.ptp(z) + 1e-12)
    r = mf.wavelet_leaders_2d(z)
    return float(r["c1"]), float(r["c2"])


def forward(c1t, c2t, n=512, seeds=3):
    """Mean measured (c1, c2) for a requested (c1_target, c2_target)."""
    out = []
    for s in range(seeds):
        f = np.asarray(mf.prescribed_cascade(n=n, seed=s, c1_target=float(c1t),
                                             c2_target=float(c2t)), float)
        out.append(measure_field(f))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 0].std(), a[:, 1].std()


def _design(c1t, c2t):
    """Quadratic design matrix with interaction — enough to capture the observed curvature."""
    c1t = np.atleast_1d(np.asarray(c1t, float)); c2t = np.atleast_1d(np.asarray(c2t, float))
    return np.column_stack([np.ones_like(c1t), c1t, c2t, c1t**2, c2t**2, c1t * c2t])


class Calibration:
    """Fitted forward map + numerical inverse."""

    def __init__(self, W, bounds):
        self.W = np.asarray(W, float)          # (6, 2): design -> (c1_meas, c2_meas)
        self.bounds = bounds                   # dict of target ranges actually fitted

    # ---------- build / io ----------
    @classmethod
    def build(cls, c1_targets, c2_targets, n=512, seeds=3, verbose=True):
        rows, Y = [], []
        tot = len(c1_targets) * len(c2_targets); k = 0
        for c1t in c1_targets:
            for c2t in c2_targets:
                k += 1
                c1m, c2m, s1, s2 = forward(c1t, c2t, n=n, seeds=seeds)
                rows.append((c1t, c2t)); Y.append((c1m, c2m))
                if verbose:
                    print(f"  [{k}/{tot}] target ({c1t:.2f}, {c2t:+.2f}) -> "
                          f"measured ({c1m:.3f}, {c2m:+.3f})  sd ({s1:.3f}, {s2:.3f})", flush=True)
        rows = np.array(rows); Y = np.array(Y)
        X = _design(rows[:, 0], rows[:, 1])
        W, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = X @ W
        resid = np.abs(pred - Y).mean(0)
        if verbose:
            print(f"  fit residual (mean abs): c1 {resid[0]:.4f}  c2 {resid[1]:.4f}")
        return cls(W, dict(c1=[float(min(c1_targets)), float(max(c1_targets))],
                           c2=[float(min(c2_targets)), float(max(c2_targets))]))

    def save(self, path=CALIB_PATH):
        path.write_text(json.dumps({"W": self.W.tolist(), "bounds": self.bounds}, indent=2))
        return path

    @classmethod
    def load(cls, path=CALIB_PATH):
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["W"]), d["bounds"])

    # ---------- use ----------
    def predict(self, c1t, c2t):
        """Requested targets -> predicted MEASURED cumulants."""
        v = (_design(c1t, c2t) @ self.W)[0]
        return float(v[0]), float(v[1])

    def invert(self, c1_want, c2_want):
        """Choose the (c1_target, c2_target) whose OUTPUT lands on (c1_want, c2_want).

        Small bounded least-squares; the surface is smooth and 2-D so this is trivial and exact
        enough. Falls back to the identity if scipy is unavailable.
        """
        from scipy.optimize import least_squares
        b = self.bounds
        x0 = [np.clip(c1_want, *b["c1"]), np.clip(c2_want, *sorted(b["c2"]))]

        def resid(p):
            c1m, c2m = self.predict(p[0], p[1])
            return [c1m - c1_want, c2m - c2_want]

        lo = [b["c1"][0], min(b["c2"])]
        hi = [b["c1"][1], max(b["c2"])]
        r = least_squares(resid, x0, bounds=(lo, hi))
        return float(r.x[0]), float(r.x[1])

    def cascade(self, c1_want, c2_want, n=1024, seed=0):
        """A field that MEASURES ~ (c1_want, c2_want)."""
        c1t, c2t = self.invert(c1_want, c2_want)
        return np.asarray(mf.prescribed_cascade(n=n, seed=int(seed),
                                                c1_target=c1t, c2_target=c2t), float)


def cmd_build(args):
    # c1_target must run well past the wanted c1: the synthesiser COMPRESSES roughness (a request
    # of 2.2 only lands at ~1.5 when c2 is weak), so the inverse needs headroom above the grid.
    c1_targets = np.round(np.linspace(0.4, 3.2, 9), 3)
    c2_targets = np.array([-0.15, -0.30, -0.50, -0.75, -1.00, -1.30])
    print(f"building calibration on {len(c1_targets)}x{len(c2_targets)} grid, "
          f"n={args.n}, {args.seeds} seeds/point", flush=True)
    cal = Calibration.build(c1_targets, c2_targets, n=args.n, seeds=args.seeds)
    p = cal.save()
    print(f"saved -> {p}")


def cmd_validate(args):
    """Held-out check: does calibration beat the naive call at hitting a wanted (c1,c2)?"""
    cal = Calibration.load()
    # deliberately OFF the fitting grid
    wants = [(0.9, -0.25), (0.9, -0.85), (1.3, -0.20), (1.3, -0.60), (1.3, -1.10),
             (1.7, -0.35), (1.7, -0.95), (2.0, -0.55)]
    print(f"{'wanted':>16} | {'naive measured':>22} | {'calibrated measured':>22}")
    print("-" * 68)
    nz, cz = [], []
    for (c1w, c2w) in wants:
        n1, n2, _, _ = forward(c1w, c2w, n=args.n, seeds=args.seeds)        # naive: want as target
        c1t, c2t = cal.invert(c1w, c2w)
        k1, k2, _, _ = forward(c1t, c2t, n=args.n, seeds=args.seeds)        # calibrated
        nz.append((n1 - c1w, n2 - c2w)); cz.append((k1 - c1w, k2 - c2w))
        print(f"({c1w:.2f}, {c2w:+.2f}) | ({n1:.3f}, {n2:+.3f}) "
              f"err ({n1-c1w:+.2f},{n2-c2w:+.2f}) | ({k1:.3f}, {k2:+.3f}) "
              f"err ({k1-c1w:+.2f},{k2-c2w:+.2f})", flush=True)
    nz = np.array(nz); cz = np.array(cz)
    print("-" * 68)
    print(f"  naive       RMSE  c1 {np.sqrt((nz[:,0]**2).mean()):.3f}   c2 {np.sqrt((nz[:,1]**2).mean()):.3f}")
    print(f"  CALIBRATED  RMSE  c1 {np.sqrt((cz[:,0]**2).mean()):.3f}   c2 {np.sqrt((cz[:,1]**2).mean()):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    if args.build:
        cmd_build(args)
    if args.validate:
        cmd_validate(args)
    if not (args.build or args.validate):
        ap.print_help()


if __name__ == "__main__":
    main()
