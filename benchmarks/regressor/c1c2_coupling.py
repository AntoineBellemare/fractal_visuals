"""
c1c2_coupling.py — CONTROL coupling: when you guide one cumulant, how does the other move?

For one representative texture per domain (water, ice, fire, rock, cloud, fluid), run two
guided sweeps and measure the resulting (c1, c2):
  * c1-sweep  : guide c1 only (w2=0), c1 target swept -> trajectory of how c2 drifts.
  * c2-sweep  : guide c2 only (w1=0), c2 target swept -> trajectory of how c1 drifts.

Plots both trajectories in the (c1, c2) plane, one panel per texture. A near-vertical
c1-sweep means c1 moves without disturbing c2 (independent); a diagonal means they're
coupled. Compares the CONTROL coupling to the NATURAL coupling (c1c2_domains.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import guided_sample as gs                       # noqa: E402
import joint_montage as jm                        # noqa: E402
from build_corpus import measure_c2               # noqa: E402

# one prompt-family per domain (must exist in prompts.py or prompts_intricate.py)
TEXTURES = [
    ("rock", "marble"), ("cloud", "cirrus"), ("fluid", "dye_diffusion"),
    ("fire", "firestorm"), ("ice", "frost_fern"), ("water", "ink_bloom"),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-vals", default="1.8,1.4,1.0,0.7")
    ap.add_argument("--c2-vals", default="-0.1,-0.35,-0.6,-0.85")
    ap.add_argument("--scale", type=float, default=150.0)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "c1c2_coupling"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    c1v = [float(x) for x in args.c1_vals.split(",")]
    c2v = [float(x) for x in args.c2_vals.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = gs.build_pipe(device); model = jm.load_joint(device)

    results = {}   # (domain, fam) -> dict(c1sweep=[(c1,c2)...], c2sweep=[...])
    for dom, fam in TEXTURES:
        prompt = jm.load_prompt(fam, ("prompts", "prompts_intricate"))
        c1s, c2s = [], []
        for c1t in c1v:                          # guide c1 only (c2 free)
            img = jm.joint_guided(pipe, model, prompt, c1t, -0.3, scale=args.scale,
                                  w1=1.0, w2=0.0, warmup=args.warmup, steps=args.steps,
                                  seed=args.seed, device=device)
            c1m, c2m = measure_c2(img); c1s.append((c1m, c2m))
            print(f"  [{fam:14s}] guide c1={c1t:+.2f} -> c1={c1m:.3f} c2={c2m:+.3f}", flush=True)
        for c2t in c2v:                          # guide c2 only (c1 free)
            img = jm.joint_guided(pipe, model, prompt, 1.3, c2t, scale=args.scale,
                                  w1=0.0, w2=1.0, warmup=args.warmup, steps=args.steps,
                                  seed=args.seed, device=device)
            c1m, c2m = measure_c2(img); c2s.append((c1m, c2m))
            print(f"  [{fam:14s}] guide c2={c2t:+.2f} -> c1={c1m:.3f} c2={c2m:+.3f}", flush=True)
        results[(dom, fam)] = dict(c1sweep=c1s, c2sweep=c2s)

    _plot(results, out / f"c1c2_coupling_s{args.seed}.png")
    import json
    (out / "coupling.json").write_text(json.dumps(
        {f"{d}:{f}": v for (d, f), v in results.items()}, indent=2))
    print(f"\nfigure -> {out/f'c1c2_coupling_s{args.seed}.png'}")


def _plot(results, fp):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results); ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.6, nrow * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, ((dom, fam), v) in zip(axes, results.items()):
        a = np.array(v["c1sweep"]); b = np.array(v["c2sweep"])
        ax.plot(a[:, 0], a[:, 1], "-s", color="#d62728", ms=6, lw=1.6,
                label="guide c1 (c2 free)")
        ax.plot(b[:, 0], b[:, 1], "-o", color="#1f77b4", ms=6, lw=1.6,
                label="guide c2 (c1 free)")
        # coupling slopes
        s1 = np.polyfit(a[:, 0], a[:, 1], 1)[0] if len(a) > 1 else float("nan")
        s2 = np.polyfit(b[:, 1], b[:, 0], 1)[0] if len(b) > 1 else float("nan")
        ax.set_title(f"{dom}: {fam}\ndc2/dc1={s1:+.2f}  dc1/dc2={s2:+.2f}", fontsize=9)
        ax.set_xlabel("c1"); ax.set_ylabel("c2"); ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Control coupling — guide one cumulant, measure the other", fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(fp, dpi=190, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
