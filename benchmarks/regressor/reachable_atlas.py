"""
reachable_atlas.py — map the reachable (c1,c2) region per texture/domain, with examples.

For one texture per domain, joint-guide a grid of (c1,c2) TARGETS and measure what's
actually ACHIEVED. Outputs:
  * reachable_region.png  — achieved (c1,c2) points per domain + convex-hull envelope,
                            over the natural corpus cloud. Shows how much of the
                            complexity x intermittency plane each texture can cover.
  * atlas_<fam>.png       — per-texture grid of the actual generated examples (rows=c1
                            target, cols=c2 target), each labelled with measured (c1,c2).
                            The visual RANGE of examples.

Uses the (enriched) joint regressor. Joint guidance w1=w2=1, warmup to avoid artifacts.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import guided_sample as gs                       # noqa: E402
import joint_montage as jm                        # noqa: E402
from build_corpus import measure_c2               # noqa: E402

TEXTURES = [("rock", "marble"), ("cloud", "cirrus"), ("fluid", "dye_diffusion"),
            ("fire", "firestorm"), ("ice", "frost_fern"), ("water", "ink_bloom")]
COLORS = {"rock": "#8c564b", "cloud": "#9467bd", "fluid": "#0aa6c2", "fire": "#d62728",
          "ice": "#17becf", "water": "#1f77b4"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-vals", default="1.8,1.3,0.8")
    ap.add_argument("--c2-vals", default="-0.15,-0.45,-0.80")
    ap.add_argument("--scale", type=float, default=150.0)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atlas-fams", default="marble,ferrofluid,frost_fern",
                    help="which families to also render as full example-grid montages")
    ap.add_argument("--out", default=str(HERE / "reachable"))
    args = ap.parse_args()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    c1v = [float(x) for x in args.c1_vals.split(",")]
    c2v = [float(x) for x in args.c2_vals.split(",")]
    atlas_fams = args.atlas_fams.split(",")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = gs.build_pipe(device); model = jm.load_joint(device)

    achieved = {}      # (dom,fam) -> list of (c1m,c2m)
    tiles = {}         # (dom,fam) -> {(ri,ci):(thumb,c1m,c2m)}
    rows = []
    # union of the named domain textures + any extra atlas families
    extra = [(None, f) for f in atlas_fams if f not in [t[1] for t in TEXTURES]]
    for dom, fam in TEXTURES + extra:
        prompt = jm.load_prompt(fam, ("prompts", "prompts_intricate"))
        achieved[(dom, fam)] = []; tiles[(dom, fam)] = {}
        for ri, c1t in enumerate(c1v):
            for ci, c2t in enumerate(c2v):
                img = jm.joint_guided(pipe, model, prompt, c1t, c2t, scale=args.scale,
                                      w1=1.0, w2=1.0, warmup=args.warmup, steps=args.steps,
                                      seed=args.seed, device=device)
                c1m, c2m = measure_c2(img)
                img.save(out / "images" / f"{fam}_c1{c1t:+.2f}_c2{c2t:+.2f}.png")
                achieved[(dom, fam)].append((c1m, c2m))
                tiles[(dom, fam)][(ri, ci)] = (img.resize((300, 300), Image.BICUBIC), c1m, c2m)
                rows.append(dict(domain=dom, family=fam, c1_target=c1t, c2_target=c2t,
                                 c1_meas=round(c1m, 3), c2_meas=round(c2m, 3)))
                print(f"  {fam:14s} target(c1={c1t:+.2f},c2={c2t:+.2f}) -> "
                      f"({c1m:.3f},{c2m:+.3f})", flush=True)

    with (out / "reachable.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "family", "c1_target", "c2_target",
                                          "c1_meas", "c2_meas"]); w.writeheader(); w.writerows(rows)
    _region_plot(achieved, out / "reachable_region.png")
    for dom, fam in TEXTURES + extra:
        if fam in atlas_fams:
            _atlas(tiles[(dom, fam)], c1v, c2v, fam, out / f"atlas_{fam}.png")
    print(f"\nfigures -> {out}/reachable_region.png + atlas_*.png")


def _region_plot(achieved, fp):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    # natural corpus cloud for context
    import pandas as pd
    d = pd.read_csv(HERE / "corpus" / "manifest.csv")
    d = d[(d.source == "diffusion") & (d.c1 > 0.5) & (d.c1 < 2.5) & (d.c2 > -1.4) & (d.c2 < 0.1)]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(d.c1, d.c2, s=6, color="#cccccc", alpha=0.4, label="corpus (natural)")
    for (dom, fam), pts in achieved.items():
        if dom is None:
            continue
        P = np.array(pts); c = COLORS.get(dom, "#444")
        ax.scatter(P[:, 0], P[:, 1], s=42, color=c, edgecolor="k", linewidth=0.4, zorder=3,
                   label=f"{dom}: {fam}")
        if len(P) >= 3:
            try:
                h = ConvexHull(P)
                poly = P[h.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color=c, alpha=0.12, zorder=2)
            except Exception:
                pass
    ax.set_xlabel("c1  (roughness / detail — lower = busier)")
    ax.set_ylabel("c2  (intermittency — lower = more clustered)")
    ax.set_title("Reachable (c1, c2) region per domain (joint guidance)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fp, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"region -> {fp}")


def _atlas(tile, c1v, c2v, fam, fp):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(c1v), len(c2v)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.4, nr * 2.55))
    axes = np.atleast_2d(axes)
    for r in range(nr):
        for c in range(nc):
            ax = axes[r][c]; t, c1m, c2m = tile[(r, c)]
            ax.imshow(t); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"c1={c1m:.2f}  c2={c2m:+.2f}", fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(f"c1 target\n{c1v[r]:+.2f}", fontsize=8)
    for c in range(nc):
        axes[0][c].text(0.5, 1.30, f"c2 target {c2v[c]:+.2f}", transform=axes[0][c].transAxes,
                        ha="center", fontsize=8, fontweight="bold")
    fig.suptitle(f"Range of examples — {fam} across the (c1, c2) grid", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97]); fig.savefig(fp, dpi=190, bbox_inches="tight")
    plt.close(fig); print(f"atlas -> {fp}")


if __name__ == "__main__":
    main()
