"""
Seed-variation sweep at mid complexity (cx=0.5), one figure per domain.

For each domain (rock, cloud, fluid, bark, fire):
  rows    = families in that domain
  columns = seeds 0, 1, 2, 3, 4
  cell    = grayscale mf.punch(image)

Outputs:
  benchmarks/figures/20_seed_sweep_rock.png
  benchmarks/figures/21_seed_sweep_cloud.png
  benchmarks/figures/22_seed_sweep_fluid.png
  benchmarks/figures/23_seed_sweep_bark.png
  benchmarks/figures/25_seed_sweep_fire.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

DOMAIN = {
    "rock":  ["marble","agate","weathered","granite","vesicular","turbulent",
              "schist","serpentinite","flow_banded","convoluted","gneiss"],
    "cloud": ["cascade_lognormal","billow","cirrus",
              "cloud_multifractional","warped_fbm","ridged","billow_smoke",
              "universal_cascade","prescribed_cascade","de_jong_attractor"],
    "fluid": ["curl_weave","eddies","vorticity","dye_diffusion",
              "rheoscopic","wave_interference"],
    "bark":  ["burled_oak","riven_oak"],
    "fire":  ["firestorm","lava_pool","volcanic_fissure"],
}
SEEDS = [0, 1, 2, 3, 4]
CX = 0.5
N = 384

DOMAIN_FIG_IDX = {
    "rock": 20, "cloud": 21, "fluid": 22,
    "bark": 23, "fire": 25,
}


def render_domain(domain: str, families: list[str], out: Path):
    nrow = len(families); ncol = len(SEEDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.5 * nrow),
                             squeeze=False)
    for ri, fam in enumerate(families):
        for ci, s in enumerate(SEEDS):
            img = mf.generate(fam, n=N, complexity=CX, seed=s)
            disp = mf.punch(img)
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"seed = {s}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle(f"Seed sweep — {domain} families  (n={N}, cx={CX})",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    for dom, fams in DOMAIN.items():
        idx = DOMAIN_FIG_IDX[dom]
        out = ROOT / "benchmarks" / "figures" / f"{idx:02d}_seed_sweep_{dom}.png"
        render_domain(dom, fams, out)
