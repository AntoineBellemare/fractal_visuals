"""
Complexity sweep for every family, one figure per domain.

For each domain (rock, cloud, fluid, bark, fire):
  rows    = families in that domain
  columns = complexity values 0.0, 0.25, 0.5, 0.75, 1.0
  cell    = grayscale image (mf.punch)

Outputs:
  benchmarks/figures/14_cx_sweep_rock.png
  benchmarks/figures/15_cx_sweep_cloud.png
  benchmarks/figures/16_cx_sweep_fluid.png
  benchmarks/figures/17_cx_sweep_bark.png
  benchmarks/figures/19_cx_sweep_fire.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mfractal as mf

DOMAIN = {
    "rock":  ["marble","agate","weathered","granite","vesicular","turbulent",
              "schist","serpentinite","flow_banded","convoluted","gneiss",
              "levy_marble"],
    "cloud": ["cascade_lognormal","stratified","billow","cirrus","cloud_mrw",
              "cloud_multifractional","warped_fbm","ridged","billow_smoke",
              "universal_cascade","prescribed_cascade","de_jong_attractor"],
    "fluid": ["curl_weave","choppy","eddies","vorticity","dye_diffusion",
              "rheoscopic","wave_interference"],
    "bark":  ["burled_oak","riven_oak"],
    "fire":  ["firestorm","lava_pool","volcanic_fissure"],
}
CXS = [0.0, 0.25, 0.5, 0.75, 1.0]
N = 384
SEED = 0

DOMAIN_FIG_IDX = {
    "rock": 14, "cloud": 15, "fluid": 16,
    "bark": 17, "fire": 19,
}


def render_domain(domain: str, families: list[str], out: Path):
    nrow = len(families); ncol = len(CXS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.5 * nrow),
                             squeeze=False)
    for ri, fam in enumerate(families):
        for ci, cx in enumerate(CXS):
            img = mf.generate(fam, n=N, complexity=cx, seed=SEED)
            disp = mf.punch(img)
            ax = axes[ri][ci]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"cx = {cx:.2f}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle(f"Complexity sweep — {domain} families  (n={N}, seed={SEED})",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    for dom, fams in DOMAIN.items():
        idx = DOMAIN_FIG_IDX[dom]
        out = ROOT / "benchmarks" / "figures" / f"{idx:02d}_cx_sweep_{dom}.png"
        render_domain(dom, fams, out)
