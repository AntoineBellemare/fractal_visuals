"""
c2_montage.py — the control figure: same prompt, same seed, dialed multifractality.

For each of N complex natural-texture substrates, fix the prompt AND the seed, then sweep
the c2 target via classifier guidance (latent regressor in the loop). The only thing that
changes across a row is the requested multifractality, so the montage shows one texture
morphing from smooth/monofractal (c2~0) to strongly intermittent (c2 strongly negative).

Rows = textures, columns = c2 levels. Each tile is annotated with its MEASURED c2.

Uses guided_sample.guided_sample (method=guidance, the clean "same seed" demo). A scaffold
variant (--method scaffold) imposes c2 structurally for substrates guidance can't push.

Usage:
  python c2_montage.py --seed 0 --guidance-scale 180
  python c2_montage.py --families marble,ferrofluid,frost_fern,ink_bloom,smoke_eddies \
      --targets=-0.7,-0.5,-0.35,-0.2,-0.08
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import guided_sample as gs                              # noqa: E402
from build_corpus import measure_c2                     # noqa: E402

# 10 complex natural textures (mixed from prompts.py + prompts_intricate.py).
DEFAULT_FAMILIES = ["marble", "agate", "ferrofluid", "frost_fern", "ink_bloom",
                    "smoke_eddies", "mycelium_net", "manganese_dendrite",
                    "cracked_glaze", "marbled_ink"]


def load_prompt_map(modules=("prompts", "prompts_intricate")):
    import importlib
    pm = {}
    for name in modules:
        m = importlib.import_module(name)
        for fam, ps in m.PROMPTS.items():
            pm.setdefault(fam, ps[0])
    return pm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    ap.add_argument("--targets", default="-0.7,-0.5,-0.35,-0.2,-0.08")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance-scale", type=float, default=180.0)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--method", choices=["guidance", "scaffold"], default="guidance")
    ap.add_argument("--strength", type=float, default=0.5, help="scaffold method only")
    ap.add_argument("--thumb", type=int, default=256, help="montage tile px")
    ap.add_argument("--out", default=str(HERE / "montage"))
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild the figure(s) from already-generated images + "
                         "montage_results.csv, for a curated --families subset (no GPU)")
    args = ap.parse_args()

    fams = args.families.split(",")
    targets = [float(x) for x in args.targets.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.assemble_only:
        import csv as _csv
        rows = list(_csv.DictReader((out / "montage_results.csv").open()))
        meas = {(r["family"], float(r["target"])): float(r["measured"]) for r in rows}
        tiles, results = {}, []
        for fam in fams:
            for ti, c2t in enumerate(targets):
                img = Image.open(out / f"{fam}_t{c2t:+.2f}.png")
                c2m = meas[(fam, c2t)]
                tiles[(fam, ti)] = (img.resize((args.thumb, args.thumb), Image.BICUBIC), c2m)
                results.append(dict(family=fam, target=c2t, measured=c2m))
        _assemble(tiles, fams, targets, args, out)
        _validation_plot(results, fams, targets, args, out)
        return

    pmap = load_prompt_map()
    missing = [f for f in fams if f not in pmap]
    if missing:
        ap.error(f"families not found in prompt modules: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = gs.build_pipe(device)
    regressor = gs.load_regressor(device) if args.method == "guidance" else None
    if args.method == "scaffold":
        import mf_scaffold as mfs
        from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
        i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        i2i.scheduler = EulerDiscreteScheduler.from_config(i2i.scheduler.config)
        i2i.to(device); i2i.set_progress_bar_config(disable=True)

    tiles = {}   # (fam, ti) -> (PIL thumb, measured c2)
    results = []
    for fam in fams:
        prompt = pmap[fam]
        for ti, c2t in enumerate(targets):
            if args.method == "guidance":
                img = gs.guided_sample(pipe, regressor, prompt, c2t, steps=args.steps,
                                       cfg=args.cfg, guidance_scale=args.guidance_scale,
                                       warmup=args.warmup, res=args.res, seed=args.seed,
                                       device=device)
            else:
                init, _ = mfs.scaffold_init(c2t, args.res, args.seed)
                g = torch.Generator(device=device).manual_seed(args.seed)
                img = i2i(prompt=prompt + mfs.TAIL, negative_prompt=mfs.NEG, image=init,
                          strength=args.strength, num_inference_steps=args.steps,
                          guidance_scale=args.cfg, generator=g).images[0]
            c2m = measure_c2(img)[1]
            img.save(out / f"{fam}_t{c2t:+.2f}.png")
            tiles[(fam, ti)] = (img.resize((args.thumb, args.thumb), Image.BICUBIC), c2m)
            results.append(dict(family=fam, target=c2t, measured=round(c2m, 4)))
            print(f"  {fam:18s} target {c2t:+.2f} -> measured {c2m:+.3f}", flush=True)

    import csv
    with (out / "montage_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "target", "measured"])
        w.writeheader(); w.writerows(results)
    _assemble(tiles, fams, targets, args, out)
    _validation_plot(results, fams, targets, args, out)


def _assemble(tiles, fams, targets, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(fams), len(targets)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.1, nr * 2.25))
    if nr == 1:
        axes = axes[None, :]
    for r, fam in enumerate(fams):
        for c in range(nc):
            ax = axes[r][c]
            thumb, c2m = tiles[(fam, c)]
            ax.imshow(thumb)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"c2={c2m:+.2f}", fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(fam, fontsize=9, rotation=0, ha="right", va="center")
        axes[r][0].yaxis.set_label_coords(-0.02, 0.5)
    for c, t in enumerate(targets):
        axes[0][c].text(0.5, 1.28, f"target {t:+.2f}", transform=axes[0][c].transAxes,
                        ha="center", fontsize=9, fontweight="bold")
    fig.suptitle(f"Controlled multifractality — same prompt + seed {args.seed}, "
                 f"c2 dialed left→right  ({args.method})", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])
    fp = out / f"c2_montage_{args.method}_s{args.seed}.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nmontage -> {fp}")


def _validation_plot(results, fams, targets, args, out):
    """Measured vs requested c2, one line per texture + y=x. Quantitative validation
    that the control is monotonic and accurate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_fam = {f: [] for f in fams}
    for r in results:
        by_fam[r["family"]].append((r["target"], r["measured"]))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = min(targets) - 0.1, max(targets) + 0.15
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="ideal (y=x)")
    cmap = plt.get_cmap("turbo")
    allt, allm = [], []
    for i, f in enumerate(fams):
        pts = sorted(by_fam[f])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        allt += xs; allm += ys
        ax.plot(xs, ys, "-o", color=cmap(i / max(1, len(fams) - 1)), ms=4, lw=1.3,
                label=f, alpha=0.9)
    allt, allm = np.array(allt), np.array(allm)
    slope, intc = np.polyfit(allt, allm, 1)
    r = float(np.corrcoef(allt, allm)[0, 1])
    ax.set_xlabel("requested c2 (target)"); ax.set_ylabel("measured c2 (wavelet-leader)")
    ax.set_title(f"Validation — measured vs requested multifractality\n"
                 f"pooled slope={slope:.2f}, r={r:.2f}  ({args.method}, seed {args.seed})",
                 fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3); ax.set_aspect("equal", "box")
    fig.tight_layout()
    fp = out / f"c2_validation_{args.method}_s{args.seed}.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"validation -> {fp}  (pooled slope {slope:.2f}, r {r:.2f})")


if __name__ == "__main__":
    main()
