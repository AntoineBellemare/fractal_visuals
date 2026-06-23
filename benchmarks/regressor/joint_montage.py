"""
joint_montage.py — control BOTH cumulants and show it.

Guides SDXL with the 2-output joint regressor (c1, c2) in the loop. Two figures:
  --mode grid2d : one texture, ROWS = c1 (perceived complexity / detail), COLS = c2
                  (intermittency). Same prompt+seed. Proves the axes are independent.
  --mode c1sweep: ROWS = textures, COLS = c1 levels (c2 held fixed) -- the perceptually
                  obvious busy<->smooth control your eye actually reads.

Each tile is annotated with measured (c1, c2).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import guided_sample as gs                               # noqa: E402
from joint_regressor import JointCNN                     # noqa: E402
from latent_regressor import SDXL_SCALING                # noqa: E402
from build_corpus import measure_c2                      # noqa: E402


def load_joint(device):
    ckpt = torch.load(HERE / "joint_model.pt", map_location=device)
    m = JointCNN(in_ch=4).to(device); m.load_state_dict(ckpt["model"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_prompt(family, modules=("prompts", "prompts_intricate")):
    import importlib
    for mod in modules:
        pm = importlib.import_module(mod)
        if family in pm.PROMPTS:
            return pm.PROMPTS[family][0]
    raise SystemExit(f"family {family} not found in {modules}")


def joint_guided(pipe, model, prompt, c1t, c2t, *, steps=45, cfg=6.0, scale=180.0,
                 w1=1.0, w2=1.0, warmup=10, res=1024, seed=0, device="cuda"):
    """DPS guidance toward (c1t, c2t) jointly. w1/w2 weight the two cumulant losses."""
    g = torch.Generator(device=device).manual_seed(seed)
    sched = pipe.scheduler; sched.set_timesteps(steps, device=device)
    abar = sched.alphas_cumprod.to(device)
    pe, npe, ppe, nppe = gs._encode_prompt(pipe, prompt, device)
    prompt_embeds = torch.cat([npe, pe]); add_text = torch.cat([nppe, ppe])
    add_time = pipe._get_add_time_ids(
        (res, res), (0, 0), (res, res), dtype=pe.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
    ).to(device).repeat(2, 1)
    added = {"text_embeds": add_text, "time_ids": add_time}
    lat = torch.randn((1, pipe.unet.config.in_channels, res // 8, res // 8),
                      generator=g, device=device, dtype=pe.dtype) * sched.init_noise_sigma
    tgt = torch.tensor([c1t, c2t], device=device)
    wts = torch.tensor([w1, w2], device=device)
    for i, t in enumerate(sched.timesteps):
        with torch.no_grad():
            li = sched.scale_model_input(torch.cat([lat] * 2), t)
            eps = pipe.unet(li, t, encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs=added).sample
            eu, ec = eps.chunk(2); eps = eu + cfg * (ec - eu)
        lat = sched.step(eps, t, lat).prev_sample
        if i >= warmup and scale > 0:
            at = abar[t].clamp(1e-6, 1)
            l = lat.detach().requires_grad_(True)
            x0 = (l - (1 - at).sqrt() * eps.detach()) / at.sqrt()
            pred = model(x0.float())[0]                  # (2,) -> (c1,c2)
            loss = (wts * (pred - tgt) ** 2).sum()
            grad = torch.autograd.grad(loss, l)[0]
            lat = (lat - scale * grad).detach().to(pe.dtype)
    with torch.no_grad():
        img = pipe.vae.decode(lat / SDXL_SCALING).sample
    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def _grid(tiles, row_vals, col_vals, row_label, col_label, title, fp, thumb):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(row_vals), len(col_vals)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.1, nr * 2.25))
    axes = np.atleast_2d(axes)
    for r in range(nr):
        for c in range(nc):
            ax = axes[r][c]; thumb_img, c1m, c2m = tiles[(r, c)]
            ax.imshow(thumb_img); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"c1={c1m:.2f} c2={c2m:+.2f}", fontsize=7, pad=2)
            if c == 0:
                ax.set_ylabel(f"{row_label}\n{row_vals[r]:+.2f}", fontsize=8)
    for c in range(nc):
        axes[0][c].text(0.5, 1.30, f"{col_label} {col_vals[c]:+.2f}",
                        transform=axes[0][c].transAxes, ha="center", fontsize=8, fontweight="bold")
    fig.suptitle(title, fontsize=12, y=0.997)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97]); fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"figure -> {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grid2d", "c1sweep"], default="grid2d")
    ap.add_argument("--family", default="marble")               # grid2d
    ap.add_argument("--families", default="marble,ferrofluid,frost_fern,ink_bloom")  # c1sweep
    ap.add_argument("--c1-vals", default="2.0,1.6,1.2,0.8")
    ap.add_argument("--c2-vals", default="-0.7,-0.4,-0.15")
    ap.add_argument("--c2-fixed", type=float, default=-0.3)      # c1sweep
    ap.add_argument("--scale", type=float, default=180.0)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--thumb", type=int, default=320)
    ap.add_argument("--prompts-module", default="prompts,prompts_intricate",
                    help="comma-separated prompt modules to search for families, e.g. "
                         "prompts_ood for the out-of-distribution probe")
    ap.add_argument("--out", default=str(HERE / "joint_montage"))
    args = ap.parse_args()
    modules = tuple(args.prompts_module.split(","))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = gs.build_pipe(device); model = load_joint(device)
    c1v = [float(x) for x in args.c1_vals.split(",")]
    c2v = [float(x) for x in args.c2_vals.split(",")]
    rows = []

    if args.mode == "grid2d":
        prompt = load_prompt(args.family, modules)
        tiles = {}
        for r, c1t in enumerate(c1v):
            for c, c2t in enumerate(c2v):
                img = joint_guided(pipe, model, prompt, c1t, c2t, scale=args.scale,
                                   w1=args.w1, w2=args.w2, seed=args.seed, steps=args.steps,
                                   device=device)
                c1m, c2m = measure_c2(img)
                img.save(out / f"{args.family}_c1{c1t:+.2f}_c2{c2t:+.2f}.png")
                tiles[(r, c)] = (img.resize((args.thumb, args.thumb), Image.BICUBIC), c1m, c2m)
                rows.append(dict(family=args.family, c1_target=c1t, c2_target=c2t,
                                 c1_meas=round(c1m, 3), c2_meas=round(c2m, 3)))
                print(f"  c1t={c1t:+.2f} c2t={c2t:+.2f} -> c1={c1m:.3f} c2={c2m:+.3f}", flush=True)
        _grid(tiles, c1v, c2v, "c1 target", "c2 target",
              f"Independent control of c1 (rows) and c2 (cols) — {args.family}, seed {args.seed}",
              out / f"joint_grid2d_{args.family}_s{args.seed}.png", args.thumb)
    else:
        fams = args.families.split(",")
        tiles = {}
        for r, fam in enumerate(fams):
            prompt = load_prompt(fam, modules)
            for c, c1t in enumerate(c1v):
                img = joint_guided(pipe, model, prompt, c1t, args.c2_fixed, scale=args.scale,
                                   w1=args.w1, w2=args.w2, seed=args.seed, steps=args.steps,
                                   device=device)
                c1m, c2m = measure_c2(img)
                img.save(out / f"{fam}_c1{c1t:+.2f}.png")
                tiles[(r, c)] = (img.resize((args.thumb, args.thumb), Image.BICUBIC), c1m, c2m)
                rows.append(dict(family=fam, c1_target=c1t, c2_target=args.c2_fixed,
                                 c1_meas=round(c1m, 3), c2_meas=round(c2m, 3)))
                print(f"  {fam:14s} c1t={c1t:+.2f} -> c1={c1m:.3f} c2={c2m:+.3f}", flush=True)
        _grid(tiles, list(range(len(fams))), c1v, "texture", "c1 target",
              f"Perceived-complexity control (c1), c2 fixed {args.c2_fixed:+.2f}, seed {args.seed}",
              out / f"joint_c1sweep_s{args.seed}.png", args.thumb)

    with (out / "joint_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "c1_target", "c2_target", "c1_meas", "c2_meas"])
        w.writeheader(); w.writerows(rows)
    print(f"csv -> {out/'joint_results.csv'}")


if __name__ == "__main__":
    main()
