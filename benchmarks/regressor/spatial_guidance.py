"""
spatial_guidance.py — SPATIALLY VARYING (c1,c2) control, on the gradient path.

Why this exists: every conditioning-IMAGE route (scaffold img2img, ControlNet) is capped by the
8-bit conditioning bottleneck (a field at c2=-0.80 measures -0.356 once encoded; no encoding,
float or uint8, avoids it). Classifier guidance escapes that cap because the control signal is a
GRADIENT, not an image — but until now it applied ONE GLOBAL target to the whole latent.

Idea: the joint regressor is `head(GAP(body(z)))`. Keep body and head; replace the GLOBAL average
pool with PER-REGION pooling. body(z) is (B,256,8,8) for a 1024 image, so a GxG split gives each
region a (8/G)x(8/G) patch of the feature map. Apply the same head per region -> a GxG map of
(c1,c2) predictions. Loss = sum over regions of ||pred_region - target_region||^2, so the gradient
pushes different parts of the image toward different statistics.

Region size matters: local cumulant estimation is noise-dominated below whole-image scale, so
G=2 (each region = 512x512 px of a 1024 image = the canonical measurement resolution) is the
reliable setting. G=4 is offered but is near the noise floor.

Usage:
  python spatial_guidance.py --mode validate      # does per-region prediction track truth?
  python spatial_guidance.py --mode gradient      # generate + measure a spatial c2 ramp
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import mfractal as mf                                  # noqa: E402
import guided_sample as gs                             # noqa: E402
import joint_montage as jm                             # noqa: E402
from build_corpus import measure_c2, load_field        # noqa: E402
from latent_regressor import SDXL_SCALING              # noqa: E402


def region_predict(model, z, grid=2):
    """(B,4,H,W) latent -> (B, grid, grid, 2) per-region (c1,c2), reusing the trained head."""
    f = model.body(z)                                   # (B,256,8,8) at 1024
    B, C, H, W = f.shape
    g = grid
    rh, rw = H // g, W // g
    f = f[:, :, :rh * g, :rw * g].reshape(B, C, g, rh, g, rw)
    pooled = f.mean(dim=(3, 5))                         # (B,C,g,g)
    pooled = pooled.permute(0, 2, 3, 1).reshape(B * g * g, C)
    out = model.head(pooled).reshape(B, g, g, -1)
    return out


def spatial_guided(pipe, model, prompt, c1_map, c2_map, *, steps=40, cfg=6.0, scale=150.0,
                   w1=1.0, w2=1.0, warmup=12, res=1024, seed=0, grid=2, device="cuda"):
    """Generate with a per-region (c1,c2) target map (grid x grid)."""
    g_ = torch.Generator(device=device).manual_seed(int(seed))
    sched = pipe.scheduler; sched.set_timesteps(steps, device=device)
    abar = sched.alphas_cumprod.to(device)
    pe, npe, ppe, nppe = gs._encode_prompt(pipe, prompt, device)
    emb = torch.cat([npe, pe]); add_text = torch.cat([nppe, ppe])
    add_time = pipe._get_add_time_ids(
        (res, res), (0, 0), (res, res), dtype=pe.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
    ).to(device).repeat(2, 1)
    added = {"text_embeds": add_text, "time_ids": add_time}
    lat = torch.randn((1, pipe.unet.config.in_channels, res // 8, res // 8),
                      generator=g_, device=device, dtype=pe.dtype) * sched.init_noise_sigma
    tgt = torch.tensor(np.stack([c1_map, c2_map], -1), dtype=torch.float32, device=device)[None]
    wts = torch.tensor([w1, w2], device=device)

    for i, t in enumerate(sched.timesteps):
        with torch.no_grad():
            li = sched.scale_model_input(torch.cat([lat] * 2), t)
            eps = pipe.unet(li, t, encoder_hidden_states=emb, added_cond_kwargs=added).sample
            eu, ec = eps.chunk(2); eps = eu + cfg * (ec - eu)
        lat = sched.step(eps, t, lat).prev_sample
        if i >= warmup and scale > 0:
            at = abar[t].clamp(1e-6, 1)
            l = lat.detach().requires_grad_(True)
            x0 = (l - (1 - at).sqrt() * eps.detach()) / at.sqrt()
            pred = region_predict(model, x0.float(), grid)          # (1,g,g,2)
            loss = (wts * (pred - tgt) ** 2).sum()
            grad = torch.autograd.grad(loss, l)[0]
            lat = (lat - scale * grad).detach().to(pe.dtype)
    with torch.no_grad():
        img = pipe.vae.decode(lat / SDXL_SCALING).sample
    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def quad_measure(img, grid=2):
    """True (c1,c2) of each image region, via the real estimator."""
    a = np.asarray(img.convert("L"), float) / 255.0
    H, W = a.shape; out = np.zeros((grid, grid, 2))
    for i in range(grid):
        for j in range(grid):
            sub = a[i * H // grid:(i + 1) * H // grid, j * W // grid:(j + 1) * W // grid]
            s = np.asarray(Image.fromarray((sub * 255).astype(np.uint8)).resize((512, 512)),
                           float) / 255.0
            r = mf.wavelet_leaders_2d(s)
            out[i, j] = (r["c1"], r["c2"])
    return out


def do_validate(args, device):
    """Does the region-pooled head actually predict each region's TRUE statistics?"""
    import pandas as pd
    from diffusers import AutoencoderKL
    import torch.nn.functional as F
    from train_stage1 import robust_unit
    model = jm.load_joint(device)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16).to(device).eval()
    d = pd.read_csv(HERE / "corpus" / "manifest.csv")
    d = d[d.source.isin(["diffusion", "macro"])].sample(args.n, random_state=0)
    P, T = [], []
    for _, r in d.iterrows():
        try:
            p = Path(r["path"]); p = p if p.is_absolute() else ROOT / p
            img = Image.open(p).convert("RGB")
            f = robust_unit(np.asarray(img.convert("L"), float))
            t = torch.from_numpy(f)[None, None].float()
            t = F.interpolate(t, size=(1024, 1024), mode="bicubic", align_corners=False)
            t = (t.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1).to(device, torch.float16)
            with torch.no_grad():
                z = vae.encode(t).latent_dist.mean * SDXL_SCALING
                pred = region_predict(model, z.float(), args.grid)[0].cpu().numpy()
            P.append(pred); T.append(quad_measure(img, args.grid))
        except Exception:
            pass
    P, T = np.array(P), np.array(T)
    print(f"\nper-region prediction vs TRUE region statistics  (n={len(P)}, grid {args.grid})")
    for k, nm in ((0, "c1"), (1, "c2")):
        p, t = P[..., k].ravel(), T[..., k].ravel()
        print(f"  {nm}: corr {np.corrcoef(p, t)[0,1]:+.3f}  RMSE {np.sqrt(((p-t)**2).mean()):.3f}"
              f"  bias {np.mean(p-t):+.3f}")
    # does it resolve WITHIN-image differences (the thing spatial guidance needs)?
    for k, nm in ((0, "c1"), (1, "c2")):
        dp = P[..., k] - P[..., k].mean(axis=(1, 2), keepdims=True)
        dt = T[..., k] - T[..., k].mean(axis=(1, 2), keepdims=True)
        print(f"  {nm} WITHIN-image deviation: corr {np.corrcoef(dp.ravel(), dt.ravel())[0,1]:+.3f}"
              f"   (this is what spatial guidance actually steers)")


def do_gradient(args, device):
    pipe = gs.build_pipe(device); model = jm.load_joint(device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    g = args.grid
    rows = []
    for name, prompt in [("marble", "polished marble slab, natural texture, photorealistic"),
                         ("forest", "dense forest canopy seen from above, photorealistic"),
                         ("frost",  "frost fern crystals on cold glass, photorealistic")]:
        # horizontal c2 ramp: left calm, right turbulent; c1 held constant
        c1_map = np.full((g, g), 1.3)
        ramp = np.linspace(args.c2_lo, args.c2_hi, g)
        c2_map = np.tile(ramp[None, :], (g, 1))
        img = spatial_guided(pipe, model, prompt, c1_map, c2_map, scale=args.scale,
                             warmup=args.warmup, steps=args.steps, res=args.res,
                             seed=args.seed, grid=g, device=device)
        img.save(out / f"spatial_{name}.png")
        meas = quad_measure(img, g)
        left = meas[:, 0, 1].mean(); right = meas[:, -1, 1].mean()
        want = args.c2_hi - args.c2_lo
        got = right - left
        rows.append(dict(substrate=name, c2_left=round(left, 3), c2_right=round(right, 3),
                         requested_delta=round(want, 3), achieved_delta=round(got, 3),
                         transmitted_pct=round(100 * got / want, 1)))
        print(f"  {name:8s} left c2 {left:+.3f}  right c2 {right:+.3f}  "
              f"-> {100*got/want:.0f}% of the requested ramp", flush=True)
    with (out / "spatial.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    best = max(r["transmitted_pct"] for r in rows)
    print(f"\nbest transmission {best:.0f}%   (ControlNet on scenes was ~5%)")
    print(f"-> {out/'spatial.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "gradient"], default="validate")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--c2-lo", type=float, default=-0.15)
    ap.add_argument("--c2-hi", type=float, default=-0.85)
    ap.add_argument("--scale", type=float, default=150.0)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "spatial_out"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (do_validate if args.mode == "validate" else do_gradient)(args, device)


if __name__ == "__main__":
    main()
