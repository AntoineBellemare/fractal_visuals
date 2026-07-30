"""
spatial_guidance.py — SPATIALLY VARYING (c1,c2) control.  *** RESULT: DOES NOT WORK ***

Kept as a documented negative result. Read this before trying spatial control again.

The idea: the joint regressor is head(GAP(body(z))). Replace the GLOBAL average pool with
PER-REGION pooling, give each region its own target, and let the gradient steer regions
differently — all on the gradient path, so it escapes the 8-bit conditioning cap.

WHAT ACTUALLY HAPPENED (measured, horizontal c2 ramp -0.15 -> -0.85):
  * HARD 2x2 blocks, summed loss, scale 150:  frost 88%, forest 27%, marble -14%.
    The 88% was NOT control — it was a SEAM. Disjoint blocks give the loss a step between
    neighbouring targets, so the image is driven into two half-images with a hard vertical
    cut, plus heavy artifacts (yellow grids, hallucinated text) because summing over 4
    regions multiplied the gradient ~4x.
  * SMOOTH overlapping windows (pool 4 / stride 2), mean loss: transmission collapses to
    6% / 3% / -11%. Raising the scale 5x (600) -> 17%, 12x (1400) -> 16%: it SATURATES, so
    this is not under-driving. Artifacts still appear at the higher scales.

ROOT CAUSE — the regressor's features are not spatially local enough. body() is four 3x3
stride-2 blocks, so each cell of the 8x8 map has a receptive field of ~31 latent px (~25% of
the image), and a 4x4 pooling window sees ~62% of it. Neighbouring "regions" therefore read
almost the same global information: the only way the network can satisfy two different
targets is a sharp discontinuity, and once the windows overlap smoothly the differentiation
washes out entirely.

This is the SAME limit found for the local cumulant map (0-47% signal below whole-image
scale): c2 cannot be reliably localised below roughly half an image — by the estimator
(noise floor) and by the regressor (receptive field). Spatial multifractal control needs a
regressor trained for locality, not a globally-trained one repurposed by pooling.

PRACTICAL NOTE: for *aesthetic* complexity gradients the scaffold/ControlNet route
(benchmarks/regressor/gradient_pass/) gives smooth, artifact-free images that read as
complexity gradients, even though its measured c2 transmission is only ~5%. Use that for
visuals; do not claim measured spatial control from it.
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
    """HARD block pooling -> (B, grid, grid, 2). Kept for the validation mode only.

    NOT suitable for guidance: disjoint blocks give the loss a STEP between neighbouring
    targets, and the gradient then drives a visible seam at the block boundary."""
    f = model.body(z)                                   # (B,256,8,8) at 1024
    B, C, H, W = f.shape
    g = grid
    rh, rw = H // g, W // g
    f = f[:, :, :rh * g, :rw * g].reshape(B, C, g, rh, g, rw)
    pooled = f.mean(dim=(3, 5))                         # (B,C,g,g)
    pooled = pooled.permute(0, 2, 3, 1).reshape(B * g * g, C)
    out = model.head(pooled).reshape(B, g, g, -1)
    return out


def region_predict_soft(model, z, pool=4, stride=2):
    """OVERLAPPING soft pooling -> (B, n, n, 2) predictions + their centre coords in [0,1].

    Adjacent windows share half their area, so the target varies CONTINUOUSLY across the
    image instead of stepping at block edges. pool=4 on the 8x8 feature map keeps each
    window at 512 px (the canonical measurement scale, above the estimator noise floor)
    while stride=2 gives 3x3 overlapping windows.
    """
    f = model.body(z)                                    # (B,C,8,8)
    B, C, H, W = f.shape
    win = torch.nn.functional.avg_pool2d(f, kernel_size=pool, stride=stride)   # (B,C,n,n)
    n = win.shape[-1]
    flat = win.permute(0, 2, 3, 1).reshape(B * n * n, C)
    out = model.head(flat).reshape(B, n, n, -1)
    # centre of each window in normalised image coords
    idx = (torch.arange(n, device=z.device) * stride + (pool - 1) / 2) / (H - 1)
    return out, idx


def spatial_guided(pipe, model, prompt, c1_fn, c2_fn, *, steps=40, cfg=6.0, scale=150.0,
                   w1=1.0, w2=1.0, warmup=12, res=1024, seed=0, pool=4, stride=2,
                   device="cuda"):
    """Generate with a CONTINUOUS spatial target.

    c1_fn / c2_fn take normalised coords (x, y in [0,1]) and return the desired value, so the
    target is smooth by construction — no block edges. The loss is MEAN (not sum) over
    overlapping windows, so the gradient magnitude matches the global-guidance case and
    `scale` keeps its usual meaning (summing over regions was what over-drove the latent
    into the yellow-grid / hallucinated-text regime)."""
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
    wts = torch.tensor([w1, w2], device=device)
    tgt = None                      # built lazily once we know the window layout

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
            pred, coord = region_predict_soft(model, x0.float(), pool, stride)   # (1,n,n,2)
            if tgt is None:
                cy, cx = torch.meshgrid(coord, coord, indexing="ij")
                tgt = torch.stack([c1_fn(cx, cy), c2_fn(cx, cy)], -1)[None].float()
            loss = (wts * (pred - tgt) ** 2).mean()        # MEAN: gradient magnitude ~ global case
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
        lo, hi = args.c2_lo, args.c2_hi
        c1_fn = lambda x, y: torch.full_like(x, 1.3)
        c2_fn = lambda x, y: lo + (hi - lo) * x          # smooth horizontal ramp
        img = spatial_guided(pipe, model, prompt, c1_fn, c2_fn, scale=args.scale,
                             warmup=args.warmup, steps=args.steps, res=args.res,
                             seed=args.seed, pool=args.pool, stride=args.stride,
                             device=device)
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
    ap.add_argument("--grid", type=int, default=2, help="validate mode only")
    ap.add_argument("--pool", type=int, default=4, help="window size on the 8x8 feature map")
    ap.add_argument("--stride", type=int, default=2, help="window stride (overlap)")
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
