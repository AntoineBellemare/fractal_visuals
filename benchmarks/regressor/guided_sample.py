"""
guided_sample.py  --  Stage 4: classifier-guided SDXL toward a target c2.

Hooks the Stage-3 latent regressor into SDXL's denoising loop. At each step we form the
predicted clean latent x0, read its c2, and nudge the latent to reduce |c2 - target|.
This is the actual experiment: does measured c2 of the output track the requested c2?

Guidance math (DPS / FreeDoM family). With epsilon-prediction and DDIM:
    x0 = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t)
    loss = (regressor(x0) - c2_target)^2
    x_t <- x_t - guidance_scale * d loss / d x_t      (after the normal scheduler step)

Two backprop modes:
  * default (cheap, stable): eps is DETACHED, so the gradient flows only through the
    explicit x0(x_t) term -- no backprop through the UNet. Memory-light, good first pass.
  * --full-dps: backprop through the UNet too (the PLAN's exact pseudocode). More
    faithful, heavier; may not fit alongside SDXL in 16 GB at 1024.

Guidance is GATED to start after --warmup steps: early x0 is blurry and the regressor
(trained on clean images) gives unreliable gradients there.

NOTE: guidance_scale is the one knob that always needs tuning per the PLAN ([50, 500]
raw, or use --normalize-grad for a latent-space step size that is scale-stable). Run the
validation grid first to find the working range before trusting any single image.

Usage:
  # single image:
  python guided_sample.py --prompt "extreme macro photograph, polished marble ..." \
      --c2-target -0.4 --guidance-scale 200 --seed 0
  # validation grid (the real deliverable):
  python guided_sample.py --grid --guidance-scale 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from latent_regressor import LatentCNN, SDXL_SCALING  # noqa: E402
from build_corpus import measure_c2                    # noqa: E402

MODEL_ID = "SG161222/RealVisXL_V4.0"
DEFAULT_PROMPTS = [
    "extreme macro photograph, polished white marble slab with delicate gray veins, "
    "professional photography, sharp focus, 8K, photorealistic",
    "extreme macro photograph, cirrus cloud wisps in a pale blue sky, "
    "professional photography, sharp focus, 8K, photorealistic",
    "extreme macro photograph, ink diffusing in water, swirling turbulent filaments, "
    "professional photography, sharp focus, 8K, photorealistic",
    "extreme macro photograph, weathered granite surface with crystalline grains, "
    "professional photography, sharp focus, 8K, photorealistic",
]
NEG = ("low quality, blurry, distorted, watermark, text, logo, frame, border, "
       "illustration, cartoon, human, person, face")


# --------------------------------------------------------------------------- #
def load_regressor(device):
    ckpt = torch.load(HERE / "latent_model.pt", map_location=device)
    model = LatentCNN(in_ch=4).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_pipe(device):
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def _encode_prompt(pipe, prompt, device):
    pe, npe, ppe, nppe = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt=NEG, negative_prompt_2=NEG)
    return pe, npe, ppe, nppe


def guided_sample(pipe, regressor, prompt, c2_target, *, steps=50, cfg=6.0,
                  guidance_scale=200.0, warmup=10, res=1024, seed=0,
                  full_dps=False, normalize_grad=False, pixel_fallback=None,
                  device="cuda"):
    """Return a PIL image steered toward c2_target."""
    g = torch.Generator(device=device).manual_seed(seed)
    sched = pipe.scheduler
    sched.set_timesteps(steps, device=device)
    timesteps = sched.timesteps
    abar = sched.alphas_cumprod.to(device)

    pe, npe, ppe, nppe = _encode_prompt(pipe, prompt, device)
    prompt_embeds = torch.cat([npe, pe], dim=0)
    add_text = torch.cat([nppe, ppe], dim=0)
    add_time = pipe._get_add_time_ids(
        (res, res), (0, 0), (res, res), dtype=pe.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
    ).to(device).repeat(2, 1)
    added = {"text_embeds": add_text, "time_ids": add_time}

    shape = (1, pipe.unet.config.in_channels, res // 8, res // 8)
    latents = torch.randn(shape, generator=g, device=device, dtype=pe.dtype) \
        * sched.init_noise_sigma

    for i, t in enumerate(timesteps):
        # --- CFG noise prediction (no grad) ---
        with torch.no_grad():
            lat_in = sched.scale_model_input(torch.cat([latents] * 2), t)
            eps = pipe.unet(lat_in, t, encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs=added).sample
            eps_u, eps_c = eps.chunk(2)
            eps = eps_u + cfg * (eps_c - eps_u)

        # --- scheduler step ---
        latents = sched.step(eps, t, latents).prev_sample

        # --- classifier guidance via x0, gated after warmup ---
        if i >= warmup and guidance_scale > 0:
            at = abar[t].clamp(1e-6, 1)
            lat = latents.detach().requires_grad_(True)
            if full_dps:
                lat_in2 = sched.scale_model_input(torch.cat([lat] * 2), t)
                e = pipe.unet(lat_in2, t, encoder_hidden_states=prompt_embeds,
                              added_cond_kwargs=added).sample
                eu, ec = e.chunk(2)
                e = eu + cfg * (ec - eu)
            else:
                e = eps.detach()
            x0 = (lat - (1 - at).sqrt() * e) / at.sqrt()
            if pixel_fallback is not None:
                c2_pred = pixel_fallback(x0)
            else:
                c2_pred = regressor(x0.float())
            loss = ((c2_pred - c2_target) ** 2).mean()
            grad = torch.autograd.grad(loss, lat)[0]
            if normalize_grad:
                grad = grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)
            latents = (latents - guidance_scale * grad).detach().to(pe.dtype)

    # decode
    with torch.no_grad():
        img = pipe.vae.decode(latents / SDXL_SCALING).sample
    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    from PIL import Image
    return Image.fromarray((img * 255).astype(np.uint8))


# optional pixel-space fallback (Stage 3 fail path): decode x0 -> DINOv2 head
def make_pixel_fallback(device, net_res=518):
    from train_stage1 import Head, IMAGENET_MEAN, IMAGENET_STD, dino_pooled
    from transformers import AutoModel
    ckpt = torch.load(HERE / "pixel_model.pt", map_location=device)
    backbone = AutoModel.from_pretrained(ckpt["backbone_id"]).to(device).eval()
    head = Head(backbone.config.hidden_size).to(device).eval()
    head.load_state_dict(ckpt["head"])
    mean = IMAGENET_MEAN.to(device); std = IMAGENET_STD.to(device)

    def fn(x0_latent):
        img = pipe_ref["pipe"].vae.decode(x0_latent / SDXL_SCALING).sample
        img = (img.clamp(-1, 1) + 1) / 2          # [0,1] rgb
        gray = img.mean(1, keepdim=True)
        gray = F.interpolate(gray, size=(net_res, net_res), mode="bicubic",
                             align_corners=False).clamp(0, 1).repeat(1, 3, 1, 1)
        gray = (gray - mean) / std
        feat = dino_pooled(backbone, gray)
        return head(feat.float())
    return fn


pipe_ref = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=DEFAULT_PROMPTS[0])
    ap.add_argument("--c2-target", type=float, default=-0.4)
    ap.add_argument("--guidance-scale", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-dps", action="store_true")
    ap.add_argument("--normalize-grad", action="store_true")
    ap.add_argument("--pixel-fallback", action="store_true",
                    help="use pixel DINOv2 regressor + VAE decode (Stage 3 fail path)")
    ap.add_argument("--grid", action="store_true", help="run the validation grid")
    ap.add_argument("--grid-targets", default="-0.7,-0.5,-0.3,-0.15,-0.05")
    ap.add_argument("--grid-seeds", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "guided"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = build_pipe(device); pipe_ref["pipe"] = pipe
    regressor = None if args.pixel_fallback else load_regressor(device)
    pixfn = make_pixel_fallback(device) if args.pixel_fallback else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def sample(prompt, c2t, seed):
        img = guided_sample(pipe, regressor, prompt, c2t, steps=args.steps, cfg=args.cfg,
                            guidance_scale=args.guidance_scale, warmup=args.warmup,
                            res=args.res, seed=seed, full_dps=args.full_dps,
                            normalize_grad=args.normalize_grad, pixel_fallback=pixfn,
                            device=device)
        _, c2m = measure_c2(img)
        return img, c2m

    if not args.grid:
        img, c2m = sample(args.prompt, args.c2_target, args.seed)
        fp = out / f"single_t{args.c2_target:+.2f}_s{args.seed}.png"
        img.save(fp)
        print(f"target {args.c2_target:+.3f} -> measured {c2m:+.3f}  ({fp})")
        return

    # validation grid: prompts x seeds x targets
    targets = [float(x) for x in args.grid_targets.split(",")]
    rows = []
    for pi, prompt in enumerate(DEFAULT_PROMPTS):
        for c2t in targets:
            for s in range(args.grid_seeds):
                img, c2m = sample(prompt, c2t, s)
                img.save(out / f"p{pi}_t{c2t:+.2f}_s{s}.png")
                rows.append(dict(prompt_idx=pi, target=c2t, seed=s, measured=c2m))
                print(f"  p{pi} t={c2t:+.2f} s{s} -> {c2m:+.3f}")

    import csv
    with (out / "grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_idx", "target", "seed", "measured"])
        w.writeheader(); w.writerows(rows)

    tgt = np.array([r["target"] for r in rows])
    mea = np.array([r["measured"] for r in rows])
    slope, intercept = np.polyfit(tgt, mea, 1)
    r = float(np.corrcoef(tgt, mea)[0, 1])
    summary = dict(n=len(rows), slope=float(slope), intercept=float(intercept),
                   pearson_r=r, guidance_scale=args.guidance_scale)
    (out / "grid_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.scatter(tgt, mea, alpha=0.6)
        lo, hi = min(targets) - 0.1, max(targets) + 0.1
        plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        plt.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], "r-",
                 label=f"fit slope={slope:.2f} r={r:.2f}")
        plt.xlabel("c2 target"); plt.ylabel("c2 measured"); plt.legend()
        plt.title(f"classifier guidance (scale={args.guidance_scale:g})")
        plt.tight_layout(); plt.savefig(out / "grid.png", dpi=130)
    except Exception as e:                       # noqa: BLE001
        print(f"(plot skipped: {e})")

    print(f"\nGRID: slope={slope:.3f}  r={r:.3f}  (PLAN target: slope>0.7, r>0.85)")
    print(f"  {'PASS' if slope > 0.7 and r > 0.85 else 'TUNE guidance_scale / warmup'}")
    print(f"  -> {out/'grid.png'}, {out/'grid_summary.json'}")


if __name__ == "__main__":
    main()
