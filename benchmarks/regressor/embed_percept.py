"""
embed_percept.py  --  Plan A: embed a percept into a fractal texture while
controlling its multifractality, via *compositional* classifier guidance.

The idea (training-free, builds directly on guided_sample.py): drive a frozen
SDXL denoising loop with TWO guidance critics at once, on the predicted clean
latent x0 at each step:

  1. c2 critic   -- the Stage-3 latent regressor. Pulls measured c2 toward a
                    target intermittency.  (controls *texture statistics*)
  2. percept critic -- CLIP image<->text similarity to a concept ("a face").
                    Pulls the image toward containing the percept.
                    (controls *semantic content*)

The two gradients are normalized SEPARATELY and combined with explicit weights:

    x_t <- x_t - ( c2_scale * g_c2_hat  +  percept_scale * g_percept_hat )

so the RATIO percept_scale / c2_scale is literally the pareidolia dial:
  percept_scale too high -> obvious pasted face (percept dominates)
  percept_scale too low  -> pure texture       (no percept)
  sweet spot             -> ambiguous-but-findable: a face *in* the texture.

Because c2 is a GLOBAL statistic and the percept is a LOCAL structure, the two
constraints are largely orthogonal -- you can move one without destroying the
other. That orthogonality is what makes this tractable (and it's the same
"intricacy != c2 -> decouple substrate from c2" finding from RESULTS.md, now
generalized to "semantic content != c2 -> guide them independently").

The c2 critic reads the latent directly (cheap). The percept critic must see
pixels, so it VAE-decodes x0 each guided step -- heavier. Mitigations:
  * --percept-every k   : apply percept guidance only every k steps
  * pipe.vae.enable_slicing()/enable_tiling() (on by default here)
  * --region x0,y0,x1,y1: crop the decoded image to a box before CLIP, so the
    percept is localized (a face in one quadrant) instead of global.

STATUS: written WITHOUT a GPU to test on (see the GPU-access question to Phil).
Mirrors guided_sample.py's verified DPS loop exactly, with the percept term
added. Before trusting outputs, run --grid to find the working (c2_scale,
percept_scale) range -- guidance scales ALWAYS need tuning per-setup.

Usage:
  # single image: a face embedded in a marble texture at c2=-0.4
  python embed_percept.py \
      --prompt "extreme macro photograph, polished marble slab, 8K photorealistic" \
      --percept-text "a human face" --c2-target -0.4 \
      --c2-scale 200 --percept-scale 60 --seed 0

  # the real deliverable: sweep the embedded-ness dial at fixed c2
  python embed_percept.py --dial --c2-target -0.4 \
      --percept-scales 0,30,60,120,240
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
from latent_regressor import LatentCNN, SDXL_SCALING       # noqa: E402
from build_corpus import measure_c2                         # noqa: E402
from guided_sample import build_pipe, load_regressor, _encode_prompt, NEG  # noqa: E402

# CLIP normalization constants (OpenAI CLIP preprocessing).
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
def load_clip(device, clip_id="openai/clip-vit-base-patch32"):
    """Return (model, encode_text, encode_image_from_pixels)."""
    from transformers import CLIPModel, CLIPTokenizerFast
    model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = CLIPTokenizerFast.from_pretrained(clip_id)
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)

    @torch.no_grad()
    def encode_text(texts):
        batch = tok(texts, return_tensors="pt", padding=True).to(device)
        emb = model.get_text_features(**batch)
        return F.normalize(emb.float(), dim=-1)

    def encode_image(pixels01):
        """pixels01: [B,3,H,W] in [0,1], differentiable. -> normalized embeds."""
        x = F.interpolate(pixels01, size=(224, 224), mode="bicubic",
                          align_corners=False).clamp(0, 1)
        x = (x - mean) / std
        emb = model.get_image_features(pixel_values=x)
        return F.normalize(emb.float(), dim=-1)

    return model, encode_text, encode_image


def _decode_x0(pipe, x0_latent):
    """Differentiable VAE decode of a clean-latent estimate -> [0,1] rgb."""
    img = pipe.vae.decode(x0_latent / SDXL_SCALING).sample
    return (img.clamp(-1, 1) + 1) / 2


def _crop(pixels01, region, res):
    """region = (x0,y0,x1,y1) in [0,1] fractions, or None."""
    if region is None:
        return pixels01
    x0, y0, x1, y1 = region
    H = W = pixels01.shape[-1]
    return pixels01[..., int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]


# --------------------------------------------------------------------------- #
def embed_sample(pipe, regressor, encode_image, txt_emb, prompt, c2_target, *,
                 steps=50, cfg=6.0, c2_scale=200.0, percept_scale=60.0,
                 warmup=10, percept_every=1, region=None, res=1024, seed=0,
                 full_dps=False, device="cuda"):
    """Sample an image steered toward c2_target AND containing the percept."""
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

    def norm_grad(grad):
        return grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)

    for i, t in enumerate(timesteps):
        # --- CFG noise prediction (no grad) ---
        with torch.no_grad():
            lat_in = sched.scale_model_input(torch.cat([latents] * 2), t)
            eps = pipe.unet(lat_in, t, encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs=added).sample
            eps_u, eps_c = eps.chunk(2)
            eps = eps_u + cfg * (eps_c - eps_u)

        latents = sched.step(eps, t, latents).prev_sample

        if i < warmup:
            continue

        at = abar[t].clamp(1e-6, 1)
        do_percept = percept_scale > 0 and (i % percept_every == 0)

        # ---- c2 guidance (reads latent directly: cheap) ----
        step_grad = torch.zeros_like(latents)
        if c2_scale > 0:
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
            loss_c2 = ((regressor(x0.float()) - c2_target) ** 2).mean()
            g_c2 = torch.autograd.grad(loss_c2, lat)[0]
            step_grad = step_grad + c2_scale * norm_grad(g_c2)

        # ---- percept guidance (needs pixels: VAE-decode x0) ----
        if do_percept:
            lat = latents.detach().requires_grad_(True)
            e = eps.detach()
            x0 = (lat - (1 - at).sqrt() * e) / at.sqrt()
            pix = _decode_x0(pipe, x0.to(pipe.vae.dtype))
            pix = _crop(pix, region, res)
            img_emb = encode_image(pix)
            sim = (img_emb * txt_emb).sum(-1)        # cosine sim, [-1,1]
            loss_p = (1.0 - sim).mean()              # minimize -> raise sim
            g_p = torch.autograd.grad(loss_p, lat)[0]
            step_grad = step_grad + percept_scale * norm_grad(g_p)

        latents = (latents - step_grad).detach().to(pe.dtype)

    with torch.no_grad():
        img = pipe.vae.decode(latents / SDXL_SCALING).sample
    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    from PIL import Image
    return Image.fromarray((img * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
def percept_score(img, encode_image, txt_emb, device):
    """CLIP cosine similarity of a PIL image to the percept text (detectability)."""
    import numpy as _np
    arr = _np.asarray(img.convert("RGB")).astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = encode_image(t)
        return float((emb * txt_emb).sum(-1).item())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt",
                    default="extreme macro photograph, polished marble slab with "
                            "delicate veins, 8K, photorealistic, sharp focus")
    ap.add_argument("--percept-text", default="a human face")
    ap.add_argument("--c2-target", type=float, default=-0.4)
    ap.add_argument("--c2-scale", type=float, default=200.0)
    ap.add_argument("--percept-scale", type=float, default=60.0)
    ap.add_argument("--percept-every", type=int, default=1)
    ap.add_argument("--region", default=None,
                    help="x0,y0,x1,y1 in [0,1] to localize the percept (e.g. 0.25,0.2,0.75,0.8)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-dps", action="store_true")
    ap.add_argument("--clip-id", default="openai/clip-vit-base-patch32")
    ap.add_argument("--dial", action="store_true",
                    help="sweep percept-scale at fixed c2 (the embedded-ness dial deliverable)")
    ap.add_argument("--percept-scales", default="0,30,60,120,240")
    ap.add_argument("--dial-seeds", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "embedded"))
    args = ap.parse_args()

    region = tuple(float(v) for v in args.region.split(",")) if args.region else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA. SDXL on CPU is unusably slow; this needs a GPU.")

    pipe = build_pipe(device)
    try:
        pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
    except Exception:                                       # noqa: BLE001
        pass
    regressor = load_regressor(device)
    _, encode_text, encode_image = load_clip(device, args.clip_id)
    txt_emb = encode_text([args.percept_text])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def run(pscale, seed):
        img = embed_sample(
            pipe, regressor, encode_image, txt_emb, args.prompt, args.c2_target,
            steps=args.steps, cfg=args.cfg, c2_scale=args.c2_scale,
            percept_scale=pscale, warmup=args.warmup, percept_every=args.percept_every,
            region=region, res=args.res, seed=seed, full_dps=args.full_dps, device=device)
        _, c2m = measure_c2(img)
        psc = percept_score(img, encode_image, txt_emb, device)
        return img, c2m, psc

    if not args.dial:
        img, c2m, psc = run(args.percept_scale, args.seed)
        fp = out / f"embed_c2{args.c2_target:+.2f}_p{args.percept_scale:g}_s{args.seed}.png"
        img.save(fp)
        print(f"c2 target {args.c2_target:+.2f} -> measured {c2m:+.3f} | "
              f"percept sim {psc:+.3f}  ({fp})")
        return

    # the embedded-ness dial: fixed c2, rising percept_scale.
    pscales = [float(x) for x in args.percept_scales.split(",")]
    rows = []
    for ps in pscales:
        for s in range(args.dial_seeds):
            img, c2m, psc = run(ps, s)
            img.save(out / f"dial_p{ps:g}_s{s}.png")
            rows.append(dict(percept_scale=ps, seed=s, c2=c2m, percept_sim=psc))
            print(f"  percept_scale={ps:6g} s{s} -> c2 {c2m:+.3f} | sim {psc:+.3f}")

    (out / "dial.json").write_text(json.dumps(rows, indent=2))
    # The deliverable read: as percept_scale rises, percept_sim should climb
    # while c2 stays ~flat (orthogonality). The pareidolia band is where sim is
    # high enough to be findable but the image still reads as texture.
    c2_arr = np.array([r["c2"] for r in rows])
    print(f"\nDIAL: c2 stayed in [{c2_arr.min():+.3f},{c2_arr.max():+.3f}] "
          f"(target {args.c2_target:+.2f}) while percept_scale swept "
          f"{pscales[0]:g}->{pscales[-1]:g}")
    print(f"  -> {out/'dial.json'}  (inspect dial_*.png for the ambiguity sweet spot)")


if __name__ == "__main__":
    main()
