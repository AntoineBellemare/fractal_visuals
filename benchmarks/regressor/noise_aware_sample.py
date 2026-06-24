"""
noise_aware_sample.py — classifier guidance with the noise-aware regressor.

The noise-aware regressor reads (c1,c2) directly from the noisy latent x_t (+ t), so we
guide from step 0 with NO x0-prediction and NO warmup. Head-to-head against the clean+x0+
warmup path (joint_montage.joint_guided): same prompt/seed/targets, compare measured (c1,c2).
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import guided_sample as gs                       # noqa: E402
import joint_montage as jm                        # noqa: E402
from build_corpus import measure_c2               # noqa: E402
from noise_aware_regressor import NoiseAwareJointCNN, SDXL_SCALING  # noqa: E402


def load_na(device):
    ck = torch.load(HERE / "noise_aware_model.pt", map_location=device)
    m = NoiseAwareJointCNN(in_ch=4).to(device); m.load_state_dict(ck["model"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def na_guided(pipe, na, prompt, c1t, c2t, *, steps=40, cfg=6.0, scale=150.0,
              w1=1.0, w2=1.0, res=1024, seed=0, device="cuda"):
    """Guide every step directly on the noisy latent x_t at timestep t (no x0, no warmup)."""
    g = torch.Generator(device=device).manual_seed(seed)
    sched = pipe.scheduler; sched.set_timesteps(steps, device=device)
    pe, npe, ppe, nppe = gs._encode_prompt(pipe, prompt, device)
    emb = torch.cat([npe, pe]); add_text = torch.cat([nppe, ppe])
    add_time = pipe._get_add_time_ids((res, res), (0, 0), (res, res), dtype=pe.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(device).repeat(2, 1)
    added = {"text_embeds": add_text, "time_ids": add_time}
    lat = torch.randn((1, pipe.unet.config.in_channels, res // 8, res // 8),
                      generator=g, device=device, dtype=pe.dtype) * sched.init_noise_sigma
    tgt = torch.tensor([c1t, c2t], device=device); wts = torch.tensor([w1, w2], device=device)
    for t in sched.timesteps:
        with torch.no_grad():
            li = sched.scale_model_input(torch.cat([lat] * 2), t)
            eps = pipe.unet(li, t, encoder_hidden_states=emb, added_cond_kwargs=added).sample
            eu, ec = eps.chunk(2); eps = eu + cfg * (ec - eu)
        lat = sched.step(eps, t, lat).prev_sample
        # noise-aware guidance on the (next) noisy latent at this timestep
        l = lat.detach().float().requires_grad_(True)
        pred = na(l, t.reshape(1).to(device))[0]
        loss = (wts * (pred - tgt) ** 2).sum()
        grad = torch.autograd.grad(loss, l)[0]
        lat = (lat - scale * grad).detach().to(pe.dtype)
    with torch.no_grad():
        img = pipe.vae.decode(lat / SDXL_SCALING).sample
    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    from PIL import Image
    return Image.fromarray((img * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="marble,ferrofluid,frost_fern,firestorm")
    ap.add_argument("--c1-target", type=float, default=1.3)
    ap.add_argument("--c2-vals", default="-0.15,-0.45,-0.75")
    ap.add_argument("--scale", type=float, default=150.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "noise_aware_test"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    c2v = [float(x) for x in args.c2_vals.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = gs.build_pipe(device); na = load_na(device); jmodel = jm.load_joint(device)

    rows = []
    print("c2-sweep (c1 fixed) — NOISE-AWARE (all steps) vs CLEAN+x0+warmup")
    for fam in args.families.split(","):
        prompt = jm.load_prompt(fam, ("prompts", "prompts_intricate"))
        for c2t in c2v:
            na_img = na_guided(pipe, na, prompt, args.c1_target, c2t, scale=args.scale,
                               steps=args.steps, seed=args.seed, device=device)
            cl_img = jm.joint_guided(pipe, jmodel, prompt, args.c1_target, c2t, scale=args.scale,
                                     w1=1.0, w2=1.0, warmup=16, steps=args.steps, seed=args.seed, device=device)
            na_c1, na_c2 = measure_c2(na_img); cl_c1, cl_c2 = measure_c2(cl_img)
            na_img.save(out / f"{fam}_c2{c2t:+.2f}_NA.png"); cl_img.save(out / f"{fam}_c2{c2t:+.2f}_clean.png")
            rows.append(dict(family=fam, c2_target=c2t,
                             na_c1=round(na_c1, 3), na_c2=round(na_c2, 3),
                             clean_c1=round(cl_c1, 3), clean_c2=round(cl_c2, 3)))
            print(f"  {fam:12s} c2t={c2t:+.2f} | NA->c2 {na_c2:+.3f} | clean->c2 {cl_c2:+.3f}", flush=True)

    with (out / "compare.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "c2_target", "na_c1", "na_c2", "clean_c1", "clean_c2"])
        w.writeheader(); w.writerows(rows)
    # tracking quality: slope/r of measured c2 vs target, per method
    t = np.array([r["c2_target"] for r in rows])
    for m in ("na", "clean"):
        y = np.array([r[f"{m}_c2"] for r in rows])
        s = np.polyfit(t, y, 1)[0]; rr = float(np.corrcoef(t, y)[0, 1])
        print(f"\n{m:5s}: c2 tracking slope={s:.2f} r={rr:.2f}")
    print(f"images + compare.csv -> {out}")


if __name__ == "__main__":
    main()
