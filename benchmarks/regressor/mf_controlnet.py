"""
mf_controlnet.py — MF-ControlNet (Stage 0, zero-shot): impose a target (c1,c2) on any
substrate by injecting the prescribed-cascade field as a ControlNet conditioning image at
EVERY UNet block / EVERY step (vs mf_scaffold's img2img init that fades).

Checkpoint: xinsir/controlnet-tile-sdxl-1.0 (tile CN conditions on a continuous-tone
reference, so it can transfer the field's structure). Base: RealVisXL_V4.0 + fp16-fix VAE.

CAVEAT (from adversarial review): the tile CN was trained to deblur/upscale real photos and
*hallucinate* fine detail, so it likely transfers the COARSE envelope (helps c2) but may
overwrite fine structure (c1) with a learned texture prior. control_guidance_end=1.0 (don't
release the late steps) helps c1; --log-leader conditioning helps too. If c1 still regresses,
that's the signal to fine-tune (Phase B). Everything is measured with joint_model.pt.

Usage:
  python mf_controlnet.py --mode single --substrate "ferrofluid" --c1-target 1.3 --c2-target -0.7
  python mf_controlnet.py --mode eval --families ferrofluid,frost_fern,smoke_eddies,marble,firestorm \
      --c1-grid 1.0,1.3,1.6 --c2-grid=-0.3,-0.6,-0.9 --cn-sweep 0.6,0.9,1.2 --guidance-end 1.0
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
import mfractal as mf                                       # noqa: E402
from mf_scaffold import robust_unit as robust_unit_disp, MODEL_ID, NEG, TAIL  # noqa: E402
from train_stage1 import robust_unit as robust_unit_meas   # measurement (lo=0.5,hi=99.5)  # noqa: E402
from joint_regressor import JointCNN                        # noqa: E402

VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
SDXL_SCALING = 0.13025
DEFAULT_CN = "xinsir/controlnet-tile-sdxl-1.0"


def build_pipe(controlnet_id=DEFAULT_CN, device="cuda"):
    from diffusers import (StableDiffusionXLControlNetPipeline, ControlNetModel,
                           AutoencoderKL, EulerDiscreteScheduler)
    cn = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)  # NO variant=
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        MODEL_ID, controlnet=cn, vae=vae, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device); pipe.set_progress_bar_config(disable=True)
    return pipe


def synth_field(c1, c2, n, seed):
    """prescribed_cascade extrapolates the (c1,c2) knobs; measure RAW-NATIVE (never
    measure_c2_field on a >512 field — it 8-bit-quantizes and reads c2 as -50)."""
    f = np.asarray(mf.prescribed_cascade(n=n, seed=int(seed), c1_target=float(c1),
                                         c2_target=float(c2)), float)
    r = mf.wavelet_leaders_2d(f)
    return f, float(r["c1"]), float(r["c2"])


def best_field(c1, c2, n, seed, n_select=3):
    """Pick the seed whose field measures closest to (c1,c2) — raw-native measurement."""
    best, berr = None, 1e18
    for s in range(seed, seed + n_select):
        f, mc1, mc2 = synth_field(c1, c2, n, s)
        e = abs(mc1 - c1) + abs(mc2 - c2)
        if e < berr:
            best, berr = (f, mc1, mc2), e
    return best                                            # (field, meas_c1, meas_c2)


def field_to_control(field, log_leader=False):
    if log_leader:                                         # channel-rich, no 1/99 stretch
        L = np.log(field + 1e-3)
        med = np.median(L); mad = np.median(np.abs(L - med)) + 1e-8
        ch0 = 1.0 / (1.0 + np.exp(-(L - med) / (1.4826 * mad)))
        ch1 = gaussian_filter(ch0, sigma=field.shape[0] / 64.0)     # coarse envelope (c2)
        ch2 = ch0 - gaussian_filter(ch0, sigma=2.0)                 # fine residual (c1)
        ch2 = (ch2 - ch2.min()) / (np.ptp(ch2) + 1e-8)
        rgb = np.clip(np.stack([ch0, ch1, ch2], -1), 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    disp = robust_unit_disp(field)                         # mf_scaffold's 1/99 display
    return Image.fromarray((disp * 255).astype(np.uint8), "L").convert("RGB")


class JointMeasurer:
    """Achieved (c1,c2) of an output PNG — replicates training preprocessing exactly."""
    def __init__(self, vae, device):
        self.vae, self.device = vae, device
        ck = torch.load(HERE / "joint_model.pt", map_location=device)
        self.m = JointCNN(in_ch=4).to(device); self.m.load_state_dict(ck["model"]); self.m.eval()

    @torch.no_grad()
    def __call__(self, pil):
        f = robust_unit_meas(np.asarray(pil.convert("L"), float))   # lo=0.5,hi=99.5
        t = torch.from_numpy(f)[None, None].float()
        if t.shape[-1] != 1024:
            t = F.interpolate(t, size=(1024, 1024), mode="bicubic", align_corners=False)
        t = (t.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1).to(self.device, torch.float16)
        z = self.vae.encode(t).latent_dist.mean * SDXL_SCALING
        c1, c2 = self.m(z.float()).cpu().numpy()[0]
        return float(c1), float(c2)


def generate(pipe, prompt, control, cn_scale, *, guidance_end=1.0, steps=40, cfg=6.0,
             seed=0, res=1024, device="cuda"):
    g = torch.Generator(device=device).manual_seed(int(seed))
    return pipe(prompt=prompt + TAIL, negative_prompt=NEG, image=control,
                height=res, width=res, num_inference_steps=steps, guidance_scale=cfg,
                controlnet_conditioning_scale=float(cn_scale),
                control_guidance_start=0.0, control_guidance_end=float(guidance_end),
                generator=g).images[0]


def load_prompt(fam):
    import importlib
    for mod in ("prompts", "prompts_intricate", "prompts_ood"):
        pm = importlib.import_module(mod)
        if fam in pm.PROMPTS:
            return pm.PROMPTS[fam][0]
    return fam                                             # treat as free text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "eval"], default="single")
    ap.add_argument("--controlnet", default=DEFAULT_CN)
    ap.add_argument("--substrate", default="ferrofluid")
    ap.add_argument("--families", default="ferrofluid,frost_fern,smoke_eddies,marble,firestorm")
    ap.add_argument("--c1-target", type=float, default=1.3)
    ap.add_argument("--c2-target", type=float, default=-0.7)
    ap.add_argument("--c1-grid", default="1.0,1.3,1.6")
    ap.add_argument("--c2-grid", default="-0.3,-0.6,-0.9")
    ap.add_argument("--cn-sweep", default="0.6,0.9,1.2")
    ap.add_argument("--guidance-end", type=float, default=1.0)
    ap.add_argument("--log-leader", action="store_true")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--out", default=str(HERE / "mf_controlnet"))
    args = ap.parse_args()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cn_scales = [float(x) for x in args.cn_sweep.split(",")]

    print(f"loading ControlNet pipe ({args.controlnet})...", flush=True)
    pipe = build_pipe(args.controlnet, device)
    meas = JointMeasurer(pipe.vae, device)

    if args.mode == "single":
        field, fc1, fc2 = best_field(args.c1_target, args.c2_target, args.res, args.seed)
        ctrl = field_to_control(field, args.log_leader)
        ctrl.save(out / f"control_c1{args.c1_target:+.2f}_c2{args.c2_target:+.2f}.png")
        print(f"scaffold field measures (c1={fc1:.2f}, c2={fc2:+.2f})")
        for s in cn_scales:
            img = generate(pipe, load_prompt(args.substrate), ctrl, s,
                           guidance_end=args.guidance_end, steps=args.steps, cfg=args.cfg,
                           seed=args.seed, res=args.res, device=device)
            c1m, c2m = meas(img)
            img.save(out / f"{args.substrate.replace(' ','_')[:20]}_cn{s:.2f}.png")
            print(f"  cn_scale {s:.2f} -> achieved (c1={c1m:.3f}, c2={c2m:+.3f})  "
                  f"target ({args.c1_target},{args.c2_target})", flush=True)
        return

    # eval grid
    fams = args.families.split(",")
    c1g = [float(x) for x in args.c1_grid.split(",")]
    c2g = [float(x) for x in args.c2_grid.split(",")]
    rows = []
    for fam in fams:
        prompt = load_prompt(fam)
        for c1t in c1g:
            for c2t in c2g:
                field, fc1, fc2 = best_field(c1t, c2t, args.res, args.seed)
                ctrl = field_to_control(field, args.log_leader)
                for s in cn_scales:
                    img = generate(pipe, prompt, ctrl, s, guidance_end=args.guidance_end,
                                   steps=args.steps, cfg=args.cfg, seed=args.seed,
                                   res=args.res, device=device)
                    c1m, c2m = meas(img)
                    fp = out / "images" / f"{fam}_c1{c1t:+.2f}_c2{c2t:+.2f}_cn{s:.2f}.png"
                    img.save(fp)
                    rows.append(dict(family=fam, c1_target=c1t, c2_target=c2t, cn_scale=s,
                                     field_c1=round(fc1, 3), field_c2=round(fc2, 3),
                                     c1_out=round(c1m, 3), c2_out=round(c2m, 3),
                                     c1_err=round(abs(c1m - c1t), 3), c2_err=round(abs(c2m - c2t), 3)))
                    print(f"  {fam:14s} t=({c1t:+.1f},{c2t:+.1f}) cn={s:.2f} -> "
                          f"({c1m:.2f},{c2m:+.2f})", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # best cn_scale per (family,target) by combined err
    import itertools
    print("\n=== best achieved per (family,target) [min c1_err+c2_err over cn-sweep] ===")
    keyf = lambda r: (r["family"], r["c1_target"], r["c2_target"])
    rows.sort(key=keyf)
    for (fam, c1t, c2t), grp in itertools.groupby(rows, keyf):
        b = min(grp, key=lambda r: r["c1_err"] + r["c2_err"])
        print(f"  {fam:14s} target({c1t:+.1f},{c2t:+.1f}) -> ({b['c1_out']:.2f},{b['c2_out']:+.2f}) "
              f"@cn{b['cn_scale']:.2f}  err(c1={b['c1_err']:.2f},c2={b['c2_err']:.2f})")
    print(f"\nresults -> {out/'results.csv'}")


if __name__ == "__main__":
    main()
