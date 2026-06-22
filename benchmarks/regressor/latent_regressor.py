"""
latent_regressor.py  --  Stage 3: a c2 regressor that reads the SDXL VAE LATENT.

At sampling time (Stage 4) the guidance gradient is taken w.r.t. the latent every
denoising step. A pixel-space regressor would need a VAE decode in that loop (slow);
a regressor that reads the 4x(H/8)x(W/8) latent directly skips the decode entirely.

This trains such a regressor on the SAME c2 labels as the pixel model, so the two are
directly comparable. Two phases:

  1. --cache-latents : encode every corpus image through the SDXL VAE once, store a
     stacked latents.npy (N,4,h,w) + latents_meta.csv (path,c2,group,source) in order.
  2. train           : a small conv net on (latent, c2) with the SAME group-aware split.

Important consistency note: the latent the diffusion loop produces lives in the SCALED
latent space (encode * vae.config.scaling_factor). We cache exactly that, so a latent
fed in Stage 4 matches what this net was trained on. The deployment domain is SDXL, so
the SDXL-generated corpus rows anchor it; procedural/prescribed rows are encoded through
the same path (robust-normalized -> VAE) and mainly extend c2 coverage.

VAE: madebyollin/sdxl-vae-fp16-fix by default (numerically stable in fp16; the stock
stabilityai/sdxl-vae NaNs in fp16). --vae-res 1024 -> 128x128 latents (matches 1024 SDXL
deployment); 512 -> 64x64 (faster, smaller cache).

Usage:
  python latent_regressor.py --cache-latents --vae-res 1024
  python latent_regressor.py --epochs 300            # reuses an existing cache
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_corpus import load_field, ROOT as _ROOT  # noqa: E402
from train_stage1 import grouped_split, regression_stats, robust_unit  # noqa: E402

SDXL_SCALING = 0.13025  # SDXL VAE config.scaling_factor


# --------------------------------------------------------------------------- #
# Phase 1: cache latents
# --------------------------------------------------------------------------- #
def cache_latents(manifest, out_npy, out_meta, vae_id, vae_res, batch=8):
    import pandas as pd
    from diffusers import AutoencoderKL
    rows = pd.read_csv(manifest).to_dict("records")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(device)
    vae.eval()

    def to_vae_input(path):
        f = robust_unit(load_field(path))                  # [0,1] 512
        t = torch.from_numpy(f)[None, None]
        if t.shape[-1] != vae_res:
            t = F.interpolate(t, size=(vae_res, vae_res), mode="bicubic",
                              align_corners=False)
        t = t.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1       # 3ch, [-1,1]
        return t[0]

    lats, meta = [], []
    buf, buf_rows = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device, torch.float16)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean * SDXL_SCALING
        for zi, r in zip(z.cpu().float().numpy(), buf_rows):
            lats.append(zi.astype(np.float16))
            meta.append(r)
        buf.clear(); buf_rows.clear()

    for i, r in enumerate(rows, 1):
        p = Path(r["path"]); p = p if p.is_absolute() else _ROOT / p
        buf.append(to_vae_input(p)); buf_rows.append(r)
        if len(buf) >= batch:
            flush()
        if i % 200 == 0:
            print(f"  encoded {i}/{len(rows)}")
    flush()

    arr = np.stack(lats)
    np.save(out_npy, arr)
    with open(out_meta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "c2", "group", "source"])
        w.writeheader()
        for r in meta:
            w.writerow({k: r[k] for k in ["path", "c2", "group", "source"]})
    print(f"cached {arr.shape} latents -> {out_npy}")


# --------------------------------------------------------------------------- #
# Phase 2: train
# --------------------------------------------------------------------------- #
class LatentSet(Dataset):
    def __init__(self, latents, rows, augment):
        self.z = latents
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        z = torch.from_numpy(self.z[i].astype(np.float32))   # (4,h,w)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                z = torch.flip(z, dims=[2])
            if torch.rand(1).item() < 0.5:
                z = torch.flip(z, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                z = torch.rot90(z, k, dims=[1, 2])
        return z, torch.tensor(float(self.rows[i]["c2"]), dtype=torch.float32), \
            self.rows[i]["source"]


class LatentCNN(nn.Module):
    """~2M params. GlobalAvgPool makes it agnostic to 64x64 vs 128x128 latents."""
    def __init__(self, in_ch=4, p=0.3):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1), nn.GroupNorm(8, co),
                                 nn.GELU())
        self.body = nn.Sequential(blk(in_ch, 64), blk(64, 128), blk(128, 256),
                                  blk(256, 256))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(256, 128), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(128, 1))

    def forward(self, z):
        h = self.body(z).mean(dim=(2, 3))      # GAP
        return self.head(h).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, srcs = [], [], []
    for z, y, src in loader:
        z = z.to(device)
        ps += model(z).cpu().tolist(); ys += y.tolist(); srcs += list(src)
    ys, ps, srcs = np.array(ys), np.array(ps), np.array(srcs)
    out = {"overall": regression_stats(ys, ps)}
    for s in sorted(set(srcs)):
        m = srcs == s
        out[s] = regression_stats(ys[m], ps[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "corpus" / "manifest.csv"))
    ap.add_argument("--latents", default=str(HERE / "corpus" / "latents.npy"))
    ap.add_argument("--meta", default=str(HERE / "corpus" / "latents_meta.csv"))
    ap.add_argument("--cache-latents", action="store_true")
    ap.add_argument("--vae-id", default="madebyollin/sdxl-vae-fp16-fix")
    ap.add_argument("--vae-res", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cache_latents:
        cache_latents(args.manifest, args.latents, args.meta, args.vae_id, args.vae_res)

    import pandas as pd
    z_all = np.load(args.latents)
    meta = pd.read_csv(args.meta).to_dict("records")
    assert len(z_all) == len(meta), "latents/meta length mismatch; re-cache"
    idx = {id(r): i for i, r in enumerate(meta)}            # row -> latent index

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = grouped_split(meta, seed=args.seed)
    print(f"latents {z_all.shape}  split: train {len(tr)} val {len(va)} test {len(te)}")

    def subset(rows):
        ix = [idx[id(r)] for r in rows]
        return z_all[ix], rows

    ztr, rtr = subset(tr); zva, rva = subset(va); zte, rte = subset(te)
    dl = lambda z, r, aug, sh: DataLoader(LatentSet(z, r, aug), batch_size=args.batch,
                                          shuffle=sh, num_workers=2)
    train_loader = dl(ztr, rtr, not args.no_aug, True)
    val_loader = dl(zva, rva, False, False)
    test_loader = dl(zte, rte, False, False)

    model = LatentCNN(in_ch=z_all.shape[1], p=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LatentCNN params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 1e-2)

    best, best_state, bad = math.inf, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for z, y, _ in train_loader:
            z, y = z.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(z), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y)
        sched.step()
        val = evaluate(model, val_loader, device)
        vmse = val["overall"]["rmse"] ** 2
        if vmse < best - 1e-5:
            best, bad, best_state = vmse, 0, {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            diff = val.get("diffusion", {}).get("rmse", float("nan"))
            print(f"ep {ep:3d} train_mse {tot/len(rtr):.4f} val_rmse "
                  f"{val['overall']['rmse']:.4f} val_diff {diff:.4f} bad {bad}")
        if bad >= args.patience:
            print(f"early stop @ {ep}"); break

    if best_state:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    torch.save({"model": model.state_dict(), "vae_id": args.vae_id,
                "vae_res": args.vae_res, "scaling": SDXL_SCALING},
               HERE / "latent_model.pt")
    (HERE / "metrics_v3.json").write_text(json.dumps({"config": vars(args),
                                                      "test": test}, indent=2))

    print("\n=== TEST (latent) ===")
    for k, v in test.items():
        print(f"  {k:12s} n={v['n']:4d} RMSE={v['rmse']:.4f} r={v['pearson_r']:.3f}")
    # compare to pixel model
    px = HERE / "metrics_v2.json"
    if px.exists():
        pj = json.loads(px.read_text())["test"]
        for s in ("diffusion", "overall"):
            lr = test.get(s, {}).get("rmse", float("nan"))
            pr = pj.get(s, {}).get("rmse", float("nan"))
            print(f"  {s}: latent {lr:.4f} vs pixel {pr:.4f}  gap {lr-pr:+.4f}")
        gap = test.get("diffusion", {}).get("rmse", 9) - pj.get("diffusion", {}).get("rmse", 0)
        print(f"\nStage-3 pass (latent within 0.02 of pixel on diffusion): "
              f"{'PASS' if gap < 0.02 else 'FAIL -> use pixel+decode in the loop'} "
              f"(gap {gap:+.4f})")


if __name__ == "__main__":
    main()
