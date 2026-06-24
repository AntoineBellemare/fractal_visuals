"""
noise_aware_regressor.py — predict (c1,c2) of the CLEAN image from a NOISED latent + t.

Why: the joint regressor reads clean images well (r 0.97) but during guidance it's applied
to blurry mid-sampling x0-predictions -> unreliable gradients -> compression, artifacts,
the warmup kludge. A timestep-conditioned regressor trained on noised latents (the
Dhariwal-Nichol classifier-guidance recipe) gives accurate gradients at every noise level,
so guidance can act from step 1 directly on the noisy latent (no x0, no warmup).

Training reuses the cached clean latents (corpus/latents.npy) + (c1,c2) labels: pick a
random timestep t, apply the SDXL forward noising, and learn clean-(c1,c2) <- (z_t, t).

Usage:
  python noise_aware_regressor.py --epochs 400
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_stage1 import grouped_split, regression_stats   # noqa: E402
from latent_regressor import SDXL_SCALING                   # noqa: E402

TARGETS = ["c1", "c2"]


def sdxl_alphas_cumprod(T=1000, beta_start=0.00085, beta_end=0.012):
    """SDXL 'scaled_linear' noise schedule -> alphas_cumprod (length T)."""
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, T, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas).astype(np.float32)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)


class NoiseAwareJointCNN(nn.Module):
    def __init__(self, in_ch=4, p=0.3, tdim=128, out_dim=2):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1), nn.GroupNorm(8, co), nn.GELU())
        self.body = nn.Sequential(blk(in_ch, 64), blk(64, 128), blk(128, 256), blk(256, 256))
        self.tdim = tdim
        self.t_mlp = nn.Sequential(nn.Linear(tdim, 256), nn.GELU(), nn.Linear(256, 256))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(256 + 256, 128), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(128, out_dim))

    def forward(self, z, t):                       # z:(B,4,h,w)  t:(B,) int
        h = self.body(z).mean(dim=(2, 3))          # (B,256)
        te = self.t_mlp(timestep_embedding(t, self.tdim))
        return self.head(torch.cat([h, te], dim=1))


class LatSet(Dataset):
    def __init__(self, z, rows, augment):
        self.z, self.rows, self.augment = z, rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        z = torch.from_numpy(self.z[i].astype(np.float32))
        if self.augment:
            if torch.rand(1).item() < 0.5: z = torch.flip(z, [2])
            if torch.rand(1).item() < 0.5: z = torch.flip(z, [1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k: z = torch.rot90(z, k, [1, 2])
        y = torch.tensor([float(self.rows[i]["c1"]), float(self.rows[i]["c2"])], dtype=torch.float32)
        return z, y


def noise_batch(z0, t, acp):
    a = acp[t].view(-1, 1, 1, 1)
    return a.sqrt() * z0 + (1 - a).sqrt() * torch.randn_like(z0)


@torch.no_grad()
def evaluate(model, loader, acp, device, t_levels=(0, 200, 500, 800)):
    model.eval()
    out = {}
    for tl in t_levels:
        ys, ps = [], []
        for z, y in loader:
            z, y = z.to(device), y.to(device)
            t = torch.full((len(z),), tl, device=device, dtype=torch.long)
            zt = noise_batch(z, t, acp)
            ps.append(model(zt, t).cpu().numpy()); ys.append(y.cpu().numpy())
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        out[f"t{tl}"] = {ax: regression_stats(ys[:, d], ps[:, d]) for d, ax in enumerate(TARGETS)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default=str(HERE / "corpus" / "latents.npy"))
    ap.add_argument("--meta", default=str(HERE / "corpus" / "latents_meta.csv"))
    ap.add_argument("--manifest", default=str(HERE / "corpus" / "manifest.csv"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=45)
    ap.add_argument("--c2-min", type=float, default=-1.5); ap.add_argument("--c2-max", type=float, default=0.2)
    ap.add_argument("--c1-min", type=float, default=0.5); ap.add_argument("--c1-max", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    z_all = np.load(args.latents)
    meta = pd.read_csv(args.meta).to_dict("records")
    man = pd.read_csv(args.manifest)
    c1map = dict(zip(man["path"].astype(str), man["c1"].astype(float)))
    keep, rows = [], []
    for i, r in enumerate(meta):
        c1 = c1map.get(str(r["path"])); c2 = float(r["c2"])
        if c1 is None or not (args.c1_min <= c1 <= args.c1_max and args.c2_min <= c2 <= args.c2_max):
            continue
        r["c1"] = float(c1); keep.append(i); rows.append(r)
    z_all = z_all[keep]
    print(f"{len(rows)} latents; noise-aware training over t in [0,999]")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acp = torch.from_numpy(sdxl_alphas_cumprod()).to(device)
    tr, va, te = grouped_split(rows, seed=args.seed)
    idx = {id(r): i for i, r in enumerate(rows)}
    sub = lambda rs: (z_all[[idx[id(r)] for r in rs]], rs)
    dl = lambda zr, aug, sh: DataLoader(LatSet(*zr, aug), batch_size=args.batch, shuffle=sh, num_workers=0)
    train_loader = dl(sub(tr), True, True); val_loader = dl(sub(va), False, False); test_loader = dl(sub(te), False, False)

    model = NoiseAwareJointCNN(in_ch=z_all.shape[1], p=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 1e-2)
    best, best_state, bad = math.inf, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for z, y in train_loader:
            z, y = z.to(device), y.to(device)
            t = torch.randint(0, 1000, (len(z),), device=device)
            zt = noise_batch(z, t, acp)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(zt, t), y)
            loss.backward(); opt.step(); tot += loss.item() * len(y)
        sched.step()
        v = evaluate(model, val_loader, acp, device, t_levels=(100, 500))
        vmse = sum(v[k][ax]["rmse"] ** 2 for k in v for ax in TARGETS)
        if vmse < best - 1e-6:
            best, bad, best_state = vmse, 0, {k: w.cpu().clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
        if ep % 20 == 0 or ep == 1:
            print(f"ep {ep:3d} train {tot/len(tr):.4f} "
                  f"val@t500 c1 {v['t500']['c1']['rmse']:.3f} c2 {v['t500']['c2']['rmse']:.3f} bad {bad}")
        if bad >= args.patience:
            print(f"early stop @ {ep}"); break

    if best_state: model.load_state_dict(best_state)
    test = evaluate(model, test_loader, acp, device, t_levels=(0, 200, 500, 800))
    torch.save({"model": model.state_dict(), "targets": TARGETS, "scaling": SDXL_SCALING,
                "noise_aware": True}, HERE / "noise_aware_model.pt")
    (HERE / "metrics_noise_aware.json").write_text(json.dumps(test, indent=2))
    print("\n=== TEST: (c1,c2) RMSE read from NOISED latents at each level ===")
    for tl, d in test.items():
        print(f"  {tl:6s}  c1 RMSE {d['c1']['rmse']:.4f} r {d['c1']['pearson_r']:.3f}  |  "
              f"c2 RMSE {d['c2']['rmse']:.4f} r {d['c2']['pearson_r']:.3f}")
    print("saved -> noise_aware_model.pt")


if __name__ == "__main__":
    main()
