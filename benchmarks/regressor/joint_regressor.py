"""
joint_regressor.py — predict BOTH cumulants (c1, c2) from the SDXL VAE latent.

Motivation: perceived "complexity" tracks c1 (mean roughness / detail density), while c2
is intermittency (how clustered the roughness is). Controlling only c2 changes a
perceptually subtle axis; controlling c1 changes what the eye actually reads as complex.
This trains one CNN with a 2-d output [c1, c2] so guidance can steer both independently.

Reuses the latents already cached by latent_regressor.py (corpus/latents.npy +
latents_meta.csv); c1 is joined from corpus/manifest.csv by path (no re-encode needed).

Usage:
  python joint_regressor.py --epochs 300        # after latent_regressor cached latents
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))
from train_stage1 import grouped_split, regression_stats  # noqa: E402
from latent_regressor import SDXL_SCALING                  # noqa: E402

TARGETS = ["c1", "c2"]


class JointSet(Dataset):
    def __init__(self, latents, rows, augment):
        self.z, self.rows, self.augment = latents, rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        z = torch.from_numpy(self.z[i].astype(np.float32))
        if self.augment:
            if torch.rand(1).item() < 0.5:
                z = torch.flip(z, dims=[2])
            if torch.rand(1).item() < 0.5:
                z = torch.flip(z, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                z = torch.rot90(z, k, dims=[1, 2])
        y = torch.tensor([float(self.rows[i]["c1"]), float(self.rows[i]["c2"])],
                         dtype=torch.float32)
        return z, y, self.rows[i]["source"]


class JointCNN(nn.Module):
    """Same backbone as LatentCNN but a 2-d head -> (c1, c2)."""
    def __init__(self, in_ch=4, p=0.3, out_dim=2):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1), nn.GroupNorm(8, co), nn.GELU())
        self.body = nn.Sequential(blk(in_ch, 64), blk(64, 128), blk(128, 256), blk(256, 256))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(256, 128), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(128, out_dim))

    def forward(self, z):
        return self.head(self.body(z).mean(dim=(2, 3)))     # (B, 2)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, srcs = [], [], []
    for z, y, src in loader:
        ps.append(model(z.to(device)).cpu().numpy()); ys.append(y.numpy()); srcs += list(src)
    ys, ps, srcs = np.concatenate(ys), np.concatenate(ps), np.array(srcs)
    out = {}
    for d, name in enumerate(TARGETS):
        out[name] = {"overall": regression_stats(ys[:, d], ps[:, d])}
        for s in sorted(set(srcs)):
            m = srcs == s
            out[name][s] = regression_stats(ys[m, d], ps[m, d])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "corpus" / "manifest.csv"))
    ap.add_argument("--latents", default=str(HERE / "corpus" / "latents.npy"))
    ap.add_argument("--meta", default=str(HERE / "corpus" / "latents_meta.csv"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--c2-min", type=float, default=-1.5)
    ap.add_argument("--c2-max", type=float, default=0.2)
    ap.add_argument("--c1-min", type=float, default=0.5)
    ap.add_argument("--c1-max", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    z_all = np.load(args.latents)
    meta = pd.read_csv(args.meta).to_dict("records")
    man = pd.read_csv(args.manifest)
    c1map = dict(zip(man["path"].astype(str), man["c1"].astype(float)))
    keep_idx, rows = [], []
    for i, r in enumerate(meta):
        c1 = c1map.get(str(r["path"]))
        c2 = float(r["c2"])
        if c1 is None or not np.isfinite(c1):
            continue
        if not (args.c1_min <= c1 <= args.c1_max and args.c2_min <= c2 <= args.c2_max):
            continue                                       # drop sparse-field artifacts
        r["c1"] = float(c1)
        keep_idx.append(i); rows.append(r)
    z_all = z_all[keep_idx]
    print(f"{len(rows)} latents with (c1,c2). "
          f"c1 range [{min(r['c1'] for r in rows):.2f},{max(r['c1'] for r in rows):.2f}]")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = grouped_split(rows, seed=args.seed)
    idx = {id(r): i for i, r in enumerate(rows)}
    sub = lambda rs: (z_all[[idx[id(r)] for r in rs]], rs)
    dl = lambda zr, aug, sh: DataLoader(JointSet(*zr, aug), batch_size=args.batch,
                                        shuffle=sh, num_workers=2)
    train_loader = dl(sub(tr), not args.no_aug, True)
    val_loader = dl(sub(va), False, False)
    test_loader = dl(sub(te), False, False)
    print(f"split: train {len(tr)} val {len(va)} test {len(te)}")

    model = JointCNN(in_ch=z_all.shape[1], p=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 1e-2)
    best, best_state, bad = math.inf, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for z, y, _ in train_loader:
            z, y = z.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(z), y)
            loss.backward(); opt.step(); tot += loss.item() * len(y)
        sched.step()
        val = evaluate(model, val_loader, device)
        vmse = val["c1"]["overall"]["rmse"] ** 2 + val["c2"]["overall"]["rmse"] ** 2
        if vmse < best - 1e-6:
            best, bad, best_state = vmse, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"ep {ep:3d} train {tot/len(tr):.4f} "
                  f"val c1_rmse {val['c1']['overall']['rmse']:.4f} "
                  f"c2_rmse {val['c2']['overall']['rmse']:.4f} bad {bad}")
        if bad >= args.patience:
            print(f"early stop @ {ep}"); break

    if best_state:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    torch.save({"model": model.state_dict(), "targets": TARGETS,
                "vae_id": "madebyollin/sdxl-vae-fp16-fix", "vae_res": 1024,
                "scaling": SDXL_SCALING}, HERE / "joint_model.pt")
    (HERE / "metrics_joint.json").write_text(json.dumps(test, indent=2))
    print("\n=== TEST (joint) ===")
    for name in TARGETS:
        print(f"  [{name}]")
        for k, v in test[name].items():
            print(f"    {k:12s} n={v['n']:4d} RMSE={v['rmse']:.4f} r={v['pearson_r']:.3f}")
    print(f"saved -> joint_model.pt, metrics_joint.json")


if __name__ == "__main__":
    main()
