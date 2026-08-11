"""
c2_regressor_v2.py — Stage 1 of the GPU plan: stronger backbone + augmentation.

What changed vs the MVP (c2_regressor.py):
  - Backbone is swappable: DINOv2-small (default, 384-d), CLIP-ViT-B/32 (768-d),
    or resnet18 (512-d, sanity-check / same as MVP).
  - Augmentation is applied EVERY epoch (random-resized-crop, flips, 90° rotations,
    brightness/contrast jitter), all c2-invariant. Because the augmented pixels
    differ each epoch, features can't be precomputed once — the (frozen) backbone
    runs inside the training loop. Only the MLP head learns.
  - Stronger regularization: dropout 0.3, weight_decay 1e-3, cosine LR, early stop
    on val MSE.

The source images are cached once at 256x256 gray (images_v2.npz) so re-runs skip
the slow procedural regeneration; the augmentation + backbone forward happen live.

Usage:
  python c2_regressor_v2.py                      # dinov2, 300 epochs
  python c2_regressor_v2.py --backbone clip
  python c2_regressor_v2.py --backbone resnet18 --epochs 200   # MVP-parity backbone
  python c2_regressor_v2.py --rebuild            # force-rebuild the image cache

Outputs (benchmarks/regressor/):
  images_v2.npz, pixel_model.pt, metrics_v1.json, scatter_v1.png

Pass criterion (see PLAN.md Stage 1): diffusion-domain RMSE < 0.15.
"""
from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mfractal as mf  # noqa: F401  (needed by regen_procedural via c2_regressor)

# Reuse the MVP's data collection so the corpus stays identical across stages.
from c2_regressor import collect_diffusion, collect_pareidolia, regen_procedural  # noqa: E402

OUT = ROOT / "benchmarks" / "regressor"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_SIZE = 256          # cached source resolution (crops sample sub-regions of this)
IMG_SIZE = 224          # backbone input


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #
IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_NORM = ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])


def make_backbone(name):
    """Return (forward_fn, feat_dim, module, (norm_mean, norm_std)). Frozen + eval."""
    name = name.lower()
    if name == "dinov2":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dinov2-small")
        feat_dim, norm = 384, IMAGENET_NORM

        def fwd(x):
            return m(pixel_values=x).pooler_output            # CLS token, [B,384]
    elif name == "clip":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        feat_dim, norm = 768, CLIP_NORM

        def fwd(x):
            return m(pixel_values=x).pooler_output            # [B,768]
    elif name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        feat_dim, norm = 512, IMAGENET_NORM

        def fwd(x):
            return m(x)                                       # [B,512]
    else:
        raise ValueError(f"unknown backbone {name!r}")

    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return fwd, feat_dim, m, norm


# --------------------------------------------------------------------------- #
# Image cache (source images at 256x256 gray, plus labels/metadata)
# --------------------------------------------------------------------------- #
def _to_gray256(arr_or_path):
    if isinstance(arr_or_path, (str, Path)):
        img = Image.open(arr_or_path).convert("L")
    else:
        a = np.clip(arr_or_path, 0, 1)
        img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize((SRC_SIZE, SRC_SIZE), Image.BICUBIC), dtype=np.uint8)


def build_image_cache():
    print("Collecting diffusion + pareidolia (disk) ...")
    rows = collect_diffusion() + collect_pareidolia()
    print(f"  disk-loaded: {len(rows)}")
    print("Regenerating procedural images ...")
    rows += regen_procedural()
    print(f"  total rows: {len(rows)}")

    imgs = np.zeros((len(rows), SRC_SIZE, SRC_SIZE), dtype=np.uint8)
    c2 = np.zeros(len(rows), dtype=np.float32)
    domains = np.empty(len(rows), dtype=object)
    sources = np.empty(len(rows), dtype=object)
    t0 = time.time()
    for i, r in enumerate(rows):
        imgs[i] = _to_gray256(r["img"] if "img" in r else r["path"])
        c2[i] = r["c2"]; domains[i] = r["domain"]; sources[i] = r["source"]
        if (i + 1) % 200 == 0:
            print(f"  cached {i+1}/{len(rows)}  t={time.time()-t0:4.0f}s")
    return imgs, c2, domains, sources


def load_image_cache(rebuild=False):
    cache = OUT / "images_v2.npz"
    if cache.exists() and not rebuild:
        print(f"Loading image cache {cache.relative_to(ROOT)}")
        d = np.load(cache, allow_pickle=True)
        return d["imgs"], d["c2"], d["domains"], d["sources"]
    imgs, c2, domains, sources = build_image_cache()
    np.savez(cache, imgs=imgs, c2=c2, domains=domains, sources=sources)
    print(f"Wrote {cache.relative_to(ROOT)}  ({len(c2)} samples)")
    return imgs, c2, domains, sources


# --------------------------------------------------------------------------- #
# Dataset with c2-invariant augmentation
# --------------------------------------------------------------------------- #
class _Rot90:
    """Random exact 90° rotation (k in 0..3) — keeps c2 invariant, no interpolation."""
    def __call__(self, x):                     # x: tensor [C,H,W]
        return torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), dims=[1, 2])


class TextureSet(Dataset):
    def __init__(self, imgs, c2, norm, train):
        self.imgs = imgs
        self.c2 = c2
        mean, std = norm
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1),
                                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                _Rot90(),
                T.Normalize(mean, std),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        pil = Image.fromarray(self.imgs[i], mode="L").convert("RGB")
        return self.tf(pil), float(self.c2[i])


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Split / train / eval
# --------------------------------------------------------------------------- #
def stratified_split(sources, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sources))
    splits = {"train": [], "val": [], "test": []}
    for src in np.unique(sources):
        sub = idx[sources == src]; rng.shuffle(sub); n = len(sub)
        n_test = max(1, int(0.10 * n)); n_val = max(1, int(0.10 * n))
        splits["test"].extend(sub[:n_test])
        splits["val"].extend(sub[n_test:n_test + n_val])
        splits["train"].extend(sub[n_test + n_val:])
    for k in splits:
        splits[k] = np.array(splits[k]); rng.shuffle(splits[k])
    return splits


@torch.no_grad()
def predict(backbone_fwd, head, loader, amp):
    head.eval()
    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            feats = backbone_fwd(xb)
        preds.append(head(feats.float()).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def train(args):
    imgs, c2, domains, sources = load_image_cache(args.rebuild)
    print(f"\nDataset: {len(c2)} samples  c2∈[{c2.min():.2f},{c2.max():.2f}]")
    for src in np.unique(sources):
        m = sources == src
        print(f"  {src:11s}: n={int(m.sum())}  c2 mean={c2[m].mean():+.3f}")

    backbone_fwd, feat_dim, _, norm = make_backbone(args.backbone)
    print(f"\nBackbone: {args.backbone}  feat_dim={feat_dim}  device={DEVICE}")

    splits = stratified_split(sources)
    print(f"  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    def loader(which, train_aug, shuffle):
        ds = TextureSet(imgs[splits[which]], c2[splits[which]], norm, train_aug)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"))

    dl_tr = loader("train", train_aug=True, shuffle=True)
    dl_val = loader("val", train_aug=False, shuffle=False)
    dl_te = loader("test", train_aug=False, shuffle=False)

    head = MLP(feat_dim, dropout=args.dropout).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()
    amp = (DEVICE.type == "cuda") and not args.no_amp

    best_val, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        head.train()
        tr = 0.0; ntr = 0
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).float()
            with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                feats = backbone_fwd(xb)
            pred = head(feats.float())
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item() * xb.size(0); ntr += xb.size(0)
        sched.step()
        tr /= ntr
        vp, vy = predict(backbone_fwd, head, dl_val, amp)
        val = float(np.mean((vp - vy) ** 2))
        history.append((ep, tr, val))
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  ep {ep+1:4d}  train={tr:.4f}  val={val:.4f}  best={best_val:.4f}"
                  f"  t={time.time()-t0:5.0f}s")
        if args.patience and bad >= args.patience:
            print(f"  early stop at epoch {ep+1} (no val improvement for {args.patience})")
            break

    head.load_state_dict(best_state)
    pred_te, y_te = predict(backbone_fwd, head, dl_te, amp)
    metrics = report(pred_te, y_te, sources[splits["test"]], domains[splits["test"]],
                     history, args.backbone)
    torch.save(dict(state_dict=head.state_dict(), backbone=args.backbone,
                    feat_dim=feat_dim, hidden=(128, 64)), OUT / "pixel_model.pt")
    print(f"Wrote {(OUT / 'pixel_model.pt').relative_to(ROOT)}")
    return metrics


def report(pred, y, sources_test, domains_test, history, backbone):
    pearson = float(np.corrcoef(pred, y)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((pred - y) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    metrics = dict(backbone=backbone, pearson=pearson, rmse=rmse, mae=mae, r2=r2,
                   n_test=int(len(y)))
    print("\n=== TEST METRICS ===")
    for k in ("pearson", "rmse", "mae", "r2"):
        print(f"  {k:8s} = {metrics[k]:+.4f}")

    per_src, per_dom = {}, {}
    for label, arr, store in [("source", sources_test, per_src),
                              ("domain", domains_test, per_dom)]:
        for v in np.unique(arr):
            m = arr == v
            if m.sum() < 3:
                continue
            p, t = pred[m], y[m]
            store[str(v)] = dict(n=int(m.sum()),
                                 pearson=float(np.corrcoef(p, t)[0, 1]) if m.sum() > 1 else float("nan"),
                                 rmse=float(np.sqrt(np.mean((p - t) ** 2))))
    print("\nPer-source:")
    for s, d in per_src.items():
        print(f"  {s:11s} n={d['n']:4d}  r={d['pearson']:+.3f}  rmse={d['rmse']:.3f}")
    metrics["per_source"] = per_src
    metrics["per_domain"] = per_dom
    diff = per_src.get("diffusion", {}).get("rmse")
    if diff is not None:
        verdict = "PASS" if diff < 0.15 else "FAIL"
        print(f"\nStage-1 gate (diffusion RMSE < 0.15): {diff:.3f}  -> {verdict}")
        metrics["stage1_gate_pass"] = bool(diff < 0.15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    colors = {"procedural": "#8c564b", "pareidolia": "#2ca02c", "diffusion": "#1f77b4"}
    for src in np.unique(sources_test):
        m = sources_test == src
        ax.scatter(y[m], pred[m], c=colors.get(src, "k"), s=22, alpha=0.7,
                   edgecolor="k", linewidth=0.3, label=f"{src} (n={int(m.sum())})")
    lo = min(y.min(), pred.min()) - 0.05; hi = max(y.max(), pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x")
    ax.set_xlabel("measured c2"); ax.set_ylabel("predicted c2")
    ax.set_title(f"{backbone}  test n={len(y)}\nr={pearson:+.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    ax = axes[1]
    h = np.array(history)
    ax.plot(h[:, 0], h[:, 1], label="train MSE", lw=1.5)
    ax.plot(h[:, 0], h[:, 2], label="val MSE", lw=1.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_yscale("log")
    ax.set_title("Training curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_v1.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {(OUT / 'scatter_v1.png').relative_to(ROOT)}")
    (OUT / "metrics_v1.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {(OUT / 'metrics_v1.json').relative_to(ROOT)}")
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="dinov2", choices=["dinov2", "clip", "resnet18"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=60, help="early-stop patience (0=off)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable fp16 autocast")
    p.add_argument("--rebuild", action="store_true", help="force-rebuild image cache")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
