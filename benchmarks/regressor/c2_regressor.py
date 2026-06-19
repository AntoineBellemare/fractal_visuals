"""
c2_regressor.py — MVP c2 regressor: predict wavelet-leader c2 of a natural-texture
image from its frozen-ResNet18 features.

Data (~2000 labeled images):
  procedural  -- regenerate from benchmarks/results/validation_manifest.csv
  pareidolia  -- read from datasets/pareidolia/images/
  diffusion   -- read from benchmarks/diffusion/out/<family>/

Pipeline:
  1. Build (image, c2) pairs from all three sources at 224x224 grayscale->RGB.
  2. Pass once through a frozen ImageNet-pretrained ResNet18 -> 512-d features.
     Cache to features.npz so the MLP head can be retrained instantly.
  3. Train a small MLP head (512 -> 128 -> 64 -> 1) with 80/10/10 split.
  4. Report Pearson r, Spearman rho, RMSE on the held-out test set.
  5. Scatter plot of predicted vs measured c2.

Outputs (benchmarks/regressor/):
  features.npz, model.pt, scatter.png, metrics.json
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mfractal as mf

OUT = ROOT / "benchmarks" / "regressor"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

PREPROCESS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),                                 # [0,1], C=3
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def to_pil_gray3(arr_or_path):
    """Accept np.ndarray float [0,1] or path. Return PIL RGB image (gray replicated)."""
    if isinstance(arr_or_path, (str, Path)):
        img = Image.open(arr_or_path).convert("L")
    else:
        a = np.clip(arr_or_path, 0, 1)
        img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    return img.convert("RGB")


def collect_diffusion():
    df = pd.read_csv(ROOT / "benchmarks" / "diffusion" / "out" / "analysis.csv")
    rows = []
    for _, r in df.iterrows():
        p = ROOT / "benchmarks" / "diffusion" / "out" / r["path"]
        if not p.exists() or not np.isfinite(r["c2"]):
            continue
        rows.append(dict(source="diffusion", path=str(p), c2=float(r["c2"]),
                         family=r["family"], domain=r["domain"]))
    return rows


def collect_pareidolia():
    df = pd.read_csv(ROOT / "datasets" / "pareidolia" / "manifest.csv")
    rows = []
    for _, r in df.iterrows():
        p = ROOT / "datasets" / "pareidolia" / r["image_path"]
        if not p.exists() or not np.isfinite(r["c2"]):
            continue
        rows.append(dict(source="pareidolia", path=str(p), c2=float(r["c2"]),
                         family=r["family"], domain=r["domain"]))
    return rows


def regen_procedural():
    """The validation_manifest doesn't store PNGs — regenerate from (family, cx, seed)."""
    df = pd.read_csv(ROOT / "benchmarks" / "results" / "validation_manifest.csv")
    df = df[(df["status"] == "ok") & df["c2"].between(-3.0, 0.5)].copy()
    rows = []
    t0 = time.time()
    for k, r in df.iterrows():
        fam = r["family"]
        # Skip families that no longer exist post-prune.
        if not hasattr(mf, fam):
            continue
        try:
            img = mf.generate(fam, n=int(r["n"]), complexity=float(r["complexity"]),
                              seed=int(r["seed"]))
        except Exception:
            continue
        rows.append(dict(source="procedural", img=img.astype(np.float32),
                         c2=float(r["c2"]), family=fam, domain=r["domain"]))
        if (k + 1) % 100 == 0:
            print(f"  procedural {k+1}/{len(df)}  t={time.time()-t0:5.0f}s")
    return rows


def build_dataset():
    print("Collecting diffusion + pareidolia (read from disk)...")
    rows = collect_diffusion() + collect_pareidolia()
    print(f"  disk-loaded: {len(rows)}")
    print("Regenerating procedural images...")
    rows += regen_procedural()
    print(f"  total: {len(rows)}")
    return rows


@torch.no_grad()
def extract_features(rows, batch_size=32):
    """One forward pass through frozen ResNet18 -> 512-d features per image."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = nn.Identity()           # output the 512-d global-average pool
    net.eval().to(DEVICE)

    feats = np.zeros((len(rows), 512), dtype=np.float32)
    c2 = np.zeros(len(rows), dtype=np.float32)
    domains = np.empty(len(rows), dtype=object)
    sources = np.empty(len(rows), dtype=object)

    batch_imgs = []
    batch_idx = []
    t0 = time.time()

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(DEVICE)
        f = net(x).cpu().numpy()
        for j, gi in enumerate(batch_idx):
            feats[gi] = f[j]
        batch_imgs.clear(); batch_idx.clear()

    for i, r in enumerate(rows):
        if "img" in r:
            pil = to_pil_gray3(r["img"])
        else:
            pil = to_pil_gray3(r["path"])
        batch_imgs.append(PREPROCESS(pil))
        batch_idx.append(i)
        c2[i] = r["c2"]
        domains[i] = r["domain"]
        sources[i] = r["source"]
        if len(batch_imgs) >= batch_size:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  features {i+1}/{len(rows)}  t={time.time()-t0:5.0f}s")
    flush()
    return feats, c2, domains, sources


class MLP(nn.Module):
    def __init__(self, in_dim=512, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(feats, c2, sources, domains, epochs=400, batch_size=128, lr=1e-3,
              weight_decay=1e-4):
    """Stratified by source so train/val/test all contain procedural + diffusion + pareidolia."""
    rng = np.random.default_rng(0)
    idx = np.arange(len(c2))
    splits = {"train": [], "val": [], "test": []}
    for src in np.unique(sources):
        sub = idx[sources == src]
        rng.shuffle(sub)
        n = len(sub)
        n_test = max(1, int(0.10 * n))
        n_val = max(1, int(0.10 * n))
        splits["test"].extend(sub[:n_test])
        splits["val"].extend(sub[n_test:n_test + n_val])
        splits["train"].extend(sub[n_test + n_val:])
    for k in splits:
        splits[k] = np.array(splits[k])
        rng.shuffle(splits[k])

    print(f"  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    for src in np.unique(sources):
        for k in splits:
            n_src = int(np.sum(sources[splits[k]] == src))
            print(f"    {k:5s}/{src:11s} = {n_src}")

    Xtr = torch.from_numpy(feats[splits["train"]]).float()
    ytr = torch.from_numpy(c2[splits["train"]]).float()
    Xval = torch.from_numpy(feats[splits["val"]]).float().to(DEVICE)
    yval = torch.from_numpy(c2[splits["val"]]).float().to(DEVICE)
    Xte = torch.from_numpy(feats[splits["test"]]).float().to(DEVICE)
    yte = torch.from_numpy(c2[splits["test"]]).float().to(DEVICE)

    ds_tr = TensorDataset(Xtr, ytr)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(ds_tr)
        sched.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval)
            val_loss = loss_fn(val_pred, yval).item()
        history.append((ep, tr_loss, val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1:4d}  train_mse={tr_loss:.4f}  val_mse={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test = model(Xte).cpu().numpy()
    y_test = yte.cpu().numpy()
    return model, splits, pred_test, y_test, history


def report(pred, y, sources_test, domains_test, history):
    pearson = float(np.corrcoef(pred, y)[0, 1])
    # Spearman without scipy
    rx = np.argsort(np.argsort(pred)) - len(pred) / 2
    ry = np.argsort(np.argsort(y)) - len(y) / 2
    spearman = float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    metrics = dict(pearson=pearson, spearman=spearman, rmse=rmse, mae=mae, r2=r2,
                   n_test=int(len(y)))
    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        print(f"  {k:10s} = {v}")

    # Per-source breakdown
    per_src = {}
    for src in np.unique(sources_test):
        m = sources_test == src
        if m.sum() < 3:
            continue
        p, t = pred[m], y[m]
        per_src[src] = dict(n=int(m.sum()),
                            pearson=float(np.corrcoef(p, t)[0, 1]),
                            rmse=float(np.sqrt(np.mean((p - t) ** 2))))
    print("\nPer-source test metrics:")
    for src, d in per_src.items():
        print(f"  {src:11s}  n={d['n']}  r={d['pearson']:+.3f}  rmse={d['rmse']:.3f}")
    metrics["per_source"] = per_src

    # scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    colors = {"procedural": "#8c564b", "pareidolia": "#2ca02c", "diffusion": "#1f77b4"}
    for src in np.unique(sources_test):
        m = sources_test == src
        ax.scatter(y[m], pred[m], c=colors.get(src, "k"), label=f"{src} (n={m.sum()})",
                   s=22, alpha=0.7, edgecolor="k", linewidth=0.3)
    lo = min(y.min(), pred.min()) - 0.05
    hi = max(y.max(), pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x")
    ax.set_xlabel("measured c2 (wavelet-leader)")
    ax.set_ylabel("predicted c2 (regressor)")
    ax.set_title(f"Held-out test set  (n={len(y)})\n"
                 f"Pearson r = {pearson:+.3f}  RMSE = {rmse:.3f}  R² = {r2:.3f}")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    hist = np.array(history)
    ax.plot(hist[:, 0], hist[:, 1], label="train MSE", lw=1.5)
    ax.plot(hist[:, 0], hist[:, 2], label="val MSE", lw=1.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.set_title("Training curve")
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUT / "scatter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {(OUT / 'scatter.png').relative_to(ROOT)}")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {(OUT / 'metrics.json').relative_to(ROOT)}")
    return metrics


def main():
    feat_cache = OUT / "features.npz"
    if feat_cache.exists():
        print(f"Loading cached features from {feat_cache.relative_to(ROOT)}")
        d = np.load(feat_cache, allow_pickle=True)
        feats, c2 = d["feats"], d["c2"]
        domains, sources = d["domains"], d["sources"]
    else:
        rows = build_dataset()
        feats, c2, domains, sources = extract_features(rows)
        np.savez(feat_cache, feats=feats, c2=c2, domains=domains, sources=sources)
        print(f"Wrote {feat_cache.relative_to(ROOT)}  ({len(c2)} samples)")

    print(f"\nDataset: {len(c2)} samples")
    print(f"  c2 range: [{c2.min():.2f}, {c2.max():.2f}]  mean={c2.mean():+.3f}  std={c2.std():.3f}")
    for src in np.unique(sources):
        m = sources == src
        print(f"  {src:11s}: n={m.sum()}  c2 mean={c2[m].mean():+.3f}")

    model, splits, pred_test, y_test, history = train_mlp(feats, c2, sources, domains)
    sources_test = sources[splits["test"]]
    domains_test = domains[splits["test"]]
    metrics = report(pred_test, y_test, sources_test, domains_test, history)

    torch.save(dict(state_dict=model.state_dict(),
                    arch=dict(in_dim=512, hidden=(128, 64))),
               OUT / "model.pt")
    print(f"Wrote {(OUT / 'model.pt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
