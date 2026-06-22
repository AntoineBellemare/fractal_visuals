"""
train_stage1.py  --  Stage 1 of the c2 regressor: DINOv2-small backbone + MLP head.

What changed from the CPU MVP (and why):
  * Backbone ResNet18 -> DINOv2-small (frozen). DINOv2 encodes texture statistics
    broadly, which is exactly what a low-level pixel-stat target like c2 needs.
  * Augmentations are c2-PRESERVING ONLY: H/V flip + 90° rotation (the dihedral
    group, 8x). Random crops + resizes are gone (they change multi-scale content);
    brightness/contrast are gone too (the [0,1] clip shifts c2 on heavy-tailed
    fields). check_c2_invariance.py is the empirical proof. This removes the biggest
    source of label noise.
  * Label and input share ONE resolution convention. Labels are measured at 512
    (build_corpus.py); the network ingests the WHOLE image resized to --net-res
    (default 518, DINOv2's native patch grid). A whole-image resize is a fixed
    deterministic map -- a learnable constant gap, not random per-sample noise.
  * Split is GROUP-AWARE: no diffusion prompt (and no procedural family / c2 band)
    straddles train/val/test, so the reported diffusion RMSE is real generalization,
    not seed memorization.
  * Grayscale->3ch input: c2 is a grayscale statistic, so color is dropped for a
    consistent modality across procedural / diffusion / photo sources.

Frozen-backbone note: with live augmentation we re-run the backbone every epoch
(can't cache features AND augment). `--cache-features` gives a fast no-aug baseline
that precomputes features once; the two paths are mutually exclusive by construction.

Run (GPU):
  python train_stage1.py --epochs 300
  python train_stage1.py --cache-features --epochs 300   # fast no-aug baseline
  python train_stage1.py --finetune --lr-backbone 1e-5   # unfreeze (decision point)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_corpus import CANON_RES, load_field  # noqa: E402  (single labeling source)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def robust_unit(f: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
    """Per-image affine map to ~[0,1] using percentiles, then clip. This spreads
    heavy-tailed generated fields so DINOv2 can read their structure, and is ~identity
    on already-spread 8-bit photos. The label was measured on the faithful field, and
    THIS same map is applied at deployment, so the net just learns input->c2."""
    a, b = np.percentile(f, lo), np.percentile(f, hi)
    if b - a < 1e-6:
        return np.zeros_like(f)
    return np.clip((f - a) / (b - a), 0, 1).astype(np.float32)


def load_gray_tensor(path: Path, res: int) -> torch.Tensor:
    """path (.npy field or image) -> grayscale -> robust-normalize -> resize(res)
    -> float[0,1] tensor (1,res,res). Identical view to what was labeled."""
    f = robust_unit(load_field(path))                       # 512, [0,1]
    t = torch.from_numpy(f)[None, None]                     # (1,1,512,512)
    if t.shape[-1] != res:
        t = F.interpolate(t, size=(res, res), mode="bicubic", align_corners=False)
    return t[0].clamp_(0, 1)                                # (1, res, res)


def c2_invariant_aug(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Apply ONLY geometric label-preserving ops to a (1,H,W) field: H/V flip + 90°
    rotation (the dihedral group, 8x expansion). check_c2_invariance.py confirms these
    leave c2 within ~0.01.

    Photometric ops (brightness/contrast) were tested and REJECTED: on faithful
    heavy-tailed fields the [0,1] clip interacts with the near-zero mass that carries
    the multifractal structure, shifting c2 by >2.0. Geometric only is the safe set."""
    if torch.rand(1, generator=rng).item() < 0.5:
        x = torch.flip(x, dims=[2])             # hflip
    if torch.rand(1, generator=rng).item() < 0.5:
        x = torch.flip(x, dims=[1])             # vflip
    k = int(torch.randint(0, 4, (1,), generator=rng).item())
    if k:
        x = torch.rot90(x, k, dims=[1, 2])
    return x


class C2Dataset(Dataset):
    def __init__(self, rows, res, augment, seed=0):
        self.rows = rows
        self.res = res
        self.augment = augment
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = Path(r["path"])
        p = p if p.is_absolute() else ROOT / p
        x = load_gray_tensor(p, self.res)
        if self.augment:
            x = c2_invariant_aug(x, self.rng)
        x = x.repeat(3, 1, 1)                    # gray -> 3ch
        x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        return x, torch.tensor(float(r["c2"]), dtype=torch.float32), r["source"]


# --------------------------------------------------------------------------- #
# Leakage-free split: whole groups go to exactly one of train/val/test, balanced
# per source so every source is ~80/10/10.
# --------------------------------------------------------------------------- #
def grouped_split(rows, val=0.1, test=0.1, seed=0):
    rng = np.random.default_rng(seed)
    by_src: dict[str, dict[str, list]] = {}
    for r in rows:
        by_src.setdefault(r["source"], {}).setdefault(r["group"], []).append(r)
    train, va, te = [], [], []
    for src, groups in by_src.items():
        gkeys = list(groups)
        rng.shuffle(gkeys)
        n = len(gkeys)
        n_te = max(1, int(round(n * test))) if n > 2 else 0
        n_va = max(1, int(round(n * val))) if n > 2 else 0
        te_k, va_k, tr_k = gkeys[:n_te], gkeys[n_te:n_te + n_va], gkeys[n_te + n_va:]
        for k in tr_k: train += groups[k]
        for k in va_k: va += groups[k]
        for k in te_k: te += groups[k]
    return train, va, te


def stratified_split(rows, val=0.1, test=0.1, seed=0):
    """Image-level split, stratified by source (the OTHER agent's MVP scheme). LEAKY:
    a family's/prompt's other seeds can land in both train and test. Kept ONLY for an
    apples-to-apples comparison that exposes how much the group split's honesty costs."""
    rng = np.random.default_rng(seed)
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    train, va, te = [], [], []
    for src, items in by_src.items():
        rng.shuffle(items)
        n = len(items)
        n_te = max(1, int(round(n * test)))
        n_va = max(1, int(round(n * val)))
        te += items[:n_te]; va += items[n_te:n_te + n_va]; train += items[n_te + n_va:]
    return train, va, te


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    def __init__(self, in_dim, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_backbone(name="facebook/dinov2-small"):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name)
    return m, m.config.hidden_size


def dino_pooled(backbone, x):
    """Version-robust forward. transformers >=4.46 interpolates pos-embeds internally
    (and rejects the kwarg); older builds need it explicitly. Try, then fall back."""
    try:
        return backbone(pixel_values=x, interpolate_pos_encoding=True).pooler_output
    except TypeError:
        return backbone(pixel_values=x).pooler_output


def backbone_features(backbone, x, finetune):
    ctx = torch.enable_grad() if finetune else torch.no_grad()
    with ctx:
        return dino_pooled(backbone, x)        # (B, hidden) CLS after layernorm


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_stats(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if len(y_true) > 1 and y_true.std() > 0 and y_pred.std() > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        r, r2 = float("nan"), float("nan")
    return dict(n=int(len(y_true)), rmse=rmse, pearson_r=r, r2=r2)


@torch.no_grad()
def evaluate(backbone, head, loader, device):
    head.eval(); backbone.eval()
    ys, ps, srcs = [], [], []
    for x, y, src in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            feat = dino_pooled(backbone, x)
            pred = head(feat.float())
        ys += y.tolist(); ps += pred.cpu().tolist(); srcs += list(src)
    ys, ps, srcs = np.array(ys), np.array(ps), np.array(srcs)
    out = {"overall": regression_stats(ys, ps)}
    for s in sorted(set(srcs)):
        m = srcs == s
        out[s] = regression_stats(ys[m], ps[m])
    return out


# --------------------------------------------------------------------------- #
# Cached-feature fast path (adopted from the other agent's MVP): one frozen-backbone
# pass over every image, then the head trains in seconds. Lets us sweep split modes
# and hyperparams cheaply. Mutually exclusive with augmentation (features are fixed).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_all_features(rows, backbone, net_res, device, batch=64):
    hidden = backbone.config.hidden_size
    feats = np.zeros((len(rows), hidden), np.float32)
    bx, bi = [], []

    def flush():
        if not bx:
            return
        x = torch.stack(bx).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            f = dino_pooled(backbone, x).float().cpu().numpy()
        for j, gi in enumerate(bi):
            feats[gi] = f[j]
        bx.clear(); bi.clear()

    for i, r in enumerate(rows):
        p = Path(r["path"]); p = p if p.is_absolute() else ROOT / p
        x = load_gray_tensor(p, net_res).repeat(3, 1, 1)
        x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        bx.append(x); bi.append(i)
        if len(bx) >= batch:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  features {i+1}/{len(rows)}")
    flush()
    return feats


def get_cached_features(rows, args, device):
    """Features aligned to `rows` order, cached on disk keyed by (backbone, net_res).
    Re-extracts only if the cache is missing or the image set changed."""
    safe = args.backbone.replace("/", "_")
    cache = HERE / "corpus" / f"features_{safe}_{args.net_res}.npz"
    paths = [str(r["path"]) for r in rows]
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if len(d["paths"]) == len(paths) and list(d["paths"]) == paths:
            print(f"loaded cached features {d['feats'].shape} <- {cache.name}")
            return d["feats"]
    backbone, _ = build_backbone(args.backbone)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print("extracting features (one frozen pass over the corpus)...")
    feats = extract_all_features(rows, backbone, args.net_res, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feats=feats, paths=np.array(paths, object))
    print(f"cached features {feats.shape} -> {cache.name}")
    return feats


def train_head_cached(Xtr, ytr, Xva, yva, Xte, yte, srcs_te, args, device):
    head = Head(Xtr.shape[1], p=args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 1e-2)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva, yva = Xva.to(device), yva.to(device)
    best, best_state, bad = math.inf, None, 0
    n, bs = len(Xtr), max(32, args.batch)
    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(head(Xtr[j]), ytr[j])
            loss.backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            vmse = F.mse_loss(head(Xva), yva).item()
        if vmse < best - 1e-6:
            best, bad = vmse, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(Xte.to(device)).cpu().numpy()
    y = yte.numpy()
    out = {"overall": regression_stats(y, pred)}
    for s in sorted(set(srcs_te)):
        m = np.array(srcs_te) == s
        out[s] = regression_stats(y[m], pred[m])
    return head, out, pred, y, best


def scatter_plot(y, pred, srcs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for s in sorted(set(srcs)):
        m = np.array(srcs) == s
        ax.scatter(y[m], pred[m], s=22, alpha=0.7, edgecolor="k", linewidth=0.3,
                   label=f"{s} (n={int(m.sum())})")
    lo, hi = float(min(y.min(), pred.min())) - 0.1, float(max(y.max(), pred.max())) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y=x")
    ax.set_xlabel("measured c2"); ax.set_ylabel("predicted c2")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "corpus" / "manifest.csv"))
    ap.add_argument("--backbone", default="facebook/dinov2-small")
    ap.add_argument("--net-res", type=int, default=518)   # multiple of patch 14
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--finetune", action="store_true", help="unfreeze the backbone")
    ap.add_argument("--cache-features", action="store_true",
                    help="precompute frozen features once; trains the head fast (no aug)")
    ap.add_argument("--split", choices=["group", "stratified"], default="group",
                    help="group=leakage-free (default); stratified=image-level (leaky, "
                         "for apples-to-apples comparison with the MVP)")
    ap.add_argument("--c2-min", type=float, default=-1.5,
                    help="drop samples with c2 below this (extreme tails inflate RMSE "
                         "and are outside the control range); set -inf to keep all")
    ap.add_argument("--c2-max", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    if args.cache_features and (args.finetune):
        ap.error("--cache-features is incompatible with --finetune (features change)")
    if args.net_res % 14 != 0:
        print(f"warning: --net-res {args.net_res} is not a multiple of 14 "
              f"(DINOv2 patch); rounding down.")
        args.net_res -= args.net_res % 14

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  | label res {CANON_RES}  | net res {args.net_res}")

    import pandas as pd
    rows = pd.read_csv(args.manifest).to_dict("records")
    n0 = len(rows)
    rows = [r for r in rows if args.c2_min <= float(r["c2"]) <= args.c2_max]
    print(f"loaded {n0} rows; kept {len(rows)} in c2 [{args.c2_min}, {args.c2_max}]")

    splitter = grouped_split if args.split == "group" else stratified_split
    train_rows, val_rows, test_rows = splitter(rows, seed=args.seed)
    print(f"split={args.split}: train {len(train_rows)}  val {len(val_rows)}  "
          f"test {len(test_rows)}")
    if args.split == "group":
        g = lambda rs: {r["group"] for r in rs}
        assert not (g(train_rows) & g(val_rows)) and not (g(train_rows) & g(test_rows)) \
            and not (g(val_rows) & g(test_rows)), "group leak across splits!"

    # ---- fast cached-feature path (frozen backbone) ----
    if args.cache_features and not args.finetune:
        feats_all = get_cached_features(rows, args, device)
        ix = {id(r): i for i, r in enumerate(rows)}
        sel = lambda rs: torch.from_numpy(feats_all[[ix[id(r)] for r in rs]]).float()
        yof = lambda rs: torch.tensor([float(r["c2"]) for r in rs], dtype=torch.float32)
        head, test, pred, ytrue, best_val = train_head_cached(
            sel(train_rows), yof(train_rows), sel(val_rows), yof(val_rows),
            sel(test_rows), yof(test_rows), [r["source"] for r in test_rows], args, device)
        out = Path(args.out)
        torch.save({"head": head.state_dict(), "backbone": None,
                    "backbone_id": args.backbone, "net_res": args.net_res,
                    "finetune": False}, out / "pixel_model.pt")
        tag = args.split
        (out / f"metrics_v2_{tag}.json").write_text(json.dumps(
            {"config": vars(args), "best_val_mse": best_val, "test": test}, indent=2))
        scatter_plot(ytrue, pred, [r["source"] for r in test_rows],
                     out / f"scatter_{tag}.png", f"Stage 1 ({tag} split)")
        print(f"\n=== TEST ({tag} split) ===")
        for k, v in test.items():
            print(f"  {k:12s} n={v['n']:4d}  RMSE={v['rmse']:.4f}  "
                  f"r={v['pearson_r']:.3f}  R2={v['r2']:.3f}")
        print(f"saved -> metrics_v2_{tag}.json, scatter_{tag}.png")
        return

    augment = not args.cache_features
    dl = lambda rs, aug, shuf: DataLoader(
        C2Dataset(rs, args.net_res, aug, seed=args.seed),
        batch_size=args.batch, shuffle=shuf, num_workers=4, pin_memory=True,
        drop_last=False)
    train_loader = dl(train_rows, augment, True)
    val_loader = dl(val_rows, False, False)
    test_loader = dl(test_rows, False, False)

    backbone, hidden = build_backbone(args.backbone)
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = args.finetune
    head = Head(hidden, p=args.dropout).to(device)

    params = [{"params": head.parameters(), "lr": args.lr}]
    if args.finetune:
        params.append({"params": backbone.parameters(), "lr": args.lr_backbone})
    opt = torch.optim.AdamW(params, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val, best_state, bad = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        head.train(); backbone.train(args.finetune)
        tot = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                feat = backbone_features(backbone, x, args.finetune)
                pred = head(feat.float())
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        train_mse = tot / len(train_rows)

        val = evaluate(backbone, head, val_loader, device)
        vmse = val["overall"]["rmse"] ** 2
        if vmse < best_val - 1e-5:
            best_val, bad = vmse, 0
            best_state = {"head": head.state_dict(),
                          "backbone": backbone.state_dict() if args.finetune else None}
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            diff = val.get("diffusion", {}).get("rmse", float("nan"))
            print(f"ep {epoch:3d}  train_mse {train_mse:.4f}  "
                  f"val_rmse {val['overall']['rmse']:.4f}  "
                  f"val_diff_rmse {diff:.4f}  best_val_mse {best_val:.4f}  bad {bad}")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    # restore best, evaluate on test
    if best_state is not None:
        head.load_state_dict(best_state["head"])
        if args.finetune and best_state["backbone"] is not None:
            backbone.load_state_dict(best_state["backbone"])
    test = evaluate(backbone, head, test_loader, device)

    out = Path(args.out)
    torch.save({"head": head.state_dict(),
                "backbone": backbone.state_dict() if args.finetune else None,
                "backbone_id": args.backbone, "net_res": args.net_res,
                "finetune": args.finetune}, out / "pixel_model.pt")
    metrics = {"config": vars(args), "best_val_mse": best_val,
               "test": test,
               "splits": {"train": len(train_rows), "val": len(val_rows),
                          "test": len(test_rows)}}
    (out / "metrics_v2.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== TEST ===")
    for k, v in test.items():
        print(f"  {k:12s} n={v['n']:4d}  RMSE={v['rmse']:.4f}  "
              f"r={v['pearson_r']:.3f}  R2={v['r2']:.3f}")
    diff = test.get("diffusion", {}).get("rmse", float("nan"))
    print(f"\nStage-1 pass criterion (diffusion RMSE < 0.15): "
          f"{'PASS' if diff < 0.15 else 'FAIL'}  ({diff:.4f})")
    print(f"saved -> {out/'pixel_model.pt'}, {out/'metrics_v2.json'}")


if __name__ == "__main__":
    main()
