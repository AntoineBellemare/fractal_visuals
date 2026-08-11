"""
probe_metal.py -- run the trained MVP c2 regressor on the 3 metal families
and check whether they sit in-distribution or out-of-distribution.

The MVP was trained on rock + cloud + fluid + diffusion + pareidolia samples;
metal (rust_bloom, rust_pitted, rust_dewy) was added to the toolbox AFTER the
MVP training run. This script:

  1. Loads model.pt + features.npz from the MVP run.
  2. Generates 3 families x 5 cx x 4 seeds = 60 metal images.
  3. Measures wavelet-leader c2 on each (ground truth) and predicts via the
     MVP regressor (frozen ResNet18 + MLP head).
  4. Plots predicted vs measured for metal overlaid on the original
     held-out test set, with per-family color.

Outputs:
  benchmarks/regressor/metal_ood_scatter.png
  benchmarks/regressor/metal_ood_metrics.json
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
PREPROCESS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CXS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3]
N = 256


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


def to_pil_gray3(arr):
    a = np.clip(arr, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8), mode="L").convert("RGB")


@torch.no_grad()
def extract(backbone, imgs):
    x = torch.stack([PREPROCESS(to_pil_gray3(i)) for i in imgs]).to(DEVICE)
    return backbone(x).cpu().numpy()


def main():
    # 1. Generate metal stimuli + measure ground-truth c2.
    print("Generating metal stimuli (3 fams x 5 cx x 4 seeds = 60 images)...")
    rows = []
    t0 = time.time()
    for fam in mf.METAL_FAMILIES:
        for cx in CXS:
            for sd in SEEDS:
                img = mf.generate(fam, n=N, complexity=cx, seed=sd)
                c2 = float(mf.wavelet_leaders_2d(img)["c2"])
                rows.append(dict(family=fam, cx=cx, seed=sd, img=img.astype(np.float32),
                                 c2=c2))
    print(f"  generated + measured: {len(rows)} samples in {time.time()-t0:.1f}s")

    # 2. Load backbone + trained head.
    print("Loading frozen ResNet18 + trained MLP head...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone.eval().to(DEVICE)

    ckpt = torch.load(OUT / "model.pt", map_location=DEVICE, weights_only=False)
    head = MLP(**ckpt["arch"]).to(DEVICE)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    # 3. Predict.
    print("Predicting c2 for metal samples...")
    feats = extract(backbone, [r["img"] for r in rows])
    with torch.no_grad():
        pred = head(torch.from_numpy(feats).float().to(DEVICE)).cpu().numpy()
    measured = np.array([r["c2"] for r in rows])
    families = np.array([r["family"] for r in rows])

    # Per-family stats.
    print("\nPer-family generalization on metal:")
    per_fam = {}
    for fam in mf.METAL_FAMILIES:
        m = families == fam
        if not m.any():
            continue
        p, t = pred[m], measured[m]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        bias = float((p - t).mean())
        r = float(np.corrcoef(p, t)[0, 1]) if t.std() > 1e-6 else float("nan")
        per_fam[fam] = dict(n=int(m.sum()),
                             rmse=rmse, bias=bias, pearson=r,
                             measured_range=[float(t.min()), float(t.max())],
                             predicted_range=[float(p.min()), float(p.max())])
        print(f"  {fam:14s}  n={m.sum():3d}  rmse={rmse:.3f}  bias={bias:+.3f}  r={r:+.3f}")
        print(f"    measured  c2: [{t.min():+.2f}, {t.max():+.2f}]")
        print(f"    predicted c2: [{p.min():+.2f}, {p.max():+.2f}]")

    overall_rmse = float(np.sqrt(np.mean((pred - measured) ** 2)))
    overall_bias = float((pred - measured).mean())
    overall_r = float(np.corrcoef(pred, measured)[0, 1])
    print(f"\nOverall: rmse={overall_rmse:.3f}  bias={overall_bias:+.3f}  r={overall_r:+.3f}")

    # 4. Plot: metal overlaid on the original held-out test scatter for context.
    d = np.load(OUT / "features.npz", allow_pickle=True)
    f_all, c2_all, sources_all = d["feats"], d["c2"], d["sources"]
    rng = np.random.default_rng(0)
    idx = np.arange(len(c2_all))
    splits_test = []
    for src in np.unique(sources_all):
        sub = idx[sources_all == src]; rng.shuffle(sub)
        n_test = max(1, int(0.10 * len(sub)))
        splits_test.extend(sub[:n_test])
    splits_test = np.array(splits_test)
    with torch.no_grad():
        ref_pred = head(torch.from_numpy(f_all[splits_test]).float().to(DEVICE)).cpu().numpy()
    ref_meas = c2_all[splits_test]
    ref_src = sources_all[splits_test]

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    src_colors = {"procedural": "#bbbbbb", "pareidolia": "#dddddd", "diffusion": "#cccccc"}
    for src in np.unique(ref_src):
        m = ref_src == src
        ax.scatter(ref_meas[m], ref_pred[m], c=src_colors.get(src, "k"),
                   s=18, alpha=0.5, edgecolor="k", linewidth=0.2,
                   label=f"test/{src} (n={m.sum()})")
    fam_colors = {"rust_bloom":  "#d62728",
                  "rust_pitted": "#1f77b4",
                  "rust_dewy":   "#2ca02c"}
    for fam, c in fam_colors.items():
        m = families == fam
        ax.scatter(measured[m], pred[m], c=c, s=32, alpha=0.85,
                   edgecolor="k", linewidth=0.4,
                   label=f"{fam} (n={m.sum()})")
    lo = min(min(ref_meas.min(), ref_pred.min()),
             min(measured.min(), pred.min())) - 0.05
    hi = max(max(ref_meas.max(), ref_pred.max()),
             max(measured.max(), pred.max())) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured c2 (wavelet-leader)")
    ax.set_ylabel("predicted c2 (MVP regressor)")
    ax.set_title(f"MVP regressor generalization to metal (post-training addition)\n"
                 f"metal overall: rmse={overall_rmse:.2f}  bias={overall_bias:+.2f}  r={overall_r:+.2f}",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "metal_ood_scatter.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png.relative_to(ROOT)}")

    metrics = dict(
        overall=dict(n=int(len(measured)), rmse=overall_rmse,
                     bias=overall_bias, pearson=overall_r),
        per_family=per_fam,
    )
    (OUT / "metal_ood_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {(OUT / 'metal_ood_metrics.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
