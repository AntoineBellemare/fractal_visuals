"""
eval_joint_compare.py — did adding diverse-prompt data help? Before vs after.

Evaluates two joint models on the SAME held-out test split (enriched-corpus latents),
broken down by content type: natural (prompts.py families) vs diverse (intricate + ood).
The 'before' model never saw any diverse content -> its error there is the OOD gap; the
'after' model trained on diverse train prompts -> measures the gain on held-out diverse.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "benchmarks" / "diffusion"))
from train_stage1 import grouped_split, regression_stats   # noqa: E402
from joint_regressor import JointCNN                         # noqa: E402

DIVERSE = set()
for mod in ("prompts_intricate", "prompts_ood"):
    import importlib
    DIVERSE |= set(importlib.import_module(mod).PROMPTS.keys())


def famtype(row):
    parts = str(row["group"]).split(":")
    fam = parts[1] if len(parts) > 1 else ""
    return "diverse" if fam in DIVERSE else "natural"


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    m = JointCNN(in_ch=4).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def predict(model, Z, device, bs=128):
    out = []
    for i in range(0, len(Z), bs):
        z = torch.from_numpy(Z[i:i+bs].astype(np.float32)).to(device)
        out.append(model(z).cpu().numpy())
    return np.concatenate(out)


def main():
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_all = np.load(HERE / "corpus" / "latents.npy")
    meta = pd.read_csv(HERE / "corpus" / "latents_meta.csv").to_dict("records")
    man = pd.read_csv(HERE / "corpus" / "manifest.csv")
    c1map = dict(zip(man["path"].astype(str), man["c1"].astype(float)))
    keep, rows = [], []
    for i, r in enumerate(meta):
        c1 = c1map.get(str(r["path"]))
        if c1 is None or not (-1.5 <= float(r["c2"]) <= 0.2 and 0.5 <= c1 <= 2.5):
            continue
        r["c1"] = float(c1); keep.append(i); rows.append(r)
    z_all = z_all[keep]
    idx = {id(r): i for i, r in enumerate(rows)}
    _, _, te = grouped_split(rows, seed=0)
    zte = z_all[[idx[id(r)] for r in te]]
    y = np.array([[r["c1"], r["c2"]] for r in te])
    types = np.array([famtype(r) for r in te])
    diff = np.array([str(r["group"]).startswith("diffusion") for r in te])
    print(f"test n={len(te)}  diffusion-source n={int(diff.sum())}  "
          f"(diverse={int((types=='diverse').sum())}, natural={int((types=='natural').sum())})\n")

    for name, path in [("BEFORE (no diverse data)", HERE / "joint_model_before.pt"),
                       ("AFTER  (+653 diverse)", HERE / "joint_model.pt")]:
        m = load_model(path, device)
        p = predict(m, zte, device)
        print(name)
        for d, lbl in [(0, "c1"), (1, "c2")]:
            # only diffusion-source test rows, split by content type
            for t in ("natural", "diverse"):
                sel = diff & (types == t)
                if sel.sum() >= 3:
                    s = regression_stats(y[sel, d], p[sel, d])
                    print(f"    {lbl} {t:8s} n={s['n']:3d}  RMSE={s['rmse']:.4f}  r={s['pearson_r']:.3f}")
        print()


if __name__ == "__main__":
    main()
