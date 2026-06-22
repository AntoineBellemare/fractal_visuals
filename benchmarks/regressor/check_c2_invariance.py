"""
check_c2_invariance.py  --  empirically prove which augmentations preserve c2.

The whole reason Stage 1 changed its augmentation set: c2 is a multi-scale scaling
slope, so geometric/photometric ops that look harmless are NOT all label-preserving.
This script measures the c2 shift each candidate augmentation induces, averaged over
a sample of corpus images, and prints a verdict.

Expected result (this is the design justification):
  KEEP  : hflip, vflip, rot90/180/270                          |dc2| ~ 0.01
  DROP  : random_crop, downscale/resize, gaussian_noise,       |dc2| large
          AND brightness/contrast -- their [0,1] clip interacts with the near-zero
          mass of heavy-tailed fields and shifts c2 by >2.0. Photometric augmentation
          is only safe on already-spread 8-bit photos, not on faithful HDR fields,
          so train_stage1.py uses GEOMETRIC augmentation only.

Run AFTER build_corpus.py:
  python check_c2_invariance.py --n 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import ROOT, CANON_RES, to_canonical_gray, load_field, measure_c2_field  # noqa: E402


def mf_c2(field):
    """c2 measured faithfully on a float field (same convention as the labeler)."""
    return measure_c2_field(field)[1]


# --- candidate augmentations, each maps a [0,1] field -> [0,1] field --------- #
def hflip(g):      return g[:, ::-1].copy()
def vflip(g):      return g[::-1, :].copy()
def rot90(g):      return np.rot90(g, 1).copy()
def rot180(g):     return np.rot90(g, 2).copy()
def rot270(g):     return np.rot90(g, 3).copy()
def brightness(g): return np.clip(g + 0.08, 0, 1)
def contrast(g):   return np.clip((g - g.mean()) * 1.1 + g.mean(), 0, 1)

# the ones we expect to BREAK c2 (kept here only to demonstrate why they are out):
def random_crop(g):
    n = g.shape[0]; c = int(n * 0.6)
    rng = np.random.default_rng(0)
    y, x = rng.integers(0, n - c, size=2)
    crop = g[y:y + c, x:x + c]
    return to_canonical_gray(crop)                 # resized back to CANON_RES
def downscale(g):
    img = Image.fromarray((g * 255).astype(np.uint8), "L").resize((224, 224), Image.BICUBIC)
    return to_canonical_gray(img)
def gaussian_noise(g):
    rng = np.random.default_rng(0)
    return np.clip(g + rng.normal(0, 0.05, g.shape), 0, 1)

KEEP = [("hflip", hflip), ("vflip", vflip), ("rot90", rot90), ("rot180", rot180),
        ("rot270", rot270)]
DROP = [("brightness+0.08", brightness), ("contrast*1.1", contrast),
        ("random_crop_0.6", random_crop), ("downscale_224", downscale),
        ("gaussian_noise_0.05", gaussian_noise)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of corpus images to sample")
    ap.add_argument("--manifest",
                    default=str(Path(__file__).resolve().parent / "corpus" / "manifest.csv"))
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.manifest)
    df = df.sample(min(args.n, len(df)), random_state=0)

    fields = []
    for _, r in df.iterrows():
        p = Path(r["path"])
        p = p if p.is_absolute() else ROOT / p
        fields.append(load_field(p))            # faithful field (.npy) or 8-bit photo
    base_c2 = np.array([mf_c2(g) for g in fields])

    print(f"sampled {len(fields)} images, base c2 mean {base_c2.mean():.3f} "
          f"std {base_c2.std():.3f}\n")
    print(f"{'augmentation':22s} {'mean|dc2|':>10s} {'max|dc2|':>10s}   verdict")
    print("-" * 60)
    for name, fn in KEEP + DROP:
        d = np.array([abs(mf_c2(fn(g)) - b) for g, b in zip(fields, base_c2)])
        # "invariant" if the shift is small vs the label spread we care about (RMSE<0.10)
        verdict = "KEEP (invariant)" if d.mean() < 0.02 else "DROP (shifts c2)"
        print(f"{name:22s} {d.mean():10.4f} {d.max():10.4f}   {verdict}")

    print("\nRule of thumb: anything with mean|dc2| >> 0.02 injects label noise into a "
          "target we want to predict to RMSE 0.10. Only KEEP augs go in train_stage1.py.")


if __name__ == "__main__":
    main()
