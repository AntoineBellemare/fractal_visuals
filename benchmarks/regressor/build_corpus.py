"""
build_corpus.py  --  Stage 1/2 unified labeled-corpus builder for the c2 regressor.

Assembles (image, c2) pairs from up to five sources into a single manifest, with
ONE consistent labeling convention so the regression target is clean:

  * c2 is measured with mfractal.wavelet_leaders_2d on the GRAYSCALE [0,1] field
    resized to CANON_RES (=512) -- the same resolution the diffusion sweep and the
    descriptor sweep used. Mixed-resolution labels (the repo previously had 256 and
    512 floating around) are re-measured here so every label is comparable.

  * For generated sources the c2 is measured on the EXACT array that is saved to
    disk (no `punch` contrast-stretch in between -- punch is a pointwise
    nonlinearity that shifts c2, so reusing build_dataset's PNGs would inject label
    noise). What you train on is what was measured.

Sources
-------
  procedural   mf.generate(family, complexity, seed)        [CPU, free]
  prescribed   mf.prescribed_cascade(c2_target=..., seed)   [CPU, free, c2-uniform coverage]
  diffusion    existing SDXL PNGs in benchmarks/diffusion/out  (re-measured at 512)
  pareidolia   datasets/pareidolia/images/*.png             (re-measured at 512)
  macro        a user-provided folder of real macro-texture photos

Output
------
  benchmarks/regressor/corpus/manifest.csv   columns: path, c1, c2, source, group
  benchmarks/regressor/corpus/images/<source>/*.png   (generated sources only)

`group` is the leakage-safe split unit (see train_stage1.py): generated images are
grouped by family / c2-target band, diffusion by prompt, so the same prompt's seeds
never straddle the train/test boundary.

Usage
-----
  # everything that is free + already on disk:
  python build_corpus.py --procedural --prescribed --diffusion --pareidolia

  # add your real photos (recursively globbed; group = parent folder name):
  python build_corpus.py --macro /path/to/macro_textures

  # scale the free coverage layer up (it is CPU and basically free):
  python build_corpus.py --prescribed --prescribed-grid 60 --prescribed-seeds 16
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mfractal as mf

# --------------------------------------------------------------------------- #
# Canonical conventions -- shared with train_stage1.py and check_c2_invariance.py
# --------------------------------------------------------------------------- #
CANON_RES = 512          # resolution at which c2 LABELS are measured (grayscale)
C2_SANITY = 3.0          # |c2| above this == sparse-field estimator artifact -> drop

OUT_DIR = Path(__file__).resolve().parent / "corpus"
IMG_DIR = OUT_DIR / "images"

# Procedural families to sweep. Curated union of the reliable domains; we skip
# volcanic_fissure (the REPORT flags it as a sparse-field c2 artifact in procedural
# form) -- it is dropped automatically by the C2_SANITY guard anyway, but naming it
# here documents the intent.
#
# FLUID_FAMILIES are intentionally EXCLUDED: the fluid generators are iterative sims
# that are pathologically slow / can hang at n=512 (eddies/vorticity/rheoscopic).
# The fluid look is already covered by the pareidolia set (choppy/eddies/dye_diffusion/
# vorticity/curl_weave/rheoscopic) and the macro WATER photos, so we lose nothing.
PROC_SKIP = {"volcanic_fissure"}


def procedural_families() -> list[str]:
    fams: list[str] = []
    for const in ("ROCK_FAMILIES", "CLOUD_FAMILIES", "BARK_FAMILIES", "FIRE_FAMILIES"):
        fams += list(getattr(mf, const))
    # de-dup, preserve order, drop the known sparse-field offender
    seen, out = set(), []
    for f in fams:
        if f in seen or f in PROC_SKIP:
            continue
        seen.add(f)
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Labeling helpers -- ONE place that defines "the c2 of an image"
# --------------------------------------------------------------------------- #
def to_canonical_gray(arr_or_img, res: int = CANON_RES) -> np.ndarray:
    """Return a float32 grayscale [0,1] field at `res` x `res`.

    Accepts either a float field (HxW or HxWxC in [0,1]) or a PIL image.
    Resize is whole-image bicubic -- NOT a crop -- so it is a fixed deterministic
    function of the input (no random label noise).
    """
    if isinstance(arr_or_img, np.ndarray):
        a = arr_or_img
        if a.ndim == 3:
            a = a.mean(axis=2)
        img = Image.fromarray(np.clip(a, 0, 1).astype(np.float32) * 255.0).convert("L")
    else:
        img = arr_or_img.convert("L")
    if img.width != res or img.height != res:
        img = img.resize((res, res), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def measure_c2(arr_or_img) -> tuple[float, float]:
    """(c1, c2) for a PHOTO (8-bit display-referred image), measured at CANON_RES.
    Correct for diffusion/pareidolia/macro: those genuinely ARE 8-bit images."""
    g = to_canonical_gray(arr_or_img)
    r = mf.wavelet_leaders_2d(g)
    return float(r["c1"]), float(r["c2"])


def measure_c2_field(field: np.ndarray) -> tuple[float, float]:
    """(c1, c2) for a GENERATED float field, measured on the faithful raw values.

    Why not reuse measure_c2: heavy-tailed procedural/cascade fields keep most of
    their mass near 0. Quantizing them to 8-bit (or `punch`-stretching them)
    destroys the multifractal structure -- empirically c2 jumps from -1.0 to -50
    (8-bit) or flattens to -0.4 (punch). So generated fields are measured AND stored
    as float; only true 8-bit images go through measure_c2."""
    f = np.asarray(field, float)
    if f.ndim == 3:
        f = f.mean(axis=2)
    if f.shape[0] != CANON_RES:                # generated at n=CANON_RES, so rare
        f = to_canonical_gray(f)               # float bilinear-ish via PIL; ok
    r = mf.wavelet_leaders_2d(f)
    return float(r["c1"]), float(r["c2"])


def load_field(path) -> np.ndarray:
    """Unified reader used by training + the invariance check. Returns a float32
    grayscale field at CANON_RES. .npy -> faithful generated field; image -> 8-bit
    grayscale /255. Keeps the train-time view identical to the labeled view."""
    p = Path(path)
    if p.suffix == ".npy":
        f = np.load(p).astype(np.float32)
        if f.ndim == 3:
            f = f.mean(axis=2)
        if f.shape[0] != CANON_RES or f.shape[1] != CANON_RES:
            f = to_canonical_gray(f)
        return f
    return to_canonical_gray(Image.open(p))


def _save_field_npy(field01: np.ndarray, path: Path) -> None:
    """Store the faithful float field (float16 to save space; the 4th decimal of a
    [0,1] field is below c2's sensitivity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = np.asarray(field01, np.float32)
    if f.ndim == 3:
        f = f.mean(axis=2)
    np.save(path, f.astype(np.float16))


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------- #
# Source builders -- each yields manifest rows {path, c1, c2, source, group}
# --------------------------------------------------------------------------- #
def build_procedural(complexities, seeds, verbose=True, families=None,
                     source="procedural") -> list[dict]:
    rows = []
    fams = procedural_families() if families is None else list(families)
    t0 = time.time()
    failed = 0
    for fam in fams:
        kept = 0
        for c in complexities:
            for s in seeds:
                try:
                    try:
                        raw = mf.generate(fam, n=CANON_RES, complexity=float(c), seed=int(s))
                    except TypeError:
                        raw = mf.generate(fam, n=CANON_RES, seed=int(s))
                    c1, c2 = measure_c2_field(raw)
                except Exception as e:               # noqa: BLE001  (don't kill a long run)
                    failed += 1
                    if failed <= 5:
                        print(f"  ! {fam} c={c} s={s}: {type(e).__name__}: {e}")
                    continue
                if not np.isfinite(c2) or abs(c2) > C2_SANITY:
                    continue
                fname = IMG_DIR / source / f"{fam}_c{float(c):.2f}_s{int(s)}.npy"
                _save_field_npy(raw, fname)
                rows.append(dict(path=_rel(fname), c1=round(c1, 5), c2=round(c2, 5),
                                 source=source, group=f"{source}:{fam}"))
                kept += 1
        if verbose:
            print(f"  {source} {fam:24s} +{kept}")
    if failed and verbose:
        print(f"  ({failed} {source} samples skipped due to generator errors)")
    if verbose:
        print(f"procedural: {len(rows)} rows in {time.time()-t0:.0f}s")
    return rows


def build_prescribed(grid, seeds, c2_lo=-1.2, c2_hi=-0.02, verbose=True) -> list[dict]:
    """Free c2-uniform coverage layer. We sweep c2_TARGET uniformly but LABEL each
    image with its MEASURED c2 (the calibration is approximate), so the labels stay
    honest while the *coverage* is forced to span the range."""
    rows = []
    targets = np.linspace(c2_lo, c2_hi, int(grid))
    t0 = time.time()
    for ti, c2t in enumerate(targets):
        band = f"{c2t:+.2f}"
        for s in range(int(seeds)):
            raw = mf.prescribed_cascade(n=CANON_RES, seed=int(s), c2_target=float(c2t))
            c1, c2 = measure_c2_field(raw)
            if not np.isfinite(c2) or abs(c2) > C2_SANITY:
                continue
            fname = IMG_DIR / "prescribed" / f"t{band}_s{s}.npy"
            _save_field_npy(raw, fname)
            rows.append(dict(path=_rel(fname), c1=round(c1, 5), c2=round(c2, 5),
                             source="prescribed", group=f"prescribed:{band}"))
    if verbose:
        meas = [r["c2"] for r in rows]
        print(f"prescribed: {len(rows)} rows in {time.time()-t0:.0f}s, "
              f"measured c2 in [{min(meas):.2f}, {max(meas):.2f}]")
    return rows


def index_diffusion(remeasure=True, verbose=True) -> list[dict]:
    """Reference the existing SDXL PNGs. Re-measure at CANON_RES for consistency
    (analysis.csv was already 512, but we re-measure to be uniform with the rest)."""
    out = ROOT / "benchmarks" / "diffusion" / "out"
    import pandas as pd
    df = pd.read_csv(out / "analysis.csv")
    rows = []
    t0 = time.time()
    for _, r in df.iterrows():
        p = out / r["path"]
        if not p.exists():
            continue
        if remeasure:
            c1, c2 = measure_c2(Image.open(p))
        else:
            c1, c2 = float(r["c1"]), float(r["c2"])
        if not np.isfinite(c2) or abs(c2) > C2_SANITY:
            continue
        # group by PROMPT (family + prompt_idx) -- the leakage unit across seeds
        grp = f"diffusion:{r['family']}:{int(r['prompt_idx'])}"
        rows.append(dict(path=_rel(p), c1=round(c1, 5), c2=round(c2, 5),
                         source="diffusion", group=grp))
    if verbose:
        print(f"diffusion: {len(rows)} rows in {time.time()-t0:.0f}s")
    return rows


def index_pareidolia(remeasure=True, verbose=True) -> list[dict]:
    base = ROOT / "datasets" / "pareidolia"
    import pandas as pd
    df = pd.read_csv(base / "manifest.csv")
    rows = []
    for _, r in df.iterrows():
        p = base / r["image_path"]
        if not p.exists():
            continue
        if remeasure:
            c1, c2 = measure_c2(Image.open(p))
        else:
            c1, c2 = float(r["c1"]), float(r["c2"])
        if not np.isfinite(c2) or abs(c2) > C2_SANITY:
            continue
        rows.append(dict(path=_rel(p), c1=round(c1, 5), c2=round(c2, 5),
                         source="pareidolia", group=f"pareidolia:{r['family']}"))
    if verbose:
        print(f"pareidolia: {len(rows)} rows")
    return rows


_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def index_macro(folder: Path, verbose=True) -> list[dict]:
    """Label a user-provided folder of real macro-texture photos. Group = the
    photo's parent-folder name (so a 'marble/' subfolder stays together in a split).

    De-dups by file-content hash: the same photo copied into two subfolders (common
    when folders are curated by hand) would otherwise leak across the train/test
    boundary as two different groups. First occurrence (sorted) wins."""
    import hashlib
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in _EXTS)
    rows, seen, dups = [], set(), 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in seen:
                dups += 1
                continue
            seen.add(digest)
            c1, c2 = measure_c2(Image.open(p))
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        if not np.isfinite(c2) or abs(c2) > C2_SANITY:
            continue
        grp = f"macro:{p.parent.name}"
        rows.append(dict(path=p.resolve().as_posix(), c1=round(c1, 5), c2=round(c2, 5),
                         source="macro", group=grp))
        if verbose and i % 50 == 0:
            print(f"  macro {i}/{len(files)} ...")
    if verbose:
        print(f"macro [{folder}]: {len(rows)} rows ({dups} dup skipped) "
              f"in {time.time()-t0:.0f}s")
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procedural", action="store_true")
    ap.add_argument("--prescribed", action="store_true")
    ap.add_argument("--diffusion", action="store_true")
    ap.add_argument("--pareidolia", action="store_true")
    ap.add_argument("--metal", action="store_true",
                    help="add the merged metal/corrosion families as a held-out OOD source")
    ap.add_argument("--macro", action="append", default=[],
                    help="folder of real macro-texture photos (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="procedural + prescribed + diffusion + pareidolia")
    # procedural knobs
    ap.add_argument("--proc-complexities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--proc-seeds", type=int, default=4)
    # prescribed coverage knobs (free -- scale up freely)
    ap.add_argument("--prescribed-grid", type=int, default=40,
                    help="number of c2-target steps across the range")
    ap.add_argument("--prescribed-seeds", type=int, default=12)
    ap.add_argument("--no-remeasure", action="store_true",
                    help="trust existing CSV c2 for diffusion/pareidolia (faster)")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing manifest instead of overwriting")
    args = ap.parse_args()

    if args.all:
        args.procedural = args.prescribed = args.diffusion = args.pareidolia = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / "manifest.csv"
    all_rows: list[dict] = []
    if args.append and manifest.exists():
        import pandas as pd
        all_rows = pd.read_csv(manifest).to_dict("records")

    def flush():
        with manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "c1", "c2", "source", "group"])
            w.writeheader()
            w.writerows(all_rows)

    # Write the manifest after EACH source so a hang/kill in a later source can't
    # discard the work already done (procedural is the slow one).
    if args.procedural:
        comps = [float(x) for x in args.proc_complexities.split(",")]
        all_rows += build_procedural(comps, range(args.proc_seeds)); flush()
    if args.metal:
        comps = [float(x) for x in args.proc_complexities.split(",")]
        all_rows += build_procedural(comps, range(args.proc_seeds),
                                     families=list(mf.METAL_FAMILIES), source="metal")
        flush()
    if args.prescribed:
        all_rows += build_prescribed(args.prescribed_grid, args.prescribed_seeds); flush()
    if args.diffusion:
        all_rows += index_diffusion(remeasure=not args.no_remeasure); flush()
    if args.pareidolia:
        all_rows += index_pareidolia(remeasure=not args.no_remeasure); flush()
    for m in args.macro:
        all_rows += index_macro(m); flush()

    if not all_rows:
        ap.error("no sources selected; pass --all and/or --macro DIR")
    flush()

    # tiny summary
    from collections import Counter
    by_src = Counter(r["source"] for r in all_rows)
    c2s = np.array([r["c2"] for r in all_rows], float)
    print(f"\nmanifest: {manifest}  ({len(all_rows)} rows)")
    for s, n in sorted(by_src.items()):
        print(f"  {s:12s} {n}")
    print(f"  c2 range [{c2s.min():.2f}, {c2s.max():.2f}]  "
          f"mean {c2s.mean():.3f}  std {c2s.std():.3f}")
    # coverage histogram across the deployment range
    edges = np.linspace(-1.0, 0.0, 11)
    hist, _ = np.histogram(c2s, bins=edges)
    print("  c2 coverage [-1..0]:", " ".join(f"{h:>4d}" for h in hist))


if __name__ == "__main__":
    main()
