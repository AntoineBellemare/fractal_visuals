"""Did the spatial multifractal gradient take? Measure c2 LOCALLY at the calm end vs the
turbulent end of each gradient image; the turbulent end should read MORE NEGATIVE c2."""
import sys, glob, re
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import mfractal as mf

# (calm box, turbulent box) as (x0,y0,x1,y1) fractions, per gradient direction
BOX = {
    "h": ((0, 0, .33, 1), (.67, 0, 1, 1)),
    "v": ((0, 0, 1, .33), (0, .67, 1, 1)),
    "d": ((0, 0, .42, .42), (.58, .58, 1, 1)),
    "radial": ((.36, .36, .64, .64), (0, 0, .30, .30)),
}


def patch_c2(a, box):
    h, w = a.shape; x0, y0, x1, y1 = box
    sub = a[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
    s = np.asarray(Image.fromarray((sub*255).astype(np.uint8)).resize((512, 512)), float)/255
    return float(mf.wavelet_leaders_2d(s)["c2"])


paths = sorted(glob.glob(str(HERE/"gradient_pass"/"img"/"*.png")))
rows = []
for p in paths:
    m = re.search(r"_(h|v|d|radial)_cn([\d.]+)_", Path(p).name)
    if not m:
        continue
    d, cn = m.group(1), float(m.group(2))
    a = np.asarray(Image.open(p).convert("L"), float)/255
    cb, tb = BOX[d]
    cc, tc = patch_c2(a, cb), patch_c2(a, tb)
    rows.append(dict(dir=d, cn=cn, c2_calm=cc, c2_turb=tc, delta=tc-cc))

if not rows:
    print("no images yet"); sys.exit()
delta = np.array([r["delta"] for r in rows])
print(f"images analysed: {len(rows)}")
print(f"mean c2 calm-end:      {np.mean([r['c2_calm'] for r in rows]):+.3f}")
print(f"mean c2 turbulent-end: {np.mean([r['c2_turb'] for r in rows]):+.3f}")
print(f"mean delta (turb-calm): {delta.mean():+.3f}   (negative = turbulent more intermittent = CONTROL)")
print(f"correct direction (delta<0): {int((delta<0).sum())}/{len(delta)} = {100*(delta<0).mean():.0f}%")
print("\nby controlnet_conditioning_scale (stronger cn should give bigger negative delta):")
for cn in sorted(set(r["cn"] for r in rows)):
    dd = np.array([r["delta"] for r in rows if r["cn"] == cn])
    print(f"  cn={cn:.1f}: mean delta {dd.mean():+.3f}  correct {100*(dd<0).mean():.0f}%  (n={len(dd)})")
print("\nby direction:")
for d in BOX:
    dd = np.array([r["delta"] for r in rows if r["dir"] == d])
    if len(dd):
        print(f"  {d:6s}: mean delta {dd.mean():+.3f}  correct {100*(dd<0).mean():.0f}%  (n={len(dd)})")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    cc = np.array([r["c2_calm"] for r in rows]); tc = np.array([r["c2_turb"] for r in rows])
    ax.scatter(cc, tc, c=[{ "h":"#1f77b4","v":"#2ca02c","d":"#d62728","radial":"#9467bd"}[r["dir"]] for r in rows],
               s=22, alpha=0.7, edgecolor="k", lw=0.3)
    lo, hi = min(cc.min(), tc.min())-0.05, max(cc.max(), tc.max())+0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x (no gradient)")
    ax.set_xlabel("c2 at CALM end"); ax.set_ylabel("c2 at TURBULENT end")
    ax.set_title(f"Spatial c2 control: turbulent end below the line = stronger c2\n"
                 f"mean delta {delta.mean():+.3f}, correct {100*(delta<0).mean():.0f}%")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(HERE/"gradient_pass"/"c2_control_check.png", dpi=150)
    print("\nplot -> gradient_pass/c2_control_check.png")
except Exception as e:
    print("plot skipped:", e)
