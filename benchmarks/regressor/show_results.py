"""Assemble a gallery of example textures + a results plot from the reachable-atlas run."""
import sys, csv, itertools
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "results" / "reachable_atlas.csv"
IMG = HERE / "reachable" / "images"
OUT = HERE / "showcase"; OUT.mkdir(exist_ok=True)
COL = {"rock":"#8c564b","cloud":"#9467bd","fluid":"#0aa6c2","fire":"#d62728",
       "ice":"#17becf","water":"#1f77b4", None:"#555"}

rows = list(csv.DictReader(CSV.open()))
for r in rows:
    for k in ("c1_target","c2_target","c1_meas","c2_meas"):
        r[k] = float(r[k])
    r["path"] = IMG / f"{r['family']}_c1{r['c1_target']:+.2f}_c2{r['c2_target']:+.2f}.png"

# ---------- gallery: rows = family, cols = 3 spread-in-c2 examples ----------
fams = [f for f in dict.fromkeys(r["family"] for r in rows)]
gal = []
for fam in fams:
    fr = sorted([r for r in rows if r["family"]==fam and r["path"].exists()],
                key=lambda r: r["c2_meas"])
    if len(fr) >= 3:
        gal.append((fam, [fr[0], fr[len(fr)//2], fr[-1]]))   # strong, mid, weak c2
gal = gal[:6]
if gal:
    nr, nc = len(gal), 3
    fig, ax = plt.subplots(nr, nc, figsize=(nc*2.5, nr*2.7))
    ax = np.atleast_2d(ax)
    for i,(fam,three) in enumerate(gal):
        for j,r in enumerate(three):
            a = ax[i][j]; a.imshow(Image.open(r["path"]).resize((300,300)))
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"c1={r['c1_meas']:.2f}  c2={r['c2_meas']:+.2f}", fontsize=8, pad=2)
            if j==0: a.set_ylabel(f"{r['family']}\n({r['domain']})", fontsize=9, rotation=0, ha="right", va="center")
        ax[i][0].yaxis.set_label_coords(-0.04,0.5)
    fig.suptitle("Example textures generated at controlled multifractality\n"
                 "(each row: same prompt+seed, c2 strong→weak left→right; tile = MEASURED c1,c2)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0.04,0,1,0.96]); fig.savefig(OUT/"gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("gallery ->", OUT/"gallery.png")

# ---------- results plot: achievable (c1,c2) region per domain ----------
from scipy.spatial import ConvexHull
fig, ax = plt.subplots(figsize=(7.5,6.2))
for dom, grp in itertools.groupby(sorted(rows, key=lambda r: str(r["domain"])), lambda r: r["domain"]):
    P = np.array([[r["c1_meas"], r["c2_meas"]] for r in grp])
    c = COL.get(dom, "#555")
    ax.scatter(P[:,0], P[:,1], s=45, color=c, edgecolor="k", lw=0.4, zorder=3, label=str(dom))
    if len(P) >= 3:
        try:
            h = ConvexHull(P); poly = P[h.vertices]
            ax.fill(poly[:,0], poly[:,1], color=c, alpha=0.13, zorder=2)
        except Exception: pass
ax.set_xlabel("c1  (roughness / detail — lower = busier)")
ax.set_ylabel("c2  (intermittency — lower = more clustered)")
ax.set_title("Results: achievable (c1, c2) region per texture domain\n(measured cumulants of generated outputs)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"results_region.png", dpi=160, bbox_inches="tight")
plt.close(fig); print("plot ->", OUT/"results_region.png")
