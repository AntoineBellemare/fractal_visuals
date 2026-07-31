"""
Per-substrate feasibility table: what part of the (c1, c2) plane each texture
substrate can actually reach, how well it responds to control, and which way an
FD ramp should run on it.

WHY
---
"Substrate feasibility" is a recurring finding in this repo: every texture family
occupies only part of the (c1, c2) plane, and c2 (intermittency) is controllable on
some substrates and essentially inert on others. That knowledge was scattered across
eight CSVs and several prose docs. This script consolidates it into one machine-readable
table plus a readable reference doc.

INPUTS (all already measured, all ground-truth `wavelet_leaders_2d` estimates)
------
  benchmarks/results/natural_atlas.csv          60 families x 9 scale bands x 5 (c1,c2) targets
  benchmarks/results/creature_atlas.csv         16 creature/face/emergent families x 5 targets
  benchmarks/results/reachable_atlas.csv         6 families x 9 targets (3x3 grid)
  benchmarks/results/c1_complexity_control.csv   4 families x 4 c1 targets (c2 fixed)
  benchmarks/results/wider_c1_control.csv        8 families x 3 c1 targets (c2 fixed)
  benchmarks/results/ood_c1_control.csv          8 out-of-distribution families x 3 c1 targets
  benchmarks/results/c2_control_montage.csv      6 families x 5 c2 targets (c1 unrecorded)
  benchmarks/results/fd_gradient_gallery.csv    18 subjects, per-region c1/c2 for an FD ramp
  benchmarks/results/mf_gradient_intermittent.csv  12 subjects, same for a c2 ramp

METHOD
------
1. Every source is normalised to one long observation frame:
       substrate, domain, source, c1_target, c2_target, c1_meas, c2_meas
   Substrate names are canonicalised through ALIASES (printed in the output doc so
   the merge is auditable); domains come from the source's own scale/domain column
   where it has one, else from DOMAIN_OF.

2. Reachable range = raw min/max of the measured values over every observation of
   that substrate. FD = 3 - c1, so fd_min = 3 - c1_max and fd_max = 3 - c1_min.

3. Control slope = WITHIN-SOURCE (fixed-effects) OLS of measured on requested.
   Observations are centred inside each (substrate, source) group before pooling, so
   a constant offset between two generation runs cannot masquerade as a slope. A group
   only contributes if it holds >= 2 distinct target values; a substrate needs >= 3
   usable points for a slope to be reported at all (else UNTESTED).
   slope = sum(xy)/sum(xx) on the centred pairs; r = the within-group Pearson r.
   Slope 1.0 = perfect obedience, 0.0 = the substrate ignores the request.
   NOTE: the natural/creature atlas design is exactly orthogonal in (c1_target,
   c2_target) (cov = 0 over its 5 target cells), so the two univariate slopes are
   unconfounded by construction.

4. VERDICT THRESHOLDS (chosen here, stated explicitly so they can be argued with).
   The ground-truth wavelet-leader estimate of c2 on a single 512px image is
   repeatable to roughly +-0.05-0.08. The requested c2 swing in these sweeps is
   0.45-0.80 wide. So:
       FEASIBLE    slope >= 0.35 and r >= 0.70
                   -> delivers >= 0.16 of measured c2 swing, ~2-3x measurement noise,
                      and does it monotonically. Usable as an experimental factor.
       WEAK        slope >= 0.15 and r >= 0.40 (and not FEASIBLE)
                   -> direction is real but the achieved swing is comparable to noise;
                      only usable with averaging over seeds.
       INFEASIBLE  anything else -> the substrate ignores the c2 request.
       UNTESTED    fewer than 3 usable points.
   The same rule is applied to c1 for symmetry (almost everything is FEASIBLE in c1).

5. FD RAMP POLARITY (fd_gradient_gallery.csv, 18 subjects).
   `delta` there is the MEASURED c1 difference across the image in the requested
   direction. delta > 0 -> the substrate renders the ramp as asked ("direct").
   delta < 0 -> the substrate renders it inverted ("inverted"): dunes, coral, forest,
   lichen_rock, volcanic. A flip test showed flipping those REDUCES the gradient
   magnitude, so inversion is a genuine per-substrate rendering preference, not a bug,
   and is recorded as a property. Strength = |delta| (measured c1 span across the image):
       STRONG   |delta| >= 0.30
       MODERATE |delta| >= 0.10
       FLAT     |delta| <  0.10

OUTPUTS
-------
  benchmarks/results/feasibility_table.csv
  benchmarks/regressor/FEASIBILITY.md
  benchmarks/figures/56_feasibility_map.png

CPU only. No torch, no GPU, no generation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
FIG = ROOT / "benchmarks" / "figures"
DOC = ROOT / "benchmarks" / "regressor" / "FEASIBILITY.md"

# ---------------------------------------------------------------- thresholds
C2_FEASIBLE_SLOPE, C2_FEASIBLE_R = 0.35, 0.70
C2_WEAK_SLOPE, C2_WEAK_R = 0.15, 0.40
MIN_PTS = 3
RAMP_STRONG, RAMP_MODERATE = 0.30, 0.10

# ------------------------------------------------------------------ aliases
# Different sweeps used slightly different prompt names for the same substrate.
# Merging them is what makes a per-substrate row possible; the map is printed in
# FEASIBILITY.md so every merge can be audited / rejected.
ALIASES = {
    # fd_gradient_gallery subject -> atlas family
    "mountains": "mountain_range",
    "tundra": "tundra_polygons",
    "volcanic": "volcanic_terrain",
    "smoke": "smoke_plume",
    "frost": "frost_pattern",
    "forest": "forest_canopy",
    "autumn": "autumn_canopy",
    "lichen_rock": "lichen",
    "dunes": "sand_dunes",
    # mf_gradient_intermittent subject -> canonical
    "cirrus_wisp": "cirrus",
    "dendrite": "manganese_dendrite",
    "dye_plume": "dye_diffusion",
    "ink_water": "ink_bloom",
    "lichen_slate": "lichen",
    # control sweeps
    "mycelium_net": "mycelium",
}

# Domain for substrates whose source CSV carries no domain column.
DOMAIN_OF = {
    "ferrofluid": "fluid",
    "ink_bloom": "fluid",
    "dye_diffusion": "fluid",
    "firestorm": "fire",
    "frost_fern": "water_ice",
    "smoke_eddies": "atmosphere",
    "smoke_backlit": "atmosphere",
    "manganese_dendrite": "rock",
    "marble": "rock",
    "granite": "rock",
    "cirrus": "atmosphere",
    "coral": "marine",
    "jellyfish": "marine",
    "cracked_glaze": "ood",
    # ood_c1_control set (never trained on)
    "circuit_board": "ood",
    "crumpled_foil": "ood",
    "honeycomb": "ood",
    "knitted_wool": "ood",
    "peacock_feather": "ood",
    "snake_scales": "ood",
    "woven_fabric": "ood",
    "cracked_desert": "ood",
}

# reachable_atlas.csv domain column -> the domain vocabulary used here
REACHABLE_DOMAIN = {
    "cloud": "atmosphere",
    "ice": "water_ice",
    "water": "fluid",
    "fluid": "fluid",
    "rock": "rock",
    "fire": "fire",
}

DOMAIN_ORDER = [
    "atmosphere", "water_ice", "terrain", "rock", "sand", "plant", "tree",
    "microbial", "microscopy", "fluid", "fire", "marine",
    "creature", "emergent", "face", "ood",
]


def canon(name: str) -> str:
    return ALIASES.get(name, name)


# ------------------------------------------------------------------- loading
def load_observations() -> pd.DataFrame:
    """Normalise every measurement CSV into one long observation frame."""
    obs = []

    def add(df, source):
        df = df.copy()
        df["source"] = source
        obs.append(df[["substrate", "domain", "source",
                       "c1_target", "c2_target", "c1_meas", "c2_meas"]])

    # --- atlases (targets in both c1 and c2) --------------------------------
    for fn, src in [("natural_atlas.csv", "natural_atlas"),
                    ("creature_atlas.csv", "creature_atlas")]:
        d = pd.read_csv(RESULTS / fn)
        d["substrate"] = d["family"].map(canon)
        d["domain"] = d["scale"]
        add(d, src)

    d = pd.read_csv(RESULTS / "reachable_atlas.csv")
    d["substrate"] = d["family"].map(canon)
    d["domain"] = d["domain"].map(lambda x: REACHABLE_DOMAIN.get(x, x))
    add(d, "reachable_atlas")

    # --- c1 sweeps (c2 held constant -> contributes to the c1 slope only) ----
    for fn, src in [("c1_complexity_control.csv", "c1_control"),
                    ("wider_c1_control.csv", "wider_c1_control"),
                    ("ood_c1_control.csv", "ood_c1_control")]:
        d = pd.read_csv(RESULTS / fn)
        d["substrate"] = d["family"].map(canon)
        d["domain"] = np.nan
        add(d, src)

    # --- c2 sweep (c1 not recorded) -----------------------------------------
    d = pd.read_csv(RESULTS / "c2_control_montage.csv")
    d["substrate"] = d["family"].map(canon)
    d["domain"] = np.nan
    d["c2_target"] = d["target"]
    d["c2_meas"] = d["measured"]
    d["c1_target"] = np.nan
    d["c1_meas"] = np.nan
    add(d, "c2_control_montage")

    # --- gradient galleries: the two half-image measurements are real -------
    # (c1,c2) observations of that substrate, so they widen the reachable range,
    # but they carry no requested target -> they never enter a slope.
    for fn, src in [("fd_gradient_gallery.csv", "fd_gradient"),
                    ("mf_gradient_intermittent.csv", "mf_gradient")]:
        g = pd.read_csv(RESULTS / fn)
        g["substrate"] = g["subject"].map(canon)
        rows = []
        for _, r in g.iterrows():
            for end in ("A", "B"):
                rows.append(dict(substrate=r["substrate"], domain=np.nan,
                                 c1_target=np.nan, c2_target=np.nan,
                                 c1_meas=r[f"c1_{end}"], c2_meas=r[f"c2_{end}"]))
        add(pd.DataFrame(rows), src)

    o = pd.concat(obs, ignore_index=True)

    # fill missing domains
    known = (o.dropna(subset=["domain"])
              .groupby("substrate")["domain"].agg(lambda s: s.mode().iat[0]))
    o["domain"] = o["domain"].fillna(o["substrate"].map(known))
    o["domain"] = o["domain"].fillna(o["substrate"].map(DOMAIN_OF))
    o["domain"] = o["domain"].fillna("other")
    # one domain per substrate, globally
    dom = o.groupby("substrate")["domain"].agg(lambda s: s.mode().iat[0])
    o["domain"] = o["substrate"].map(dom)
    return o


# -------------------------------------------------------------------- slopes
def within_source_slope(g: pd.DataFrame, tcol: str, mcol: str):
    """Fixed-effects (within-source) OLS slope + Pearson r of measured on requested.

    Each (source) group is centred on its own mean before pooling, so between-run
    offsets cannot create or destroy a slope. Returns (slope, r, n_points).
    """
    xs, ys = [], []
    for _, sub in g.groupby("source"):
        s = sub[[tcol, mcol]].dropna()
        if len(s) < 2 or s[tcol].nunique() < 2:
            continue
        x = s[tcol].to_numpy(float)
        y = s[mcol].to_numpy(float)
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    if not xs:
        return np.nan, np.nan, 0
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    n = len(x)
    if n < MIN_PTS or np.allclose(x, 0):
        return np.nan, np.nan, n
    slope = float(x @ y / (x @ x))
    denom = np.sqrt((x @ x) * (y @ y))
    r = float(x @ y / denom) if denom > 0 else np.nan
    return slope, r, n


def verdict(slope, r, n):
    if not np.isfinite(slope) or n < MIN_PTS:
        return "UNTESTED"
    if slope >= C2_FEASIBLE_SLOPE and r >= C2_FEASIBLE_R:
        return "FEASIBLE"
    if slope >= C2_WEAK_SLOPE and r >= C2_WEAK_R:
        return "WEAK"
    return "INFEASIBLE"


def ramp_class(delta):
    if not np.isfinite(delta):
        return ""
    a = abs(delta)
    return "STRONG" if a >= RAMP_STRONG else ("MODERATE" if a >= RAMP_MODERATE else "FLAT")


# --------------------------------------------------------------------- build
def build_table(obs: pd.DataFrame) -> pd.DataFrame:
    fd = pd.read_csv(RESULTS / "fd_gradient_gallery.csv")
    fd["substrate"] = fd["subject"].map(canon)
    fd = fd.set_index("substrate")
    mfg = pd.read_csv(RESULTS / "mf_gradient_intermittent.csv")
    mfg["substrate"] = mfg["subject"].map(canon)
    mfg = mfg.set_index("substrate")

    rows = []
    for sub, g in obs.groupby("substrate"):
        c1 = g["c1_meas"].dropna()
        c2 = g["c2_meas"].dropna()
        s1, r1, n1 = within_source_slope(g, "c1_target", "c1_meas")
        s2, r2, n2 = within_source_slope(g, "c2_target", "c2_meas")
        row = dict(
            substrate=sub,
            domain=g["domain"].iat[0],
            n_images=int(len(g)),
            n_c1_pts=n1,
            n_c2_pts=n2,
            sources="+".join(sorted(g["source"].unique())),
            c1_min=c1.min() if len(c1) else np.nan,
            c1_max=c1.max() if len(c1) else np.nan,
            c1_span=(c1.max() - c1.min()) if len(c1) else np.nan,
            fd_min=(3 - c1.max()) if len(c1) else np.nan,
            fd_max=(3 - c1.min()) if len(c1) else np.nan,
            c2_min=c2.min() if len(c2) else np.nan,
            c2_max=c2.max() if len(c2) else np.nan,
            c2_span=(c2.max() - c2.min()) if len(c2) else np.nan,
            c1_slope=s1, c1_r=r1, c1_verdict=verdict(s1, r1, n1),
            c2_slope=s2, c2_r=r2, c2_verdict=verdict(s2, r2, n2),
        )
        if sub in fd.index:
            f = fd.loc[sub]
            row.update(
                fd_ramp_axis=f["direction"],
                fd_ramp_delta=f["delta"],
                fd_ramp_strength=abs(f["delta"]),
                fd_ramp_polarity="inverted" if f["delta"] < 0 else "direct",
                fd_ramp_class=ramp_class(f["delta"]),
                fd_ramp_c1_A=f["c1_A"], fd_ramp_c1_B=f["c1_B"],
            )
        else:
            row.update(fd_ramp_axis="", fd_ramp_delta=np.nan, fd_ramp_strength=np.nan,
                       fd_ramp_polarity="", fd_ramp_class="",
                       fd_ramp_c1_A=np.nan, fd_ramp_c1_B=np.nan)
        if sub in mfg.index:
            m = mfg.loc[sub]
            row.update(mf_ramp_axis=m["direction"], mf_ramp_delta=m["delta"],
                       mf_ramp_strength=abs(m["delta"]))
        else:
            row.update(mf_ramp_axis="", mf_ramp_delta=np.nan, mf_ramp_strength=np.nan)
        rows.append(row)

    t = pd.DataFrame(rows)
    t["domain"] = pd.Categorical(t["domain"],
                                 categories=[d for d in DOMAIN_ORDER
                                             if d in set(t["domain"])] +
                                            sorted(set(t["domain"]) - set(DOMAIN_ORDER)),
                                 ordered=True)
    t = t.sort_values(["domain", "c2_slope"], ascending=[True, False],
                      na_position="last").reset_index(drop=True)
    return t


# -------------------------------------------------------------------- figure
def _hull_poly(p: np.ndarray):
    """Convex hull vertices of a point cloud; bounding box if degenerate."""
    if len(p) >= 3:
        try:
            return p[ConvexHull(p).vertices]
        except Exception:
            pass
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    pad = 0.025
    return np.array([[x0 - pad, y0 - pad], [x1 + pad, y0 - pad],
                     [x1 + pad, y1 + pad], [x0 - pad, y1 + pad]])


def make_figure(obs: pd.DataFrame, t: pd.DataFrame, path: Path):
    doms = [d for d in DOMAIN_ORDER if d in set(t["domain"].astype(str))]
    cmap = plt.get_cmap("tab20")
    colors = {d: cmap(i % 20) for i, d in enumerate(doms)}

    fig = plt.figure(figsize=(16.5, 15.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.19,
                             left=0.055, right=0.985, top=0.925, bottom=0.05)

    # ---- panel A: one small multiple per domain (overplotting one axes with
    #      16 hulls is unreadable mush, so each domain gets its own frame with
    #      the whole corpus shown behind it in grey for reference) ------------
    pts = obs.dropna(subset=["c1_meas", "c2_meas"])
    allp = pts[["c1_meas", "c2_meas"]].to_numpy(float)
    ncol = 4
    nrow = int(np.ceil(len(doms) / ncol))
    grid = outer[0].subgridspec(nrow, ncol, hspace=0.16, wspace=0.06)
    xlim = (allp[:, 0].min() - 0.06, allp[:, 0].max() + 0.06)
    ylim = (allp[:, 1].min() - 0.08, allp[:, 1].max() + 0.10)
    for i, d in enumerate(doms):
        ax = fig.add_subplot(grid[i // ncol, i % ncol])
        ax.scatter(allp[:, 0], allp[:, 1], s=3.5, color="0.78", alpha=0.55,
                   linewidths=0, zorder=1)
        p = pts[pts["domain"] == d][["c1_meas", "c2_meas"]].to_numpy(float)
        c = colors[d]
        poly = _hull_poly(p)
        ax.add_patch(Polygon(poly, closed=True, facecolor=c, alpha=0.22,
                             edgecolor=c, lw=1.7, zorder=2))
        ax.scatter(p[:, 0], p[:, 1], s=13, color=c, alpha=0.9, linewidths=0, zorder=3)
        sub = t[t["domain"].astype(str) == d]
        nf = int((sub["c2_verdict"] == "FEASIBLE").sum())
        ax.set_title(f"{d}   {len(sub)} sub · {len(p)} pts · c2 ok {nf}/{len(sub)}",
                     fontsize=9, loc="left", color="0.15", pad=3.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axhline(0, color="0.55", lw=0.7, ls=":")
        ax.grid(alpha=0.18, lw=0.5)
        ax.tick_params(labelsize=8)
        if i % ncol != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("c2", fontsize=9.5)
        if i // ncol != nrow - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("c1   (FD = 3 − c1)", fontsize=9.5)
    fig.text(0.055, 0.938,
             f"A · Reachable (c1, c2) region per domain — convex hull of every measured image "
             f"(n = {len(pts)}); grey = whole corpus",
             fontsize=12.5, fontweight="bold", ha="left")

    # ---- panel B: control quality ------------------------------------------
    bottom = outer[1].subgridspec(1, 2, wspace=0.20)
    axb = fig.add_subplot(bottom[0, 0])
    q = t.dropna(subset=["c1_slope", "c2_slope"])
    for d in doms:
        s = q[q["domain"].astype(str) == d]
        if len(s) == 0:
            continue
        axb.scatter(s["c1_slope"], s["c2_slope"], s=46, color=colors[d],
                    edgecolor="white", lw=0.7, label=d, zorder=3)
    axb.axhline(C2_FEASIBLE_SLOPE, color="seagreen", lw=1.3, ls="--")
    axb.axhline(C2_WEAK_SLOPE, color="darkorange", lw=1.3, ls="--")
    axb.text(0.012, C2_FEASIBLE_SLOPE + 0.012, f"c2 FEASIBLE  (slope ≥ {C2_FEASIBLE_SLOPE}, r ≥ {C2_FEASIBLE_R})",
             color="seagreen", fontsize=8.5, transform=axb.get_yaxis_transform())
    axb.text(0.012, C2_WEAK_SLOPE + 0.012, f"c2 WEAK  (slope ≥ {C2_WEAK_SLOPE})",
             color="darkorange", fontsize=8.5, transform=axb.get_yaxis_transform())
    top = q.nlargest(6, "c2_slope")
    bot = q.nsmallest(3, "c2_slope")
    wide = q.nlargest(2, "c1_slope")
    for k, (_, r) in enumerate(pd.concat([top, bot, wide]).drop_duplicates("substrate").iterrows()):
        dx, dy = (7, 5) if k % 2 == 0 else (7, -11)
        axb.annotate(r["substrate"], (r["c1_slope"], r["c2_slope"]),
                     textcoords="offset points", xytext=(dx, dy), fontsize=8, color="0.25")
    axb.set_xlabel("c1 control slope  (measured Δc1 per requested Δc1)", fontsize=10.5)
    axb.set_ylabel("c2 control slope", fontsize=10.5)
    axb.set_title("B · Control response per substrate — 1.0 would be perfect",
                  fontsize=12, loc="left", fontweight="bold")
    axb.grid(alpha=0.22, lw=0.6)
    axb.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)

    # ---- panel C: FD ramp polarity -----------------------------------------
    axc = fig.add_subplot(bottom[0, 1])
    r18 = t.dropna(subset=["fd_ramp_delta"]).sort_values("fd_ramp_delta")
    y = np.arange(len(r18))
    cols = ["#c2452d" if v < 0 else "#2d6ec2" for v in r18["fd_ramp_delta"]]
    axc.barh(y, r18["fd_ramp_delta"], color=cols, height=0.72)
    axc.set_yticks(y)
    axc.set_yticklabels(r18["substrate"], fontsize=8.5)
    axc.axvline(0, color="0.2", lw=1.0)
    for v in (-RAMP_STRONG, RAMP_STRONG):
        axc.axvline(v, color="0.55", lw=1.0, ls=":")
    axc.set_xlabel("measured Δc1 across the image  (fd_gradient_gallery)", fontsize=10.5)
    axc.set_title("C · FD-ramp polarity and strength — red = renders inverted",
                  fontsize=12, loc="left", fontweight="bold")
    axc.grid(axis="x", alpha=0.22, lw=0.6)
    axc.legend(handles=[Line2D([], [], color="#2d6ec2", lw=7, label="direct (as requested)"),
                        Line2D([], [], color="#c2452d", lw=7, label="inverted (prefers reversed)"),
                        Line2D([], [], color="0.55", ls=":", label=f"|Δ| = {RAMP_STRONG} (STRONG)")],
               fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle("Substrate feasibility map — where each texture family lives in the (c1, c2) plane, "
                 "how well it obeys control, and how it ramps",
                 fontsize=15, fontweight="bold", x=0.055, y=0.982, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- markdown
def fmt(v, nd=2):
    return "—" if (v is None or not np.isfinite(v)) else f"{v:.{nd}f}"


def rng(lo, hi, nd=2):
    """Range cell. Bracketed because these values are often negative."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "—"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def sl(slope, r):
    """slope · correlation cell."""
    if not np.isfinite(slope):
        return "—"
    return f"{slope:.2f} · {r:.2f}"


def md_table(t: pd.DataFrame) -> str:
    head = ("| substrate | domain | n | c1 range | FD range | c2 range | "
            "c1 slope·r | c2 slope·r | c2 verdict | FD ramp |")
    sep = "|" + "---|" * 10
    lines = [head, sep]
    for _, r in t.iterrows():
        ramp = ""
        if np.isfinite(r["fd_ramp_delta"]):
            ramp = (f"{r['fd_ramp_polarity']} {r['fd_ramp_strength']:.2f} "
                    f"({r['fd_ramp_class']}, {r['fd_ramp_axis']})")
        lines.append(
            f"| `{r['substrate']}` | {r['domain']} | {r['n_images']} | "
            f"{rng(r['c1_min'], r['c1_max'])} | {rng(r['fd_min'], r['fd_max'])} | "
            f"{rng(r['c2_min'], r['c2_max'])} | "
            f"{sl(r['c1_slope'], r['c1_r'])} | {sl(r['c2_slope'], r['c2_r'])} | "
            f"**{r['c2_verdict']}** | {ramp} |")
    return "\n".join(lines)


def _med(s: pd.Series) -> float:
    """Median that returns NaN (quietly) when every value is missing."""
    s = s.dropna()
    return float(s.median()) if len(s) else np.nan


def domain_summary(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in t.groupby("domain", observed=True):
        rows.append(dict(
            domain=d, n_substrates=len(g), n_images=int(g["n_images"].sum()),
            c1_min=g["c1_min"].min(), c1_max=g["c1_max"].max(),
            fd_min=3 - g["c1_max"].max(), fd_max=3 - g["c1_min"].min(),
            c2_min=g["c2_min"].min(), c2_max=g["c2_max"].max(),
            c1_slope=_med(g["c1_slope"]), c2_slope=_med(g["c2_slope"]),
            n_feasible=int((g["c2_verdict"] == "FEASIBLE").sum()),
            n_weak=int((g["c2_verdict"] == "WEAK").sum()),
            n_infeasible=int((g["c2_verdict"] == "INFEASIBLE").sum()),
        ))
    return pd.DataFrame(rows)


def write_doc(t: pd.DataFrame, obs: pd.DataFrame, path: Path):
    ds = domain_summary(t)
    tested = t[t["c2_verdict"] != "UNTESTED"]
    feas = tested[tested["c2_verdict"] == "FEASIBLE"].sort_values("c2_slope", ascending=False)
    wide_fd = t.dropna(subset=["c1_span"]).sort_values("c1_span", ascending=False)
    ramps = t.dropna(subset=["fd_ramp_delta"]).sort_values("fd_ramp_strength", ascending=False)
    both = tested[(tested["c2_verdict"] == "FEASIBLE") & (tested["c1_span"] >= 0.45)] \
        .sort_values("c2_slope", ascending=False)

    L = []
    A = L.append
    A("# Substrate feasibility reference\n")
    A(f"*Generated by `benchmarks/regressor/feasibility_table.py` from "
      f"{len(obs)} measured images across {obs['source'].nunique()} sweeps. "
      f"{len(t)} substrates, {t['domain'].nunique()} domains.*\n")
    A("Machine-readable version: `benchmarks/results/feasibility_table.csv`. "
      "Figure: `benchmarks/figures/56_feasibility_map.png`.\n")

    A("## How to use this\n")
    A("Every texture substrate reaches only part of the (c1, c2) plane. Before you design a "
      "stimulus set, look your substrate up here and check three things:\n")
    A("1. **Is the (c1, c2) you want inside its achieved range?** The ranges are raw "
      "min/max of *measured* values, not targets. `FD = 3 − c1`, so a wide c1 range is a wide FD range.\n"
      "2. **Does it obey?** The control slope is measured Δ per requested Δ; 1.0 is perfect. "
      "c1 slopes are high nearly everywhere; c2 slopes are the discriminating number.\n"
      "3. **If you want a spatial ramp, which way does it run?** `inverted` substrates render "
      "an FD ramp backwards relative to the request — and flipping the request makes the gradient "
      "*weaker*, so this is a rendering preference to accept, not a bug to fix.\n")
    A("**Verdict thresholds (c2)** — chosen against a ground-truth c2 measurement repeatability "
      f"of ≈0.05–0.08 on a single image and a requested c2 swing of 0.45–0.80:\n")
    A(f"- `FEASIBLE`   slope ≥ {C2_FEASIBLE_SLOPE} **and** r ≥ {C2_FEASIBLE_R} — delivers ≥ 0.16 of "
      "measured c2 swing, monotonically. Usable as an experimental factor.\n"
      f"- `WEAK`       slope ≥ {C2_WEAK_SLOPE} **and** r ≥ {C2_WEAK_R} — direction real, magnitude ≈ noise. "
      "Needs seed averaging.\n"
      f"- `INFEASIBLE` otherwise — the substrate ignores the c2 request.\n"
      f"- `UNTESTED`   fewer than {MIN_PTS} usable points.\n")
    A("Slopes are **within-source** fits (each generation run is centred on its own mean before "
      "pooling), so a run-to-run offset cannot fake a slope.\n")

    A("## Headline counts\n")
    vc = t["c2_verdict"].value_counts()
    A(f"- c2 `FEASIBLE`: **{vc.get('FEASIBLE', 0)}** substrates · `WEAK`: {vc.get('WEAK', 0)} · "
      f"`INFEASIBLE`: {vc.get('INFEASIBLE', 0)} · `UNTESTED`: {vc.get('UNTESTED', 0)}")
    vc1 = t["c1_verdict"].value_counts()
    A(f"- c1 `FEASIBLE`: **{vc1.get('FEASIBLE', 0)}** · `WEAK`: {vc1.get('WEAK', 0)} · "
      f"`INFEASIBLE`: {vc1.get('INFEASIBLE', 0)} · `UNTESTED`: {vc1.get('UNTESTED', 0)}")
    A(f"- achieved c1 overall: {t['c1_min'].min():.2f} – {t['c1_max'].max():.2f} "
      f"(FD {3 - t['c1_max'].max():.2f} – {3 - t['c1_min'].min():.2f})")
    A(f"- achieved c2 overall: {t['c2_min'].min():.2f} – {t['c2_max'].max():.2f}")
    A(f"- median c1 slope {t['c1_slope'].median():.2f}, median c2 slope {t['c2_slope'].median():.2f}\n")

    A("## Pick a substrate for goal X\n")
    A("### X = a wide FD / c1 ramp (parametric complexity ladder)\n")
    A("| substrate | domain | c1 range (span) | FD range | c1 slope · r | n |")
    A("|---|---|---|---|---|---|")
    for _, r in wide_fd.head(10).iterrows():
        A(f"| `{r['substrate']}` | {r['domain']} | {rng(r['c1_min'], r['c1_max'])} "
          f"({fmt(r['c1_span'])}) | {rng(r['fd_min'], r['fd_max'])} | "
          f"{sl(r['c1_slope'], r['c1_r'])} | {r['n_images']} |")
    A("")
    A("### X = strong, obedient c2 (multifractality as an experimental factor)\n")
    A("| substrate | domain | c2 slope · r | c2 range | n |")
    A("|---|---|---|---|---|")
    for _, r in feas.head(12).iterrows():
        A(f"| `{r['substrate']}` | {r['domain']} | {sl(r['c2_slope'], r['c2_r'])} | "
          f"{rng(r['c2_min'], r['c2_max'])} | {r['n_images']} |")
    A("")
    A("### X = both at once (joint (c1, c2) grid on one substrate)\n")
    A(f"Substrates that are c2-`FEASIBLE` **and** span ≥ 0.45 in c1 — {len(both)} of {len(tested)} tested:\n")
    A("| substrate | domain | c1 span | c2 slope · r | c2 range | n |")
    A("|---|---|---|---|---|---|")
    for _, r in both.iterrows():
        A(f"| `{r['substrate']}` | {r['domain']} | {fmt(r['c1_span'])} | "
          f"{sl(r['c2_slope'], r['c2_r'])} | {rng(r['c2_min'], r['c2_max'])} | "
          f"{r['n_images']} |")
    A("")
    A("### X = a spatial FD gradient inside one image\n")
    A("Sorted by measured |Δc1| across the image. `inverted` = the substrate renders the ramp "
      "backwards relative to the request; a flip test showed flipping *reduces* the gradient, "
      "so run it in the substrate's preferred polarity.\n")
    A("| subject | domain | axis | c1 A → B | Δc1 | strength | polarity | class |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in ramps.iterrows():
        A(f"| `{r['substrate']}` | {r['domain']} | {r['fd_ramp_axis']} | "
          f"{r['fd_ramp_c1_A']:.2f} → {r['fd_ramp_c1_B']:.2f} | "
          f"{r['fd_ramp_delta']:+.3f} | {r['fd_ramp_strength']:.3f} | "
          f"**{r['fd_ramp_polarity']}** | {r['fd_ramp_class']} |")
    A("")
    npol = ramps["fd_ramp_polarity"].value_counts()
    A(f"{npol.get('direct', 0)} direct / {npol.get('inverted', 0)} inverted; "
      f"{int((ramps['fd_ramp_class'] == 'STRONG').sum())} of {len(ramps)} reach |Δc1| ≥ "
      f"{RAMP_STRONG}. The inverted substrates — "
      + ", ".join(f"`{s}`" for s in ramps[ramps['fd_ramp_polarity'] == 'inverted']['substrate'])
      + " — were re-run with the request flipped, and flipping *reduced* |Δc1|. Inversion is a "
        "rendering preference of the substrate, so render these with the high-c1 (low-FD) end "
        "first and relabel, rather than fighting it.\n")

    A("## Per-domain summary\n")
    A("| domain | substrates | images | c1 range | FD range | c2 range | med c1 slope | "
      "med c2 slope | c2 FEAS/WEAK/INFEAS |")
    A("|" + "---|" * 9)
    for _, r in ds.iterrows():
        A(f"| **{r['domain']}** | {r['n_substrates']} | {r['n_images']} | "
          f"{rng(r['c1_min'], r['c1_max'])} | {rng(r['fd_min'], r['fd_max'])} | "
          f"{rng(r['c2_min'], r['c2_max'])} | {fmt(r['c1_slope'])} | {fmt(r['c2_slope'])} | "
          f"{r['n_feasible']} / {r['n_weak']} / {r['n_infeasible']} |")
    A("")

    A("## Full table\n")
    A("Sorted by domain, then by c2 slope (best-controlled substrate first). "
      "`n` = measured images backing the row; ranges are `[min, max]` of measured values. "
      "The CSV additionally carries `c1_span`/`c2_span`, `c1_verdict`, `n_c1_pts`/`n_c2_pts` "
      "(points entering each slope), `sources`, `fd_ramp_c1_A`/`_B`, and `mf_ramp_*` "
      "(the c2-mode gradient sweep, 12 subjects).\n")
    A(md_table(t))
    A("")

    A("## Caveats — read before quoting a number\n")
    hi_c1 = t[t["c1_max"] > 2.0].sort_values("c1_max", ascending=False)
    pos_c2 = t[t["c2_max"] > 0.05].sort_values("c2_max", ascending=False)
    A("- **Ranges include the gradient half-images.** `fd_gradient_gallery` / "
      "`mf_gradient_intermittent` measure the two ends of a ramped image separately, so those "
      "two values enter the reachable range (they are genuine measurements of that substrate) "
      "but never enter a slope (no requested target attaches to them). Half-image estimates are "
      "noisier than full-image ones.\n"
      "- **c1 > 2 means FD < 1, which is not physical for a surface** — it is the wavelet-leader "
      "estimator saturating on a very smooth / blurred field. Affected: "
      + ", ".join(f"`{r['substrate']}` (max c1 {r['c1_max']:.2f})" for _, r in hi_c1.iterrows())
      + ". Treat the top of those c1 ranges as a soft ceiling, not a usable target.\n"
      "- **c2 > 0 is meaningless as intermittency** (it should be ≤ 0); small positive values are "
      "estimator noise on near-monofractal fields. Affected: "
      + ", ".join(f"`{r['substrate']}` ({r['c2_max']:+.2f})" for _, r in pos_c2.iterrows()) + ".\n"
      "- **Slopes pool across sweeps run months apart** with different guidance scales. The "
      "within-source centring removes level differences but not slope differences; where a "
      "substrate has several sources the reported slope is the pooled within-source estimate, "
      "which can sit below the best single sweep (e.g. `marble` c2 is 1.32 in "
      "`c2_control_montage` alone, 0.59 in `reachable_atlas`, 0.42 in `natural_atlas`, "
      "**0.80 pooled**).\n"
      f"- **Small n.** {int((t['n_images'] < 5).sum())} substrates are backed by fewer than 5 "
      "images; the `n` column is in the table for exactly this reason. Single-seed rows "
      "(`natural_atlas`, `creature_atlas` are seed 0 only) carry seed noise in the range, "
      "not just substrate structure.\n")

    A("## Where this contradicts the repo's own prose\n")
    A("The CSVs are the ground truth; `APPLICATIONS.md` is the prose. Three places disagree:\n")
    dense = t[t["domain"].astype(str).isin(["microscopy", "plant", "sand", "microbial"])]
    dense_f = dense[dense["c2_verdict"] == "FEASIBLE"].sort_values("c2_slope", ascending=False)
    A("1. **\"c2 is essentially not controllable on microscopy / plant / sand / microbial\"** is "
      f"too strong. Those four bands do have the worst *median* c2 slope, but "
      f"{len(dense_f)} of their {len(dense)} substrates clear the FEASIBLE bar: "
      + ", ".join(f"`{r['substrate']}` ({r['c2_slope']:.2f}·{r['c2_r']:.2f})"
                  for _, r in dense_f.iterrows())
      + ". The band is not the unit of feasibility — the *substrate* is.\n")
    A("2. **The per-band correlations quoted in `APPLICATIONS.md` (c2 r ≈ 0.11–0.52) are pooled "
      "across families**, which mixes in each family's own c2 offset and understates how "
      f"monotonic control is *within* a family. Within-substrate, the median c2 correlation here "
      f"is {_med(t['c2_r']):.2f} and {int((t['c2_r'] >= 0.7).sum())} substrates exceed r = 0.70.\n")
    neg = t[(t["c2_slope"] < 0) & (t["c2_r"] < 0)]
    if len(neg):
        A("3. **Nothing in the docs mentions sign reversal**, but "
          + ", ".join(f"`{r['substrate']}` (slope {r['c2_slope']:+.2f}, r {r['c2_r']:+.2f})"
                      for _, r in neg.iterrows())
          + " responds to a c2 request *backwards*: asking for more intermittency makes it less "
            "intermittent. Do not use it as a c2 axis in either direction.\n")

    A("## Name aliasing\n")
    A("Different sweeps named the same substrate differently; these merges make the "
      "per-substrate rows possible. Audit them here:\n")
    A("| sweep name | canonical |")
    A("|---|---|")
    for k, v in sorted(ALIASES.items()):
        A(f"| `{k}` | `{v}` |")
    A("")
    A("Source CSVs: " + ", ".join(f"`{s}`" for s in sorted(obs["source"].unique())) + ".\n")

    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------- driver
def main():
    obs = load_observations()
    t = build_table(obs)
    out_csv = RESULTS / "feasibility_table.csv"
    t.to_csv(out_csv, index=False, float_format="%.4f")
    make_figure(obs, t, FIG / "56_feasibility_map.png")
    write_doc(t, obs, DOC)

    print(f"observations : {len(obs)} rows from {obs['source'].nunique()} sweeps")
    print(f"substrates   : {len(t)} in {t['domain'].nunique()} domains")
    print(f"c2 verdicts  : {t['c2_verdict'].value_counts().to_dict()}")
    print(f"c1 verdicts  : {t['c1_verdict'].value_counts().to_dict()}")
    print(f"c1 achieved  : {t['c1_min'].min():.3f} .. {t['c1_max'].max():.3f}  "
          f"(FD {3 - t['c1_max'].max():.3f} .. {3 - t['c1_min'].min():.3f})")
    print(f"c2 achieved  : {t['c2_min'].min():.3f} .. {t['c2_max'].max():.3f}")
    print(f"ramp polarity: {t['fd_ramp_polarity'].replace('', np.nan).value_counts().to_dict()}")
    print(f"ramp class   : {t['fd_ramp_class'].replace('', np.nan).value_counts().to_dict()}")
    print(f"-> {out_csv}")
    print(f"-> {DOC}")
    print(f"-> {FIG / '56_feasibility_map.png'}")


if __name__ == "__main__":
    main()
