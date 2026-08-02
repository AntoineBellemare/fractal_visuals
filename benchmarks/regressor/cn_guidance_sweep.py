"""
cn_guidance_sweep.py — is the control-vs-semantics tension structural, or just a bad operating point?

THE QUESTION. Conditioning strength buys FD control and costs semantics: at cn_scale 0.90 the
melting clocks become dunes, at 0.40 the axis stops moving (figure 73). But that was measured at a
single `guidance_end = 0.60`. Those are two different knobs and they were never swept together.

THE HYPOTHESIS worth testing: impose the FD structure HARD but release it EARLY. The conditioning
only has to win during the steps that lay down structure; if it lets go soon enough, the late steps
may re-establish semantics on top of an already-set fractal dimension. If that works there is a
sweet spot at high cn_scale and low guidance_end, and the tension is an operating-point problem.
If control and semantics fall together no matter how the two are combined, the tension is
structural and should be documented as the ceiling of a conditioning-image route.

MEASURING BOTH AXES:
  control   = achieved FD span across the frame sweep, plus tracking corr(field c1, image c1)
  semantics = DINOv2 cosine similarity between the conditioned render and an UNCONDITIONED render
              (cn_scale = 0) from the same prompt and seed. That reference is "what the prompt
              alone gives", so the similarity says how much of the intended scene survived.
              DINOv2 is already the repo's pixel backbone, so this adds no new dependency.

The result is a Pareto plot: every (cn_scale, guidance_end) cell placed by what it costs and what
it buys.

Usage:
  python cn_guidance_sweep.py --frames 8
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "diffusion"))
from cascade_calibration import Calibration                              # noqa: E402
from complexity_animation import (SURREAL, SUBJECTS, measure, uniform_us,  # noqa: E402
                                  frame_fields, TAIL)
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN  # noqa: E402

# Subjects whose SEMANTICS are unambiguous, so "did the scene survive" is a real question.
PROBES = ["melting_clocks", "moth_machine", "eye_storm", "clock_forest"]


class Dino:
    """DINOv2 embedding + cosine similarity — the semantic-preservation proxy."""
    def __init__(self, device):
        from transformers import AutoModel, AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.device = device

    @torch.no_grad()
    def emb(self, pil):
        x = self.proc(images=pil.convert("RGB"), return_tensors="pt").to(self.device)
        v = self.m(**x).last_hidden_state[:, 0]          # CLS token
        return torch.nn.functional.normalize(v, dim=-1)[0].float().cpu().numpy()

    def sim(self, a, b):
        return float(np.dot(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cn-scales", default="0.65,0.90,1.15")
    ap.add_argument("--guidance-ends", default="0.25,0.40,0.60,0.80")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--c1-lo", type=float, default=0.35)
    ap.add_argument("--c1-hi", type=float, default=1.45)
    ap.add_argument("--c2", type=float, default=-0.45)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--subjects", default=",".join(PROBES))
    ap.add_argument("--out", default=str(HERE / "creative" / "cn_ge_sweep"))
    args = ap.parse_args()

    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    cns = [float(x) for x in args.cn_scales.split(",")]
    ges = [float(x) for x in args.guidance_ends.split(",")]
    subs = [s for s in args.subjects.split(",") if s in SURREAL or s in SUBJECTS]
    cal = Calibration.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ControlNet ({DEFAULT_CN})...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)
    dino = Dino(device)
    us, _ = uniform_us(args.c1_lo, args.c1_hi, args.c2, args.frames, cal)

    rows = []
    t0 = time.time(); k = 0
    n_tot = len(subs) * (1 + len(cns) * len(ges) * args.frames)
    for si, sub in enumerate(subs):
        prompt = (SURREAL.get(sub) or SUBJECTS[sub]) + TAIL
        seed = args.seed + si
        flds = frame_fields(args.c1_lo, args.c1_hi, args.c2, args.frames, args.res,
                            seed, cal, us=us)
        # reference: the prompt with NO conditioning at all
        k += 1
        ref = generate(pipe, prompt, field_to_control(flds[0]), 0.0,
                       guidance_end=0.6, steps=args.steps, cfg=args.cfg,
                       seed=seed, res=args.res, device=device)
        ref.save(out / "img" / f"{sub}_REF.png", optimize=True)
        eref = dino.emb(ref)
        print(f"  [{k}/{n_tot}] {sub}: reference rendered", flush=True)

        for cn in cns:
            for ge in ges:
                c1s, sims = [], []
                for i, fld in enumerate(flds):
                    k += 1
                    img = generate(pipe, prompt, field_to_control(fld), cn,
                                   guidance_end=ge, steps=args.steps, cfg=args.cfg,
                                   seed=seed, res=args.res, device=device)
                    if i in (0, args.frames - 1):
                        img.save(out / "img" / f"{sub}_cn{cn:.2f}_ge{ge:.2f}_f{i:02d}.png",
                                 optimize=True)
                    c1m, _ = measure(np.asarray(img.convert("L"), float) / 255.0)
                    c1s.append(c1m); sims.append(dino.sim(eref, dino.emb(img)))
                fc = [measure(f)[0] for f in flds]
                span = max(c1s) - min(c1s)
                track = float(np.corrcoef(fc, c1s)[0, 1])
                rows.append(dict(subject=sub, cn_scale=cn, guidance_end=ge,
                                 span=round(span, 4), track=round(track, 4),
                                 semantic=round(float(np.mean(sims)), 4),
                                 c1_lo=round(min(c1s), 3), c1_hi=round(max(c1s), 3)))
                print(f"  [{k}/{n_tot}] {sub:15s} cn {cn:.2f} ge {ge:.2f}  "
                      f"span {span:.3f}  track {track:+.2f}  semantic {np.mean(sims):.3f}  "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
                _flush(out, rows)
    _flush(out, rows)

    import pandas as pd
    d = pd.DataFrame(rows)
    print("\n=== mean over subjects ===")
    piv_s = d.pivot_table(index="cn_scale", columns="guidance_end", values="span")
    piv_m = d.pivot_table(index="cn_scale", columns="guidance_end", values="semantic")
    print("\nFD span (control):\n" + piv_s.round(3).to_string())
    print("\nDINOv2 similarity to unconditioned (semantics):\n" + piv_m.round(3).to_string())
    g = d.groupby(["cn_scale", "guidance_end"]).agg(span=("span", "mean"),
                                                    semantic=("semantic", "mean")).reset_index()
    # Pareto front: no other cell beats it on BOTH axes
    front = [r for _, r in g.iterrows()
             if not ((g.span > r.span) & (g.semantic > r.semantic)).any()]
    print("\nPARETO FRONT (nothing beats these on both control and semantics):")
    for r in sorted(front, key=lambda r: -r.span):
        print(f"  cn {r.cn_scale:.2f}  ge {r.guidance_end:.2f}   "
              f"span {r.span:.3f}   semantic {r.semantic:.3f}")


def _flush(out, rows):
    with (out / "sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()


def figure(src=None):
    """75_cn_guidance_sweep.png — the two knobs are SEPARABLE, and the old default was dominated."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src = Path(src or (HERE / "creative" / "cn_ge_sweep"))
    FIGD = ROOT / "benchmarks" / "figures"
    d = pd.read_csv(src / "sweep.csv")
    d.to_csv(ROOT / "benchmarks" / "results" / "cn_guidance_sweep.csv", index=False)
    S = d.pivot_table(index="cn_scale", columns="guidance_end", values="span")
    M = d.pivot_table(index="cn_scale", columns="guidance_end", values="semantic")

    subs = ["melting_clocks", "moth_machine", "eye_storm"]
    cells = [("REFERENCE", None), (0.65, 0.70), (0.90, 0.70), (1.15, 0.50), (1.15, 0.35)]
    fig = plt.figure(figsize=(19.5, 12.4))
    gs = fig.add_gridspec(4, len(cells), height_ratios=[1, 1, 1, 1.15], hspace=0.22, wspace=0.05)
    bot = gs[3, :].subgridspec(1, 4, wspace=0.42, width_ratios=[1, 1, 0.12, 2.1])
    for i, sub in enumerate(subs):
        for j, (cn, ge) in enumerate(cells):
            ax = fig.add_subplot(gs[i, j]); ax.set_xticks([]); ax.set_yticks([])
            fp = (src / "img" / f"{sub}_REF.png") if ge is None else \
                 (src / "img" / f"{sub}_cn{cn:.2f}_ge{ge:.2f}_f00.png")
            ax.imshow(Image.open(fp).resize((400, 400)))
            if i == 0:
                ax.set_title("prompt alone\n(no conditioning)" if ge is None
                             else f"cn {cn:.2f}   ge {ge:.2f}", fontsize=11.5, fontweight="bold")
            if j == 0:
                ax.set_ylabel(sub, fontsize=11, fontweight="bold")
    for k, (T, lab, cmap) in enumerate(((S, "FD span  (control)", "viridis"),
                                        (M, "DINOv2 similarity  (semantics)", "magma"))):
        ax = fig.add_subplot(bot[0, k])
        im = ax.imshow(T.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(T.columns))); ax.set_xticklabels(T.columns)
        ax.set_yticks(range(len(T.index))); ax.set_yticklabels(T.index)
        ax.set_xlabel("guidance_end")
        ax.set_ylabel("cn_scale" if k == 0 else "")
        for a in range(T.shape[0]):
            for b in range(T.shape[1]):
                ax.text(b, a, f"{T.values[a, b]:.2f}", ha="center", va="center",
                        color="w", fontsize=10, fontweight="bold")
        ax.set_title(lab, fontsize=11, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.040, pad=0.04)
    ax = fig.add_subplot(bot[0, 3])
    g = d.groupby(["cn_scale", "guidance_end"]).agg(span=("span", "mean"),
                                                    semantic=("semantic", "mean")).reset_index()
    for cn, mk in zip(sorted(g.cn_scale.unique()), ("o", "s", "^")):
        s = g[g.cn_scale == cn].sort_values("guidance_end")
        ax.plot(s.semantic, s.span, mk + "-", lw=2, ms=8, label=f"cn_scale {cn:.2f}")
        for _, r in s.iterrows():
            ax.annotate(f"{r.guidance_end:.2f}", (r.semantic, r.span), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("semantics kept  (DINOv2 similarity to the unconditioned render)")
    ax.set_ylabel("control  (achieved FD span)")
    ax.legend(fontsize=9.5); ax.grid(alpha=0.25)
    ax.set_title("Up and to the right is better. Raising cn_scale moves the whole curve UP at "
                 "almost no horizontal cost —\nthe semantic price is paid by guidance_end "
                 "(labelled on each point), not by conditioning strength.", fontsize=10.5,
                 loc="left")
    fig.suptitle("Is the control-vs-semantics tension structural?  NO — the old default was simply "
                 "not on the Pareto front.  The two knobs are SEPARABLE:\ncn_scale buys FD control "
                 "nearly for free (semantics at ge 0.50: 0.52 → 0.57 → 0.59 as cn rises 0.65 → "
                 "1.15), while guidance_end is what\nactually costs semantics (0.76 → 0.33 as it "
                 "rises 0.20 → 0.70).  Recommended: cn_scale 1.15 with guidance_end 0.50 for "
                 "texture,\n0.35 for figurative content — both dominate the previous cn 0.90 / "
                 "ge 0.60 default.", fontsize=12.5, y=0.999)
    fig.savefig(FIGD / "75_cn_guidance_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("-> 75_cn_guidance_sweep.png")
