"""
cond_encoding_sweep.py — can ANY 8-bit conditioning-image encoding carry c2 (multifractality)?

CONTEXT
-------
The MF-ControlNet conditioning image is 8-bit.  The established diagnosis in this repo is that
c1 (roughness / FD) survives that bottleneck but c2 (intermittency) does not:

    c1 is a SLOPE of <log leader> vs log scale  -> invariant to any affine rescale of the field.
    c2 is a VARIANCE of log leader vs log scale -> destroyed by the percentile clip + quantisation.

The current encoder is mf_controlnet.field_to_control(), i.e.
    mf_scaffold.robust_unit(f, 1, 99)  ->  uint8 "L"  ->  RGB.

This script tests ~20 alternative encodings on two axes:

  E1 GLOBAL TRANSMISSION  regress c2(encoded uint8) on c2(raw float field) over a c2 sweep.
                          A separate c1 sweep gives an UNCONFOUNDED c1 slope (in the c2 sweep
                          c1 drifts as a side effect of the c1-c2 coupling of prescribed_cascade,
                          so the c1 slope measured there is not interpretable on its own).

  E2 SPATIAL TRANSMISSION build a c2 gradient field the way mf_creative.gradient_field does,
                          measure c2 in the left third vs the right third of the ENCODED image,
                          and compare to the same delta on the RAW float field (the ceiling).

E2 is run under two field CONSTRUCTIONS, because the headline result of this script is that the
encoding is only the *second* bottleneck:

  "current"    field = flo*(1-m) + fhi*m                      <- exactly mf_creative.gradient_field
  "scalematch" field = zscore(flo)*(1-m) + zscore(fhi)*m      <- each component divided by its MAD
  "histmatch"  field = flo*(1-m) + hist_match(fhi, flo)*m     <- what mf_creative.shape_field
                                                                 already does for hidden shapes

prescribed_cascade returns _normalize01() of a heavy-tailed signed field, so the *bulk* amplitude
of a c2 = -0.90 field is ~40x smaller than that of a c2 = -0.20 field.  In the "current" linear
blend the low-|c2| component therefore dominates the bulk EVERYWHERE, including on the side that
is supposed to be high-|c2| -- so the raw float field does not actually carry a c2 gradient at all.

Usage:
    python benchmarks/regressor/cond_encoding_sweep.py                 # full run
    python benchmarks/regressor/cond_encoding_sweep.py --quick         # 2 seeds, fewer targets
    python benchmarks/regressor/cond_encoding_sweep.py --skip-e1

CPU only.  No torch, no GPU.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import mfractal as mf  # noqa: E402

OUT = HERE / "cond_encoding"
EPS = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (np.ptp(x) + EPS)


def mad_scale(x):
    """Robust sigma = 1.4826 * MAD.  Non-zero by construction."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826 + EPS)


def zscore_robust(x):
    return (x - np.median(x)) / mad_scale(x)


def rank_uniform(x):
    """Rank transform -> uniform on [0,1].  Monotone, marginal-destroying."""
    flat = np.asarray(x, float).ravel()
    r = flat.argsort(kind="stable").argsort(kind="stable").astype(float)
    return (r / max(r.size - 1, 1)).reshape(np.shape(x))


def rank_gauss(x, clip=4.0):
    u = np.clip(rank_uniform(x), 1e-7, 1 - 1e-7)
    z = np.clip(np.sqrt(2.0) * erfinv(2 * u - 1), -clip, clip)
    return (z + clip) / (2 * clip)


def hist_match(src, ref):
    """Remap src onto ref's marginal histogram, preserving src's rank order."""
    ranks = np.asarray(src, float).ravel().argsort(kind="stable").argsort(kind="stable")
    return np.sort(np.asarray(ref, float).ravel())[ranks].reshape(np.shape(src))


def _tri_dither(rng, shape):
    """Triangular-pdf dither, +-1 LSB, the standard choice for breaking quantisation banding."""
    return rng.random(shape) - rng.random(shape)


def to_u8(u, dither=False, rng=None):
    """[0,1] float -> uint8, optionally with triangular dither before rounding."""
    v = np.clip(np.asarray(u, float), 0, 1) * 255.0
    if dither:
        v = v + _tri_dither(rng, v.shape)
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def gray_rgb(u8):
    return Image.fromarray(u8, "L").convert("RGB")


def luma(img):
    """The naive decode any downstream model can do: RGB -> float greyscale."""
    return np.asarray(img.convert("L"), float)


# ---------------------------------------------------------------------------
# ENCODINGS.  each: field(float 2D) -> (PIL RGB uint8 image, decode_fn, meta)
#   decode_fn(img) -> float 2D array that the *estimator* is run on.
#   meta["oracle"] is True when decode_fn needs side information the ControlNet would not have.
# ---------------------------------------------------------------------------
def _pack(u8, meta=None):
    img = gray_rgb(u8)
    return img, luma, (meta or {})


def enc_robust_unit_1_99(f, rng):
    """BASELINE == mf_scaffold.robust_unit(f, 1, 99) -> uint8 L -> RGB."""
    a, b = np.percentile(f, 1.0), np.percentile(f, 99.0)
    return _pack(to_u8(np.clip((f - a) / (b - a + 1e-8), 0, 1)))


def enc_robust_unit_01_999(f, rng):
    a, b = np.percentile(f, 0.1), np.percentile(f, 99.9)
    return _pack(to_u8(np.clip((f - a) / (b - a + 1e-8), 0, 1)))


def enc_robust_unit_5_95(f, rng):
    a, b = np.percentile(f, 5.0), np.percentile(f, 95.0)
    return _pack(to_u8(np.clip((f - a) / (b - a + 1e-8), 0, 1)))


def enc_minmax(f, rng):
    return _pack(to_u8(minmax(f)))


def enc_log_minmax(f, rng):
    return _pack(to_u8(minmax(np.log(f - f.min() + 1e-3))))


def enc_log_robust(f, rng):
    L = np.log(f - f.min() + 1e-3)
    a, b = np.percentile(L, 1.0), np.percentile(L, 99.0)
    return _pack(to_u8(np.clip((L - a) / (b - a + 1e-8), 0, 1)))


def enc_hist_eq(f, rng):
    return _pack(to_u8(rank_uniform(f)))


def enc_hist_eq_dither(f, rng):
    return _pack(to_u8(rank_uniform(f), dither=True, rng=rng))


def enc_normal_score(f, rng):
    return _pack(to_u8(rank_gauss(f)))


def enc_normal_score_dither(f, rng):
    return _pack(to_u8(rank_gauss(f), dither=True, rng=rng))


def enc_log_dither(f, rng):
    return _pack(to_u8(minmax(np.log(f - f.min() + 1e-3)), dither=True, rng=rng))


def _log_local_norm(f, sigma):
    """log -> subtract a WIDE gaussian blur -> renormalise.  'local affine renormalisation':
    c1 is affine-invariant so it should survive, and no single extreme region can dominate the
    global stretch and flatten the rest of the frame."""
    L = np.log(f - f.min() + 1e-3)
    L = L - gaussian_filter(L, sigma)
    return minmax(np.arcsinh(L / mad_scale(L)))


def enc_log_local_norm_s12(f, rng):
    return _pack(to_u8(_log_local_norm(f, 12.0)))


def enc_log_local_norm_s48(f, rng):
    return _pack(to_u8(_log_local_norm(f, 48.0)))


def enc_log_local_norm_s192(f, rng):
    return _pack(to_u8(_log_local_norm(f, 192.0)))


def _asinh(f, a):
    """Symmetric SOFT clip about the median: monotone, tail-compressing, nothing truncated.
    The field is normalize01() of a *signed* symmetric heavy-tailed variable, so the natural
    compression is symmetric about the median, not a log."""
    return minmax(np.arcsinh(zscore_robust(f) / a))


def enc_asinh_a1(f, rng):
    return _pack(to_u8(_asinh(f, 1.0)))


def enc_asinh_a025(f, rng):
    return _pack(to_u8(_asinh(f, 0.25)))


def enc_asinh_a4(f, rng):
    return _pack(to_u8(_asinh(f, 4.0)))


def enc_signpow_03(f, rng):
    t = zscore_robust(f)
    return _pack(to_u8(minmax(np.sign(t) * np.abs(t) ** 0.3)))


def enc_local_affine_s96(f, rng):
    """Full local affine renormalisation: subtract local mean AND divide by local scale."""
    t = zscore_robust(f)
    mu = gaussian_filter(t, 96.0)
    d = t - mu
    sd = np.sqrt(np.maximum(gaussian_filter(d * d, 96.0), EPS))
    return _pack(to_u8(minmax(np.arcsinh(d / sd))))


def enc_bitsplit16(f, rng):
    """16-bit normal score split across two channels: R = high byte, G = low byte, B = 8-bit copy.
    Attacks quantisation head-on -- 65536 levels instead of 256.  The decode is exact and needs
    no side information, so it is a clean upper bound on 'is 8 bits the binding constraint?'."""
    v = np.clip(rank_gauss(f), 0, 1) * 65535.0
    q = np.clip(np.round(v), 0, 65535).astype(np.uint16)
    hi = (q >> 8).astype(np.uint8)
    lo = (q & 0xFF).astype(np.uint8)
    rgb = np.stack([hi, lo, hi], -1)
    img = Image.fromarray(rgb, "RGB")

    def dec(im):
        a = np.asarray(im, float)
        return a[..., 0] * 256.0 + a[..., 1]

    return img, dec, {"note": "exact 16-bit decode, no side info"}


def enc_multichannel(f, rng):
    """R = coarse envelope, G = mid band, B = fine residual, each stretched separately.
    3x the effective bits, and the band that carries intermittency gets its own full range."""
    t = np.arcsinh(zscore_robust(f))
    coarse = gaussian_filter(t, 16.0)
    mid = gaussian_filter(t, 4.0) - coarse
    fine = t - gaussian_filter(t, 4.0)
    chans, scales = [], []
    for c in (coarse, mid, fine):
        lo_, hi_ = c.min(), np.ptp(c) + EPS
        scales.append((lo_, hi_))
        chans.append(to_u8((c - lo_) / hi_))
    img = Image.fromarray(np.stack(chans, -1), "RGB")

    def dec(im):
        a = np.asarray(im, float) / 255.0
        out = np.zeros(a.shape[:2])
        for k, (lo_, hi_) in enumerate(scales):
            out = out + (a[..., k] * hi_ + lo_)
        return out

    return img, dec, {"oracle": True, "note": "decode needs the 3 per-channel scales"}


def enc_multiscale_rgb(f, rng):
    """R = normal score (the ControlNet-friendly image), G = coarse envelope, B = fine residual.
    Unlike multichannel, R alone is a valid standalone conditioning image."""
    t = np.arcsinh(zscore_robust(f))
    r = rank_gauss(f)
    g = minmax(gaussian_filter(t, 16.0))
    b = minmax(t - gaussian_filter(t, 4.0))
    img = Image.fromarray(np.stack([to_u8(r), to_u8(g), to_u8(b)], -1), "RGB")
    return img, (lambda im: np.asarray(im, float)[..., 0]), {"note": "decode = R channel"}


# --- float controls (NO 8-bit step) : isolate transform damage from quantisation damage ------
def enc_identity_f32(f, rng):
    return None, None, {"float": lambda x: np.asarray(x, float)}


def enc_robust_1_99_nq(f, rng):
    a, b = np.percentile(f, 1.0), np.percentile(f, 99.0)
    return None, None, {"float": lambda x, a=a, b=b: np.clip((x - a) / (b - a + 1e-8), 0, 1)}


def enc_normal_score_nq(f, rng):
    return None, None, {"float": lambda x: rank_gauss(x)}


def enc_minmax_nq(f, rng):
    return None, None, {"float": lambda x: minmax(x)}


ENCODINGS = {
    # --- required set -------------------------------------------------------
    "robust_unit_1_99": enc_robust_unit_1_99,          # BASELINE (current behaviour)
    "minmax": enc_minmax,
    "log_minmax": enc_log_minmax,
    "log_robust": enc_log_robust,
    "normal_score": enc_normal_score,
    "hist_eq": enc_hist_eq,
    "log_local_norm_s12": enc_log_local_norm_s12,
    "log_local_norm_s48": enc_log_local_norm_s48,
    "log_local_norm_s192": enc_log_local_norm_s192,
    "log_dither": enc_log_dither,
    "multichannel": enc_multichannel,
    # --- added ---------------------------------------------------------------
    "robust_unit_0.1_99.9": enc_robust_unit_01_999,
    "robust_unit_5_95": enc_robust_unit_5_95,
    "hist_eq_dither": enc_hist_eq_dither,
    "normal_score_dither": enc_normal_score_dither,
    "asinh_mad_a0.25": enc_asinh_a025,
    "asinh_mad_a1": enc_asinh_a1,
    "asinh_mad_a4": enc_asinh_a4,
    "signpow_0.3": enc_signpow_03,
    "local_affine_s96": enc_local_affine_s96,
    "bitsplit16": enc_bitsplit16,
    "multiscale_rgb": enc_multiscale_rgb,
    # --- float controls (diagnostics) ---------------------------------------
    "CTRL_identity_f32": enc_identity_f32,
    "CTRL_minmax_noquant": enc_minmax_nq,
    "CTRL_robust_1_99_noquant": enc_robust_1_99_nq,
    "CTRL_normal_score_noquant": enc_normal_score_nq,
}

BASELINE = "robust_unit_1_99"


def encode_and_decode(name, f, rng):
    """-> (decoded float array measured by the estimator, diagnostics dict)."""
    img, dec, meta = ENCODINGS[name](f, rng)
    if "float" in meta:                                   # float control: no uint8 round trip
        g = meta["float"](f)
        return np.asarray(g, float), {"levels": np.nan, "clip_frac": np.nan,
                                      "oracle": False, "float_ctrl": True}
    a = np.asarray(img, np.uint8)
    lum = np.asarray(img.convert("L"), np.uint8)
    diag = {"levels": int(np.unique(lum).size),
            "clip_frac": float(np.mean((lum == 0) | (lum == 255))),
            "oracle": bool(meta.get("oracle", False)), "float_ctrl": False}
    return np.asarray(dec(img), float), diag, img


def _enc(name, f, rng):
    """Uniform wrapper: -> (decoded, luma_decoded_or_None, diag)."""
    out = encode_and_decode(name, f, rng)
    if len(out) == 2:
        return out[0], None, out[1]
    dec, diag, img = out
    return dec, luma(img), diag


def measure(a):
    a = np.asarray(a, float)
    if not np.isfinite(a).all() or np.ptp(a) < 1e-12:
        return np.nan, np.nan
    r = mf.wavelet_leaders_2d(a)
    return float(r["c1"]), float(r["c2"])


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) < 1e-9 or np.ptp(y[ok]) < 1e-12:
        return np.nan, np.nan
    s = float(np.polyfit(x[ok], y[ok], 1)[0])
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return s, r


# ---------------------------------------------------------------------------
# E1  global transmission
# ---------------------------------------------------------------------------
def run_e1(args, names):
    c2_targets = ([-0.15, -0.45, -0.75, -1.05] if args.quick else
                  [-0.15, -0.30, -0.45, -0.60, -0.75, -0.90, -1.05, -1.20])
    c1_targets = [1.0, 1.15, 1.3, 1.45, 1.6]
    seeds = list(range(args.seeds_e1))

    rows = []
    for sweep, targets in (("c2", c2_targets), ("c1", c1_targets)):
        for t in targets:
            c1t, c2t = (1.3, t) if sweep == "c2" else (t, -0.45)
            for sd in seeds:
                f = np.asarray(mf.prescribed_cascade(n=args.n_e1, seed=sd,
                                                     c1_target=c1t, c2_target=c2t), float)
                in1, in2 = measure(f)                      # raw, native resolution -- never 8-bit
                for nm in names:
                    rng = np.random.default_rng(1234 + sd)
                    dec, lum, diag = _enc(nm, f, rng)
                    o1, o2 = measure(dec)
                    l1, l2 = measure(lum) if lum is not None else (np.nan, np.nan)
                    rows.append(dict(sweep=sweep, target=t, seed=sd, enc=nm,
                                     c1_in=in1, c2_in=in2, c1_out=o1, c2_out=o2,
                                     c1_out_luma=l1, c2_out_luma=l2, **diag))
            print(f"  [E1 {sweep}] target {t:+.2f} done", flush=True)

    with open(OUT / "e1_global_raw.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summ = []
    for nm in names:
        r2 = [r for r in rows if r["enc"] == nm and r["sweep"] == "c2"]
        r1 = [r for r in rows if r["enc"] == nm and r["sweep"] == "c1"]
        s2, rr2 = fit([r["c2_in"] for r in r2], [r["c2_out"] for r in r2])
        s2l, rr2l = fit([r["c2_in"] for r in r2], [r["c2_out_luma"] for r in r2])
        s1, rr1 = fit([r["c1_in"] for r in r1], [r["c1_out"] for r in r1])
        s1c, rr1c = fit([r["c1_in"] for r in r2], [r["c1_out"] for r in r2])
        lv = np.nanmean([r["levels"] for r in r2])
        cf = np.nanmean([r["clip_frac"] for r in r2])
        summ.append(dict(enc=nm, c2_slope=s2, c2_r=rr2, c2_slope_luma=s2l, c2_r_luma=rr2l,
                         c1_slope=s1, c1_r=rr1, c1_slope_c2sweep=s1c, c1_r_c2sweep=rr1c,
                         levels=lv, clip_frac=cf,
                         oracle=bool(r2[0]["oracle"]), float_ctrl=bool(r2[0]["float_ctrl"])))
    with open(OUT / "e1_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    return rows, summ


# ---------------------------------------------------------------------------
# E2  spatial transmission  (the one that matters)
# ---------------------------------------------------------------------------
CONSTRUCTIONS = ["current", "scalematch", "histmatch", "null_samec2", "null_ampramp"]


def build_gradient(construction, flo, fhi, m, fnull=None, amp_ratio=1.0):
    """flo/fhi are the c2_lo / c2_hi components.  fnull is a SAME-c2 partner for flo, used only
    by the null constructions."""
    if construction == "current":                  # exactly mf_creative.gradient_field
        return flo * (1 - m) + fhi * m
    if construction == "scalematch":               # each component divided by its own MAD first
        return zscore_robust(flo) * (1 - m) + zscore_robust(fhi) * m
    if construction == "histmatch":                # fhi remapped onto flo's marginal histogram
        return flo * (1 - m) + hist_match(fhi, flo) * m

    # ---- NULL CONTROLS.  Both have NO c2 gradient.  Any encoding that reports a consistent
    # non-zero dc2 here is manufacturing the signal, not transmitting it.
    if construction == "null_samec2":              # two same-c2 fields, plain blend
        return flo * (1 - m) + fnull * m
    if construction == "null_ampramp":
        # same-c2 fields, but the right component is scaled down by the SAME bulk-amplitude
        # ratio a real c2=-0.90 field would have.  Isolates "does the encoding respond to the
        # amplitude ramp rather than to intermittency?"
        return flo * (1 - m) + (fnull * amp_ratio) * m
    raise ValueError(construction)


def thirds(a, n, side_px):
    """Left-third and right-third square crops, taken at the vertical centre so both are
    isotropic.  Resized to 512 only if the third is not already 512 wide."""
    s = n // 3
    y0 = (n - s) // 2
    L = a[y0:y0 + s, 0:s]
    R = a[y0:y0 + s, n - s:n]
    if s != side_px:
        L = np.asarray(Image.fromarray(L.astype(np.float32), "F")
                       .resize((side_px, side_px), Image.BICUBIC), float)
        R = np.asarray(Image.fromarray(R.astype(np.float32), "F")
                       .resize((side_px, side_px), Image.BICUBIC), float)
    return L, R


def run_e2(args, names):
    n = args.n_e2
    m = np.mgrid[0:n, 0:n][1] / (n - 1)                    # horizontal ramp == grad_mask("h")
    constructions = [c for c in args.constructions.split(",") if c]
    seeds = list(range(args.seeds_e2))
    need_null = any(c.startswith("null_") for c in constructions)
    rows = []
    for sd in seeds:
        t0 = time.time()
        flo = np.asarray(mf.prescribed_cascade(n=n, seed=sd, c1_target=args.c1,
                                               c2_target=args.c2_lo), float)
        fhi = np.asarray(mf.prescribed_cascade(n=n, seed=sd, c1_target=args.c1,
                                               c2_target=args.c2_hi), float)
        fnull, amp_ratio = None, 1.0
        if need_null:
            fnull = np.asarray(mf.prescribed_cascade(n=n, seed=sd + 500, c1_target=args.c1,
                                                     c2_target=args.c2_lo), float)
            amp_ratio = mad_scale(fhi) / mad_scale(flo)     # the real bulk-amplitude ratio
        for con in constructions:
            fld = build_gradient(con, flo, fhi, m, fnull, amp_ratio)
            L, R = thirds(fld, n, args.region)
            rc1L, rc2L = measure(L)
            rc1R, rc2R = measure(R)
            for nm in names:
                rng = np.random.default_rng(1234 + sd)
                dec, _lum, diag = _enc(nm, fld, rng)
                eL, eR = thirds(dec, n, args.region)
                ec1L, ec2L = measure(eL)
                ec1R, ec2R = measure(eR)
                rows.append(dict(construction=con, seed=sd, enc=nm,
                                 dc1_raw=rc1R - rc1L, dc2_raw=rc2R - rc2L,
                                 dc1_enc=ec1R - ec1L, dc2_enc=ec2R - ec2L,
                                 c2_raw_L=rc2L, c2_raw_R=rc2R,
                                 c2_enc_L=ec2L, c2_enc_R=ec2R, **diag))
        print(f"  [E2] seed {sd} done ({time.time() - t0:.0f}s)", flush=True)

    with open(OUT / "e2_spatial_raw.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summ = []
    for con in constructions:
        sub0 = [r for r in rows if r["construction"] == con and r["enc"] == names[0]]
        draw2 = np.array([r["dc2_raw"] for r in sub0])
        draw1 = np.array([r["dc1_raw"] for r in sub0])
        for nm in names:
            sub = [r for r in rows if r["construction"] == con and r["enc"] == nm]
            d2 = np.array([r["dc2_enc"] for r in sub], float)
            d1 = np.array([r["dc1_enc"] for r in sub], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                t2 = np.nanmean(d2) / np.nanmean(draw2)
                t1 = np.nanmean(d1) / np.nanmean(draw1)
            sd2 = float(np.nanstd(d2, ddof=1)) if np.isfinite(d2).sum() > 1 else np.nan
            sd1 = float(np.nanstd(d1, ddof=1)) if np.isfinite(d1).sum() > 1 else np.nan
            summ.append(dict(construction=con, enc=nm,
                             dc2_raw_mean=float(np.nanmean(draw2)),
                             dc2_raw_sd=float(np.nanstd(draw2, ddof=1)),
                             dc2_enc_mean=float(np.nanmean(d2)), dc2_enc_sd=sd2,
                             dc2_transmission=float(t2),
                             dc2_snr=float(abs(np.nanmean(d2)) / (sd2 + EPS)),
                             dc1_raw_mean=float(np.nanmean(draw1)),
                             dc1_enc_mean=float(np.nanmean(d1)), dc1_enc_sd=sd1,
                             dc1_transmission=float(t1),
                             dc1_snr=float(abs(np.nanmean(d1)) / (sd1 + EPS))))
    with open(OUT / "e2_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    return rows, summ


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(e1_rows, e1_summ, e2_summ, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = [s for s in e1_summ if not s["float_ctrl"]] if e1_summ else []
    fig, ax = plt.subplots(2, 3, figsize=(21, 11))

    if e1_summ:
        order = sorted(real, key=lambda s: -abs(s["c2_r"] if np.isfinite(s["c2_r"]) else 0))
        nm = [s["enc"] for s in order]
        col = ["tab:red" if x == BASELINE else "tab:blue" for x in nm]

        a = ax[0, 0]
        a.barh(range(len(nm)), [abs(s["c2_r"]) for s in order], color=col)
        a.set_yticks(range(len(nm)));  a.set_yticklabels(nm, fontsize=7)
        a.invert_yaxis();  a.axvline(abs([s for s in real if s["enc"] == BASELINE][0]["c2_r"]),
                                     ls="--", c="k", lw=1)
        a.set_xlabel("|r|  (E1: encoded c2 vs raw c2)")
        a.set_title("E1 global: c2 discriminability (higher = better)")

        a = ax[0, 1]
        v = [np.clip(s["c2_slope"], -2, 5) for s in order]
        a.barh(range(len(nm)), v, color=col)
        a.set_yticks(range(len(nm)));  a.set_yticklabels(nm, fontsize=7)
        a.invert_yaxis();  a.axvline(1.0, ls="--", c="k", lw=1)
        a.set_xlabel("c2 slope (clipped to [-2,5] for display)")
        a.set_title("E1 global: c2 slope (1.0 = faithful)")

        a = ax[0, 2]
        v = [np.clip(s["c1_slope"], -2, 5) for s in order]
        a.barh(range(len(nm)), v, color=col)
        a.set_yticks(range(len(nm)));  a.set_yticklabels(nm, fontsize=7)
        a.invert_yaxis();  a.axvline(1.0, ls="--", c="k", lw=1)
        a.set_xlabel("c1 slope (dedicated c1 sweep)")
        a.set_title("E1 global: c1 slope -- does any encoding break the WORKING FD path?")

    if e2_summ:
        cons = [c for c in CONSTRUCTIONS if any(s["construction"] == c for s in e2_summ)][:3]
        for k, con in enumerate(cons):
            a = ax[1, k]
            sub = [s for s in e2_summ if s["construction"] == con
                   and not s["enc"].startswith("CTRL_")]
            sub = sorted(sub, key=lambda s: s["dc2_enc_mean"])
            nm = [s["enc"] for s in sub]
            mu = [np.clip(s["dc2_enc_mean"], -3, 3) for s in sub]
            er = [np.clip(s["dc2_enc_sd"], 0, 3) for s in sub]
            col = ["tab:red" if x == BASELINE else "tab:blue" for x in nm]
            a.barh(range(len(nm)), mu, xerr=er, color=col, error_kw=dict(lw=0.8))
            a.set_yticks(range(len(nm)));  a.set_yticklabels(nm, fontsize=7)
            a.invert_yaxis()
            raw = sub[0]["dc2_raw_mean"]
            a.axvline(raw, ls="--", c="g", lw=1.5, label=f"raw ceiling {raw:+.2f}")
            a.axvline(0, c="k", lw=0.7)
            a.legend(fontsize=8)
            a.set_xlabel("dc2 (right third - left third) of ENCODED image")
            a.set_title(f"E2 spatial -- construction '{con}'\n(target dc2 is NEGATIVE)")

    fig.suptitle("Conditioning-image encodings: can c2 survive the 8-bit bottleneck?", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "cond_encoding_summary.png", dpi=110)
    print(f"figure -> {OUT / 'cond_encoding_summary.png'}")


def make_null_figure(e2_rows):
    """THE decisive figure: what each encoding reports on fields that contain NO c2 gradient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(con, enc, key="dc2_enc"):
        s = [r for r in e2_rows if r["construction"] == con and r["enc"] == enc]
        s.sort(key=lambda r: int(r["seed"]))
        return np.array([float(r[key]) for r in s])

    encs = sorted({r["enc"] for r in e2_rows})
    keep = [e for e in encs if np.abs(series("current", e)).max() < 20
            and np.abs(series("null_ampramp", e)).max() < 20]
    rec = []
    for e in keep:
        cur, na, ns = series("current", e), series("null_ampramp", e), series("null_samec2", e)
        d = cur - na                                    # paired artifact subtraction
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        rec.append((e, cur.mean(), cur.std(ddof=1), na.mean(), ns.mean(), d.mean(), t))
    rec.sort(key=lambda r: r[1])
    nm = [r[0] for r in rec]
    col = ["tab:red" if x == BASELINE else ("tab:green" if x.startswith("CTRL_") else "tab:blue")
           for x in nm]

    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    a = ax[0]
    a.barh(range(len(nm)), [r[1] for r in rec], xerr=[r[2] for r in rec], color=col,
           error_kw=dict(lw=0.8))
    a.axvline(0, c="k", lw=0.8)
    a.set_title("(1) as-shipped gradient field\ntrue dc2 is NEGATIVE -> bars should point LEFT")
    a.set_xlabel("dc2 reported (right third - left third)")

    a = ax[1]
    a.barh(range(len(nm)), [r[3] for r in rec], color=col)
    a.barh(range(len(nm)), [r[4] for r in rec], color="0.6", height=0.4)
    a.axvline(0, c="k", lw=0.8)
    a.set_title("(2) NULL CONTROLS -- no c2 gradient exists\n"
                "colour = amplitude ramp only;  grey = plain same-c2 blend\n"
                "anything away from 0 is MANUFACTURED signal")
    a.set_xlabel("dc2 reported on a field with zero true dc2")

    a = ax[2]
    a.barh(range(len(nm)), [r[5] for r in rec], color=col)
    a.axvline(0, c="k", lw=0.8)
    a.set_title("(3) artifact-corrected: (1) minus (2)\nthe c2 signal that is genuinely there")
    a.set_xlabel("paired dc2(current) - dc2(null_ampramp)")

    for a in ax:
        a.set_yticks(range(len(nm)))
        a.set_yticklabels(nm, fontsize=8)
        a.invert_yaxis()
    fig.suptitle("Why spatial c2 gradients read as noise: the amplitude-ramp confound", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "null_controls.png", dpi=110)
    print(f"figure -> {OUT / 'null_controls.png'}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-e1", type=int, default=512)
    p.add_argument("--n-e2", type=int, default=1536,
                   help="gradient field size; 1536 makes each third exactly 512 px (no resample)")
    p.add_argument("--region", type=int, default=512)
    p.add_argument("--seeds-e1", type=int, default=3)
    p.add_argument("--seeds-e2", type=int, default=6)
    p.add_argument("--c1", type=float, default=1.3)
    p.add_argument("--c2-lo", type=float, default=-0.20)
    p.add_argument("--c2-hi", type=float, default=-0.90)
    p.add_argument("--constructions", default=",".join(CONSTRUCTIONS),
                   help="gradient-field constructions to test (see CONSTRUCTIONS)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-e1", action="store_true")
    p.add_argument("--skip-e2", action="store_true")
    p.add_argument("--only", default="", help="comma-separated encoding subset")
    p.add_argument("--save-samples", action="store_true",
                   help="write one example conditioning image per encoding")
    p.add_argument("--figures-only", action="store_true",
                   help="rebuild the figures from the CSVs already in cond_encoding/")
    args = p.parse_args()

    if args.figures_only:
        def _num(d):
            out = {}
            for k, v in d.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            return out
        e1s = [_num(r) for r in csv.DictReader(open(OUT / "e1_summary.csv"))]
        for r in e1s:
            r["float_ctrl"] = (r["float_ctrl"] == "True")
        e2s = [_num(r) for r in csv.DictReader(open(OUT / "e2_summary.csv"))]
        e2r = list(csv.DictReader(open(OUT / "e2_spatial_raw.csv")))
        make_figure(None, e1s, e2s, [r["enc"] for r in e1s])
        make_null_figure(e2r)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    names = [x for x in args.only.split(",") if x] or list(ENCODINGS)
    for x in names:
        assert x in ENCODINGS, f"unknown encoding {x}"
    print(f"{len(names)} encodings -> {OUT}")

    t0 = time.time()
    e1_rows = e1_summ = e2_summ = e2_rows = None
    if not args.skip_e1:
        print("E1 global transmission ...")
        e1_rows, e1_summ = run_e1(args, names)
        print(f"\n{'encoding':<26}{'c2_slope':>10}{'c2_r':>8}{'c1_slope':>10}{'levels':>9}{'clip%':>8}")
        for s in sorted(e1_summ, key=lambda s: -abs(s["c2_r"] if np.isfinite(s["c2_r"]) else 0)):
            print(f"{s['enc']:<26}{s['c2_slope']:>10.3f}{s['c2_r']:>8.3f}"
                  f"{s['c1_slope']:>10.3f}{s['levels']:>9.1f}{100 * s['clip_frac']:>8.2f}")
    if not args.skip_e2:
        print("\nE2 spatial transmission ...")
        e2_rows, e2_summ = run_e2(args, names)
        for con in CONSTRUCTIONS:
            sub = [s for s in e2_summ if s["construction"] == con]
            if not sub:
                continue
            tag = "  <<< NULL CONTROL: true dc2 is 0" if con.startswith("null_") else ""
            print(f"\n--- construction '{con}':  raw ceiling dc2 = "
                  f"{sub[0]['dc2_raw_mean']:+.3f} +- {sub[0]['dc2_raw_sd']:.3f}  "
                  f"(dc1 raw {sub[0]['dc1_raw_mean']:+.3f}){tag}")
            print(f"{'encoding':<26}{'dc2_enc':>10}{'+-sd':>8}{'transm':>9}{'snr':>7}{'dc1_enc':>10}")
            for s in sorted(sub, key=lambda s: s["dc2_enc_mean"]):
                print(f"{s['enc']:<26}{s['dc2_enc_mean']:>10.3f}{s['dc2_enc_sd']:>8.3f}"
                      f"{s['dc2_transmission']:>9.2f}{s['dc2_snr']:>7.2f}{s['dc1_enc_mean']:>10.3f}")

    if args.save_samples:
        sd = (OUT / "samples"); sd.mkdir(exist_ok=True)
        f = np.asarray(mf.prescribed_cascade(n=512, seed=0, c1_target=1.3, c2_target=-0.75), float)
        for nm in names:
            out = ENCODINGS[nm](f, np.random.default_rng(0))
            if out[0] is not None:
                out[0].save(sd / f"{nm}.png")
        print(f"samples -> {sd}")

    if e1_summ or e2_summ:
        make_figure(e1_rows, e1_summ, e2_summ, names)
    if e2_rows and any(r["construction"].startswith("null_") for r in e2_rows):
        make_null_figure(e2_rows)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
