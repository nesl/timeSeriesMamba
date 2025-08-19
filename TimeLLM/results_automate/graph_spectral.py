#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-domain plots & stats + aggregate (no AllDomains plots).

Improvements:
  • Paper-ready plotting: constrained layout, tight bbox, minimal margins
  • Larger, consistent font sizes & markers for readability
  • Flexible Omega label: Ω / "Spectral predictability" / "Spectral predictability (Ω)"
  • Smaller whitespace around axes, grid kept subtle
  • CLI knobs for figsize, dpi, fonts, margins
  • NEW: GPT2 model support (detection, styling, comparisons)

Usage
-----
python graph_spectral.py --x-metric ALL --out-dir ./out --mse-out-dir ./out_mse \
  --omega-label both --fig-w 6.2 --fig-h 4.0 --dpi 600 --font 11 --tick-font 10 --legend-font 10 \
  --x-margin 0.02 --y-margin 0.06 --tight-bbox
"""

import os
import re
import sys
import math
import argparse
from glob import glob
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Optional deps (used if present)
try:
    from scipy import stats as ss
except Exception:
    ss = None
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None
# Robust/Mixed models (optional)
try:
    import statsmodels.api as sm
except Exception:
    sm = None

# ------------------- Defaults -------------------
DEFAULT_LOG_DIRS = [
    "../results/pems_eval/",
    "../results/fitbit_eval/",
    "../results/carbon_eval/",
    "../results/spectralUniTest/",
    "../results/uniSynth_eval/",
]
DEFAULT_OUT_DIR = "./out"

DOMAIN_NAME_FIX = {
    "spectralUniTest": "CarbonCast",
    "carbon_eval": "CarbonCast",
    "pems_eval": "PEMS",
    "fitbit_eval": "Fitbit",
    "uniSynth_eval": "UniSynth",
}

DOMAINS_CANON = ("CarbonCast", "PEMS", "Fitbit", "UniSynth")

# ------------------- Plot styling -------------------
MODEL_COLOR = {
    "Language Pretrained": "tab:blue",
    "Random Init": "tab:orange",
    "DLinear": "tab:green",
    "GPT2": "tab:blue",  # same family color as Language Pretrained
}
MODEL_MARKER = {
    "Language Pretrained": "^",
    "Random Init": "^",
    "DLinear": "o",
    "GPT2": "s",  # square to distinguish from Language Pretrained
}
ALPHA_RAW = 0.28
ALPHA_MARKER = 0.9   # slightly higher for clearer means

# ------------------- Metric keys -------------------
NUM_PAT = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

SMAPE_PATS = [
    re.compile(r"\bsmape(_loss)?\b"),
    re.compile(r"\bsym(_)?mape\b"),
]
MSE_PATS = [
    re.compile(r"\bmse(_loss)?\b"),
    re.compile(r"\bmean_squar(?:ed|e)_error\b"),
    re.compile(r"\bmse\b"),
]

X_METRIC_PATS = {
    "SE":     [re.compile(r"\bse_ctx_mean\b"), re.compile(r"\bspectral_entropy(_ctx)?_mean\b"), re.compile(r"\bse_mean\b")],
    "LLE":    [re.compile(r"\blle_ctx_mean\b")],
    "Omega":  [re.compile(r"\bomega_ctx_mean\b")],
    "SNR":    [re.compile(r"\bsnr_ctx_proxy\b")],
    "Season": [re.compile(r"\bseason_ctx_mean\b")],
}

# ------------------- Helpers -------------------
def norm_key(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())

def parse_run_summary(lines: List[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    in_block = False
    for line in lines:
        if "wandb: Run summary:" in line:
            in_block = True
            continue
        if not in_block:
            continue
        m = re.match(r"^\s*wandb:\s+(.+?)\s+(" + NUM_PAT + r")\s*$", line)
        if m:
            key = norm_key(m.group(1))
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            metrics[key] = val
    return metrics

def choose_metric(metrics: Dict[str, Any], patterns: List[re.Pattern]) -> Optional[float]:
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
            if any(pat.search(norm_key(k)) for pat in patterns):
                return float(v)
    return None

def infer_model_from_path(path: str) -> Optional[str]:
    b = os.path.basename(path)
    b_up = b.upper()
    b_lo = b.lower()
    # DLinear
    if "dlin" in b_lo:
        return "DLinear"
    # GPT2 (e.g., ..._mGPT2_... or ..._GPT2_...)
    if "mgpt2" in b_lo or re.search(r"\bgpt2\b", b_lo):
        return "GPT2"
    # LLAMA3.2 families with r0/r1 flag
    if "LLAMA3.2" in b_up:
        if "_R0_" in b_up:
            return "Language Pretrained"
        if "_R1_" in b_up:
            return "Random Init"
        return None
    return None

def parse_seed_init(path: str) -> Dict[str, Optional[int]]:
    b = os.path.basename(path)
    m_seed = re.search(r"(?:_seed|_s)(\d+)", b, re.IGNORECASE)
    m_init = re.search(r"(?:_init|_i)(\d+)", b, re.IGNORECASE)
    return {"seed": int(m_seed.group(1)) if m_seed else None,
            "init_seed": int(m_init.group(1)) if m_init else None}

def make_base_key(path: str) -> str:
    b = os.path.basename(path)
    b = re.sub(r"_r[01]_", "_", b, flags=re.IGNORECASE)
    b = re.sub(r"_(?:seed|s)\d+", "", b, flags=re.IGNORECASE)
    b = re.sub(r"_(?:init|i)\d+", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\.txt$", "", b, flags=re.IGNORECASE)
    b = re.sub(r"__+", "_", b).strip("_")
    return b

def infer_domain_from_text(text: str) -> Optional[str]:
    t = text.lower()
    if "unisynth" in t:
        return "UniSynth"
    if "carbon" in t or "spectralunitest" in t:
        return "CarbonCast"
    if "fitbit" in t:
        return "Fitbit"
    if "pems" in t:
        return "PEMS"
    return None

def domain_name_from_dir(dir_path: str) -> str:
    base = os.path.basename(os.path.normpath(dir_path))
    return DOMAIN_NAME_FIX.get(base, base)

# ---------- New: plotting style controls ----------
def apply_rc(font=11, tick_font=10, legend_font=10):
    matplotlib.rcParams.update({
        "font.size": font,
        "axes.titlesize": font,
        "axes.labelsize": font,
        "xtick.labelsize": tick_font,
        "ytick.labelsize": tick_font,
        "legend.fontsize": legend_font,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.30,
        "legend.frameon": False,
        "figure.autolayout": False,  # we use constrained layout per-figure
    })

def label_for_omega(mode: str) -> str:
    mode = mode.lower()
    if mode == "greek": return "Ω"
    if mode == "text":  return "Spectral predictability"
    # default: both
    return "Spectral predictability (Ω)"

def label_for_xmetric(xm: str, omega_mode: str) -> str:
    if xm.lower() == "omega":
        return label_for_omega(omega_mode)
    if xm == "SE":      return "SE"
    if xm == "LLE":     return "LLE"
    if xm == "SNR":     return "SNR (proxy)"
    if xm == "Season":  return "Seasonality strength"
    return xm

# ------------------- Load raw (across dirs) -------------------
def load_all_dirs(log_dirs: List[str], x_metric_name: str) -> pd.DataFrame:
    pats_x = X_METRIC_PATS[x_metric_name]
    rows = []
    for d in log_dirs:
        files = sorted(glob(os.path.join(d, "**", "*.txt"), recursive=True))
        fallback_domain = domain_name_from_dir(d)
        for path in files:
            try:
                model = infer_model_from_path(path)
                if model is None:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                metrics = parse_run_summary(text.splitlines())
                if not metrics:
                    continue
                smape = choose_metric(metrics, SMAPE_PATS)
                mse   = choose_metric(metrics, MSE_PATS)
                # pick the first matching x metric (keeps behavior)
                x = None
                for pat in pats_x:
                    x = choose_metric(metrics, [pat])
                    if x is not None:
                        break
                if smape is None or x is None:
                    continue
                seeds = parse_seed_init(path)
                domain = infer_domain_from_text(os.path.basename(path)) or fallback_domain
                rows.append({
                    "domain": domain,
                    "log_path": path,
                    "model": model,
                    "base_key": make_base_key(path),
                    "seed": seeds["seed"],
                    "init_seed": seeds["init_seed"],
                    "metric": float(x),
                    "smape": float(smape),
                    "mse": float(mse) if mse is not None else np.nan,
                    "mtime": os.path.getmtime(path),
                })
            except Exception as e:
                print(f"[warn] failed to parse {path}: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=[
            "domain","log_path","model","base_key","seed","init_seed","metric","smape","mse"
        ])

    df = pd.DataFrame(rows).sort_values("mtime").drop_duplicates(subset=["log_path"], keep="last")
    df = df.drop(columns=["mtime"], errors="ignore")
    return df

# ------------------- Aggregation (base plots) -------------------
def aggregate_by_model_metric_y(df_raw: pd.DataFrame, ykey: str, round_digits: int = 3) -> pd.DataFrame:
    if df_raw.empty or ykey not in df_raw.columns:
        return pd.DataFrame(columns=[
            "model","metric_bin","n_raw","n_seeds","n_inits","n_base_keys",
            "metric_mean","y_mean","y_low","y_high"
        ])
    df = df_raw.copy()
    df = df[np.isfinite(df["metric"]) & np.isfinite(df[ykey])]
    if df.empty:
        return pd.DataFrame(columns=[
            "model","metric_bin","n_raw","n_seeds","n_inits","n_base_keys",
            "metric_mean","y_mean","y_low","y_high"
        ])
    df["metric_bin"] = df["metric"].round(round_digits)
    rows = []
    for (model, mbin), g in df.groupby(["model","metric_bin"]):
        rows.append({
            "model": model,
            "metric_bin": float(mbin),
            "n_raw": int(len(g)),
            "n_seeds": int(g["seed"].nunique()),
            "n_inits": int(g["init_seed"].nunique()),
            "n_base_keys": int(g["base_key"].nunique()),
            "metric_mean": float(g["metric"].mean()),
            "y_mean": float(g[ykey].mean()),
            "y_low": float(np.nanmin(g[ykey])),
            "y_high": float(np.nanmax(g[ykey])),
        })
    return pd.DataFrame(rows).sort_values(["model","metric_bin"])

# ------------------- Plot (base) -------------------
def _data_span(a: np.ndarray, pad_frac: float = 0.02) -> Tuple[float, float]:
    if a.size == 0 or not np.isfinite(a).any():
        return (0.0, 1.0)
    lo = float(np.nanmin(a)); hi = float(np.nanmax(a))
    span = max(hi - lo, 1e-9)
    pad = pad_frac * span
    return (lo - pad, hi + pad)

def plot_by_model_metric(df_raw: pd.DataFrame,
                         agg: pd.DataFrame,
                         x_metric_label: str,
                         y_label: str,
                         title: str,
                         out_png: str,
                         jitter_x_frac: float,
                         fig_w: float, fig_h: float, dpi: int,
                         x_margin: float, y_margin: float,
                         tick_font: int,
                         legend_font: int,
                         tight_bbox: bool) -> None:
    if agg.empty:
        print(f"[info] No data for {title}")
        return

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")

    # raw points (semi-transparent)
    if not df_raw.empty:
        ycol = "mse" if y_label.lower() == "mse" else "smape"
        for model, g in df_raw.groupby("model"):
            g = g[np.isfinite(g["metric"]) & np.isfinite(g[ycol])]
            if g.empty:
                continue
            ax.scatter(
                g["metric"], g[ycol],
                marker=MODEL_MARKER.get(model, "^"),
                s=28, alpha=ALPHA_RAW, edgecolors="none",
                c=MODEL_COLOR.get(model, "tab:gray"),
                label="_nolegend_", zorder=2,
            )

    # jitter separation based on data span
    x_vals = df_raw["metric"].to_numpy(float)
    x_vals = x_vals[np.isfinite(x_vals)]
    x_span = float(np.nanmax(x_vals) - np.nanmin(x_vals)) if x_vals.size else 0.0
    x_mean_abs = float(np.nanmean(np.abs(x_vals))) if x_vals.size else 0.0
    jitter_abs = jitter_x_frac * x_span if x_span > 0 else (1e-3 * max(x_mean_abs, 1e-6))

    def model_offset(m: str) -> float:
        if jitter_abs <= 0:
            return 0.0
        # Spread models slightly so markers don't overlap
        if m == "Language Pretrained": return -jitter_abs
        if m == "Random Init":         return +jitter_abs
        if m == "GPT2":                return +2.0 * jitter_abs
        if m == "DLinear":             return 0.0
        return 0.0

    used = set()
    for _, r in agg.iterrows():
        model = r["model"]
        color = MODEL_COLOR.get(model, "tab:gray")
        marker = MODEL_MARKER.get(model, "^")
        label = model if model not in used else "_nolegend_"
        used.add(model)
        x = r["metric_mean"] + model_offset(model)

        yerr = [[r["y_mean"] - r["y_low"]], [r["y_high"] - r["y_mean"]]]
        msize = 8 if model == "DLinear" else 10

        ax.errorbar(
            x, r["y_mean"], yerr=yerr, xerr=None,
            fmt=marker, linestyle="none", capsize=2.5,
            markersize=msize, alpha=ALPHA_MARKER,
            color=color, ecolor=color,
            markeredgewidth=0.9, markeredgecolor="black",
            elinewidth=1.2, zorder=3, label=label
        )

    ax.set_xlabel(x_metric_label, labelpad=2)
    ax.set_ylabel(y_label, labelpad=2)
    ax.set_title(title, pad=2)
    if used:
        ax.legend(title="Model", fontsize=legend_font, title_fontsize=legend_font, loc="best")

    # tighter limits & margins to reduce whitespace
    if x_vals.size:
        lo, hi = _data_span(x_vals, pad_frac=x_margin)
        if "Ω" in x_metric_label or "Spectral predictability" in x_metric_label:
            lo = max(0.0, lo)  # Omega shown non-negative
        ax.set_xlim(lo, hi)
    y_vals = df_raw["mse"].to_numpy(float) if y_label.lower() == "mse" else df_raw["smape"].to_numpy(float)
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size:
        ylo, yhi = _data_span(y_vals, pad_frac=y_margin)
        ax.set_ylim(ylo, yhi)

    ax.grid(True, linestyle="--", alpha=0.30)
    ax.tick_params(axis="both", which="major", labelsize=tick_font)
    ax.margins(x=0, y=0)

    if tight_bbox:
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    else:
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_png}")

# ------------------- Stats (per-domain) -------------------
def _pearson_r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3: return np.nan
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return np.nan if den == 0 else float(np.dot(x, y) / den)

def _fisher_ci(r, n):
    if not np.isfinite(r) or n < 4: return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / math.sqrt(max(n - 3, 1))
    zcrit = 1.959963984540054
    return (float(np.tanh(z - zcrit*se)), float(np.tanh(z + zcrit*se)))

def _rank(a):
    if ss is not None and hasattr(ss, "rankdata"):
        return ss.rankdata(np.asarray(a, float), method="average")
    return pd.Series(a, dtype=float).rank(method="average").to_numpy()

def _spearman(x, y):
    xr = _rank(x); yr = _rank(y)
    return _pearson_r(xr, yr)

def _kendall_tau_b(x, y):
    if ss is not None and hasattr(ss, "kendalltau"):
        try:
            res = ss.kendalltau(x, y, variant="b", nan_policy="omit")
            return float(res.statistic)
        except Exception:
            return np.nan
    return np.nan

def _distance_correlation(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 2: return np.nan
    X = np.abs(x.reshape(-1,1) - x.reshape(1,-1))
    Y = np.abs(y.reshape(-1,1) - y.reshape(1,-1))
    def _dc(D):
        r = D.mean(axis=1, keepdims=True); c = D.mean(axis=0, keepdims=True); a = D.mean()
        return D - r - c + a
    A = _dc(X); B = _dc(Y)
    dcov2 = np.mean(A*B); dvarx = np.mean(A*A); dvary = np.mean(B*B)
    denom = math.sqrt(max(dvarx,0)*max(dvary,0))
    return np.nan if denom == 0 else float(max(dcov2,0)/denom)

def _isotonic_r2(x, y):
    if IsotonicRegression is None:
        return np.nan
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3: return np.nan
    order = np.argsort(x); xr, yr = x[order], y[order]
    iso = IsotonicRegression(increasing="auto")
    yhat = iso.fit_transform(xr, yr)
    ss_res = float(np.sum((yr - yhat)**2)); ss_tot = float(np.sum((yr - yr.mean())**2))
    return np.nan if ss_tot == 0 else 1.0 - ss_res/ss_tot

def _theilsen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 2: return (np.nan, np.nan)
    slopes = []
    for i in range(n-1):
        dx = x[i+1:] - x[i]
        v = dx != 0
        if np.any(v):
            slopes.extend(((y[i+1:][v] - y[i]) / dx[v]).tolist())
    if not slopes: return (np.nan, float(np.median(y)))
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope*x))
    return slope, intercept

def _quartile_effect(x, y, q=0.25):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 4: return (np.nan, np.nan, np.nan, np.nan, np.nan)
    lo, hi = np.quantile(x, q), np.quantile(x, 1-q)
    y_lo = y[x <= lo]; y_hi = y[x >= hi]
    if y_lo.size == 0 or y_hi.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    d_mean = float(np.mean(y_hi) - np.mean(y_lo))
    s_lo, s_hi = float(np.std(y_lo, ddof=1)), float(np.std(y_hi, ddof=1))
    n_lo, n_hi = int(y_lo.size), int(y_hi.size)
    sp = math.sqrt(((n_lo-1)*s_lo**2 + (n_hi-1)*s_hi**2) / max(n_lo + n_hi - 2, 1))
    d_cohen = np.nan if sp == 0 else d_mean / sp
    return d_mean, d_cohen, float(n_lo), float(n_hi), np.nan

def stats_csv_for_domain(df_dom: pd.DataFrame, xm: str, out_dir: str):
    x = df_dom["metric"].to_numpy(float)
    y = df_dom["smape"].to_numpy(float)
    n = int(np.sum(np.isfinite(x) & np.isfinite(y)))
    pear = _pearson_r(x, y); r_lo, r_hi = _fisher_ci(pear, n)
    spear = _spearman(x, y); kend = _kendall_tau_b(x, y)
    dcor = _distance_correlation(x, y); iso = _isotonic_r2(x, y)
    ts_slope, ts_intercept = _theilsen(x, y)
    d_mean, d_cohen, n_lo, n_hi, _ = _quartile_effect(x, y, q=0.25)

    df_stats = pd.DataFrame([{
        "domain": df_dom["domain"].iloc[0],
        "x_metric": xm, "n": n,
        "pearson_r": pear, "r_95ci_low": r_lo, "r_95ci_high": r_hi,
        "spearman_rho": spear, "kendall_tau_b": kend,
        "distance_corr": dcor, "isotonic_R2": iso,
        "theilsen_slope": ts_slope, "theilsen_intercept": ts_intercept,
        "q25q75_delta_mean": d_mean, "q25q75_cohens_d": d_cohen,
        "q25_n": n_lo, "q75_n": n_hi
    }])
    dom = df_dom["domain"].iloc[0]
    csv_path = os.path.join(out_dir, f"{dom}_stats_sMAPE_vs_{xm}.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"[stats] wrote {csv_path}")

# -------- Aggregate (across domains) helpers --------
def _fisher_z(r): return np.arctanh(np.clip(r, -0.999999, 0.999999))
def _inv_fisher_z(z): return float(np.tanh(z))

def _meta_fisher(rs: List[float], ns: List[int], weighted: bool=True):
    vals = [(r, n) for r, n in zip(rs, ns) if np.isfinite(r) and n is not None and n >= 4]
    if not vals:
        return dict(r=np.nan, lo=np.nan, hi=np.nan, Q=np.nan, df=np.nan, p=np.nan, I2=np.nan)
    zs = np.array([_fisher_z(r) for r, _ in vals], float)
    if weighted:
        w = np.array([max(n-3, 1) for _, n in vals], float)  # var(z)≈1/(n-3)
    else:
        w = np.ones_like(zs)
    zbar = float(np.sum(w*zs) / np.sum(w))
    rbar = _inv_fisher_z(zbar)
    se  = 1.0 / math.sqrt(np.sum(w))
    lo  = _inv_fisher_z(zbar - 1.959963984540054*se)
    hi  = _inv_fisher_z(zbar + 1.959963984540054*se)
    Q   = float(np.sum(w*(zs - zbar)**2))
    df  = len(zs) - 1
    I2  = np.nan if Q <= 0 else max(0.0, (Q - df) / Q)
    pQ  = float(ss.chi2.sf(Q, df)) if (ss is not None and hasattr(ss, "chi2")) else np.nan
    return dict(r=rbar, lo=lo, hi=hi, Q=Q, df=df, p=pQ, I2=I2)

def _binom_p_ge(k_success: int, n_trials: int) -> float:
    from math import comb
    if n_trials <= 0: return np.nan
    return float(sum(comb(n_trials, i) for i in range(k_success, n_trials+1)) / (2**n_trials))

def aggregate_summary_across_domains(df_all: pd.DataFrame, xm: str, out_dir: str):
    rows = []
    pear_rs, pear_ns = [], []
    spear_rs, spear_ns = [], []
    taus, dcors, isoR2s, slopes, dmeans, dcohens = [], [], [], [], [], []
    neg_slope = 0; K = 0

    for dom in DOMAINS_CANON:
        g = df_all[df_all["domain"] == dom]
        if g.empty: continue
        x = g["metric"].to_numpy(float); y = g["smape"].to_numpy(float)
        n = int(np.sum(np.isfinite(x) & np.isfinite(y)))
        pear = _pearson_r(x, y); r_lo, r_hi = _fisher_ci(pear, n)
        spear = _spearman(x, y)
        tau = _kendall_tau_b(x, y)
        dcor = _distance_correlation(x, y)
        iso = _isotonic_r2(x, y)
        slope, _ = _theilsen(x, y)
        d_mean, d_cohen, _, _, _ = _quartile_effect(x, y, q=0.25)

        rows.append({
            "domain": dom, "x_metric": xm, "n": n,
            "pearson_r": pear, "r_95ci_low": r_lo, "r_95ci_high": r_hi,
            "spearman_rho": spear, "kendall_tau_b": tau,
            "distance_corr": dcor, "isotonic_R2": iso,
            "theilsen_slope": slope,
            "q25q75_delta_mean": d_mean, "q25q75_cohens_d": d_cohen
        })
        if np.isfinite(pear): pear_rs.append(pear); pear_ns.append(n)
        if np.isfinite(spear): spear_rs.append(spear); spear_ns.append(n)
        if np.isfinite(tau): taus.append(tau)
        if np.isfinite(dcor): dcors.append(dcor)
        if np.isfinite(iso): isoR2s.append(iso)
        if np.isfinite(slope): slopes.append(slope); neg_slope += (slope < 0)
        if np.isfinite(d_mean): dmeans.append(d_mean)
        if np.isfinite(d_cohen): dcohens.append(d_cohen)
        K += 1

    meta_r_w = _meta_fisher(pear_rs, pear_ns, weighted=True)
    meta_r_u = _meta_fisher(pear_rs, pear_ns, weighted=False)
    meta_s_w = _meta_fisher(spear_rs, spear_ns, weighted=True)
    meta_s_u = _meta_fisher(spear_rs, spear_ns, weighted=False)

    sign_p = _binom_p_ge(neg_slope, K) if K else np.nan

    rows.append({
        "domain": "META_Pearson_weighted", "x_metric": xm, "n": int(np.nansum(pear_ns)) if pear_ns else 0,
        "pearson_r": meta_r_w["r"], "r_95ci_low": meta_r_w["lo"], "r_95ci_high": meta_r_w["hi"],
        "spearman_rho": np.nan, "kendall_tau_b": np.nan,
        "distance_corr": np.nan, "isotonic_R2": np.nan,
        "theilsen_slope": np.nan, "q25q75_delta_mean": np.nan, "q25q75_cohens_d": np.nan,
        "Q": meta_r_w["Q"], "df": meta_r_w["df"], "I2": meta_r_w["I2"], "Q_pvalue": meta_r_w["p"],
    })
    rows.append({
        "domain": "META_Pearson_macro", "x_metric": xm, "n": K,
        "pearson_r": meta_r_u["r"], "r_95ci_low": meta_r_u["lo"], "r_95ci_high": meta_r_u["hi"],
        "spearman_rho": np.nan, "kendall_tau_b": np.nan,
        "distance_corr": np.nan, "isotonic_R2": np.nan,
        "theilsen_slope": np.nan, "q25q75_delta_mean": np.nan, "q25q75_cohens_d": np.nan,
        "Q": meta_r_u["Q"], "df": meta_r_u["df"], "I2": meta_r_u["I2"], "Q_pvalue": meta_r_u["p"],
    })
    rows.append({
        "domain": "META_Spearman_weighted", "x_metric": xm, "n": int(np.nansum(spear_ns)) if spear_ns else 0,
        "pearson_r": np.nan, "r_95ci_low": np.nan, "r_95ci_high": np.nan,
        "spearman_rho": meta_s_w["r"], "kendall_tau_b": np.nan,
        "distance_corr": np.nan, "isotonic_R2": np.nan,
        "theilsen_slope": np.nan, "q25q75_delta_mean": np.nan, "q25q75_cohens_d": np.nan,
        "Q": meta_s_w["Q"], "df": meta_s_w["df"], "I2": meta_s_w["I2"], "Q_pvalue": meta_s_w["p"],
    })
    rows.append({
        "domain": "META_Spearman_macro", "x_metric": xm, "n": K,
        "pearson_r": np.nan, "r_95ci_low": np.nan, "r_95ci_high": np.nan,
        "spearman_rho": meta_s_u["r"], "kendall_tau_b": np.nan,
        "distance_corr": np.nan, "isotonic_R2": np.nan,
        "theilsen_slope": np.nan, "q25q75_delta_mean": np.nan, "q25q75_cohens_d": np.nan,
        "Q": meta_s_u["Q"], "df": meta_s_u["df"], "I2": meta_s_u["I2"], "Q_pvalue": meta_s_u["p"],
    })
    rows.append({
        "domain": "MEANS/OTHER", "x_metric": xm, "n": K,
        "pearson_r": np.nan,
        "spearman_rho": np.nan,
        "kendall_tau_b": float(np.mean(taus)) if taus else np.nan,
        "distance_corr": float(np.mean(dcors)) if dcors else np.nan,
        "isotonic_R2": float(np.mean(isoR2s)) if isoR2s else np.nan,
        "theilsen_slope": float(np.median(slopes)) if slopes else np.nan,
        "q25q75_delta_mean": float(np.mean(dmeans)) if dmeans else np.nan,
        "q25q75_cohens_d": float(np.mean(dcohens)) if dcohens else np.nan,
        "neg_slope_domains": int(neg_slope), "K_domains": int(K), "sign_test_p(one-sided)": sign_p
    })

    agg_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f"Aggregate_sMAPE_vs_{xm}_summary.csv")
    agg_df.to_csv(out_csv, index=False)
    print(f"[agg] wrote {out_csv}")

# ------------------- Relative-gain by x-bin -------------------
def _aggregate_by_xbin(df_raw: pd.DataFrame, ykey: str, round_digits: int) -> pd.DataFrame:
    if ykey not in df_raw.columns:
        return pd.DataFrame(columns=["domain","metric_bin","model","metric_mean","err_mean","n"])
    df = df_raw.copy()
    df = df[np.isfinite(df["metric"]) & np.isfinite(df[ykey])]
    if df.empty:
        return pd.DataFrame(columns=["domain","metric_bin","model","metric_mean","err_mean","n"])
    df["metric_bin"] = df["metric"].round(round_digits)
    agg = (df.groupby(["domain","metric_bin","model"], dropna=False)
             .agg(metric_mean=("metric","mean"),
                  err_mean=(ykey,"mean"),
                  n=("log_path","count"))
             .reset_index())
    return agg

def _relative_gain_by_xbin(df_raw: pd.DataFrame,
                           model_a: str,
                           model_b: str,
                           ykey: str,
                           round_digits: int) -> pd.DataFrame:
    agg = _aggregate_by_xbin(df_raw, ykey=ykey, round_digits=round_digits)
    if agg.empty:
        return agg
    A = agg[agg["model"] == model_a][["domain","metric_bin","metric_mean","err_mean","n"]]
    B = agg[agg["model"] == model_b][["domain","metric_bin","metric_mean","err_mean","n"]]
    A = A.rename(columns={"metric_mean":"omega_A", "err_mean":"errA", "n":"nA"})
    B = B.rename(columns={"metric_mean":"omega_B", "err_mean":"errB", "n":"nB"})
    M = pd.merge(A, B, on=["domain","metric_bin"], how="inner")
    if M.empty:
        return pd.DataFrame(columns=["domain","metric_bin","omega","rel_gain_pct","nA","nB"])
    M["omega"] = 0.5 * (M["omega_A"] + M["omega_B"])
    denom = M["errA"].replace(0, np.nan)
    M["rel_gain_pct"] = 100.0 * (M["errA"] - M["errB"]) / denom
    M = M[["domain","metric_bin","omega","rel_gain_pct","nA","nB"]].dropna(subset=["rel_gain_pct"])
    return M.sort_values(["domain","metric_bin"]).reset_index(drop=True)

def _plot_rel_gain_vs_x(df_pairs: pd.DataFrame,
                        title: str,
                        out_png: str,
                        xlabel: str,
                        ylabel: str,
                        fig_w: float, fig_h: float, dpi: int,
                        x_margin: float, y_margin: float,
                        tick_font: int,
                        tight_bbox: bool) -> None:
    if df_pairs.empty:
        print(f"[info] No matched x-bins for {title}")
        return
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    ax.scatter(df_pairs["omega"], df_pairs["rel_gain_pct"], s=30, alpha=0.8, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="No gain")
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, pad=2)
    ax.legend(loc="best")

    # tight limits to reduce whitespace
    xv = df_pairs["omega"].to_numpy(float); yv = df_pairs["rel_gain_pct"].to_numpy(float)
    if xv.size:
        lo, hi = _data_span(xv, pad_frac=x_margin)
        if "Ω" in xlabel or "Spectral predictability" in xlabel:
            lo = max(0.0, lo)
        ax.set_xlim(lo, hi)
    if yv.size:
        ylo, yhi = _data_span(yv, pad_frac=y_margin)
        ax.set_ylim(ylo, yhi)

    ax.tick_params(axis="both", which="major", labelsize=tick_font)
    if tight_bbox:
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    else:
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_png}")

# ---------- NEW: output layout helpers ----------
def make_out_dirs(base_dir: str, xm: str) -> Dict[str, str]:
    root = os.path.join(base_dir, xm)
    sub = {
        "base": os.path.join(root, "base"),
        "base_mse": os.path.join(root, "base_mse"),
        "rel": os.path.join(root, "rel"),
        "delta": os.path.join(root, "delta"),
        "stats": os.path.join(root, "stats"),
    }
    for p in sub.values():
        os.makedirs(p, exist_ok=True)
    return sub

# ---------- NEW: compute Δ vs x per domain ----------
def _delta_pairs_unbinned(df_dom: pd.DataFrame,
                          model_num: str,
                          model_den: str,
                          ykey: str) -> pd.DataFrame:
    if ykey not in df_dom.columns or not np.isfinite(df_dom[ykey]).any():
        return pd.DataFrame(columns=["domain","omega","delta","nA","nB"])
    A = _aggregate_by_xbin(df_dom[df_dom["model"] == model_num], ykey=ykey, round_digits=3)
    B = _aggregate_by_xbin(df_dom[df_dom["model"] == model_den], ykey=ykey, round_digits=3)
    A = A[["domain","metric_bin","metric_mean","err_mean","n"]].rename(
        columns={"metric_mean":"xA","err_mean":"errA","n":"nA"})
    B = B[["domain","metric_bin","metric_mean","err_mean","n"]].rename(
        columns={"metric_mean":"xB","err_mean":"errB","n":"nB"})
    M = pd.merge(A, B, on=["domain","metric_bin"], how="inner")
    if M.empty:
        return pd.DataFrame(columns=["domain","omega","delta","nA","nB"])
    M["omega"] = 0.5*(M["xA"] + M["xB"])
    M["delta"] = M["errA"] - M["errB"]
    keep = ["domain","metric_bin","omega","delta","nA","nB"]
    return M[keep].dropna().sort_values(["domain","metric_bin"]).reset_index(drop=True)

# ---------- NEW: robust fit, Ω* and CIs ----------
def _robust_fit_and_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, random_state: int = 0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    slope, intercept = _theilsen(x, y)
    omega_star = np.nan
    if np.isfinite(slope) and slope != 0:
        omega_star = -intercept / slope
    rng = np.random.default_rng(random_state)
    boots = []
    if x.size >= 5:
        idx = np.arange(x.size)
        for _ in range(n_boot):
            b = rng.choice(idx, size=idx.size, replace=True)
            s, b0 = _theilsen(x[b], y[b])
            if np.isfinite(s):
                xstar = np.nan if s == 0 else (-b0 / s)
                boots.append((s, xstar))
    if boots:
        S = np.array([b[0] for b in boots], float)
        Xs = np.array([b[1] for b in boots], float)
        slo, shi = np.nanpercentile(S, [2.5, 97.5])
        xlo, xhi = np.nanpercentile(Xs, [2.5, 97.5])
    else:
        slo=shi=xlo=xhi=np.nan
    return dict(slope=slope, intercept=intercept,
                slope_lo=slo, slope_hi=shi,
                omega_star=omega_star, omega_star_lo=xlo, omega_star_hi=xhi)

# ---------- NEW: Δ vs Ω plot ----------
def plot_delta_vs_x(df_delta: pd.DataFrame,
                    xlabel: str,
                    title: str,
                    out_png: str,
                    fig_w: float, fig_h: float, dpi: int,
                    x_margin: float, y_margin: float,
                    tick_font: int,
                    tight_bbox: bool) -> Dict[str, float]:
    if df_delta.empty:
        print(f"[info] No delta data for {title}")
        return {}
    x = df_delta["omega"].to_numpy(float)
    y = df_delta["delta"].to_numpy(float)

    fit = _robust_fit_and_ci(x, y)
    rho = _spearman(x, y)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    ax.scatter(x, y, s=32, alpha=0.85, edgecolors="none", label="Δ per bin", zorder=2)
    if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ys = fit["intercept"] + fit["slope"]*xs
        ax.plot(xs, ys, linewidth=1.8, label=f"Theil–Sen fit, ρ={rho:.2f}", zorder=3)
        if np.isfinite(fit["omega_star"]):
            ax.axvline(fit["omega_star"], linestyle="--", linewidth=1.2, color="black",
                       label=f"Ω* ≈ {fit['omega_star']:.3g}", zorder=3)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel("Δ error (num − den)", labelpad=2)
    ax.set_title(title, pad=2)
    ax.legend(loc="best")

    # tighter bounds
    lo, hi = _data_span(x, pad_frac=x_margin)
    if "Ω" in xlabel or "Spectral predictability" in xlabel:
        lo = max(0.0, lo)
    ax.set_xlim(lo, hi)
    ylo, yhi = _data_span(y, pad_frac=y_margin)
    ax.set_ylim(ylo, yhi)

    ax.tick_params(axis="both", which="major", labelsize=tick_font)
    if tight_bbox:
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    else:
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_png}")

    fit["spearman_rho"] = rho
    return fit

# ---------- NEW: across-domain mixed-effects ----------
def mixed_effects_delta(df_all_deltas: pd.DataFrame) -> Dict[str, float]:
    if sm is None or df_all_deltas.empty:
        return {}
    D = df_all_deltas.copy()
    D = D[np.isfinite(D["omega"]) & np.isfinite(D["delta"])]
    if D.empty:
        return {}
    try:
        X = sm.add_constant(D["omega"])
        model = sm.MixedLM(D["delta"], X, groups=D["domain"])
        res = model.fit(reml=True, method="lbfgs", disp=False)
        slope = float(res.params.get("omega", np.nan))
        se    = float(res.bse.get("omega", np.nan))
        z = slope / se if (np.isfinite(slope) and se and np.isfinite(se)) else np.nan
        p = float(2.0*sm.stats.norm.sf(np.abs(z))) if np.isfinite(z) else np.nan
        return {"slope": slope, "se": se, "p": p}
    except Exception as e:
        print(f"[warn] MixedLM failed: {e}")
        return {}

# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", type=str, default=",".join(DEFAULT_LOG_DIRS),
                    help="Comma-separated domain directories (recursive scan for *.txt).")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                    help="Directory for per-domain sMAPE base PNGs and stats CSVs.")
    ap.add_argument("--mse-out-dir", type=str, default="./out_mse",
                    help="Directory for per-domain Omega-vs-MSE base PNGs.")
    ap.add_argument("--rel-out-dir", type=str, default="./out_rel",
                    help="Directory for per-domain relative-gain PNGs & CSVs (Omega only).")
    ap.add_argument("--x-metric", type=str, default="ALL",
                    choices=["ALL"] + list(X_METRIC_PATS.keys()),
                    help="Which context feature to use on x-axis. 'ALL' runs all.")
    ap.add_argument("--round", type=int, default=3,
                    help="Round x-metric to this many decimals for binning.")
    ap.add_argument("--jitter_x_frac", type=float, default=0.02,
                    help="Horizontal separation as a fraction of x-span (per plot).")

    # ---- New: paper/plotting controls
    ap.add_argument("--omega-label", type=str, default="both",
                    choices=["greek", "text", "both"],
                    help="Use 'Ω', 'Spectral predictability', or both in x label.")
    ap.add_argument("--fig-w", type=float, default=6.2, help="Figure width in inches.")
    ap.add_argument("--fig-h", type=float, default=4.0, help="Figure height in inches.")
    ap.add_argument("--dpi", type=int, default=600, help="Figure DPI.")
    ap.add_argument("--font", type=int, default=11, help="Base font size.")
    ap.add_argument("--tick-font", type=int, default=10, help="Tick label font size.")
    ap.add_argument("--legend-font", type=int, default=10, help="Legend font size.")
    ap.add_argument("--x-margin", type=float, default=0.06, help="Fractional x padding of data span.")
    ap.add_argument("--y-margin", type=float, default=0.06, help="Fractional y padding of data span.")
    ap.add_argument("--tight-bbox", action="store_true",
                    help="Use bbox_inches='tight' in savefig to shave borders further.")

    args = ap.parse_args()

    # rc params once
    apply_rc(font=args.font, tick_font=args.tick_font, legend_font=args.legend_font)

    log_dirs = [d.strip() for d in args.log_dirs.split(",") if d.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.mse_out_dir, exist_ok=True)
    os.makedirs(args.rel_out_dir, exist_ok=True)

    metrics_to_run = list(X_METRIC_PATS.keys()) if args.x_metric == "ALL" else [args.x_metric]

    for xm in metrics_to_run:
        df_all = load_all_dirs(log_dirs, xm)
        if df_all.empty:
            print(f"[info] No parsed runs for x-metric={xm}. Check paths/patterns.")
            continue

        if xm.lower() == "omega":
            before = len(df_all)
            df_all = df_all[df_all["metric"] >= 0].copy()
            dropped = before - len(df_all)
            if dropped > 0:
                print(f"[filter] Omega: dropped {dropped} rows with negative values")

        OUT = make_out_dirs(args.out_dir, xm)
        x_label = label_for_xmetric(xm, args.omega_label)

        # ---------- Per-domain sMAPE base plots + stats CSV ----------
        for domain, df_dom in df_all.groupby("domain", sort=False):
            agg = aggregate_by_model_metric_y(df_dom, ykey="smape", round_digits=args.round)
            png = os.path.join(OUT["base"], f"{domain}_sMAPE_vs_{xm}_by_model.png")
            plot_by_model_metric(
                df_dom, agg, x_label, "sMAPE",
                title=f"sMAPE vs {x_label} — {domain}",
                out_png=png, jitter_x_frac=args.jitter_x_frac,
                fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                x_margin=args.x_margin, y_margin=args.y_margin,
                tick_font=args.tick_font, legend_font=args.legend_font,
                tight_bbox=args.tight_bbox
            )
            stats_csv_for_domain(df_dom, xm, out_dir=OUT["stats"])

        # ---------- Omega-only: per-domain MSE base plots ----------
        if xm.lower() == "omega":
            for domain, df_dom in df_all.groupby("domain", sort=False):
                if "mse" not in df_dom.columns or not np.isfinite(df_dom["mse"]).any():
                    print(f"[info] {domain}: no usable MSE; skipping Omega-vs-MSE plot.")
                    continue
                agg_mse = aggregate_by_model_metric_y(df_dom, ykey="mse", round_digits=args.round)
                png = os.path.join(OUT["base_mse"], f"{domain}_MSE_vs_Omega_by_model.png")
                plot_by_model_metric(
                    df_dom, agg_mse, x_label, "MSE",
                    title=f"MSE vs {x_label} — {domain}",
                    out_png=png, jitter_x_frac=args.jitter_x_frac,
                    fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                    x_margin=args.x_margin, y_margin=args.y_margin,
                    tick_font=args.tick_font, legend_font=args.legend_font,
                    tight_bbox=args.tight_bbox
                )

        # ---------- per-domain relative-gain plots + CSVs ----------
        pairs = [
            ("DLinear", "Language Pretrained"),
            ("Language Pretrained", "Random Init"),
            ("Random Init", "DLinear"),
            ("GPT2", "DLinear"),  # NEW: show GPT2 vs DLinear
        ]
        metrics = [("smape", "sMAPE"), ("mse", "MSE")]
        for domain, df_dom in df_all.groupby("domain", sort=False):
            for (A, B) in pairs:
                for ykey, ylab in metrics:
                    if ykey not in df_dom.columns or not np.isfinite(df_dom[ykey]).any():
                        continue
                    P = _relative_gain_by_xbin(df_dom, A, B, ykey=ykey, round_digits=args.round)
                    csv_name = f"{domain}_RELGAIN_{A.replace(' ','')}_to_{B.replace(' ','')}_{ylab}.csv"
                    csv_path = os.path.join(OUT["base_mse"] if ylab == "MSE" else OUT["rel"], csv_name)
                    P.to_csv(csv_path, index=False)
                    png = os.path.join(OUT["base_mse"] if ylab == "MSE" else OUT["rel"],
                        f"{domain}_RELGAIN_{A.replace(' ','')}_to_{B.replace(' ','')}_vs_{xm}_{ylab}.png")
                    _plot_rel_gain_vs_x(
                        P,
                        title=f"{ylab} relative gain: {A} → {B} vs {x_label} — {domain}",
                        out_png=png, xlabel=x_label,
                        ylabel=f"Relative gain (%): {A} → {B}",
                        fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                        x_margin=args.x_margin, y_margin=args.y_margin,
                        tick_font=args.tick_font, tight_bbox=args.tight_bbox
                    )

        # ---------- Δ(Ω) plots + robust stats ----------
        delta_specs = [
            ("Language Pretrained", "DLinear", "LLM_minus_DLinear"),
            ("Random Init", "DLinear", "RandInit_minus_DLinear"),
            ("GPT2", "DLinear", "GPT2_minus_DLinear"),  # NEW
        ]
        all_deltas = []
        for (num, den, tag) in delta_specs:
            for domain, df_dom in df_all.groupby("domain", sort=False):
                D = _delta_pairs_unbinned(df_dom, num, den, ykey="smape")
                if D.empty:
                    continue
                all_deltas.append(D.assign(comp=tag))
                png = os.path.join(OUT["delta"], f"{domain}_DELTA_{tag}_vs_{xm}.png")
                fit = plot_delta_vs_x(
                    D, xlabel=x_label,
                    title=f"Δ sMAPE: {num} − {den} vs {x_label} — {domain}",
                    out_png=png,
                    fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                    x_margin=args.x_margin, y_margin=args.y_margin,
                    tick_font=args.tick_font, tight_bbox=args.tight_bbox
                )
                row = dict(domain=domain, comp=tag, x_metric=xm, n=len(D),
                           spearman_rho=fit.get("spearman_rho", np.nan),
                           theilsen_slope=fit.get("slope", np.nan),
                           slope_ci_lo=fit.get("slope_lo", np.nan),
                           slope_ci_hi=fit.get("slope_hi", np.nan),
                           omega_star=fit.get("omega_star", np.nan),
                           omega_star_lo=fit.get("omega_star_lo", np.nan),
                           omega_star_hi=fit.get("omega_star_hi", np.nan))
                pd.DataFrame([row]).to_csv(
                    os.path.join(OUT["stats"], f"{domain}_DELTA_{tag}_stats_{xm}.csv"),
                    index=False)

        if all_deltas:
            DF = pd.concat(all_deltas, ignore_index=True)
            mix = mixed_effects_delta(DF.rename(columns={"metric_bin":"bin"}))
            if mix:
                pd.DataFrame([dict(x_metric=xm, **mix)]).to_csv(
                    os.path.join(OUT["stats"], f"MIXEDEFFECTS_DELTA_{xm}.csv"), index=False)

        # ---------- Aggregate summaries ----------
        aggregate_summary_across_domains(df_all, xm, out_dir=OUT["stats"])

if __name__ == "__main__":
    main()
