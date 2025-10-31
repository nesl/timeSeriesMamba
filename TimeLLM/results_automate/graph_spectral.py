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
  • >>> NEW: Thick, always-visible error bars with tunable width/caps and minimum length
  • >>> NEW: Bootstrap CIs for RelGain vs x and error bars on those points
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
import matplotlib.ticker as mticker

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
    "uniSynth_eval": "Synthetic",
}

DOMAINS_CANON = ("CarbonCast", "PEMS", "Fitbit", "Synthetic")

# ------------------- Plot styling -------------------
MODEL_COLOR = {
    "Language Pretrained": "tab:green",
    "Random Init": "tab:brown",
    "DLinear": "tab:cyan",
    "GPT2": "tab:purple",  # same family color as Language Pretrained
}

MODEL_MARKER = {
    "Language Pretrained": "v",
    "Random Init": "^",
    "DLinear": "X",
    "GPT2": "o",  
}
ALPHA_RAW = 0.28
ALPHA_MARKER = 0.9

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
def _save_multi(fig, out_path_png: str, tight_bbox: bool, extra_exts=("pdf", "svg")):
    base, _ = os.path.splitext(out_path_png)
    if tight_bbox:
        fig.savefig(out_path_png, bbox_inches="tight", dpi=plt.rcParams.get("savefig.dpi", 600))
    else:
        fig.savefig(out_path_png, dpi=plt.rcParams.get("savefig.dpi", 600))
    for ext in extra_exts:
        out_vec = f"{base}.{ext}"
        if tight_bbox:
            fig.savefig(out_vec, bbox_inches="tight")
        else:
            fig.savefig(out_vec)
    print(f"Saved {out_path_png} and {[f'{base}.{e}' for e in extra_exts]}")

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
    if "dlin" in b_lo:
        return "DLinear"
    if "mgpt2" in b_lo or re.search(r"\bgpt2\b", b_lo):
        return "GPT2"
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
        return "Synthetic"
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

# ---------- Plot rc ----------
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
        "figure.autolayout": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # >>> NEW: make default errorbar caps visible if user forgets flags
        "errorbar.capsize": 0.0,
    })

def label_for_omega(mode: str) -> str:
    mode = mode.lower()
    if mode == "greek": return "Ω"
    if mode == "text":  return "Spectral predictability"
    return "Spectral predictability (Ω)"

def label_for_xmetric(xm: str, omega_mode: str) -> str:
    if xm.lower() == "omega":
        return label_for_omega(omega_mode)
    if xm == "SE":      return "SE"
    if xm == "LLE":     return "LLE"
    if xm == "SNR":     return "SNR (proxy)"
    if xm == "Season":  return "Seasonality strength"
    return xm

# ------------------- Load raw -------------------
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
                })
            except Exception as e:
                print(f"[warn] failed to parse {path}: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=[
            "domain","log_path","model","base_key","seed","init_seed","metric","smape","mse"
        ])

    df = pd.DataFrame(rows).drop_duplicates(subset=["log_path"], keep="last")
    return df

# ------------------- Aggregation -------------------
def aggregate_by_model_metric_y(df_raw: pd.DataFrame,
                                ykey: str,
                                round_digits: int = 3,
                                ci_type: str = "sem",
                                ci_level: float = 0.95,
                                ci_group: str = "base_key",
                                ci_bootstrap_B: int = 2000,
                                min_bin_n: int = 2) -> pd.DataFrame:
    """
    Aggregate y vs x-bin per model and compute 95% CIs.
    ci_group controls independence: 'base_key' | 'seed' | 'init' | 'raw'
    """
    if df_raw.empty or ykey not in df_raw.columns:
        return pd.DataFrame(columns=[
            "model","metric_bin","n_raw","n_repl","n_seeds","n_inits","n_base_keys",
            "metric_mean","y_mean","y_low","y_high","ci_low","ci_high"
        ])
    df = df_raw.copy()
    df = df[np.isfinite(df["metric"]) & np.isfinite(df[ykey])]
    if df.empty:
        return pd.DataFrame(columns=[
            "model","metric_bin","n_raw","n_repl","n_seeds","n_inits","n_base_keys",
            "metric_mean","y_mean","y_low","y_high","ci_low","ci_high"
        ])

    df["metric_bin"] = df["metric"].round(round_digits)

    if ci_group == "base_key":
        repl_key = "base_key"
    elif ci_group == "seed":
        repl_key = "seed"
    elif ci_group == "init":
        repl_key = "init_seed"
    else:
        repl_key = None  # 'raw'

    rows = []
    for (model, mbin), g in df.groupby(["model","metric_bin"]):
        n_raw = int(len(g))
        n_seeds = int(g["seed"].nunique())
        n_inits = int(g["init_seed"].nunique())
        n_bk = int(g["base_key"].nunique())
        xbar = float(g["metric"].mean())

        if repl_key is None:
            vals = g[ykey].to_numpy(float)
        else:
            vals = (g.groupby(repl_key, dropna=False)[ykey].mean().to_numpy(float))
        vals = vals[np.isfinite(vals)]
        n_repl = int(vals.size)

        y_mean = float(np.mean(vals)) if n_repl else np.nan
        y_low  = float(np.nanmin(vals)) if n_repl else np.nan
        y_high = float(np.nanmax(vals)) if n_repl else np.nan

        if n_repl >= min_bin_n:
            if ci_type == "sem":
                s = float(np.std(vals, ddof=1)) if n_repl >= 2 else np.nan
                ci_lo, ci_hi = _sem_ci(y_mean, s, n_repl, level=ci_level)
            elif ci_type == "bootstrap":
                ci_lo, ci_hi = _bootstrap_ci(vals, level=ci_level, B=ci_bootstrap_B)
            else:
                ci_lo, ci_hi = (np.nan, np.nan)
        else:
            ci_lo, ci_hi = (np.nan, np.nan)

        rows.append({
            "model": model,
            "metric_bin": float(mbin),
            "n_raw": n_raw,
            "n_repl": n_repl,
            "n_seeds": n_seeds,
            "n_inits": n_inits,
            "n_base_keys": n_bk,
            "metric_mean": xbar,
            "y_mean": y_mean,
            "y_low": y_low,
            "y_high": y_high,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        })
    return pd.DataFrame(rows).sort_values(["model","metric_bin"])

# ---------- CIs ----------
def _sem_ci(mean: float, s: float, n: int, level: float = 0.95):
    if n is None or n < 2 or not np.isfinite(s):
        return (np.nan, np.nan)
    z = 1.959963984540054 if abs(level - 0.95) < 1e-9 else (ss.norm.ppf(0.5 + level/2.0) if ss else 1.96)
    half = z * (s / math.sqrt(n))
    return (float(mean - half), float(mean + half))

def _bootstrap_ci(vals: np.ndarray, level: float = 0.95, B: int = 2000, rng: Optional[np.random.Generator] = None):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if v.size < 2:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    idx = np.arange(v.size)
    boots = []
    for _ in range(B):
        b = rng.choice(idx, size=idx.size, replace=True)
        boots.append(np.mean(v[b]))
    lo = float(np.nanpercentile(boots, (1.0-level)*50))
    hi = float(np.nanpercentile(boots, 100 - (1.0-level)*50))
    return (lo, hi)

# ------------------- Plot (base) -------------------
def _data_span(a: np.ndarray, pad_frac: float = 0.02) -> Tuple[float, float]:
    if a.size == 0 or not np.isfinite(a).any():
        return (0.0, 1.0)
    lo = float(np.nanmin(a)); hi = float(np.nanmax(a))
    span = max(hi - lo, 1e-9)
    pad = pad_frac * span
    return (lo - pad, hi + pad)

# >>> NEW: helper to enforce a minimum visible errorbar length
def _ensure_visible_yerr(ymean: float,
                         lo: float, hi: float,
                         y_data_span: float,
                         min_frac: float) -> Tuple[float,float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = ymean; hi = ymean
    lower = ymean - lo
    upper = hi - ymean
    if not np.isfinite(lower): lower = 0.0
    if not np.isfinite(upper): upper = 0.0
    # If both are tiny/zero, inflate symmetrically to a small fraction of the data span
    eps = max(min_frac * max(y_data_span, 1e-12), 0.0)
    if (abs(lower) < 1e-12) and (abs(upper) < 1e-12) and eps > 0:
        lower = upper = eps
    return lower, upper

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
                         tight_bbox: bool,
                         show_legend: bool = False,
                         # visible errorbar styling
                         err_eline: float = 2.75,
                         err_cap: float = 4.0,
                         err_capthick: float = 2.0,
                         err_min_frac: float = 0.0) -> None:
    """
    Clean version for paper:
      - NO faded per-run scatter
      - ONLY jittered aggregate markers with error bars that summarize spread.
    """
    if agg.empty:
        print(f"[info] No data for {title}")
        return

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")

    # Which column is our y?
    ycol = "mse" if y_label.lower() == "mse" else "smape"

    # We'll still compute jitter scale + span limits from the raw df
    x_vals = df_raw["metric"].to_numpy(float); x_vals = x_vals[np.isfinite(x_vals)]
    y_vals_all = df_raw[ycol].to_numpy(float); y_vals_all = y_vals_all[np.isfinite(y_vals_all)]

    x_span = float(np.nanmax(x_vals) - np.nanmin(x_vals)) if x_vals.size else 0.0
    x_mean_abs = float(np.nanmean(np.abs(x_vals))) if x_vals.size else 0.0
    jitter_abs = jitter_x_frac * x_span if x_span > 0 else (1e-3 * max(x_mean_abs, 1e-6))

    def model_offset(m: str) -> float:
        if jitter_abs <= 0:
            return 0.0
        if m == "Language Pretrained": return -jitter_abs
        if m == "Random Init":         return +jitter_abs
        if m == "GPT2":                return +2.0 * jitter_abs
        if m == "DLinear":             return 0.0
        return 0.0

    # span for enforcing a visible stub when n=1
    y_span_for_min = float(np.nanmax(y_vals_all) - np.nanmin(y_vals_all)) if y_vals_all.size else 1.0

    used = set()
    for _, r in agg.iterrows():
        model = r["model"]
        color = MODEL_COLOR.get(model, "tab:gray")
        marker = MODEL_MARKER.get(model, "^")
        label = model if model not in used else "_nolegend_"
        used.add(model)

        # x position (Ω etc.) + horizontal jitter so models don't overlap
        x = r["metric_mean"] + model_offset(model)

        # mean error for this model at this Ω
        ymean = r["y_mean"]

        # spread: prefer CI if present, otherwise min/max
        lo = r.get("ci_low", np.nan); hi = r.get("ci_high", np.nan)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo = r.get("y_low", np.nan); hi = r.get("y_high", np.nan)

        # ensure bar is visible even if n=1 (lo==hi==ymean)
        lower, upper = _ensure_visible_yerr(ymean, lo, hi, y_span_for_min, err_min_frac)
        yerr = [[lower], [upper]]

        msize = 10

        ax.errorbar(
            x, ymean, yerr=yerr, xerr=None,
            fmt=marker, linestyle="none",
            capsize=err_cap, capthick=err_capthick,
            markersize=msize, alpha=ALPHA_MARKER,
            color=color, ecolor=color,
            markeredgewidth=0.9, markeredgecolor="black",
            elinewidth=err_eline, zorder=3,
            label=label
        )

    # axes / labels / title
    ax.set_xlabel(x_metric_label, labelpad=2,fontweight='bold')
    ax.set_ylabel(y_label, labelpad=2,fontweight='bold')
    ax.set_title(title, pad=2,fontweight='bold')

    # nice limits
    if x_vals.size:
        lo, hi = _data_span(x_vals, pad_frac=x_margin)
        if "Ω" in x_metric_label or "Spectral predictability" in x_metric_label:
            lo = max(0.0, lo)  # Ω shouldn't go <0 visually
        ax.set_xlim(lo, hi)

    if y_vals_all.size:
        ylo, yhi = _data_span(y_vals_all, pad_frac=y_margin)
        ax.set_ylim(ylo, yhi)

    # ticks / grid / legend
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.tick_params(axis="both", which="major", labelsize=tick_font)
    ax.margins(x=0, y=0)

    if show_legend:
        ax.legend(loc="best", fontsize=legend_font, frameon=False)

    _save_multi(fig, out_png, tight_bbox, extra_exts=("pdf", "svg"))
    plt.close(fig)

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

# -------- Aggregate helpers (unchanged) --------
def _fisher_z(r): return np.arctanh(np.clip(r, -0.999999, 0.999999))
def _inv_fisher_z(z): return float(np.tanh(z))
# ... (meta-analysis helpers unchanged for brevity) ...

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

# >>> NEW: RelGain with bootstrap CIs from raw rows per bin
def _relative_gain_by_xbin_with_ci(df_dom: pd.DataFrame,
                                   model_a: str,
                                   model_b: str,
                                   ykey: str,
                                   round_digits: int,
                                   ci_level: float = 0.95,
                                   B: int = 2000,
                                   ci_group: str = "base_key") -> pd.DataFrame:
    """
    Builds per-bin bootstrap CIs of RelGain = 100*(A-B)/A using replicate means
    defined by ci_group within each (domain, bin, model).
    """
    if ykey not in df_dom.columns:
        return pd.DataFrame(columns=["domain","metric_bin","omega","rel_gain_pct","ci_low","ci_high","nA","nB"])
    D = df_dom.copy()
    D = D[np.isfinite(D["metric"]) & np.isfinite(D[ykey])]
    if D.empty: 
        return pd.DataFrame(columns=["domain","metric_bin","omega","rel_gain_pct","ci_low","ci_high","nA","nB"])
    D["metric_bin"] = D["metric"].round(round_digits)

    # pick replicate key
    if ci_group == "base_key":
        rk = "base_key"
    elif ci_group == "seed":
        rk = "seed"
    elif ci_group == "init":
        rk = "init_seed"
    else:
        rk = None  # 'raw'

    out = []
    rng = np.random.default_rng(0)
    zlo = (1.0-ci_level)*50.0; zhi = 100 - zlo

    for mbin, Gbin in D.groupby("metric_bin"):
        # A/B replicate means in this bin
        def repl_means(model):
            g = Gbin[Gbin["model"] == model]
            if g.empty: 
                return np.array([], float)
            if rk is None:
                vals = g[ykey].to_numpy(float)
            else:
                vals = g.groupby(rk, dropna=False)[ykey].mean().to_numpy(float)
            return vals[np.isfinite(vals)]

        A = repl_means(model_a)
        Bv = repl_means(model_b)
        if A.size == 0 or Bv.size == 0:
            continue

        # point estimate using mean of replicates
        muA = float(np.mean(A)); muB = float(np.mean(Bv))
        if not np.isfinite(muA) or muA == 0:
            continue
        point = 100.0 * (muA - muB) / muA

        # bootstrap over replicate means (not raw rows)
        ci_lo = np.nan; ci_hi = np.nan
        if A.size >= 2 or Bv.size >= 2:
            NA, NB = A.size, Bv.size
            ia = np.arange(NA); ib = np.arange(NB)
            boots = []
            for _ in range(B):
                sa = rng.choice(ia, size=NA, replace=True)
                sb = rng.choice(ib, size=NB, replace=True)
                a = float(np.mean(A[sa])); b = float(np.mean(Bv[sb]))
                if np.isfinite(a) and a != 0:
                    boots.append(100.0 * (a - b) / a)
            if boots:
                ci_lo = float(np.nanpercentile(boots, zlo))
                ci_hi = float(np.nanpercentile(boots, zhi))

        omega = float(Gbin["metric"].mean())
        out.append(dict(domain=df_dom["domain"].iloc[0],
                        metric_bin=float(mbin),
                        omega=omega,
                        rel_gain_pct=point,
                        ci_low=ci_lo, ci_high=ci_hi,
                        nA=int(A.size), nB=int(Bv.size)))
    return pd.DataFrame(out).sort_values("metric_bin")

def _plot_rel_gain_vs_x(df_pairs: pd.DataFrame,
                        title: str,
                        out_png: str,
                        xlabel: str,
                        ylabel: str,
                        fig_w: float, fig_h: float, dpi: int,
                        x_margin: float, y_margin: float,
                        tick_font: int,
                        tight_bbox: bool,
                        # >>> NEW errorbar knobs:
                        err_eline: float = 2.75,
                        err_cap: float = 4.0,
                        err_capthick: float = 2.0,
                        err_min_frac: float = 0.0) -> None:
    if df_pairs.empty:
        print(f"[info] No matched x-bins for {title}")
        return
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")

    xv = df_pairs["omega"].to_numpy(float)
    yv = df_pairs["rel_gain_pct"].to_numpy(float)
    y_span_for_min = float(np.nanmax(yv) - np.nanmin(yv)) if yv.size else 1.0

    # Draw each point with its CI if present
    for _, r in df_pairs.iterrows():
        x = float(r["omega"]); y = float(r["rel_gain_pct"])
        lo = r.get("ci_low", np.nan); hi = r.get("ci_high", np.nan)
        lower, upper = _ensure_visible_yerr(y, lo, hi, y_span_for_min, err_min_frac)
        ax.errorbar(
            x, y, yerr=[[lower],[upper]], fmt="o", linestyle="none",
            elinewidth=err_eline, capsize=err_cap, capthick=err_capthick,
            markersize=6, alpha=0.9, color="tab:purple", ecolor="tab:purple",
            markeredgecolor="black", markeredgewidth=0.9, zorder=3
        )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="No gain")
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, pad=2)

    ax.minorticks_off()
    ax.grid(False, which="minor")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    if xv.size:
        lo, hi = _data_span(xv, pad_frac=x_margin)
        if "Ω" in xlabel or "Spectral predictability" in xlabel:
            lo = max(0.0, lo)
        ax.set_xlim(lo, hi)
    if yv.size:
        ylo, yhi = _data_span(yv, pad_frac=y_margin)
        ax.set_ylim(ylo, yhi)

    ax.tick_params(axis="both", which="major", labelsize=tick_font)
    _save_multi(fig, out_png, tight_bbox, extra_exts=("pdf", "svg"))
    plt.close(fig)

# ------------------- (rest of your stats / tables code unchanged) -------------------
# ... keep your build_table_mse_vs_omega, build_table_error_increase_vs_omega, etc. ...

def make_out_dirs(base_dir: str, xm: str) -> Dict[str, str]:
    root = os.path.join(base_dir, xm)
    sub = {
        "base": os.path.join(root, "base"),
        "base_mse": os.path.join(root, "base_mse"),
        "rel": os.path.join(root, "rel"),
        "delta": os.path.join(root, "delta"),
        "stats": os.path.join(root, "stats"),
        "tables": os.path.join(root, "tables"),
    }
    for p in sub.values():
        os.makedirs(p, exist_ok=True)
    return sub

# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", type=str, default=",".join(DEFAULT_LOG_DIRS))
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--mse-out-dir", type=str, default="./out_mse")
    ap.add_argument("--rel-out-dir", type=str, default="./out_rel")
    ap.add_argument("--x-metric", type=str, default="ALL",
                    choices=["ALL"] + list(X_METRIC_PATS.keys()))
    ap.add_argument("--round", type=int, default=3)
    ap.add_argument("--jitter_x_frac", type=float, default=0.02)

    # Plot controls
    ap.add_argument("--omega-label", type=str, default="both",
                    choices=["greek", "text", "both"])
    ap.add_argument("--fig-w", type=float, default=5.2)
    ap.add_argument("--fig-h", type=float, default=3.5)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--font", type=int, default=19)
    ap.add_argument("--tick-font", type=int, default=17)
    ap.add_argument("--legend-font", type=int, default=12)
    ap.add_argument("--x-margin", type=float, default=0.06)
    ap.add_argument("--y-margin", type=float, default=0.06)
    ap.add_argument("--tight-bbox", action="store_true")

    # CI controls for base plots
    ap.add_argument("--ci", type=str, default="sem",
                    choices=["none", "sem", "bootstrap"])
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--ci-group", type=str, default="raw",
                    choices=["base_key", "seed", "init", "raw"])
    ap.add_argument("--ci-bootstrap-B", type=int, default=2000)
    ap.add_argument("--min-bin-n", type=int, default=2)

    # >>> NEW: visible errorbar styling (both base & relgain)
    ap.add_argument("--err-eline", type=float, default=3.75, help="Errorbar line width.")
    ap.add_argument("--err-cap", type=float, default=6.0, help="Errorbar cap size (points).")
    ap.add_argument("--err-capthick", type=float, default=4.0, help="Errorbar cap thickness.")
    ap.add_argument("--err-min-frac", type=float, default=0,
                    help="If CI collapses to zero, draw a min symmetric bar equal to this fraction of y-span.")

    # >>> NEW: RelGain bootstrap CI controls
    ap.add_argument("--rel-ci", type=str, default="bootstrap", choices=["bootstrap", "none"],
                    help="How to compute CI for RelGain per bin.")
    ap.add_argument("--rel-ci-level", type=float, default=0.95)
    ap.add_argument("--rel-ci-B", type=int, default=2000)
    ap.add_argument("--rel-ci-group", type=str, default="base_key",
                    choices=["base_key", "seed", "init", "raw"],
                    help="Replicate unit for RelGain bootstrap.")

    args = ap.parse_args()
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
            agg = aggregate_by_model_metric_y(
                df_dom, ykey="smape", round_digits=args.round,
                ci_type=args.ci, ci_level=args.ci_level,
                ci_group=args.ci_group, ci_bootstrap_B=args.ci_bootstrap_B,
                min_bin_n=args.min_bin_n
            )

            domainTitle = ""
            if domain == "CarbonCast":
                domainTitle = "CarbonCast (Energy)"
            elif domain == "PEMS":
                domainTitle = "PEMS (Traffic)"
            elif domain == "Fitbit":
                domainTitle = "Fitbit (Health)"
            else:
                domainTitle = "Synthetic"


            png = os.path.join(OUT["base"], f"{domain}_sMAPE_vs_{xm}_by_model.png")
            plot_by_model_metric(
                df_dom, agg, x_label, "sMAPE",
                title=f"sMAPE: {domainTitle}",
                out_png=png, jitter_x_frac=args.jitter_x_frac,
                fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                x_margin=args.x_margin, y_margin=args.y_margin,
                tick_font=args.tick_font, legend_font=args.legend_font,
                tight_bbox=args.tight_bbox,
                show_legend=(domain == "Synthetic"),
                err_eline=args.err_eline, err_cap=args.err_cap,
                err_capthick=args.err_capthick, err_min_frac=args.err_min_frac
            )
            stats_csv_for_domain(df_dom, xm, out_dir=OUT["stats"])

        # ---------- Omega-only: per-domain MSE base plots ----------
        if xm.lower() == "omega":
            for domain, df_dom in df_all.groupby("domain", sort=False):
                if "mse" not in df_dom.columns or not np.isfinite(df_dom["mse"]).any():
                    print(f"[info] {domain}: no usable MSE; skipping Omega-vs-MSE plot.")
                    continue
                agg_mse = aggregate_by_model_metric_y(
                    df_dom, ykey="mse", round_digits=args.round,
                    ci_type=args.ci, ci_level=args.ci_level,
                    ci_group=args.ci_group, ci_bootstrap_B=args.ci_bootstrap_B,
                    min_bin_n=args.min_bin_n
                )
                png = os.path.join(OUT["base_mse"], f"{domain}_MSE_vs_Omega_by_model.png")
                plot_by_model_metric(
                    df_dom, agg_mse, x_label, "MSE",
                    title=f"MSE Error: {domain}",
                    out_png=png, jitter_x_frac=args.jitter_x_frac,
                    fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                    x_margin=args.x_margin, y_margin=args.y_margin,
                    tick_font=args.tick_font, legend_font=args.legend_font,
                    tight_bbox=args.tight_bbox,
                    err_eline=args.err_eline, err_cap=args.err_cap,
                    err_capthick=args.err_capthick, err_min_frac=args.err_min_frac
                )

        # ---------- per-domain relative-gain plots + CSVs ----------
        pairs = [
            ("Language Pretrained","DLinear"),
            ("Language Pretrained","Random Init"),
            ("Language Pretrained","GPT2"),
        ]
        metrics = [("smape", "sMAPE"), ("mse", "MSE")]
        for domain, df_dom in df_all.groupby("domain", sort=False):
            for (A, B) in pairs:
                for ykey, ylab in metrics:
                    if ykey not in df_dom.columns or not np.isfinite(df_dom[ykey]).any():
                        continue

                    if args.rel_ci == "bootstrap":
                        P = _relative_gain_by_xbin_with_ci(
                            df_dom, A, B, ykey=ykey, round_digits=args.round,
                            ci_level=args.rel_ci_level, B=args.rel_ci_B,
                            ci_group=args.rel_ci_group
                        )
                    else:
                        P = _relative_gain_by_xbin(df_dom, A, B, ykey=ykey, round_digits=args.round)

                    csv_name = f"{domain}_RELGAIN_{A.replace(' ','')}_to_{B.replace(' ','')}_{ylab}.csv"
                    csv_path = os.path.join(OUT["base_mse"] if ylab == "MSE" else OUT["rel"], csv_name)
                    P.to_csv(csv_path, index=False)

                    png = os.path.join(OUT["base_mse"] if ylab == "MSE" else OUT["rel"],
                        f"{domain}_RELGAIN_{A.replace(' ','')}_to_{B.replace(' ','')}_vs_{xm}_{ylab}.png")
                    _plot_rel_gain_vs_x(
                        P,
                        title=f"Error Increase ∆: {domain}",
                        out_png=png, xlabel=x_label,
                        ylabel=f"Error Increase ∆ (%)",
                        fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                        x_margin=args.x_margin, y_margin=args.y_margin,
                        tick_font=args.tick_font, tight_bbox=args.tight_bbox,
                        err_eline=args.err_eline, err_cap=args.err_cap,
                        err_capthick=args.err_capthick, err_min_frac=args.err_min_frac
                    )

        # ---------- Aggregate summaries ----------
        

if __name__ == "__main__":
    main()
