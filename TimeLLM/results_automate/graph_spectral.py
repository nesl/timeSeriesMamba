#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sMAPE vs context metric (SE/LLE/Omega/SNR/Season) with seeds+init_seeds collapsed.

Updates:
  • DLinear markers smaller circles with clear error bars.
  • Per-domain figures + one combined "AllDomains" figure.
  • By default, loop through all x-metrics; optionally pick one with --x-metric.

Domain inference (filename keywords): unisynth → UniSynth, carbon → CarbonCast, fitbit → Fitbit, pems → PEMS.
DLinear detection: any filename containing "dlin" (case-insensitive).
"""

import os
import re
import sys
import argparse
from glob import glob
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ------------------- Plot styling -------------------
MODEL_COLOR = {
    "Language Pretrained": "tab:blue",
    "Random Init": "tab:orange",
    "DLinear": "tab:green",
}
MODEL_MARKER = {
    "Language Pretrained": "^",  # triangle
    "Random Init": "^",
    "DLinear": "o",              # circle
}
ALPHA_RAW = 0.28       # faint raw dots
ALPHA_MARKER = 0.78    # translucent mean markers & error bars

# ------------------- Metric keys -------------------
NUM_PAT = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
SMAPE_PATS = [re.compile(r"\bsmape(_loss)?\b"), re.compile(r"\bsym(_)?mape\b")]

X_METRIC_PATS = {
    "SE":     [re.compile(r"\bse_ctx_mean\b"), re.compile(r"\bspectral_entropy(_ctx)?_mean\b"), re.compile(r"\bse_mean\b")],
    "LLE":    [re.compile(r"\blle_ctx_mean\b")],
    "Omega":  [re.compile(r"\bomega_ctx_mean\b")],
    "SNR":    [re.compile(r"\bsnr_ctx_proxy\b")],
    "Season": [re.compile(r"\bseason_ctx_mean\b")],
}
X_METRIC_LABEL = {
    "SE": "SE",
    "LLE": "LLE",
    "Omega": "Omega (spectral predictivity)",
    "SNR": "SNR (proxy)",
    "Season": "Seasonality strength",
}

# ------------------- Helpers -------------------
import math

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 3:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)

def _fisher_ci(r: float, n: int, alpha: float = 0.05):
    """95% CI via Fisher z transform (approx), returns (low, high)."""
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / math.sqrt(max(n - 3, 1))
    z_crit = 1.959963984540054  # ~N(0,1) 97.5th percentile
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return (float(lo), float(hi))

def _ols_slope_intercept(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept) from simple OLS; NaN-safe, needs >=2 points."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 2:
        return (np.nan, np.nan)
    x_mean = x.mean(); y_mean = y.mean()
    sxx = np.sum((x - x_mean)**2)
    if sxx == 0.0:
        return (np.nan, y_mean)
    slope = np.sum((x - x_mean)*(y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    return (float(slope), float(intercept))

def _corr_per_domain(df: pd.DataFrame) -> dict:
    """
    Compute Pearson r per domain from RAW points (not binned).
    Returns dict[domain] = {n, r, r_lo, r_hi, slope, intercept}.
    """
    out = {}
    for domain, g in df.groupby("domain", sort=False):
        x = g["metric"].to_numpy(float)
        y = g["smape"].to_numpy(float)
        n = int(np.isfinite(x).sum() if np.isfinite(x).sum() == np.isfinite(y).sum()
                else np.sum(np.isfinite(x) & np.isfinite(y)))
        r = _pearson_r(x, y)
        r_lo, r_hi = _fisher_ci(r, n)
        slope, intercept = _ols_slope_intercept(x, y)
        out[domain] = {
            "n": n, "r": r, "r_lo": r_lo, "r_hi": r_hi,
            "slope": slope, "intercept": intercept,
        }
    return out

def _fisher_mean(rs: List[float], ns: List[int]) -> float:
    """
    Macro-average r across domains via Fisher z (unweighted by default).
    """
    vals = [r for r in rs if np.isfinite(r)]
    if not vals:
        return np.nan
    zs = [np.arctanh(np.clip(r, -0.999999, 0.999999)) for r in vals]
    z_mean = float(np.mean(zs))
    return float(np.tanh(z_mean))


def norm_key(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())

def parse_run_summary(lines: List[str]) -> Dict[str, Any]:
    """Collect numeric metrics after 'wandb: Run summary:'."""
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
    """LLAMA3.2_r0 → Language Pretrained; LLAMA3.2_r1 → Random Init; *dlin* → DLinear."""
    b = os.path.basename(path)
    b_up = b.upper()
    b_lo = b.lower()
    if "dlin" in b_lo:
        return "DLinear"
    if "LLAMA3.2" in b_up:
        if "_R0_" in b_up:
            return "Language Pretrained"
        if "_R1_" in b_up:
            return "Random Init"
        return None  # ambiguous LLAMA3.2 w/o r0/r1
    return None

def parse_seed_init(path: str) -> Dict[str, Optional[int]]:
    """Accept both: ..._seed3/_init11 and ..._s3/_i11."""
    b = os.path.basename(path)
    m_seed = re.search(r"(?:_seed|_s)(\d+)", b, re.IGNORECASE)
    m_init = re.search(r"(?:_init|_i)(\d+)", b, re.IGNORECASE)
    return {
        "seed": int(m_seed.group(1)) if m_seed else None,
        "init_seed": int(m_init.group(1)) if m_init else None,
    }

def make_base_key(path: str) -> str:
    """basename minus _r[01]_, seed/init tokens (for debugging only)."""
    b = os.path.basename(path)
    b = re.sub(r"_r[01]_", "_", b, flags=re.IGNORECASE)
    b = re.sub(r"_(?:seed|s)\d+", "", b, flags=re.IGNORECASE)
    b = re.sub(r"_(?:init|i)\d+", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\.txt$", "", b, flags=re.IGNORECASE)
    b = re.sub(r"__+", "_", b).strip("_")
    return b

def infer_domain_from_text(text: str) -> Optional[str]:
    """Map filename keywords to one of: UniSynth, CarbonCast, Fitbit, PEMS."""
    t = text.lower()
    if "unisynth" in t:
        return "UniSynth"
    if "carbon" in t:
        return "CarbonCast"
    if "fitbit" in t:
        return "Fitbit"
    if "pems" in t:
        return "PEMS"
    return None

def domain_name_from_dir(dir_path: str) -> str:
    base = os.path.basename(os.path.normpath(dir_path))
    return DOMAIN_NAME_FIX.get(base, base)

# ------------------- Load raw (across dirs), add per-row domain -------------------
def load_all_dirs(log_dirs: List[str], x_metric_name: str) -> pd.DataFrame:
    """Return raw runs across all dirs with inferred domain per row."""
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
                y = choose_metric(metrics, SMAPE_PATS)
                x = choose_metric(metrics, pats_x)
                if y is None or x is None:
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
                    "smape": float(y),
                    "mtime": os.path.getmtime(path),
                })
            except Exception as e:
                print(f"[warn] failed to parse {path}: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=["domain","log_path","model","base_key","seed","init_seed","metric","smape"])

    df = pd.DataFrame(rows).sort_values("mtime").drop_duplicates(subset=["log_path"], keep="last")
    df = df.drop(columns=["mtime"], errors="ignore")
    return df

# ------------------- Aggregate to one point per (model × metric_bin) -------------------
def aggregate_by_model_metric(df_raw: pd.DataFrame, round_digits: int = 3) -> pd.DataFrame:
    """
    Pool all raw runs across seeds *and* init_seeds within each (model, metric_bin).
    Error bars = range across the pooled samples (min..max).
    """
    if df_raw.empty:
        return pd.DataFrame(columns=[
            "model","metric_bin","n_raw","n_seeds","n_inits","n_base_keys",
            "metric_mean","smape_mean","smape_low","smape_high"
        ])

    df = df_raw.copy()
    
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
            "smape_mean": float(g["smape"].mean()),
            "smape_low": float(g["smape"].min()),
            "smape_high": float(g["smape"].max()),
        })
    return pd.DataFrame(rows).sort_values(["model","metric_bin"])

# ------------------- Plot -------------------
def plot_by_model_metric(df_raw: pd.DataFrame,
                         agg: pd.DataFrame,
                         x_metric_label: str,
                         title: str,
                         out_png: str,
                         jitter_x_frac: float = 0.08,
                         round_digits: int = 3) -> None:
    plt.figure(figsize=(9, 6))

    # faint raw dots by model
    if not df_raw.empty:
        for model, g in df_raw.groupby("model"):
            plt.scatter(
                g["metric"], g["smape"],
                marker=MODEL_MARKER.get(model, "^"),
                s=26, alpha=ALPHA_RAW, edgecolors="none",
                c=MODEL_COLOR.get(model, "tab:gray"),
                label="_nolegend_"
            )

    # compute proportional jitter (per-plot)
    if not df_raw.empty:
        x_vals = df_raw["metric"].to_numpy(float)
        x_span = float(np.nanmax(x_vals) - np.nanmin(x_vals)) if len(x_vals) else 0.0
        x_mean_abs = float(np.nanmean(np.abs(x_vals))) if len(x_vals) else 0.0
    else:
        x_span, x_mean_abs = 0.0, 0.0

    # Primary: fraction of span. If span ~0, use a tiny fallback relative to magnitude.
    jitter_abs = jitter_x_frac * x_span if x_span > 0 else (1e-3 * max(x_mean_abs, 1e-6))

    # small horizontal jitter so models don't perfectly overlap at same x
    def model_offset(m: str) -> float:
        if jitter_abs <= 0:
            return 0.0
        if m == "Language Pretrained":
            return -jitter_abs
        if m == "Random Init":
            return +jitter_abs
        # DLinear centered
        return 0.0

    used = set()
    for _, r in agg.iterrows():
        model = r["model"]
        color = MODEL_COLOR.get(model, "tab:gray")
        marker = MODEL_MARKER.get(model, "^")
        label = model if model not in used else "_nolegend_"
        used.add(model)
        x = r["metric_mean"] + model_offset(model)

        # range error bars (min..max)
        yerr = [[r["smape_mean"] - r["smape_low"]],
                [r["smape_high"] - r["smape_mean"]]]

        # DLinear circles smaller so bars remain visible
        msize = 7 if model == "DLinear" else 10

        plt.errorbar(
            x, r["smape_mean"],
            yerr=yerr, xerr=None,
            fmt=marker, linestyle="none", capsize=3,
            markersize=msize,
            alpha=ALPHA_MARKER, color=color, ecolor=color,
            markeredgewidth=1.0, markeredgecolor="black",
            elinewidth=1.5,
            zorder=3, label=label
        )

    plt.xlabel(x_metric_label)
    plt.ylabel("sMAPE")
    plt.title(title)
    if used:
        plt.legend(title="Model", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved {out_png}")


# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", type=str, default=",".join(DEFAULT_LOG_DIRS),
                    help="Comma-separated domain directories (recursive scan for *.txt).")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                    help="Directory to write PNGs.")
    ap.add_argument("--x-metric", type=str, default="ALL",
                    choices=["ALL"] + list(X_METRIC_PATS.keys()),
                    help="Which context feature to use on x-axis. 'ALL' runs all.")
    ap.add_argument("--round", type=int, default=3,
                    help="Round x-metric to this many decimals for binning.")
    ap.add_argument("--jitter_x_frac", type=float, default=0.02,
                    help="Horizontal separation as a fraction of x-span (per plot).")
    args = ap.parse_args()

    log_dirs = [d.strip() for d in args.log_dirs.split(",") if d.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    # Which x metrics to run
    metrics_to_run = list(X_METRIC_PATS.keys()) if args.x_metric == "ALL" else [args.x_metric]

    for xm in metrics_to_run:
        df_all = load_all_dirs(log_dirs, xm)
        if df_all.empty:
            print(f"[info] No parsed runs for x-metric={xm}. Check paths/patterns.")
            continue

        metric_label = X_METRIC_LABEL[xm]

        # ---------- Per-domain plots ----------
        for domain, df_dom in df_all.groupby("domain", sort=False):
            agg = aggregate_by_model_metric(df_dom, round_digits=args.round)
            png = os.path.join(args.out_dir, f"{domain}_sMAPE_vs_{xm}_by_model.png")
            plot_by_model_metric(
                df_dom, agg, metric_label,
                title=f"sMAPE vs {metric_label} — {domain}",
                out_png=png, jitter_x_frac=args.jitter_x_frac
            )

        # ---------- Combined (AllDomains) plot ----------
        agg_all = aggregate_by_model_metric(df_all, round_digits=args.round)
        png_all = os.path.join(args.out_dir, f"AllDomains_sMAPE_vs_{xm}_by_model.png")
        # Title shows how many domains were included
        doms = ", ".join(sorted(df_all["domain"].unique()))
        plot_by_model_metric(
            df_all, agg_all, metric_label,
            title=f"sMAPE vs {metric_label} — AllDomains [{doms}]",
            out_png=png_all, jitter_x_frac=args.jitter_x_frac
        )
            # ---------- Correlation stats per domain (exclude AllDomains row) ----------
        corr_map = _corr_per_domain(df_all)
        # Filter to only the four named domains you actually emit (ignore weird folder names)
        domains = [d for d in corr_map.keys() if d in ("CarbonCast","PEMS","Fitbit","UniSynth")]

        rows = []
        rs_for_avg, ns_for_avg = [], []
        for d in domains:
            c = corr_map[d]
            rows.append({
                "x_metric": xm,
                "domain": d,
                "n": c["n"],
                "pearson_r": c["r"],
                "r_95ci_low": c["r_lo"],
                "r_95ci_high": c["r_hi"],
                "ols_slope": c["slope"],
                "ols_intercept": c["intercept"],
            })
            if np.isfinite(c["r"]):
                rs_for_avg.append(c["r"])
                ns_for_avg.append(c["n"])

        macro_r = _fisher_mean(rs_for_avg, ns_for_avg)
        rows.append({
            "x_metric": xm,
            "domain": "MacroAvg(excl-AllDomains)",
            "n": int(np.nansum([corr_map[d]["n"] for d in domains])) if domains else 0,
            "pearson_r": macro_r,
            "r_95ci_low": np.nan,   # CI for macro-avg omitted; can add if needed
            "r_95ci_high": np.nan,
            "ols_slope": np.nan,
            "ols_intercept": np.nan,
        })

        corr_df = pd.DataFrame(rows)
        out_csv = os.path.join(args.out_dir, f"correlation_summary_{xm}.csv")
        corr_df.to_csv(out_csv, index=False)
        print(f"[stats] wrote {out_csv}")

if __name__ == "__main__":
    main()
