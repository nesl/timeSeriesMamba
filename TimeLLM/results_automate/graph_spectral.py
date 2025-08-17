#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sMAPE vs context metric (SE/LLE/Omega/SNR/Season) with seeds+init_seeds collapsed.

For each DOMAIN directory:
  • Parse *.txt logs for:
      - model from filename: LLAMA3.2_r0 → "Language Pretrained", LLAMA3.2_r1 → "Random Init", "DLinear" → "DLinear"
      - seed, init_seed from filename
      - sMAPE and chosen x-metric from the "wandb: Run summary:" block
  • Round the chosen metric to bins (default 3 decimals) and, for each (model, metric_bin),
    pool ALL raw runs (seeds × init_seeds) → one mean marker with **range** (min–max) error bar.
    => One blue point (n=3) and one orange point (n=9) per bin, when both exist.
  • Plot faint raw points; translucent markers and error bars; optional tiny x-jitter to avoid overlap.

Outputs (per domain only):
  <domain>_sMAPE_vs_<metric>_by_model.png
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
    "../results/spectralUniTest/",
    "../results/uniSynthPSD_eval/",
]
DEFAULT_OUT_DIR = "./out"
DOMAIN_NAME_FIX = {"spectralUniTest": "CarbonCast"}

# ------------------- Plot styling -------------------
MARKER = "^"
MODEL_COLOR = {
    "Language Pretrained": "tab:blue",
    "Random Init": "tab:orange",
    "DLinear": "tab:green",
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

# ------------------- Helpers -------------------
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
    u = os.path.basename(path).upper()
    if "DLINEAR" in u:
        return "DLinear"
    if "LLAMA3.2" in u:
        if "_R0_" in u:
            return "Language Pretrained"
        if "_R1_" in u:
            return "Random Init"
        return None  # ambiguous LLAMA3.2 w/o r0/r1
    return None

def parse_seed_init(path: str) -> Dict[str, Optional[int]]:
    b = os.path.basename(path)
    seed = re.search(r"_seed(\d+)", b)
    init = re.search(r"_init(\d+)", b)
    return {
        "seed": int(seed.group(1)) if seed else None,
        "init_seed": int(init.group(1)) if init else None,
    }

def make_base_key(path: str) -> str:
    """basename minus _r[01]_, _seed\d+, _init\d+ (for debugging only)."""
    b = os.path.basename(path)
    b = re.sub(r"_r[01]_", "_", b)
    b = re.sub(r"_seed\d+", "", b)
    b = re.sub(r"_init\d+", "", b)
    b = re.sub(r"\.txt$", "", b)
    b = re.sub(r"__+", "_", b).strip("_")
    return b

def domain_name_from_dir(dir_path: str) -> str:
    base = os.path.basename(os.path.normpath(dir_path))
    return DOMAIN_NAME_FIX.get(base, base)

# ------------------- Load raw -------------------
def load_domain(dir_path: str, x_metric_name: str) -> pd.DataFrame:
    """Return raw runs: [log_path, model, base_key, seed, init_seed, metric, smape]."""
    pats_x = X_METRIC_PATS[x_metric_name]
    files = sorted(glob(os.path.join(dir_path, "**", "*.txt"), recursive=True))
    rows = []
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
            rows.append({
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
        return pd.DataFrame(columns=["log_path","model","base_key","seed","init_seed","metric","smape"])

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
        n = len(g)  # LP ~3, RI ~9
        m_mean = float(g["metric"].mean())
        smape_mean = float(g["smape"].mean())
        smape_min = float(g["smape"].min())
        smape_max = float(g["smape"].max())
        rows.append({
            "model": model,
            "metric_bin": float(mbin),
            "n_raw": int(n),
            "n_seeds": int(g["seed"].nunique()),
            "n_inits": int(g["init_seed"].nunique()),
            "n_base_keys": int(g["base_key"].nunique()),
            "metric_mean": m_mean,
            "smape_mean": smape_mean,
            "smape_low": smape_min,
            "smape_high": smape_max,
        })
    return pd.DataFrame(rows).sort_values(["model","metric_bin"])

# ------------------- Plot -------------------
def plot_by_model_metric(df_raw: pd.DataFrame,
                         agg: pd.DataFrame,
                         x_metric_label: str,
                         title: str,
                         out_png: str,
                         jitter_x: float = 0.0) -> None:
    plt.figure(figsize=(9, 6))

    # faint raw dots
    if not df_raw.empty:
        for model, g in df_raw.groupby("model"):
            plt.scatter(
                g["metric"], g["smape"],
                marker=MARKER, s=26, alpha=ALPHA_RAW, edgecolors="none",
                c=MODEL_COLOR.get(model, "tab:gray"), label="_nolegend_"
            )

    # small horizontal jitter so models don't perfectly overlap at same x
    def model_offset(m: str) -> float:
        if jitter_x <= 0:
            return 0.0
        if m == "Language Pretrained":
            return -jitter_x
        if m == "Random Init":
            return +jitter_x
        return 0.0

    used = set()
    for _, r in agg.iterrows():
        model = r["model"]; color = MODEL_COLOR.get(model, "tab:gray")
        label = model if model not in used else "_nolegend_"; used.add(model)
        x = r["metric_mean"] + model_offset(model)

        # range error bars (min..max)
        yerr = [[r["smape_mean"] - r["smape_low"]],
                [r["smape_high"] - r["smape_mean"]]]

        plt.errorbar(
            x, r["smape_mean"],
            yerr=yerr, xerr=None,
            fmt=MARKER, linestyle="none", capsize=3, markersize=10,
            alpha=ALPHA_MARKER, color=color, ecolor=color,
            markeredgewidth=1.0, markeredgecolor="black",
            elinewidth=1.2, zorder=3, label=label
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
    ap.add_argument("--x-metric", type=str, default="SE",
                    choices=list(X_METRIC_PATS.keys()),
                    help="Which context feature to use on x-axis.")
    ap.add_argument("--round", type=int, default=3,
                    help="Round x-metric to this many decimals for binning.")
    ap.add_argument("--jitter-x", type=float, default=0.002,
                    help="Small horizontal offset applied by model to reduce overlap (0 to disable).")
    args = ap.parse_args()

    log_dirs = [d.strip() for d in args.log_dirs.split(",") if d.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    for d in log_dirs:
        domain = domain_name_from_dir(d)
        df_raw = load_domain(d, args.x_metric)
        # Aggregate to one point per model × metric_bin
        agg = aggregate_by_model_metric(df_raw, round_digits=args.round)
        # Plot
        metric_label = {
            "SE": "SE",
            "LLE": "LLE",
            "Omega": "Omega (spectral predictivity)",
            "SNR": "SNR (proxy)",
            "Season": "Seasonality strength",
        }[args.x_metric]
        png = os.path.join(args.out_dir, f"{domain}_sMAPE_vs_{args.x_metric}_by_model.png")
        plot_by_model_metric(
            df_raw, agg, metric_label,
            title=f"sMAPE vs {metric_label} — {domain}",
            out_png=png, jitter_x=args.jitter_x
        )

if __name__ == "__main__":
    main()
