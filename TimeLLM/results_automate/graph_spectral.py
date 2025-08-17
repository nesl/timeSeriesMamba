#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate sMAPE vs SE by "setting" (basename minus _rX_, _seedY, _initZ), split r0/r1.
Two-stage aggregation:
  1) Within each (domain, model, base_key, seed): average across init_seeds.
  2) Across seeds: mean + 95% CI for (SE, sMAPE).

Models (from filename only):
  - contains "DLinear"  -> DLinear
  - contains "LLAMA3.2" + "_r0_" -> Language Pretrained
  - contains "LLAMA3.2" + "_r1_" -> Random Init
  - else -> skip

Domains are the top-level result directories you pass (each gets its own CSV/PNG),
plus an ALL combined.

Usage:
  python smape_se_hier.py --log-dirs "../results/pems_eval,../results/fitbit_eval,../results/spectralUniTest,../results/uniSynthPSD_eval" --out-dir ./out
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

# optional: exact t critical; fallback to normal if SciPy isn't available
try:
    from scipy.stats import t as _t_dist  # type: ignore
except Exception:  # noqa: BLE001
    _t_dist = None

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
ALPHA_RAW = 0.28

# ------------------- Parsing helpers -------------------
NUM_PAT = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
SMAPE_PATS = [re.compile(r"\bsmape(_loss)?\b"), re.compile(r"\bsym(_)?mape\b")]
SE_PATS = [re.compile(r"\bse_ctx_mean\b"), re.compile(r"\bspectral_entropy(_ctx)?_mean\b"), re.compile(r"\bse_mean\b")]

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

def choose_metric(row: Dict[str, Any], patterns: List[re.Pattern]) -> Optional[float]:
    for k, v in row.items():
        if not isinstance(v, (int, float, np.floating)):
            continue
        if not np.isfinite(v):
            continue
        if any(pat.search(norm_key(k)) for pat in patterns):
            return float(v)
    return None

def infer_model_and_r(path: str) -> Optional[str]:
    u = os.path.basename(path).upper()
    if "DLINEAR" in u:
        return "DLinear"
    if "LLAMA3.2" in u:
        if "_R0_" in u:
            return "Language Pretrained"
        if "_R1_" in u:
            return "Random Init"
        return None  # ambiguous LLAMA3.2 w/o r0/r1 -> skip
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
    """
    Base key = basename without: _r[01]_, _seed\d+, _init\d+
    This groups runs that are identical except for r/seed/init.
    """
    b = os.path.basename(path)
    b = re.sub(r"_r[01]_", "_", b)     # drop r flag
    b = re.sub(r"_seed\d+", "", b)     # drop seed
    b = re.sub(r"_init\d+", "", b)     # drop init
    b = re.sub(r"\.txt$", "", b)
    b = re.sub(r"__+", "_", b).strip("_")
    return b

def domain_name_from_dir(dir_path: str) -> str:
    base = os.path.basename(os.path.normpath(dir_path))
    return DOMAIN_NAME_FIX.get(base, base)

def tcrit95(n: int) -> float:
    if n <= 1:
        return 0.0
    if _t_dist is not None:
        return float(_t_dist.ppf(0.975, n - 1))
    return 1.96  # normal approx

def ci_halfwidth(std: float, n: int) -> float:
    if not np.isfinite(std) or n <= 1:
        return 0.0
    return tcrit95(n) * (std / np.sqrt(n))

# ------------------- Data loading -------------------
def load_domain(dir_path: str) -> pd.DataFrame:
    """Return raw runs: [log_path, model, base_key, seed, init_seed, se, smape]."""
    files = sorted(glob(os.path.join(dir_path, "**", "*.txt"), recursive=True))
    rows = []
    for path in files:
        try:
            model = infer_model_and_r(path)
            if model is None:
                continue  # ignore models we don't care about or ambiguous LLAMA
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            lines = text.splitlines()
            metrics = parse_run_summary(lines)
            if not metrics:
                continue
            smape = choose_metric(metrics, SMAPE_PATS)
            se = choose_metric(metrics, SE_PATS)
            if smape is None or se is None:
                continue

            seeds = parse_seed_init(path)
            base_key = make_base_key(path)
            rows.append({
                "log_path": path,
                "model": model,
                "base_key": base_key,
                "seed": seeds["seed"],
                "init_seed": seeds["init_seed"],
                "se": float(se),
                "smape": float(smape),
                "mtime": os.path.getmtime(path),
            })
        except Exception as e:
            print(f"[warn] failed to parse {path}: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=["log_path","model","base_key","seed","init_seed","se","smape"])

    df = pd.DataFrame(rows)
    # keep newest duplicate log_path if any
    df = df.sort_values("mtime").drop_duplicates(subset=["log_path"], keep="last")
    df = df.drop(columns=["mtime"], errors="ignore")
    return df

# ------------------- Aggregation -------------------
def aggregate_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage:
      A) seed-level means (avg across init_seed) per (model, base_key, seed)
      B) across seeds: mean ± 95% CI per (model, base_key)
    """
    if df.empty:
        return df

    # A) per-seed means (handle missing seed by treating each as its own)
    df_seed = df.copy()
    df_seed["seed_filled"] = df_seed["seed"].fillna(-1).astype(int)
    seed_means = df_seed.groupby(["model","base_key","seed_filled"], as_index=False).agg(
        se_mean_seed=("se","mean"),
        smape_mean_seed=("smape","mean"),
        n_inits=("se","count"),
    )

    # B) across seeds
    agg = seed_means.groupby(["model","base_key"], as_index=False).agg(
        se_mean=("se_mean_seed","mean"),
        se_std=("se_mean_seed","std"),
        smape_mean=("smape_mean_seed","mean"),
        smape_std=("smape_mean_seed","std"),
        n_seeds=("se_mean_seed","count"),
        n_runs_total=("n_inits","sum"),
    )
    # CI halfwidths across seeds
    agg["se_ci"] = [ci_halfwidth(s, int(n)) for s, n in zip(agg["se_std"], agg["n_seeds"])]
    agg["smape_ci"] = [ci_halfwidth(s, int(n)) for s, n in zip(agg["smape_std"], agg["n_seeds"])]
    agg[["se_std","smape_std","se_ci","smape_ci"]] = agg[["se_std","smape_std","se_ci","smape_ci"]].fillna(0.0)
    return agg, seed_means

# ------------------- Plotting -------------------
def plot_domain(df_raw: pd.DataFrame, agg: pd.DataFrame, title: str, out_png: str) -> None:
    plt.figure(figsize=(9, 6))

    # raw points (semi-transparent)
    if not df_raw.empty:
        for model, g in df_raw.groupby("model"):
            plt.scatter(
                g["se"], g["smape"],
                marker=MARKER, s=32, alpha=ALPHA_RAW, edgecolors="none",
                c=MODEL_COLOR.get(model, "tab:gray"), label="_nolegend_"
            )

    # per-setting means ± 95% CI (one marker per model×base_key)
    used = set()
    for _, row in agg.iterrows():
        model = row["model"]
        color = MODEL_COLOR.get(model, "tab:gray")
        label = f"{model}" if model not in used else "_nolegend_"
        used.add(model)
        caption = f"{model} • {row['base_key']} (seeds={int(row['n_seeds'])}, runs={int(row['n_runs_total'])})"
        # draw errorbar
        plt.errorbar(
            row["se_mean"], row["smape_mean"],
            xerr=row["se_ci"], yerr=row["smape_ci"],
            fmt=MARKER, linestyle="none", capsize=3, alpha=0.95, markersize=9,
            markeredgewidth=1.0, markeredgecolor="black", color=color, label=label,
        )
        # optional annotate lightly (comment out if too busy)
        # plt.annotate(caption, (row["se_mean"], row["smape_mean"]), fontsize=7, alpha=0.7)

    plt.xlabel("SE")
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
                    help="Directory to write CSVs/PNGs.")
    args = ap.parse_args()

    log_dirs = [d.strip() for d in args.log_dirs.split(",") if d.strip()]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    all_raw_frames, all_agg_frames = [], []

    for d in log_dirs:
        domain = domain_name_from_dir(d)
        df_raw = load_domain(d)
        raw_csv = os.path.join(out_dir, f"{domain}_raw.csv")
        df_raw.to_csv(raw_csv, index=False)
        print(f"Saved {raw_csv} with {len(df_raw)} rows.")

        agg, seed_means = aggregate_hierarchical(df_raw)
        agg_csv = os.path.join(out_dir, f"{domain}_agg_by_setting.csv")
        agg.to_csv(agg_csv, index=False)
        seed_csv = os.path.join(out_dir, f"{domain}_per_seed_means.csv")
        seed_means.to_csv(seed_csv, index=False)
        print(f"Saved {agg_csv} with {len(agg)} rows. Also {seed_csv}.")

        png = os.path.join(out_dir, f"{domain}_sMAPE_vs_SE_mean_CI_by_setting.png")
        plot_domain(df_raw, agg, f"sMAPE vs SE — {domain}", png)

        if not df_raw.empty:
            df_tmp = df_raw.copy()
            df_tmp["domain"] = domain
            all_raw_frames.append(df_tmp)
        if not agg.empty:
            agg_tmp = agg.copy()
            agg_tmp["domain"] = domain
            all_agg_frames.append(agg_tmp)

    # Combined (ALL domains)
    all_raw = pd.concat(all_raw_frames, ignore_index=True) if all_raw_frames else pd.DataFrame(
        columns=["log_path","model","base_key","seed","init_seed","se","smape","domain"]
    )
    all_agg = pd.concat(all_agg_frames, ignore_index=True) if all_agg_frames else pd.DataFrame(
        columns=["model","base_key","se_mean","se_std","smape_mean","smape_std","n_seeds","n_runs_total","se_ci","smape_ci","domain"]
    )

    all_raw_csv = os.path.join(out_dir, "ALL_raw.csv")
    all_raw.to_csv(all_raw_csv, index=False)
    print(f"Saved {all_raw_csv} with {len(all_raw)} rows.")

    all_agg_csv = os.path.join(out_dir, "ALL_agg_by_setting.csv")
    all_agg.to_csv(all_agg_csv, index=False)
    print(f"Saved {all_agg_csv} with {len(all_agg)} rows.")

    # combined plot: all settings across domains
    if not all_agg.empty:
        plt.figure(figsize=(9.5, 6.5))
        # raw points faint (colored by model)
        if not all_raw.empty:
            for model, g in all_raw.groupby("model"):
                plt.scatter(
                    g["se"], g["smape"],
                    marker=MARKER, s=26, alpha=ALPHA_RAW, edgecolors="none",
                    c=MODEL_COLOR.get(model, "tab:gray"), label="_nolegend_"
                )
        used = set()
        for _, row in all_agg.iterrows():
            model = row["model"]
            color = MODEL_COLOR.get(model, "tab:gray")
            label = f"{model}" if model not in used else "_nolegend_"
            used.add(model)
            plt.errorbar(
                row["se_mean"], row["smape_mean"],
                xerr=row["se_ci"], yerr=row["smape_ci"],
                fmt=MARKER, linestyle="none", capsize=3, alpha=0.95, markersize=9,
                markeredgewidth=1.0, markeredgecolor="black", color=color, label=label
            )
        plt.xlabel("SE")
        plt.ylabel("sMAPE")
        plt.title("sMAPE vs SE — ALL Domains (per-setting means ± 95% CI)")
        if used:
            plt.legend(title="Model", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_png_all = os.path.join(out_dir, "ALL_sMAPE_vs_SE_mean_CI_by_setting.png")
        plt.savefig(out_png_all, dpi=300)
        plt.close()
        print(f"Saved {out_png_all}")

if __name__ == "__main__":
    main()
