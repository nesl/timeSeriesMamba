#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a tidy CSV from wandb-style text logs, then plot sMAPE vs spectral entropy.
- Strictly parse tier/rand_init/seed/init_seed from filename pattern.
- Average over BOTH seed and init_seed -> one point per (tier, rand_init).
- Overlay all raw runs (faint triangles) behind mean±std markers.
- Robust de-duplication: keep newest per (tier, rand_init, seed, init_seed, run_id/log_path).
"""

import os
import re
import sys
from glob import glob
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- config ----------------
LOG_DIR = "../results/pems_eval/"
OUT_CSV = "local_wandb_summary.csv"
OUT_AGG = "local_wandb_summary_agg.csv"
OUT_PNG = "sMAPE_vs_se.png"

# Colors/markers: triangles for all; rand_init=1 -> orange, 0 -> blue
COLOR_MAP = {0: "tab:blue", 1: "tab:orange"}
MARKER = "^"

# numeric literal
NUM_PAT = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

# strict filename parser
FILE_RE = re.compile(
    r"""
    ^pems_testing_
    (?P<tier>low|medium|high)      # tier
    .*?                            # anything
    _r(?P<rand>[01])_              # rand_init
    .*?
    _seed(?P<seed>\d+)             # seed
    _init(?P<init>\d+)             # init_seed
    \.txt$
    """,
    re.IGNORECASE | re.VERBOSE
)

# ---------------- helpers ----------------
def norm_key(s: str) -> str:
    """normalize keys to snake_case, lowercase: 'sMAPE loss' -> 'smape_loss'."""
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    return s.lower()

def parse_run_summary(lines: List[str]) -> Dict[str, Any]:
    """Scan lines; after 'wandb: Run summary:' collect numeric metric lines."""
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

def parse_from_filename(path: str) -> Dict[str, Any]:
    """Parse tier/rand_init/seed/init_seed strictly from basename."""
    b = os.path.basename(path)
    m = FILE_RE.match(b)
    if not m:
        return {}
    return {
        "tier": m.group("tier").lower(),
        "rand_init": int(m.group("rand")),
        "seed": int(m.group("seed")),
        "init_seed": int(m.group("init")),
    }

def coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected metric columns exist and coerce numeric."""
    expected = [
        "se_ctx_mean", "smape_loss", "mase_loss", "mae_loss", "mse_loss",
        "lle_ctx_mean", "omega_ctx_mean", "snr_ctx_proxy",
        "season_ctx_mean", "var_ctx"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    # coerce numeric metrics
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------- main ----------------
def main() -> None:
    files = sorted(glob(os.path.join(LOG_DIR, "**", "*.txt"), recursive=True))
    if not files:
        print(f"No .txt logs found under: {LOG_DIR}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            lines = text.splitlines()
            metrics = parse_run_summary(lines)
            if not metrics:
                continue
            rows.append({
                **metrics,
                "log_path": path,
                "mtime": os.path.getmtime(path)
            })
        except Exception as e:
            print(f"[warn] failed to parse {path}: {e}", file=sys.stderr)

    if not rows:
        print("Parsed zero summaries. Check your log patterns.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = coerce_cols(df)

    # attach filename metadata strictly
    meta = df["log_path"].apply(parse_from_filename)
    meta_df = pd.DataFrame(list(meta))
    df = pd.concat([df, meta_df], axis=1)

    # require valid tier / rand_init
    df = df[
        df["tier"].isin(["low", "medium", "high"]) &
        df["rand_init"].isin([0, 1])
    ].copy()

    # non-finite cleanup
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["smape_loss", "se_ctx_mean", "seed", "init_seed"], how="any")

    # optional: capture run_id from URL if present in file text (best-effort)
    # (we didn't keep full text; okay—use log_path as identity)
    df["identity"] = df["log_path"]

    # robust de-duplication: keep newest mtime per (tier, rand_init, seed, init_seed, identity)
    df["dedup_key"] = list(zip(
        df["tier"], df["rand_init"], df["seed"], df["init_seed"], df["identity"]
    ))
    df = df.sort_values("mtime").drop_duplicates(subset=["dedup_key"], keep="last")
    df = df.drop_duplicates(subset=["log_path"])  # belt + suspenders

    # save tidy
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} with {len(df)} rows.")

    # aggregate across BOTH seed and init_seed → one row per (tier, rand_init)
    agg = df.groupby(["tier", "rand_init"], as_index=False).agg(
        smape_mean=("smape_loss", "mean"),
        smape_std=("smape_loss", "std"),
        se_mean=("se_ctx_mean", "mean"),
        se_std=("se_ctx_mean", "std"),
        n=("smape_loss", "count"),
    )
    for c in ["smape_std", "se_std"]:
        agg[c] = agg[c].fillna(0.0)

    # sanity: ensure at most one row per (tier, rand_init)
    dups = agg.groupby(["tier", "rand_init"]).size()
    bad = dups[dups > 1]
    if not bad.empty:
        print("[WARN] duplicate groups in agg; collapsing again.\n", bad)
        agg = agg.groupby(["tier", "rand_init"], as_index=False).agg({
            "smape_mean": "mean",
            "smape_std": "mean",
            "se_mean": "mean",
            "se_std": "mean",
            "n": "sum"
        })

    # stable order
    tier_order = {"low": 0, "medium": 1, "high": 2}
    agg["__ord"] = agg["tier"].map(tier_order)
    agg = agg.sort_values(["__ord", "rand_init"]).drop(columns="__ord")

    agg.to_csv(OUT_AGG, index=False)
    print(f"Saved {OUT_AGG} with {len(agg)} rows.")

    # ---------------- plot ----------------
    plt.figure(figsize=(7, 5))

    # raw runs (faint)
    for ri, g in df.groupby("rand_init"):
        if g.empty:
            continue
        plt.scatter(
            g["se_ctx_mean"], g["smape_loss"],
            marker=MARKER, s=35, alpha=0.25, edgecolors="none",
            c=COLOR_MAP.get(int(ri), "tab:blue"),
            label="_nolegend_"
        )

    # means ± std (exactly two points per tier if both r0/r1 exist)
    for _, row in agg.iterrows():
        tier, ri = row["tier"], int(row["rand_init"])
        color = COLOR_MAP.get(ri, "tab:blue")
        label = f"{tier} | rand_init={ri} (n={int(row['n'])})"
        plt.errorbar(
            row["se_mean"], row["smape_mean"],
            xerr=row["se_std"], yerr=row["smape_std"],
            fmt=MARKER, linestyle="none", capsize=3, alpha=0.95, markersize=8,
            markeredgewidth=0.9, markeredgecolor="black", color=color, label=label
        )

    plt.xlabel("se_ctx_mean")
    plt.ylabel("smape_loss")
    plt.title("PEMS: sMAPE vs SE (mean ± std over seeds & init_seeds)")
    plt.legend(title="Condition", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Saved {OUT_PNG}")

if __name__ == "__main__":
    main()
