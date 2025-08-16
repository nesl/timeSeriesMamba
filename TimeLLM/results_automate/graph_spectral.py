#!/usr/bin/env python3
import os
import re
import sys
import json
from glob import glob
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- config --------
LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "./logs"
OUT_CSV = "local_wandb_summary.csv"
OUT_PNG = "wape_vs_se.png"

# Colors/markers: triangles for all; rand_init=1 -> orange, 0 -> blue
COLOR_MAP = {0: "tab:blue", 1: "tab:orange"}
MARKER = "^"

# -------- helpers --------
num_pat = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

def norm_key(s: str) -> str:
    """normalize 'WAPE loss' -> 'wape_loss', 'SE_ctx_mean' -> 'se_ctx_mean'"""
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    return s.lower()

def parse_run_summary(lines: List[str]) -> Dict[str, Any]:
    """Scan lines; after 'wandb: Run summary:' collect wandb metric lines if numeric."""
    metrics: Dict[str, Any] = {}
    in_block = False
    for line in lines:
        if "wandb: Run summary:" in line:
            in_block = True
            continue
        if not in_block:
            continue
        # Try to match a metric line like: "wandb:       WAPE loss 0.15352"
        m = re.match(r"^\s*wandb:\s+(.+?)\s+(" + num_pat + r")\s*$", line)
        if m:
            key = norm_key(m.group(1))
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            metrics[key] = val
            continue
        # Non-metric wandb lines or an empty stretch can appear; keep scanning.
        # We don't hard-stop; we just skip non-matching lines.
    return metrics

def parse_tail_metadata(text: str) -> Dict[str, Any]:
    md: Dict[str, Any] = {}

    # Example: "Evaluation for high with init_seed 13 and seed 2 completed"
    m = re.search(r"Evaluation\s+for\s+(\w+)\s+with\s+init_seed\s+(\d+)\s+and\s+seed\s+(\d+)", text)
    if m:
        md["tier"] = m.group(1)
        md["init_seed"] = int(m.group(2))
        md["seed"] = int(m.group(3))

    # Pull run_id from URL if present:
    m = re.search(r"runs/([a-z0-9]+)", text)
    if m:
        md["run_id"] = m.group(1)

    # Parse rand_init and tier from file paths in "Logging to ..." or result lines
    # e.g., ... pems_testing_high_l0_d32_e10_m..._r1_..._seed3_init11.txt
    m = re.search(r"pems_testing_(low|medium|high)", text, re.IGNORECASE)
    if m and "tier" not in md:
        md["tier"] = m.group(1)

    m = re.search(r"_r([01])_", text)
    if m:
        md["rand_init"] = int(m.group(1))

    # If seed/init appear inside filename (_seed3_ / _init11_)
    m = re.search(r"_seed(\d+)_", text)
    if m and "seed" not in md:
        md["seed"] = int(m.group(1))
    m = re.search(r"_init(\d+)", text)
    if m and "init_seed" not in md:
        md["init_seed"] = int(m.group(1))

    return md

def coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist; fill missing with NaN
    expected = ["se_ctx_mean", "wape_loss", "mase_loss", "mae_loss", "mse_loss",
                "lle_ctx_mean", "omega_ctx_mean", "snr_ctx_proxy",
                "season_ctx_mean", "var_ctx"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    # metadata defaults
    for col, default in [("tier", "unknown"), ("init_seed", np.nan),
                         ("seed", np.nan), ("rand_init", 0)]:
        if col not in df.columns:
            df[col] = default
    return df

# -------- main scrape --------
rows = []
files = sorted(glob(os.path.join(LOG_DIR, "**", "*.txt"), recursive=True))
if not files:
    print(f"No .txt logs found under: {LOG_DIR}", file=sys.stderr)
    sys.exit(1)

for path in files:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        lines = text.splitlines()

        metrics = parse_run_summary(lines)
        if not metrics:
            # Some logs might be short; skip silently
            continue

        md = parse_tail_metadata(text)
        row = {**metrics, **md, "log_path": path}
        rows.append(row)
    except Exception as e:
        print(f"[warn] failed to parse {path}: {e}", file=sys.stderr)

if not rows:
    print("Parsed zero summaries. Check your log patterns.", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)
df = coerce_cols(df)

# Persist tidy CSV
df.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with {len(df)} rows.")

# -------- aggregate + plot --------
# Group by condition (tier, rand_init, init_seed) and aggregate over repeated 'seed'
group_cols = ["tier", "rand_init", "init_seed"]
agg = df.groupby(group_cols, dropna=False).agg(
    wape_mean=("wape_loss", "mean"),
    wape_std=("wape_loss", "std"),
    se_mean=("se_ctx_mean", "mean"),
    se_std=("se_ctx_mean", "std"),
    n=("wape_loss", "count"),
).reset_index()

# If std is NaN because only one sample, set to 0 so errorbar draws as a point
for c in ["wape_std", "se_std"]:
    agg[c] = agg[c].fillna(0.0)

plt.figure(figsize=(7, 5))
for (tier, rand_init), g in agg.groupby(["tier", "rand_init"], dropna=False):
    color = COLOR_MAP.get(int(rand_init) if pd.notna(rand_init) else 0, "tab:blue")
    label = f"{tier} | rand_init={int(rand_init)}" if pd.notna(rand_init) else f"{tier} | rand_init=?"
    plt.errorbar(
        g["se_mean"], g["wape_mean"],
        xerr=g["se_std"], yerr=g["wape_std"],
        fmt=MARKER, linestyle="none", capsize=3, label=label, alpha=0.9, markersize=7,
        markeredgewidth=0.8, markeredgecolor="black", color=color,
    )

plt.xlabel("se_ctx_mean")
plt.ylabel("wape_loss")
plt.title("WAPE vs SE (aggregated by tier, rand_init, init_seed)")
plt.legend(title="Condition", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()
print(f"Saved {OUT_PNG}")
