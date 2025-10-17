#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_gift_results.py — Collect all GIFT-Eval all_results.csv files into one tidy CSV.

Run from inside gift_eval/:
    python merge_gift_results.py

Result:
    merged_gift_results.csv  (written in current directory)

What it does:
-------------
- Recursively finds every all_results.csv under ./git_repo/
- Reads and normalizes the columns (removes "eval_metrics/" prefixes)
- Adds a provenance column ("source_csv")
- Concatenates all rows and writes a single merged CSV
"""

import pandas as pd
from pathlib import Path

# root path (assuming you're inside gift_eval/)
root = Path("git_repo")

# find all_results.csv recursively
files = sorted(root.rglob("all_results.csv"))
if not files:
    raise FileNotFoundError("No all_results.csv files found under ./git_repo")

frames = []

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Failed to read {f}: {e}")
        continue

    # strip eval_metrics/ prefix
    df.columns = [c.replace("eval_metrics/", "") for c in df.columns]

    # add provenance and infer model/dataset if not present
    if "model" not in df.columns:
        df["model"] = f.parent.name
    if "dataset" not in df.columns:
        df["dataset"] = f.stem

    df["source_csv"] = str(f)
    frames.append(df)

if not frames:
    raise RuntimeError("No valid CSVs could be read.")

merged = pd.concat(frames, ignore_index=True)

# optional: reorder most important columns to the front
front = ["dataset","model","MSE[mean]","MSE[0.5]","MAE[0.5]","MAPE[0.5]",
         "sMAPE[0.5]","RMSE[mean]","NRMSE[mean]","domain","num_variates","source_csv"]
rest = [c for c in merged.columns if c not in front]
merged = merged[[c for c in front if c in merged.columns] + rest]

# write output
out_path = Path("merged_gift_results.csv")
merged.to_csv(out_path, index=False)
print(f"[OK] Merged {len(files)} files -> {len(merged):,} rows written to {out_path.resolve()}")
