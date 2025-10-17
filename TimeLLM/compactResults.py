#!/usr/bin/env python3
"""
Scrape wandb "Run summary" metrics from results/*.txt logs and export tidy CSVs.

Usage (defaults to ETTh1 + DLinear):
  python scrape_results.py \
    --results-dir results \
    --out-dir results_compact \
    --datasets ETTh1 \
    --models DLinear

To add more:
  --datasets ETTh1 ETTm1 PEMS Fitbit CarbonCast
  --models DLinear TimeLLM Mamba AutoARIMA

The script:
  • Recurses into results/<DATASET>/
  • Parses files like: DLinear_ETTh1_e10_f1_t100_c100_seq512_pred96_seed8.txt
  • Extracts config (model, dataset, seq, pred, seed, etc.)
  • Reads the "Run summary" section (handles spacing) and collects:
      MAE loss, mae, mse, mape, mspe, rmse, train loss, vali loss, test loss, actual epochs
    Treats lines like `mse 8 0.37049` as just `mse = 0.37049` (seed ignored)
  • Writes one CSV per (dataset, model) to --out-dir
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

# Keys you care about (case-insensitive match on the left; CSV columns are snake_cased)
WANTED_KEYS = [
    "MAE loss",
    "mae",
    "mse",
    "mape",
    "mspe",
    "rmse",
    "train loss",
    "vali loss",
    "test loss",
    "actual epochs",
]

# Filename pattern example:
# DLinear_ETTh1_e10_f1_t100_c100_seq512_pred96_seed8.txt
FILE_RE = re.compile(
    r'^(?P<model>[^_]+)_(?P<dataset>[^_]+)'
    r'(?:_e(?P<epochs>\d+))?'
    r'(?:_f(?P<f>\d+))?'
    r'(?:_t(?P<t>\d+))?'
    r'(?:_c(?P<c>\d+))?'
    r'(?:_seq(?P<seq>\d+))?'
    r'(?:_pred(?P<pred>\d+))?'
    r'_seed(?P<seed>\d+)\.txt$'
)

# Matches lines like any of:
#   "wandb:    MAE loss 8 0.39158"
#   "wandb:         mse 8 0.37049"
#   "wandb: actual epochs 9"
# Captures key + final numeric value; ignores the (seed) middle number.
LINE_RE = re.compile(
    r'^\s*wandb:\s*(?P<key>.+?)\s+(?:\d+\s+)?(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$'
)

def snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s

def key_is_wanted(k: str) -> Optional[str]:
    k_clean = k.strip().lower()
    for target in WANTED_KEYS:
        if k_clean == target.lower():
            return snake(target)
    return None

def parse_config_from_filename(name: str) -> Optional[Dict[str, str]]:
    m = FILE_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    # Convert numeric strings when present
    for fld in ["epochs", "f", "t", "c", "seq", "pred", "seed"]:
        if d.get(fld) is not None:
            d[fld] = int(d[fld])
    return d

def read_run_summary_metrics(text: str) -> Dict[str, float]:
    """
    Extract wanted metrics from "wandb: ..." lines.
    We don't require a literal "Run summary" sentinel; we just parse all wandb lines and
    keep the last occurrence per key, which is commonly the summary at the end.
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if "wandb:" not in line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        raw_key = m.group("key")
        val_str  = m.group("val")
        col = key_is_wanted(raw_key)
        if col is None:
            continue
        try:
            out[col] = float(val_str)
        except ValueError:
            # Skip if not a float
            pass
    return out

def tail_read(p: Path, max_bytes: int = 120_000) -> str:
    """
    Read only the last `max_bytes` of the file (faster for long logs),
    but will gracefully fallback to full read if file is smaller.
    """
    try:
        size = p.stat().st_size
        if size <= max_bytes:
            return p.read_text(errors="ignore")
        with p.open("rb") as fh:
            fh.seek(size - max_bytes)
            data = fh.read()
        # Try to decode from a safe boundary
        return data.decode(errors="ignore")
    except Exception:
        return p.read_text(errors="ignore")

def scrape_one_dataset_model(results_dir: Path, out_dir: Path, dataset: str, model: str) -> Path:
    ds_dir = results_dir / dataset
    rows: List[Dict[str, object]] = []

    if not ds_dir.is_dir():
        print(f"[WARN] Missing dataset dir: {ds_dir}")
        out_path = out_dir / f"metrics_{dataset}_{model}.csv"
        # Still write an empty CSV with headers for consistency
        write_csv(out_path, [])
        return out_path

    # Match files that begin with "{model}_{dataset}_" and end with ".txt"
    for p in ds_dir.glob(f"{model}_{dataset}_*.txt"):
        conf = parse_config_from_filename(p.name)
        if not conf:
            continue
        text = tail_read(p)
        metrics = read_run_summary_metrics(text)
        if not metrics:
            # No summary metrics found; skip but log
            print(f"[INFO] No metrics found in: {p}")
            continue

        row = {
            "file": p.name,
            "path": str(p),
            "model": conf["model"],
            "dataset": conf["dataset"],
            "seed": conf.get("seed"),
            "epochs_tag": conf.get("epochs"),   # from filename tag e10
            "f_tag": conf.get("f"),
            "t_tag": conf.get("t"),
            "c_tag": conf.get("c"),
            "seq_len": conf.get("seq"),
            "pred_len": conf.get("pred"),
        }
        row.update(metrics)  # add scraped metrics
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{dataset}_{model}.csv"
    write_csv(out_path, rows)
    print(f"[OK] Wrote {len(rows)} rows -> {out_path}")
    return out_path

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    # Collect all keys to make a stable header
    fieldnames = [
        "file","path","model","dataset","seed","epochs_tag","f_tag","t_tag","c_tag","seq_len","pred_len",
        # metric columns (ensure stable order)
        "mae_loss","mae","mse","mape","mspe","rmse","train_loss","vali_loss","test_loss","actual_epochs",
    ]
    # Ensure presence of all found keys as well
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_compact"))
    ap.add_argument("--datasets", nargs="+", default=["ETTh1"])
    ap.add_argument("--models", nargs="+", default=["DLinear"])
    args = ap.parse_args()

    for ds in args.datasets:
        for md in args.models:
            scrape_one_dataset_model(args.results_dir, args.out_dir, ds, md)

if __name__ == "__main__":
    main()
