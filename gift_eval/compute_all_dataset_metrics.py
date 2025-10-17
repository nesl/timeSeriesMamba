#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Ω, spectral entropy, LLE, ApEn, SampEn, PermEn, Wavelet entropy
for ALL GIFT HF-on-disk datasets under ./series/, aggregate to dataset-level
(wide columns), and join domain/frequency from git_repo/notebooks/dataset_properties.json.

Outputs (in CWD)
----------------
- metrics_summary_wide.csv
- (optional) metrics_per_series_wide.csv  [if --write-series]

Usage
-----
python compute_dataset_metrics_wide.py
python compute_dataset_metrics_wide.py --max-series 100 --write-series
python compute_dataset_metrics_wide.py --apen-m 2 --apen-r 0.2 --sampen-m 2 --sampen-r 0.2 \
  --permen-m 3 --permen-tau 1 --lle-m 10 --lle-tau 1
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# ---------- Optional dependency: PyWavelets ----------
try:
    import pywt
except Exception:
    pywt = None  # we'll handle missing PyWavelets gracefully


# =========================
# Utility
# =========================

def canon(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def base_from_label(label: str) -> str:
    return canon(label.split("/")[0])

def is_hf_disk_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and any(path.glob("*.arrow"))

def collect_all_hf_datasets(root: Path) -> List[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_dir() and is_hf_disk_dataset(p):
            out.append(p)
    return sorted(out)

def load_from_disk(path: Path):
    from datasets import load_from_disk
    return load_from_disk(str(path))

def human_label(ds_path: Path) -> str:
    parent = ds_path.parent
    if parent and not is_hf_disk_dataset(parent):
        return f"{parent.name}/{ds_path.name}"
    return ds_path.name

def is_multivariate_target(target: Any) -> bool:
    return isinstance(target, (list, tuple)) and len(target) > 0 and isinstance(target[0], (list, tuple))

def iter_series(records: List[Dict[str, Any]]):
    for rec in records:
        sid = rec.get("item_id", 0)
        tgt = rec.get("target", None)
        if tgt is None: 
            continue
        if is_multivariate_target(tgt):
            for d, arr in enumerate(tgt):
                yield str(sid) if sid is not None else "0", int(d), np.asarray(arr, dtype=float)
        else:
            yield str(sid) if sid is not None else "0", None, np.asarray(tgt, dtype=float)


# =========================
# Metrics
# =========================

def _safe_rfft_power(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 4 or np.allclose(x.std(), 0.0):
        return np.array([1.0], dtype=float)
    x = x - x.mean()
    spec = np.fft.rfft(x, n=x.size)
    power = (spec.real**2 + spec.imag**2)
    power = np.clip(power, 0.0, None)
    return power if power.sum() > 0 else np.array([1.0], dtype=float)

def spectral_entropy(x: np.ndarray) -> float:
    pwr = _safe_rfft_power(x)
    p = pwr / pwr.sum()
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p))
    Hmax = np.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 1.0

def omega(x: np.ndarray) -> float:
    return 1.0 - spectral_entropy(x)

# ApEn and SampEn (m, r*std)
def _phi(x: np.ndarray, m: int, r: float) -> float:
    N = len(x)
    if N <= m + 1:
        return np.nan
    x = np.asarray(x, dtype=float)
    sd = np.std(x, ddof=0)
    tol = r * sd
    # Build m-length vectors
    Xm = np.lib.stride_tricks.sliding_window_view(x, window_shape=m)
    count = 0
    total = 0
    for i in range(len(Xm)):
        dist = np.max(np.abs(Xm - Xm[i]), axis=1)
        C = np.sum(dist <= tol)  # includes self
        total += C
        count += 1
    return total / (count * len(Xm))

def approximate_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    phi_m = _phi(x, m, r)
    phi_m1 = _phi(x, m + 1, r)
    if phi_m <= 0 or phi_m1 <= 0 or np.isnan(phi_m) or np.isnan(phi_m1):
        return np.nan
    return float(np.log(phi_m) - np.log(phi_m1))

def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N <= m + 1:
        return np.nan
    sd = np.std(x, ddof=0)
    tol = r * sd
    Xm = np.lib.stride_tricks.sliding_window_view(x, window_shape=m)
    Xm1 = np.lib.stride_tricks.sliding_window_view(x, window_shape=m+1)
    B = 0  # matches of length m
    A = 0  # matches of length m+1
    for i in range(len(Xm)):
        # exclude self-match by slicing out i
        dist_m = np.max(np.abs(Xm - Xm[i]), axis=1)
        B += np.sum((dist_m <= tol)) - 1
        if i < len(Xm1):
            dist_m1 = np.max(np.abs(Xm1 - Xm1[i]), axis=1)
            A += np.sum((dist_m1 <= tol)) - 1
    if B == 0 or A == 0:
        return np.nan
    # SampEn = -ln( A / B )
    return float(-np.log(A / B))

# Permutation entropy (Bandt & Pompe)
def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    N = len(x)
    L = N - tau * (m - 1)
    if m < 2 or tau < 1 or L <= 0:
        return np.nan
    # Build ordinal patterns
    patterns = []
    for i in range(L):
        window = x[i:i + tau * m:tau]
        ranks = np.argsort(window, kind="mergesort")
        patterns.append(tuple(ranks))
    # Count frequencies
    from collections import Counter
    cnt = Counter(patterns)
    p = np.array(list(cnt.values()), dtype=float)
    p /= p.sum()
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p))
    Hmax = np.log(np.math.factorial(m))
    return float(H / Hmax) if Hmax > 0 else np.nan

# Wavelet entropy (relative energy distribution across DWT levels)
def wavelet_entropy(x: np.ndarray, wavelet: str = "db4") -> float:
    if pywt is None:
        return np.nan
    x = np.asarray(x, dtype=float)
    try:
        coeffs = pywt.wavedec(x, wavelet=wavelet, mode="periodization")
    except Exception:
        return np.nan
    energies = np.array([np.sum(c**2) for c in coeffs], dtype=float)
    if not np.isfinite(energies).any() or energies.sum() <= 0:
        return np.nan
    p = energies / energies.sum()
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p))
    Hmax = np.log(len(p))
    return float(H / Hmax) if Hmax > 0 else np.nan

# Rosenstein LLE (simple, robust-ish); returns slope of divergence (per step)
def lle_rosenstein(x: np.ndarray, m: int = 10, tau: int = 1, k: int = 1, fit_max_steps: int = 20) -> float:
    x = np.asarray(x, dtype=float)
    N = len(x)
    emb_len = N - (m - 1) * tau
    if emb_len <= 2 or m < 2:
        return np.nan
    # Embed
    X = np.column_stack([x[i:i+emb_len] for i in range(0, m*tau, tau)])
    # Nearest neighbors (exclude temporal neighbors within Theiler window)
    theiler = tau * 2
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    # Exclude neighbors too close in time
    for i in range(len(X)):
        lo = max(0, i - theiler); hi = min(len(X), i + theiler + 1)
        dists[i, lo:hi] = np.inf
    nn = np.argmin(dists, axis=1)
    # Track divergence over time
    max_t = min(fit_max_steps, emb_len - 1)
    if max_t < 2:
        return np.nan
    div = []
    for t in range(1, max_t + 1):
        idx = np.arange(0, emb_len - t)
        jdx = nn[idx]
        valid = jdx + t < emb_len
        idx = idx[valid]; jdx = jdx[valid]
        if idx.size == 0:
            div.append(np.nan); continue
        d = np.linalg.norm(X[idx + t] - X[jdx + t], axis=1)
        d = d[d > 0]
        if d.size == 0:
            div.append(np.nan); continue
        div.append(np.log(d).mean())
    div = np.array(div)
    # linear fit over times where div is finite
    t = np.arange(1, len(div) + 1, dtype=float)
    mask = np.isfinite(div)
    if mask.sum() < 3:
        return np.nan
    slope = np.polyfit(t[mask], div[mask], 1)[0]
    return float(slope)


# =========================
# Main orchestration
# =========================

ALIASES = {
    # Add any special folder->props mappings here if a mismatch pops up
    # "electricity_hourly": "electricity",
}

def load_properties(props_path: Path) -> pd.DataFrame:
    with props_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    df = (pd.DataFrame.from_dict(raw, orient="index")
            .reset_index()
            .rename(columns={"index": "dataset_base_raw"}))
    df["dataset_base"] = df["dataset_base_raw"].apply(canon)
    return df[["dataset_base","domain","frequency","num_variates"]]

def compute_metrics_for_split(records: List[Dict[str, Any]],
                              apen_m: int, apen_r: float,
                              sampen_m: int, sampen_r: float,
                              permen_m: int, permen_tau: int,
                              lle_m: int, lle_tau: int) -> List[Dict[str, Any]]:
    rows = []
    for sid, dim, arr in iter_series(records):
        try:
            om = omega(arr)
            se = spectral_entropy(arr)
            ap = approximate_entropy(arr, m=apen_m, r=apen_r)
            sa = sample_entropy(arr, m=sampen_m, r=sampen_r)
            pe = permutation_entropy(arr, m=permen_m, tau=permen_tau)
            we = wavelet_entropy(arr)
            ll = lle_rosenstein(arr, m=lle_m, tau=lle_tau)
        except Exception:
            om = se = ap = sa = pe = we = ll = np.nan

        rows.append({
            "series_id": sid, "dim": dim,
            "omega": om,
            "spectral_entropy": se,
            "apen": ap,
            "sampen": sa,
            "permen": pe,
            "wavelet_entropy": we,
            "lle": ll,
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", default="series", help="Root containing HF datasets")
    ap.add_argument("--props", default="git_repo/notebooks/dataset_properties.json",
                    help="JSON mapping dataset -> {domain, frequency, num_variates}")
    ap.add_argument("--max-series", type=int, default=0, help="Limit #series per split")
    # metric params
    ap.add_argument("--apen-m", type=int, default=2)
    ap.add_argument("--apen-r", type=float, default=0.2)
    ap.add_argument("--sampen-m", type=int, default=2)
    ap.add_argument("--sampen-r", type=float, default=0.2)
    ap.add_argument("--permen-m", type=int, default=3)
    ap.add_argument("--permen-tau", type=int, default=1)
    ap.add_argument("--lle-m", type=int, default=10)
    ap.add_argument("--lle-tau", type=int, default=1)
    ap.add_argument("--write-series", action="store_true", help="Also write per-series wide CSV")
    ap.add_argument("--strict-domains", action="store_true", help="Error if any dataset fails to map to a domain")
    args = ap.parse_args()

    series_root = Path(args.series_root).expanduser().resolve()
    props_path = Path(args.props).expanduser().resolve()
    if not props_path.exists():
        raise FileNotFoundError(f"Properties file not found: {props_path}")

    props_df = load_properties(props_path)

    ds_dirs = collect_all_hf_datasets(series_root)
    if not ds_dirs:
        raise FileNotFoundError(f"No HF datasets found under {series_root}")

    all_series_rows = []
    all_summaries = []

    for ds_path in ds_dirs:
        label = human_label(ds_path)
        obj = load_from_disk(ds_path)
        split_items = obj.items() if isinstance(obj, dict) else [("all", obj)]

        for split_name, dset in split_items:
            records = dset.to_list()
            if args.max_series and len(records) > args.max_series:
                records = records[:args.max_series]

            series_rows = compute_metrics_for_split(
                records,
                apen_m=args.apen_m, apen_r=args.apen_r,
                sampen_m=args.sampen_m, sampen_r=args.sampen_r,
                permen_m=args.permen_m, permen_tau=args.permen_tau,
                lle_m=args.lle_m, lle_tau=args.lle_tau
            )
            # Attach identifiers and collect
            for r in series_rows:
                r.update({
                    "dataset_label": label,
                    "split": split_name,
                    "dataset_path": str(ds_path),
                })
            all_series_rows.extend(series_rows)

            # Aggregate to dataset-level (mean across series; NaNs ignored)
            if series_rows:
                df = pd.DataFrame(series_rows)
                agg = df[["omega","spectral_entropy","lle","apen","sampen","permen","wavelet_entropy"]].astype(float)
                summary = {
                    "dataset_label": label,
                    "split": split_name,
                    "n_series": int(len(df)),
                    "omega": float(np.nanmean(agg["omega"])),
                    "spectral_entropy": float(np.nanmean(agg["spectral_entropy"])),
                    "lle": float(np.nanmean(agg["lle"])),
                    "apen": float(np.nanmean(agg["apen"])),
                    "sampen": float(np.nanmean(agg["sampen"])),
                    "permen": float(np.nanmean(agg["permen"])),
                    "wavelet_entropy": float(np.nanmean(agg["wavelet_entropy"])),
                    "dataset_path": str(ds_path),
                }
                all_summaries.append(summary)

    # Build DataFrames
    per_series_df = pd.DataFrame(all_series_rows)
    summary_df = pd.DataFrame(all_summaries)

    # Canonical base, aliases, and join properties
    for df in (summary_df, per_series_df):
        if not df.empty:
            df["dataset_base"] = df["dataset_label"].apply(base_from_label)
    # ALIASES hook
    # (Add entries to ALIASES if you see mismatches between folder names and props keys)
    # summary_df["dataset_base"] = summary_df["dataset_base"].apply(lambda x: ALIASES.get(x, x))
    # per_series_df["dataset_base"] = per_series_df["dataset_base"].apply(lambda x: ALIASES.get(x, x))

    summary_df = summary_df.merge(props_df, how="left", on="dataset_base")
    if not per_series_df.empty:
        per_series_df = per_series_df.merge(props_df, how="left", on="dataset_base")

    unmatched = sorted(summary_df.loc[summary_df["domain"].isna(), "dataset_base"].unique())
    if unmatched:
        msg = "[WARN] Unmatched dataset_base (no domain in props): " + ", ".join(unmatched)
        if args.strict_domains:
            raise RuntimeError(msg)
        else:
            print(msg)

    # Order columns & write
    summary_cols = ["dataset_label","dataset_base","domain","frequency","num_variates",
                    "split","n_series",
                    "omega","spectral_entropy","lle","apen","sampen","permen","wavelet_entropy",
                    "dataset_path"]
    summary_df = summary_df[summary_cols].sort_values(["dataset_label","split"])
    summary_df.to_csv("metrics_summary_wide.csv", index=False)
    print(f"[OK] wrote metrics_summary_wide.csv  ({len(summary_df)} rows)")

    if args.write_series and not per_series_df.empty:
        per_cols = ["dataset_label","dataset_base","domain","frequency","num_variates",
                    "split","series_id","dim",
                    "omega","spectral_entropy","lle","apen","sampen","permen","wavelet_entropy",
                    "dataset_path"]
        per_series_df = per_series_df[per_cols]
        per_series_df.to_csv("metrics_per_series_wide.csv", index=False)
        print(f"[OK] wrote metrics_per_series_wide.csv ({len(per_series_df)} rows)")


if __name__ == "__main__":
    main()
