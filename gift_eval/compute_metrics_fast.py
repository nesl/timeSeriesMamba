#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute dataset-level time-series difficulty metrics for all HF-on-disk datasets
under ./series/ (recursive), and join domain/frequency from
git_repo/notebooks/dataset_properties.json.

FAST DEFAULTS:
  --metrics omega,spectral_entropy
  --truncate 4096
  --downsample 2
  --batch-size 128
  --num-proc (cpu_count - 1)

Outputs (in CWD):
  - metrics_summary_wide.csv          # dataset/split-level means (wide columns per metric)
  - metrics_per_series_wide.csv       # optional, if --write-series

Usage:
  python compute_metrics_fast.py
  python compute_metrics_fast.py --metrics omega,spectral_entropy,permen,wavelet_entropy
  python compute_metrics_fast.py --metrics apen,sampen,lle --max-series 200 --truncate 2048 --downsample 4
  python compute_metrics_fast.py --strict-domains
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

# Optional speed libraries (used only if you request the heavy metrics)
try:
    import antropy as ant  # ApEn/SampEn/PermEn
except Exception:
    ant = None
try:
    import nolds as _nolds  # Lyapunov
except Exception:
    _nolds = None
try:
    import pywt  # Wavelet entropy
except Exception:
    pywt = None


# -------------------- Utilities --------------------

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
    # HF + with_format("numpy") can give np.ndarray with shape (dims, T)
    if isinstance(target, np.ndarray):
        return target.ndim == 2  # (dims, T)
    # classic list-of-lists
    if isinstance(target, (list, tuple)) and len(target) > 0:
        return isinstance(target[0], (list, tuple, np.ndarray))
    return False


# -------------------- Core FFT-based metrics --------------------
def _safe_rfft_power(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=float).ravel()

    # not enough points to trust FFT shape
    if x.size < 4:
        print(f"[DEBUG] Series too short: {x.size}")
        return None

    std = x.std()
    if np.allclose(std, 0.0):
        # perfectly (or near) constant -> treat as pure DC spike
        # we'll special-case this later instead of faking uniform
        print(f"[DEBUG] Zero variance detected: std={std}, mean={x.mean()}, min={x.min()}, max={x.max()}")
        return np.array([1.0], dtype=float)  # meaning "all DC"

    x = x - x.mean()
    spec = np.fft.rfft(x, n=x.size)
    power = (spec.real**2 + spec.imag**2)
    power = np.clip(power, 0.0, None)

    if power.sum() == 0:
        print(f"[DEBUG] Zero total power after FFT")
        return None

    return power

def spectral_entropy(x: np.ndarray) -> float:
    pwr = _safe_rfft_power(x)

    # Case 1: unusable / too short / numerical trash
    if pwr is None:
        return np.nan  # don't hallucinate 1.0

    # Case 2: pure DC or near-constant -> expect all mass in one bin
    if pwr.size == 1:
        # all energy in a single frequency bin => entropy 0
        return 0.0

    # Normal path
    p = pwr / pwr.sum()
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p))
    Hmax = np.log(len(p))
    return float(H / Hmax) if Hmax > 0 else np.nan


def omega_from_arr(x: np.ndarray) -> float:
    return 1.0 - spectral_entropy(x)


# -------------------- Heavy metrics (optional) --------------------

# Fallback ApEn/SampEn if antropy missing (O(N^2))
def _phi(x: np.ndarray, m: int, r: float) -> float:
    N = len(x)
    if N <= m + 1:
        return np.nan
    sd = np.std(x, ddof=0)
    tol = r * sd
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
    Xm1 = np.lib.stride_tricks.sliding_window_view(x, window_shape=m + 1)
    B = 0; A = 0
    for i in range(len(Xm)):
        dist_m = np.max(np.abs(Xm - Xm[i]), axis=1)
        B += np.sum((dist_m <= tol)) - 1
        if i < len(Xm1):
            dist_m1 = np.max(np.abs(Xm1 - Xm1[i]), axis=1)
            A += np.sum((dist_m1 <= tol)) - 1
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))

def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    L = N - tau * (m - 1)
    if m < 2 or tau < 1 or L <= 0:
        return np.nan

    # Build ordinal patterns as *Python tuples of ints* (hashable)
    patterns = []
    for i in range(L):
        window = x[i:i + tau * m:tau]
        # ranks as plain Python ints
        ranks = np.argsort(window, kind="mergesort")
        ranks_tuple = tuple(int(v) for v in ranks.tolist())
        patterns.append(ranks_tuple)

    from collections import Counter
    if not patterns:
        return np.nan
    cnt = Counter(patterns)
    p = np.array(list(cnt.values()), dtype=float)
    p /= p.sum()
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p))
    Hmax = np.log(np.math.factorial(m))
    return float(H / Hmax) if Hmax > 0 else np.nan


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

def lle_rosenstein(x: np.ndarray, m: int = 8, tau: int = 1, fit_max_steps: int = 15) -> float:
    x = np.asarray(x, dtype=float)
    N = len(x)
    emb_len = N - (m - 1) * tau
    if emb_len <= 2 or m < 2:
        #print("emblen < 2")
        return np.nan
    X = np.column_stack([x[i:i+emb_len] for i in range(0, m*tau, tau)])
    theiler = tau * 2
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    for i in range(len(X)):
        lo = max(0, i - theiler); hi = min(len(X), i + theiler + 1)
        dists[i, lo:hi] = np.inf
    nn = np.argmin(dists, axis=1)
    max_t = min(fit_max_steps, emb_len - 1)
    if max_t < 2:
        #print("max_t < 2")

        return np.nan
    div = []
    for t in range(1, max_t + 1):
        idx = np.arange(0, emb_len - t)
        jdx = nn[idx]
        valid = jdx + t < emb_len
        idx = idx[valid]; jdx = jdx[valid]
        if idx.size == 0:
            #print("idx size 0")
            div.append(np.nan); continue
        d = np.linalg.norm(X[idx + t] - X[jdx + t], axis=1)
        d = d[d > 0]
        if d.size == 0:
            #print("d size 0")
            div.append(np.nan); continue
        div.append(np.log(d).mean())
    div = np.array(div)
    t = np.arange(1, len(div) + 1, dtype=float)
    mask = np.isfinite(div)
    if mask.sum() < 3:
        #print("mask.sum < 3")

        return np.nan
    slope = np.polyfit(t[mask], div[mask], 1)[0]
    return float(slope)


# -------------------- Fast per-split computation (datasets.map) --------------------

def prepare_series(arr: np.ndarray, truncate: int, downsample: int) -> np.ndarray:
    if truncate and arr.size > truncate:
        arr = arr[:truncate]
    if downsample and downsample > 1:
        arr = arr[::downsample]
    return arr

def compute_metric_row(
    arr: np.ndarray, metrics: Sequence[str],
    apen_m=2, apen_r=0.2, sampen_m=2, sampen_r=0.2,
    permen_m=3, permen_tau=1, lle_m=8, lle_tau=1, lle_steps=15
) -> dict:
    out = {}
    if arr.std() < 1e-6:
        print(f"[SKIP] Near-constant series: std={arr.std()}")
        return {m: np.nan for m in metrics}

    if "omega" in metrics or "spectral_entropy" in metrics:
        se = spectral_entropy(arr)
        out["spectral_entropy"] = se
        out["omega"] = 1.0 - se
    if "permen" in metrics:
        if ant is not None:
            try:
                out["permen"] = float(ant.perm_entropy(arr, order=permen_m, normalize=True, delay=permen_tau))
            except Exception:
                out["permen"] = permutation_entropy(arr, m=permen_m, tau=permen_tau)
        else:
            out["permen"] = permutation_entropy(arr, m=permen_m, tau=permen_tau)
    if "wavelet_entropy" in metrics:
        out["wavelet_entropy"] = wavelet_entropy(arr)
    if "apen" in metrics:
        if ant is not None:
            try:
                out["apen"] = float(ant.app_entropy(arr, order=apen_m, metric="chebyshev",
                                                    approximate=True, r=apen_r*np.std(arr)))
            except Exception:
                out["apen"] = approximate_entropy(arr, m=apen_m, r=apen_r)
        else:
            out["apen"] = approximate_entropy(arr, m=apen_m, r=apen_r)
    if "sampen" in metrics:
        if ant is not None:
            try:
                out["sampen"] = float(ant.sample_entropy(arr, order=sampen_m, r=sampen_r*np.std(arr)))
            except Exception:
                out["sampen"] = sample_entropy(arr, m=sampen_m, r=sampen_r)
        else:
            out["sampen"] = sample_entropy(arr, m=sampen_m, r=sampen_r)
    if "lle" in metrics:
        if _nolds is not None:
            try:
                out["lle"] = float(_nolds.lyap_r(arr, emb_dim=lle_m, lag=lle_tau,
                                                 min_tsep=2*lle_tau, trajectory_len=lle_steps))
            except Exception:
                out["lle"] = lle_rosenstein(arr, m=lle_m, tau=lle_tau, fit_max_steps=lle_steps)
        else:
            out["lle"] = lle_rosenstein(arr, m=lle_m, tau=lle_tau, fit_max_steps=lle_steps)
    return out

def compute_split_fast(
    dset, label: str, split_name: str, ds_path: Path,
    metrics: Sequence[str],
    num_proc: int, batch_size: int, max_series: int,
    truncate: int, downsample: int,
    metric_kwargs: dict
):
    # Optional subset for speed/debug
    if max_series and len(dset) > max_series:
        dset = dset.select(range(max_series))

    dset = dset.with_format("numpy")

    def _batch_fn(batch):
        out = {}
        series_ids = batch.get("item_id", np.arange(len(batch["target"])).astype(str))
        targets = batch["target"]

        rows = {m: [] for m in ["omega","spectral_entropy","apen","sampen","permen","wavelet_entropy","lle"]}
        sid_out = []
        dim_out = []

        for i, tgt in enumerate(targets):
            # multivariate: compute per-dim then average (simple, fast)
            # inside _batch_fn, in the for-loop over `targets`
            if is_multivariate_target(tgt):
                # if NumPy array, iterate over first axis; if list/tuple, iterate elements
                if isinstance(tgt, np.ndarray):
                    iter_dims = [tgt[d, :] for d in range(tgt.shape[0])]
                else:
                    iter_dims = tgt

                vals = []
                for arr in iter_dims:
                    arr = prepare_series(np.asarray(arr, dtype=float), truncate, downsample)
                    if arr.std() < 1e-6:
                        print(f"[WARN] {label}/{split_name} series {i}: near-constant (std={arr.std():.2e}, mean={arr.mean():.2f})")
                    vals.append(compute_metric_row(arr, metrics, **metric_kwargs))
                # average across dims
                merged = {
                    k: float(np.nanmean([v.get(k, np.nan) for v in vals
                                        if np.isfinite(v.get(k, np.nan))]))
                    if any(k in v for v in vals) else np.nan
                    for k in ["omega","spectral_entropy","apen","sampen","permen","wavelet_entropy","lle"]
                }
                for k in rows:
                    rows[k].append(merged[k])
                sid_out.append(str(series_ids[i]) if series_ids is not None else str(i))
                dim_out.append(None)
            else:
                arr = prepare_series(np.asarray(tgt, dtype=float), truncate, downsample)
                res = compute_metric_row(arr, metrics, **metric_kwargs)
                for k in rows: rows[k].append(res.get(k, np.nan))
                sid_out.append(str(series_ids[i]) if series_ids is not None else str(i))
                dim_out.append(None)


        out["series_id"] = sid_out
        out["dim"] = dim_out
        for k, v in rows.items():
            out[k] = v
        return out

    # Cap workers for small splits to avoid HF's warning/errors
    effective_num_proc = max(1, min(num_proc, len(dset)))
    effective_batch = min(batch_size, max(1, len(dset)))

    mapped = dset.map(
        _batch_fn,
        batched=True,
        batch_size=effective_batch,
        num_proc=effective_num_proc,
        desc=f"{label}/{split_name}"
    )


    cols = ["series_id","dim","omega","spectral_entropy","apen","sampen","permen","wavelet_entropy","lle"]
    present = [c for c in cols if c in mapped.column_names]
    per_series_df = pd.DataFrame({c: mapped[c] for c in present})
    per_series_df["dataset_label"] = label
    per_series_df["split"] = split_name
    per_series_df["dataset_path"] = str(ds_path)

   # summary = mean across series per metric (only if column exists and has any finite values)
    summary = {
        "dataset_label": label,
        "split": split_name,
        "n_series": int(len(per_series_df)),
        "dataset_path": str(ds_path),
    }
    for m in ["omega","spectral_entropy","lle","apen","sampen","permen","wavelet_entropy"]:
        if m in per_series_df.columns:
            col = pd.to_numeric(per_series_df[m], errors="coerce")
            finite = col[np.isfinite(col.values)]
            summary[m] = float(np.nanmean(finite)) if finite.size else np.nan

    return per_series_df, summary


# -------------------- Properties join --------------------

ALIASES = {
    # Add folder->props mappings here if needed, e.g.:
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


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", default="series", help="Root containing HF datasets on disk")
    ap.add_argument("--props", default="git_repo/notebooks/dataset_properties.json",
                    help="JSON mapping dataset -> {domain, frequency, num_variates}")
    ap.add_argument("--metrics", default="omega,spectral_entropy",
                    help="Comma-separated list. Options: omega,spectral_entropy,permen,wavelet_entropy,apen,sampen,lle")
    ap.add_argument("--max-series", type=int, default=0, help="Limit #series per split (0 = all)")
    ap.add_argument("--truncate", type=int, default=100000, help="Head truncate each series before metrics")
    ap.add_argument("--downsample", type=int, default=1, help="Stride for downsampling (1 = none)")
    ap.add_argument("--batch-size", type=int, default=128, help="datasets.map batch size")
    ap.add_argument("--num-proc", type=int, default=max((os.cpu_count() or 2) - 1, 1),
                    help="Parallel workers for datasets.map")
    # heavy metric params
    ap.add_argument("--apen-m", type=int, default=2)
    ap.add_argument("--apen-r", type=float, default=0.2)
    ap.add_argument("--sampen-m", type=int, default=2)
    ap.add_argument("--sampen-r", type=float, default=0.2)
    ap.add_argument("--permen-m", type=int, default=3)
    ap.add_argument("--permen-tau", type=int, default=1)
    ap.add_argument("--lle-m", type=int, default=4)
    ap.add_argument("--lle-tau", type=int, default=1)
    ap.add_argument("--lle-steps", type=int, default=10)
    ap.add_argument("--write-series", action="store_true", help="Also write per-series wide CSV (large)")
    ap.add_argument("--strict-domains", action="store_true", help="Raise if any dataset fails to map to domain")
    args = ap.parse_args()

    series_root = Path(args.series_root).expanduser().resolve()
    props_path = Path(args.props).expanduser().resolve()
    if not props_path.exists():
        raise FileNotFoundError(f"Properties file not found: {props_path}")

    props_df = load_properties(props_path)

    ds_dirs = collect_all_hf_datasets(series_root)
    if not ds_dirs:
        raise FileNotFoundError(f"No HF datasets found under {series_root}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metric_kwargs = dict(
        apen_m=args.apen_m, apen_r=args.apen_r,
        sampen_m=args.sampen_m, sampen_r=args.sampen_r,
        permen_m=args.permen_m, permen_tau=args.permen_tau,
        lle_m=args.lle_m, lle_tau=args.lle_tau, lle_steps=args.lle_steps
    )

    all_series_rows = []
    all_summaries = []

    for ds_path in ds_dirs:
        label = human_label(ds_path)
        obj = load_from_disk(ds_path)
        split_items = obj.items() if isinstance(obj, dict) else [("all", obj)]

        for split_name, dset in split_items:
            ps_df, sum_row = compute_split_fast(
                dset, label, split_name, ds_path,
                metrics=metrics,
                num_proc=args.num_proc,
                batch_size=args.batch_size,
                max_series=args.max_series,
                truncate=args.truncate,
                downsample=args.downsample,
                metric_kwargs=metric_kwargs
            )
            all_summaries.append(sum_row)
            if args.write_series and not ps_df.empty:
                all_series_rows.append(ps_df)

    # Build final dataframes
    summary_df = pd.DataFrame(all_summaries)
    if args.write_series and all_series_rows:
        per_series_df = pd.concat(all_series_rows, ignore_index=True)
    else:
        per_series_df = pd.DataFrame(columns=["dataset_label"])

    # Canonical base + properties join
    if not summary_df.empty:
        summary_df["dataset_base"] = summary_df["dataset_label"].apply(base_from_label)
        summary_df["dataset_base"] = summary_df["dataset_base"].apply(lambda x: ALIASES.get(x, x))
        summary_df = summary_df.merge(props_df, how="left", on="dataset_base")

        unmatched = sorted(summary_df.loc[summary_df["domain"].isna(), "dataset_base"].unique())
        if unmatched:
            msg = "[WARN] Unmatched dataset_base (no domain in props): " + ", ".join(unmatched)
            if args.strict_domains:
                raise RuntimeError(msg)
            else:
                print(msg)

        # Order columns: dynamic based on metrics present
        metric_cols = [m for m in ["omega","spectral_entropy","permen","wavelet_entropy","apen","sampen","lle"]
                       if m in summary_df.columns]
        summary_cols = ["dataset_label","dataset_base","domain","frequency","num_variates",
                        "split","n_series"] + metric_cols + ["dataset_path"]
        summary_df = summary_df[summary_cols].sort_values(["dataset_label","split"])
        summary_df.to_csv("metrics_summary_wide.csv", index=False)
        print(f"[OK] wrote metrics_summary_wide.csv  ({len(summary_df)} rows)")

    if args.write_series and not per_series_df.empty:
        per_series_df["dataset_base"] = per_series_df["dataset_label"].apply(base_from_label)
        per_series_df["dataset_base"] = per_series_df["dataset_base"].apply(lambda x: ALIASES.get(x, x))
        per_series_df = per_series_df.merge(props_df, how="left", on="dataset_base")

        metric_cols = [m for m in ["omega","spectral_entropy","permen","wavelet_entropy","apen","sampen","lle"]
                       if m in per_series_df.columns]
        per_cols = ["dataset_label","dataset_base","domain","frequency","num_variates",
                    "split","series_id","dim"] + metric_cols + ["dataset_path"]
        per_series_df = per_series_df[per_cols]
        per_series_df.to_csv("metrics_per_series_wide.csv", index=False)
        print(f"[OK] wrote metrics_per_series_wide.csv ({len(per_series_df)} rows)")


if __name__ == "__main__":
    main()
