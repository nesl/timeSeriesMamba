#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-run Ω-conditioned stats & ablations.

- Scans default result directories for CSV files (recursively).
- Infers domain & model from PATH (per your rules).
- Auto-detects MSE and Ω columns from common names/patterns.
- Computes:
    • Base error vs Ω stats (per domain×model): Spearman ρ (+ CI), Pearson r,
      Kendall τ, Theil–Sen slope (+ CI), sign-test p.
    • Δ(Ω) stats for:
         - Language Pretrained → DLinear
         - Pretrained → Random Init
         - LLaMA → GPT-2
- Writes CSVs to DEFAULT_OUT_DIR.

Run:
    python omega_stats_autorun.py
"""

import os
import re
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

# Toggle small QC plots (PNG)
SAVE_QC_PLOTS = False

# ------------------- Column detection patterns -------------------
MSE_PATS = [
    re.compile(r"\bmse(_loss)?\b", re.I),
    re.compile(r"\bmean_squar(?:ed|e)_error\b", re.I),
]
OMEGA_PATS = [
    re.compile(r"\bomega(_ctx)?(_mean)?\b", re.I),
    re.compile(r"\bomega\b", re.I),
]

# ------------------- Model inference -------------------
def infer_model_from_path(path: str) -> Optional[str]:
    b = os.path.basename(path)
    b_up = b.upper()
    b_lo = b.lower()
    # DLinear
    if "dlin" in b_lo:
        return "DLinear"
    # GPT2 (e.g., ..._mGPT2_... or ..._GPT2_...)
    if "mgpt2" in b_lo or re.search(r"\bgpt2\b", b_lo):
        return "GPT2"
    # LLAMA3.2 families with r0/r1 flag
    if "LLAMA3.2" in b_up:
        if "_R0_" in b_up:
            return "Language Pretrained"
        if "_R1_" in b_up:
            return "Random Init"
        return "LLAMA"  # fallback (size ablation bucket)
    return None

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

# ------------------- Small stats helpers -------------------
def _spearman(x, y):
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])

def _pearson(x, y):
    if len(x) < 2: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def _kendall_tau(x, y):
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(x, y, nan_policy="omit")[0])
    except Exception:
        return np.nan

def _theilsen_slope(x, y, max_pairs=20000, rng: Optional[np.random.Generator]=None):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n < 2: return np.nan
    if rng is None: rng = np.random.default_rng(0)
    # random i<j sampling for speed
    m = min(max_pairs, max(1, n*(n-1)//2))
    i = rng.integers(0, n-1, size=m)
    j = rng.integers(i+1, n, size=m)
    dx = x[j] - x[i]
    mask = np.abs(dx) > 1e-12
    slopes = (y[j] - y[i])[mask] / dx[mask]
    if slopes.size == 0: return np.nan
    return float(np.median(slopes))

def _bootstrap_ci(stat_fn, x, y, iters=2000, alpha=0.05, rng=None):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n < 3:
        s = stat_fn(x, y)
        return s, np.nan, np.nan
    if rng is None: rng = np.random.default_rng(0)
    s0 = stat_fn(x, y)
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(stat_fn(x[idx], y[idx]))
    lo = float(np.nanpercentile(boots, 100*alpha/2))
    hi = float(np.nanpercentile(boots, 100*(1 - alpha/2)))
    return s0, lo, hi

def _sign_test_on_slope(x, y, iters=1000, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    s_obs = _theilsen_slope(x, y, max_pairs=min(5000, len(x)*(len(x)-1)//2), rng=rng)
    if not np.isfinite(s_obs): return np.nan
    y_cent = np.asarray(y) - np.median(y)
    sims = []
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=len(y_cent))
        sims.append(_theilsen_slope(x, y_cent*signs, max_pairs=3000, rng=rng))
    sims = np.asarray(sims)
    p = 2 * min((np.sum(sims >= s_obs) + 1) / (iters + 1),
                (np.sum(sims <= s_obs) + 1) / (iters + 1))
    return float(min(1.0, p))

def _equal_count_bins(v, n_bins):
    v = pd.Series(v)
    try:
        bins = pd.qcut(v, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        uq = v.nunique()
        bins = pd.qcut(v, q=max(2, min(n_bins, uq)), labels=False, duplicates="drop")
    return bins.to_numpy()

def _first_crossing_x(xs, ys):
    xs = np.asarray(xs); ys = np.asarray(ys)
    for i in range(len(xs)-1):
        y1, y2 = ys[i], ys[i+1]
        if np.sign(y1) == np.sign(y2): continue
        t = abs(y1) / (abs(y1) + abs(y2))
        return float(xs[i] + t*(xs[i+1]-xs[i]))
    return np.nan

# ------------------- Load & normalize -------------------
def find_txts_from_defaults() -> List[str]:
    files = []
    for d in DEFAULT_LOG_DIRS:
        files.extend(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
    return sorted(set(files))

def detect_col(df: pd.DataFrame, pats: List[re.Pattern]) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        for p in pats:
            if p.search(c):
                return c
    return None

def add_domain_model(df: pd.DataFrame, src_path: str) -> pd.DataFrame:
    # Domain from directory name or path text
    dom = infer_domain_from_text(src_path)
    if dom is None:
        dom = domain_name_from_dir(os.path.dirname(src_path))
    df["domain"] = dom if dom in DOMAINS_CANON else dom
    # Model from basename pattern
    mdl = infer_model_from_path(src_path)
    df["model"] = mdl if mdl is not None else "Unknown"
    df["__src__"] = os.path.basename(src_path)
    return df

def load_all() -> pd.DataFrame:
    files = find_txts_from_defaults()
    if not files:
        raise FileNotFoundError("No txt found under DEFAULT_LOG_DIRS.")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df = add_domain_model(df, f)
        frames.append(df)
    if not frames:
        raise RuntimeError("No readable txt.")
    df = pd.concat(frames, ignore_index=True)

    # Detect columns
    col_mse = detect_col(df, MSE_PATS) or ("mse" if "mse" in df.columns else None)
    col_om  = detect_col(df, OMEGA_PATS) or ("omega" if "omega" in df.columns else None)
    if col_mse is None or col_om is None:
        raise KeyError(f"Could not detect MSE/Ω columns. Found: {list(df.columns)[:20]} ...")
    df = df.rename(columns={col_mse: "mse", col_om: "omega"})

    # Seed optional
    if "seed" not in df.columns:
        df["seed"] = 0

    # Basic clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["omega", "mse"])
    df["omega"] = df["omega"].clip(0.0, 1.0)

    # Keep only known domains/models
    df = df[df["model"].isin(["DLinear", "GPT2", "Language Pretrained", "Random Init", "LLAMA"])]

    return df

# ------------------- Base stats -------------------
def base_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dom, mod), g in df.groupby(["domain", "model"]):
        g = g[["omega", "mse"]].dropna()
        if len(g) < 5: 
            continue
        x = g["omega"].to_numpy()
        y = g["mse"].to_numpy()
        pear = _pearson(x, y)
        rho, lo_rho, hi_rho = _bootstrap_ci(_spearman, x, y, iters=2000, alpha=0.05)
        tau = _kendall_tau(x, y)
        slope_fn = lambda X, Y: _theilsen_slope(X, Y, max_pairs=20000)
        slp, sl_lo, sl_hi = _bootstrap_ci(slope_fn, x, y, iters=2000, alpha=0.05)
        p_sign = _sign_test_on_slope(x, y, iters=1000)
        rows.append({
            "domain": dom, "model": mod, "n": len(g),
            "pearson_r": pear, "spearman_rho": rho,
            "spearman_ci_low": lo_rho, "spearman_ci_high": hi_rho,
            "kendall_tau": tau,
            "theilsen_slope": slp, "slope_ci_low": sl_lo, "slope_ci_high": sl_hi,
            "sign_test_p": p_sign
        })
    return pd.DataFrame(rows)

# ------------------- Δ(Ω) engine -------------------
def delta_by_bins_single_domain(df_dom: pd.DataFrame,
                                label_A: str, label_B: str,
                                bins: int = 10, min_bin_count: int = 8):
    """
    Build equal-count Ω bins, aggregate median MSE per model per bin,
    compute Δ_{A→B}(bin) = 100*(MSE_A - MSE_B)/MSE_A.
    """
    bins_lab = _equal_count_bins(df_dom["omega"], bins)
    dd = df_dom.copy()
    dd["omega_bin"] = bins_lab

    agg = dd.groupby(["omega_bin", "model"]).agg(
        omega_med=("omega", "median"),
        mse_med=("mse", "median"),
        n=("__src__", "count")
    ).reset_index()

    A_tbl = agg[agg["model"] == label_A]
    B_tbl = agg[agg["model"] == label_B]
    if A_tbl.empty or B_tbl.empty:
        return pd.DataFrame(), pd.DataFrame()

    bins_common = sorted(set(A_tbl["omega_bin"]).intersection(set(B_tbl["omega_bin"])))
    rows = []
    for b in bins_common:
        a = A_tbl[A_tbl["omega_bin"] == b]
        bb = B_tbl[B_tbl["omega_bin"] == b]
        if a.empty or bb.empty:
            continue
        mseA = float(a["mse_med"].iloc[0])
        mseB = float(bb["mse_med"].iloc[0])
        om   = float(np.median(dd[dd["omega_bin"]==b]["omega"]))
        nbin = int(dd[dd["omega_bin"]==b].shape[0])
        if not np.isfinite(mseA) or not np.isfinite(mseB) or mseA <= 0:
            continue
        delta_pct = 100.0 * (mseA - mseB) / mseA  # positive => B better
        rows.append({"omega_bin": int(b), "omega_med": om, "n_bin": nbin, "delta_pct": delta_pct})

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    tbl = pd.DataFrame(rows).sort_values("omega_med")
    tbl = tbl[tbl["n_bin"] >= min_bin_count].reset_index(drop=True)
    if len(tbl) < 3:
        return tbl, pd.DataFrame()

    x = tbl["omega_med"].to_numpy()
    y = tbl["delta_pct"].to_numpy()
    pear = _pearson(x, y)
    rho, lo_rho, hi_rho = _bootstrap_ci(_spearman, x, y, iters=2000, alpha=0.05)
    slope_fn = lambda X, Y: _theilsen_slope(X, Y, max_pairs=20000)
    slp, sl_lo, sl_hi = _bootstrap_ci(slope_fn, x, y, iters=2000, alpha=0.05)
    p_sign = _sign_test_on_slope(x, y, iters=1000)
    omega_star = _first_crossing_x(x, y)
    stats_row = pd.DataFrame([{
        "n_bins": len(tbl),
        "pearson_r": pear,
        "spearman_rho": rho,
        "spearman_ci_low": lo_rho,
        "spearman_ci_high": hi_rho,
        "theilsen_slope": slp,
        "slope_ci_low": sl_lo,
        "slope_ci_high": sl_hi,
        "sign_test_p": p_sign,
        "omega_star": omega_star
    }])
    return tbl, stats_row

def run_delta_suite(df: pd.DataFrame, out_dir: str, save_qc: bool = False):
    """
    Execute the three Δ(Ω) analyses domain-wise and save CSVs (+ optional QC plots).
    """
    pairs = [
        ("LangPretrained_to_DLinear", "Language Pretrained", "DLinear"),
        ("Pretrained_to_RandomInit",  "Language Pretrained", "Random Init"),
        ("LLaMA_to_GPT2",             "LLAMA",               "GPT2"),  # size ablation bucket
    ]

    for tag, A, B in pairs:
        all_stats, all_bins = [], []
        for dom, g in df.groupby("domain"):
            gd = g[g["model"].isin([A, B])].copy()
            if gd.empty:
                continue
            bins_tbl, stats_row = delta_by_bins_single_domain(gd, A, B, bins=10, min_bin_count=8)
            if bins_tbl.empty or stats_row.empty:
                continue
            bins_tbl.insert(0, "domain", dom)
            stats_row.insert(0, "domain", dom)
            all_bins.append(bins_tbl)
            all_stats.append(stats_row)

            if save_qc:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(4.2, 3.2), dpi=140)
                    plt.scatter(bins_tbl["omega_med"], bins_tbl["delta_pct"], s=28, alpha=0.85)
                    ord_idx = np.argsort(bins_tbl["omega_med"].to_numpy())
                    med = pd.Series(bins_tbl["delta_pct"].to_numpy())[ord_idx].rolling(
                        3, min_periods=1, center=True
                    ).median()
                    plt.plot(bins_tbl["omega_med"].to_numpy()[ord_idx], med.to_numpy(), lw=1.3)
                    plt.axhline(0.0, lw=1.0, ls="--", color="k", alpha=0.6)
                    if np.isfinite(float(stats_row["omega_star"].iloc[0])):
                        plt.axvline(float(stats_row["omega_star"].iloc[0]), lw=1.0, ls=":", alpha=0.7)
                    plt.xlabel("Ω (bin median)")
                    plt.ylabel(f"Δ % ({tag.replace('_',' ')})")
                    plt.title(f"{dom}: Δ vs Ω")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"{dom}_{tag}_qc.png"))
                    plt.close()
                except Exception as e:
                    print(f"[warn] QC plot failed for {dom}/{tag}: {e}")

        if all_stats:
            stats_df = pd.concat(all_stats, ignore_index=True)
            stats_df.to_csv(os.path.join(out_dir, f"delta_stats_{tag}.csv"), index=False)
            print(f"[ok] delta_stats_{tag}.csv written ({len(stats_df)} domains)")
        if all_bins:
            bins_df = pd.concat(all_bins, ignore_index=True)
            bins_df.to_csv(os.path.join(out_dir, f"delta_bins_{tag}.csv"), index=False)
            print(f"[ok] delta_bins_{tag}.csv written ({len(bins_df)} rows)")

# ------------------- Main -------------------
def main():
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    df = load_all()

    # Base stats
    base_df = base_error_stats(df)
    base_df.sort_values(["domain", "model"]).to_csv(
        os.path.join(DEFAULT_OUT_DIR, "base_error_vs_omega_stats.csv"), index=False
    )
    print(f"[ok] base_error_vs_omega_stats.csv written ({len(base_df)} rows)")

    # Δ suites
    run_delta_suite(df, DEFAULT_OUT_DIR, save_qc=SAVE_QC_PLOTS)

    print(f"[DONE] Outputs in: {DEFAULT_OUT_DIR}")

if __name__ == "__main__":
    main()
