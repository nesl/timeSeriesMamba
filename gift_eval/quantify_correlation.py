#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify correlation strength with CIs, robust variants, and outlier sensitivity.
Also emits both winsorized and non-winsorized scatter plots.

Usage
-----
python quantify_correlation.py \
  --joined corr_out/joined_dataset_table.csv \
  --x omega \
  --y "sMAPE[0.5]" \
  --label-col dataset_base \
  --domain-col domain \
  --winsor 0.02 \
  --max-trim 3 \
  --figdir corr_out/figures \
  --raw-figdir corr_out/figures_raw \
  --fig-w 6.4 --fig-h 4.4 --dpi 300 --x-margin 0.02 --y-margin 0.05 --tight-bbox
"""

from __future__ import annotations
import argparse, numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None  # We'll fall back to simple calcs if SciPy not present

# ---------- math helpers ----------
def fisher_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float,float]:
    """95% CI for Pearson using Fisher z-transform (approx, needs n>3)."""
    if not np.isfinite(r) or n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3.0)
    zcrit = stats.norm.ppf(1 - alpha/2) if stats else 1.96
    lo = np.tanh(z - zcrit*se)
    hi = np.tanh(z + zcrit*se)
    return float(lo), float(hi)

def winsorize(s: pd.Series, p: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def pearson(x: pd.Series, y: pd.Series) -> Tuple[float,int]:
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return (np.nan, int(m.sum()))
    r = float(pd.concat([x[m], y[m]], axis=1).corr(method="pearson").iloc[0,1])
    return r, int(m.sum())

def spearman(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return (np.nan, np.nan, int(m.sum()))
    if stats:
        rho, p = stats.spearmanr(x[m], y[m], nan_policy="omit")
        return float(rho), float(p), int(m.sum())
    else:
        rho = float(pd.concat([x[m], y[m]], axis=1).corr(method="spearman").iloc[0,1])
        return rho, np.nan, int(m.sum())

def leverage_and_residuals(x: np.ndarray, y: np.ndarray):
    """Return standardized residuals and leverage to rank influential points."""
    X = np.c_[np.ones_like(x), x]
    XtX_inv = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)  # diagonal of hat matrix
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat
    s2 = (resid**2).sum() / (len(x) - 2)
    stud = resid / np.sqrt(s2 * (1 - h))
    return stud, h

def trim_topk_by_influence(df: pd.DataFrame, xcol: str, ycol: str, label: Optional[pd.Series], k: int):
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m].values, y[m].values
    if len(xv) < 5 or k <= 0:
        return df[m], []
    stud, h = leverage_and_residuals(xv, yv)
    cooks = (stud**2) * h / (2 * (1 - h))  # Cook's distance (p=2)
    idx = np.argsort(cooks)[::-1][:min(k, len(cooks))]
    kept = np.ones_like(cooks, dtype=bool); kept[idx] = False
    trimmed = df[m].iloc[kept]
    removed_labels = (label[m].iloc[idx].tolist() if label is not None else [str(i) for i in idx])
    return trimmed, removed_labels

def partial_corr_within_domain(df: pd.DataFrame, xcol: str, ycol: str, domain_col: str):
    if domain_col not in df.columns: return np.nan, np.nan, 0
    dom = df[domain_col].astype(str).fillna("")
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    x_res = x - df.groupby(dom)[xcol].transform("mean")
    y_res = y - df.groupby(dom)[ycol].transform("mean")
    m = x_res.notna() & y_res.notna() & np.isfinite(x_res) & np.isfinite(y_res)
    if m.sum() < 3: return np.nan, np.nan, 0
    r = float(pd.concat([x_res[m], y_res[m]], axis=1).corr(method="pearson").iloc[0,1])
    ci = fisher_ci(r, m.sum())
    return r, ci, int(m.sum())

def permutation_pvalue(x: pd.Series, y: pd.Series, B: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation test for Pearson r."""
    rng = np.random.default_rng(seed)
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return np.nan
    xv, yv = x[m].values, y[m].values
    r_obs = np.corrcoef(xv, yv)[0,1]
    count = 0
    for _ in range(B):
        y_perm = rng.permutation(yv)
        r_perm = np.corrcoef(xv, y_perm)[0,1]
        if abs(r_perm) >= abs(r_obs): count += 1
    return (count + 1) / (B + 1)

# ---------- plotting ----------
def make_scatter(figdir: Path, x, y, xlab: str, ylab: str, title: str, fname: str,
                 fig_w: float, fig_h: float, dpi: int,
                 x_margin: float, y_margin: float, tight_bbox: bool):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return
    xv, yv = x[m].values, y[m].values

    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    plt.scatter(xv, yv, s=18, alpha=0.75)
    try:
        m_, b_ = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        plt.plot(xs, m_*xs + b_)
    except Exception:
        pass

    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    ax = plt.gca()
    # tighten axis margins (fraction of data range)
    xr = xv.max() - xv.min(); yr = yv.max() - yv.min()
    if xr > 0:
        ax.set_xlim(xv.min() - xr * x_margin, xv.max() + xr * x_margin)
    if yr > 0:
        ax.set_ylim(yv.min() - yr * y_margin, yv.max() + yr * y_margin)
    if tight_bbox:
        plt.tight_layout()
        plt.savefig(figdir / fname, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.savefig(figdir / fname)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default="corr_out/joined_dataset_table.csv")
    ap.add_argument("--x", default="omega")
    ap.add_argument("--y", default="sMAPE[0.5]")
    ap.add_argument("--label-col", default="dataset_base")
    ap.add_argument("--domain-col", default="domain")
    ap.add_argument("--winsor", type=float, default=0.02, help="proportion to cap on each tail for winsorized Pearson")
    ap.add_argument("--max-trim", type=int, default=3, help="trim top-k influential points (0..K)")
    ap.add_argument("--out", default="corr_out/quant_summary.csv")

    # NEW: figure controls
    ap.add_argument("--figdir", default="corr_out/figures", help="winsorized plots")
    ap.add_argument("--raw-figdir", default="corr_out/figures_raw", help="non-winsorized plots")
    ap.add_argument("--fig-w", type=float, default=6.4)
    ap.add_argument("--fig-h", type=float, default=4.4)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--x-margin", type=float, default=0.02, help="fractional x padding")
    ap.add_argument("--y-margin", type=float, default=0.05, help="fractional y padding")
    ap.add_argument("--tight-bbox", action="store_true", help="use bbox_inches='tight' on savefig")
    args = ap.parse_args()

    df = pd.read_csv(args.joined)

    # base Pearson
    r, n = pearson(df[args.x], df[args.y])
    r_lo, r_hi = fisher_ci(r, n)
    # Spearman
    rho, p_spear, n_s = spearman(df[args.x], df[args.y])
    # Winsorized Pearson
    xw = winsorize(df[args.x], args.winsor); yw = winsorize(df[args.y], args.winsor)
    rw, nw = pearson(xw, yw); rw_lo, rw_hi = fisher_ci(rw, nw)

    # Jackknife (leave-one-out) range
    x = pd.to_numeric(df[args.x], errors="coerce"); y = pd.to_numeric(df[args.y], errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m].values, y[m].values
    loolist = []
    for i in range(len(xv)):
        mask = np.ones(len(xv), dtype=bool); mask[i] = False
        loolist.append(np.corrcoef(xv[mask], yv[mask])[0,1])
    loo_min, loo_max = (float(np.nanmin(loolist)), float(np.nanmax(loolist))) if loolist else (np.nan, np.nan)
    # Permutation p-value (two-sided) for Pearson
    p_perm = permutation_pvalue(df[args.x], df[args.y], B=5000, seed=0)
    # Partial (within-domain)
    r_dom, r_dom_ci, n_dom = partial_corr_within_domain(df, args.x, args.y, args.domain_col)

    # Summary table
    rows = []
    base = {"variant":"pearson", "r":r, "n":n, "ci_lo":r_lo, "ci_hi":r_hi, "note":"raw"}
    wins = {"variant":"pearson_winsor", "r":rw, "n":nw, "ci_lo":rw_lo, "ci_hi":rw_hi, "note":f"winsor p={args.winsor}"}
    spear = {"variant":"spearman", "r":rho, "n":n_s, "ci_lo":np.nan, "ci_hi":np.nan, "p_value":p_spear, "note":"rank"}
    domp = {"variant":"pearson_within_domain", "r":r_dom, "n":n_dom,
            "ci_lo":(r_dom_ci[0] if isinstance(r_dom_ci, tuple) else np.nan),
            "ci_hi":(r_dom_ci[1] if isinstance(r_dom_ci, tuple) else np.nan),
            "note":"residualized by domain"}
    perm = {"variant":"pearson_perm_test_p", "r":r, "n":n, "ci_lo":np.nan, "ci_hi":np.nan, "p_value":p_perm, "note":"two-sided permutation"}
    rows.extend([base, wins, spear, domp, perm])

    # influence-based trimming
    for k in range(1, max(0, args.max_trim) + 1):
        trimmed, removed = trim_topk_by_influence(df, args.x, args.y, df.get(args.label_col), k)
        rk, nk = pearson(trimmed[args.x], trimmed[args.y])
        rk_lo, rk_hi = fisher_ci(rk, nk)
        rows.append({"variant":f"pearson_trim_top{k}", "r":rk, "n":nk, "ci_lo":rk_lo, "ci_hi":rk_hi,
                     "note":f"removed={removed}"})

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # ---------- plots ----------
    figdir = Path(args.figdir); rawdir = Path(args.raw-figdir if hasattr(args, "raw-figdir") else args.raw_figdir)
    # compat for "-" vs "_" in argparse dest
    if isinstance(rawdir, str): rawdir = Path(rawdir)
    figdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    # winsorized plot
    make_scatter(
        figdir, xw, yw,
        args.x, args.y,
        f"{args.y} vs {args.x} (winsor p={args.winsor})",
        f"scatter_{args.x}_vs_{args.y}_winsor.png",
        args.fig_w, args.fig_h, args.dpi, args.x_margin, args.y_margin, args.tight_bbox
    )
    # raw plot
    make_scatter(
        rawdir, df[args.x], df[args.y],
        args.x, args.y,
        f"{args.y} vs {args.x} (raw)",
        f"scatter_{args.x}_vs_{args.y}_raw.png",
        args.fig_w, args.fig_h, args.dpi, args.x_margin, args.y_margin, args.tight_bbox
    )

    # console summary
    print(f"\n=== {args.x} vs {args.y} ===")
    print(out.to_string(index=False))
    print(f"\nLeave-one-out r range: [{loo_min:.3f}, {loo_max:.3f}] (n={n})")
    print(f"\n[OK] Plots saved to:\n  winsorized -> {figdir}\n  raw        -> {rawdir}")

if __name__ == "__main__":
    main()
