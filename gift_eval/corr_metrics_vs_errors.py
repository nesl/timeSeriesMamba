#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlate dataset difficulty metrics (Ω, spectral entropy, etc.)
with normalized accuracy metrics (default: sMAPE, NRMSE, ND, MASE).

Defaults:
  --error-metrics sMAPE[0.5],NRMSE[mean],ND[0.5],MASE[0.5]
  --metrics-cols omega,spectral_entropy,permen,wavelet_entropy,apen,sampen,lle

Outputs in corr_out/:
  - corr_overall.csv
  - corr_within_domain.csv           (if --within-domain)
  - corr_by_model.csv                (if --by-model)
  - joined_dataset_table.csv         (mean-aggregated join)
  - figures/scatter_*.png
Also prints diagnostics on name matching between metrics and results.

Usage:
  python corr_metrics_vs_errors.py \
    --metrics-csv metrics_summary_wide.csv \
    --results-csv merged_gift_results.csv \
    --within-domain --by-model
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def winsorize(s: pd.Series, p: float = 0.02) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def scatter_robust(figdir: Path, x, y, labels, xlab, ylab, title, fname,
                   winsor_p: float = 0.02, label_topk: int = 3):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    lab = labels.fillna("").astype(str)

    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4: 
        return

    xv = winsorize(x[mask], p=winsor_p).values
    yv = winsorize(y[mask], p=winsor_p).values
    labs = lab[mask].values

    plt.figure()
    plt.scatter(xv, yv, s=16, alpha=0.75)

    # robust-ish line: least squares on winsorized values
    try:
        m, b = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        plt.plot(xs, m*xs + b)
    except Exception:
        pass

    # label top-k absolute residuals
    if label_topk > 0:
        resid = np.abs(yv - (m * xv + b)) if 'm' in locals() else np.abs(yv - np.median(yv))
        idx = np.argsort(resid)[-label_topk:]
        for i in idx:
            plt.annotate(labs[i], (xv[i], yv[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout()
    plt.savefig(figdir / fname, dpi=150)
    plt.close()


def canon(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def dataset_base_from_results(name: str) -> str:
    return canon(str(name).split("/")[0])

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def add_scatter(figdir: Path, x, y, xlab, ylab, title, fname):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return
    xv, yv = x[mask].values, y[mask].values
    plt.figure()
    plt.scatter(xv, yv, s=16, alpha=0.75)
    try:
        m, b = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        plt.plot(xs, m*xs + b)
    except Exception:
        pass
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(figdir / fname, dpi=150); plt.close()

def load_aliases(path: str | None) -> dict:
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=False, default="metrics_summary_wide.csv")
    ap.add_argument("--results-csv", required=False, default="merged_gift_results.csv")
    # Default to normalized error metrics
    ap.add_argument("--error-metrics", default="sMAPE[0.5],NRMSE[mean],ND[0.5],MASE[0.5]",
                    help="Comma-separated error metrics to analyze from results CSV")
    ap.add_argument("--metrics-cols", default="omega,spectral_entropy,permen,wavelet_entropy,apen,sampen,lle",
                    help="Comma-separated difficulty metrics to analyze (use those present)")
    ap.add_argument("--within-domain", action="store_true", help="Report domain-demeaned correlations")
    ap.add_argument("--by-model", action="store_true", help="Also compute correlations per model")
    ap.add_argument("--aliases-json", default=None,
                    help="Optional JSON file mapping dataset_base aliases, e.g. {'electricity_hourly':'electricity'}")
    ap.add_argument("--figdir", default="corr_out/figures")
    ap.add_argument("--outdir", default="corr_out")
    ap.add_argument("--topn_print", type=int, default=30, help="How many unmatched names to print per side")
    args = ap.parse_args()

    outdir = Path(args.outdir); figdir = Path(args.figdir)
    ensure_dirs(outdir, figdir)

    aliases = load_aliases(args.aliases_json)

    # ----- load metrics side -----
    mdf = pd.read_csv(args.metrics_csv)
    # ensure dataset_base
    if "dataset_base" not in mdf.columns:
        if "dataset_path" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_path"].astype(str).apply(lambda p: canon(Path(p).parent.name))
        elif "dataset_label" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_label"].astype(str).apply(lambda s: canon(s.split("/")[0]))
        else:
            raise ValueError("metrics CSV missing dataset_base and dataset_path/dataset_label to infer it.")
    mdf["dataset_base"] = mdf["dataset_base"].astype(str).apply(canon).apply(lambda x: aliases.get(x, x))

    want_metrics = [c.strip() for c in args.metrics_cols.split(",") if c.strip()]
    have_metrics = [c for c in want_metrics if c in mdf.columns]
    if not have_metrics:
        raise ValueError("No requested difficulty metric columns found in metrics CSV.")

    # Aggregate metrics per dataset_base (mean across splits)
    met_agg = mdf.groupby("dataset_base", as_index=False)[have_metrics].mean(numeric_only=True)
    for aux in ["domain","frequency","num_variates"]:
        if aux in mdf.columns:
            met_agg[aux] = (mdf.groupby("dataset_base")[aux]
                              .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""))

    # ----- load results side -----
    rdf = pd.read_csv(args.results_csv)
    rdf.columns = [c.replace("eval_metrics/", "") for c in rdf.columns]  # flatten
    if "dataset" not in rdf.columns:
        cand = [c for c in rdf.columns if "dataset" in c.lower()]
        if not cand:
            raise ValueError("results CSV missing 'dataset' column.")
        rdf["dataset"] = rdf[cand[0]]
    rdf["dataset_base"] = rdf["dataset"].astype(str).apply(dataset_base_from_results).apply(lambda x: aliases.get(x, x))
    # pick error metrics
    err_cols_req = [c.strip() for c in args.error_metrics.split(",") if c.strip()]
    err_cols = [c for c in err_cols_req if c in rdf.columns]
    if not err_cols:
        raise ValueError(f"None of the requested error metrics are present: {err_cols_req}")
    for c in err_cols:
        rdf[c] = pd.to_numeric(rdf[c], errors="coerce")

    # ----- diagnostics on matching -----
    left = set(met_agg["dataset_base"].unique())
    right = set(rdf["dataset_base"].unique())
    inter = sorted(left & right)
    only_metrics = sorted(left - right)
    only_results = sorted(right - left)

    print("\n--- Matching diagnostics ---")
    print(f"metrics side unique datasets: {len(left)}")
    print(f"results side unique datasets: {len(right)}")
    print(f"INTERSECTION (will be plotted): {len(inter)}")
    if only_metrics:
        print(f"\nPresent in metrics but missing in results (showing up to {args.topn_print}):")
        for s in only_metrics[:args.topn_print]: print("  ", s)
    if only_results:
        print(f"\nPresent in results but missing in metrics (showing up to {args.topn_print}):")
        for s in only_results[:args.topn_print]: print("  ", s)
    print("Tip: add remaps to --aliases-json if names differ (e.g., 'electricity_hourly'→'electricity').\n")

    # ----- aggregation strategies across models/horizons -----
    agg_strategies = {
        "mean": lambda df: df.groupby("dataset_base", as_index=False)[err_cols].mean(numeric_only=True),
        "median": lambda df: df.groupby("dataset_base", as_index=False)[err_cols].median(numeric_only=True),
        "best": lambda df: df.groupby("dataset_base", as_index=False)[err_cols].min(numeric_only=True),  # lower=better
    }

    corrs = []
    joined_tables = {}

    for name, fn in agg_strategies.items():
        err_agg = fn(rdf)
        joined = met_agg.merge(err_agg, how="inner", on="dataset_base")
        joined_tables[name] = joined.copy()

        for m in have_metrics:
            for e in err_cols:
                x = pd.to_numeric(joined[m], errors="coerce")
                y = pd.to_numeric(joined[e], errors="coerce")
                xw = winsorize(x, p=0.02); yw = winsorize(y, p=0.02)

                mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                pear = float(pd.concat([xw[mask], yw[mask]], axis=1).corr(method="pearson").iloc[0,1]) if mask.sum()>=4 else np.nan
                spear = float(pd.concat([x[mask], y[mask]], axis=1).corr(method="spearman").iloc[0,1]) if mask.sum()>=4 else np.nan
                corrs.append({"aggregation": name, "metric": m, "error": e, "pearson": pear, "spearman": spear, "n": int(mask.sum())})

        # plots for 'mean' aggregation
        if name == "mean":
            for m in have_metrics:
                for e in err_cols:
                    #add_scatter(figdir, joined[m], joined[e], m, e, f"{m} vs {e} ({name})", f"scatter_{m}_vs_{e}_{name}.png")
                    # label with dataset names to spot the offenders
                    labels = joined.get("dataset_base", pd.Series([None]*len(joined)))
                    scatter_robust(figdir, joined[m], joined[e], labels,
                                m, e, f"{m} vs {e} (mean; winsorized)", f"scatter_{m}_vs_{e}_mean.png",
                                winsor_p=0.02, label_topk=5)
    pd.DataFrame(corrs).sort_values(["aggregation","error","metric"]).to_csv(outdir / "corr_overall.csv", index=False)

    # within-domain (optional)
    if args.within_domain and "domain" in met_agg.columns:
        wd_rows = []
        for name, joined in joined_tables.items():
            if "domain" not in joined.columns: continue
            dom = joined["domain"].astype(str).fillna("")
            for m in have_metrics:
                for e in err_cols:
                    x = pd.to_numeric(joined[m], errors="coerce")
                    y = pd.to_numeric(joined[e], errors="coerce")
                    x_resid = x - joined.groupby(dom)[m].transform("mean")
                    y_resid = y - joined.groupby(dom)[e].transform("mean")
                    mask = x_resid.notna() & y_resid.notna() & np.isfinite(x_resid) & np.isfinite(y_resid)
                    pear = float(pd.concat([x_resid[mask], y_resid[mask]], axis=1).corr(method="pearson").iloc[0,1]) if mask.sum()>=4 else np.nan
                    spear = float(pd.concat([x_resid[mask], y_resid[mask]], axis=1).corr(method="spearman").iloc[0,1]) if mask.sum()>=4 else np.nan
                    wd_rows.append({"aggregation": name, "metric": m, "error": e, "pearson": pear, "spearman": spear, "n": int(mask.sum())})
        if wd_rows:
            pd.DataFrame(wd_rows).sort_values(["aggregation","error","metric"]).to_csv(outdir / "corr_within_domain.csv", index=False)

    # per-model (optional)
    if args.by_model and "model" in rdf.columns:
        rows = []
        for model, sub in rdf.groupby("model"):
            err_agg = sub.groupby("dataset_base", as_index=False)[err_cols].mean(numeric_only=True)
            joined = met_agg.merge(err_agg, how="inner", on="dataset_base")
            for m in have_metrics:
                for e in err_cols:
                    x = pd.to_numeric(joined[m], errors="coerce")
                    y = pd.to_numeric(joined[e], errors="coerce")
                    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                    pear = float(pd.concat([x[mask], y[mask]], axis=1).corr(method="pearson").iloc[0,1]) if mask.sum()>=4 else np.nan
                    spear = float(pd.concat([x[mask], y[mask]], axis=1).corr(method="spearman").iloc[0,1]) if mask.sum()>=4 else np.nan
                    rows.append({"model": model, "metric": m, "error": e, "pearson": pear, "spearman": spear, "n": int(mask.sum())})
        if rows:
            pd.DataFrame(rows).sort_values(["model","error","metric"]).to_csv(outdir / "corr_by_model.csv", index=False)

    # export main joined table
    joined_tables["mean"].to_csv(outdir / "joined_dataset_table.csv", index=False)
    print(f"\n[OK] wrote correlations & tables to {outdir} and plots to {figdir}")

if __name__ == "__main__":
    main()
