#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlate dataset difficulty metrics (Ω, spectral entropy, etc.)
with normalized accuracy metrics (default: sMAPE, NRMSE, ND, MASE),
and analyze relationships by model type (pretrained, deep, statistical, fine-tuned).

Outputs:
  corr_out/
    - corr_overall.csv
    - corr_within_domain.csv
    - corr_by_model.csv
    - corr_by_modeltype.csv
    - omega_slope_by_modeltype.csv
    - joined_dataset_table.csv
    - figures/scatter_*.png

Usage:
  python corr_metrics_vs_errors_with_modeltypes.py \
    --metrics-csv metrics_summary_wide.csv \
    --results-csv merged_gift_results.csv \
    --modeltype-json model_types.json \
    --within-domain --by-model
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ---------- helpers ----------
def winsorize(s: pd.Series, p: float = 0.02) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def ensure_dirs(*paths: Path):
    for p in paths: p.mkdir(parents=True, exist_ok=True)

def canon(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def dataset_base_from_results(name: str) -> str:
    return canon(str(name).split("/")[0])

def first_nonnull(s, default=""):
    for v in s:
        if pd.notna(v) and v != "":
            return v
    return default
def load_aliases(path: str | None) -> dict:
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_model_types(path: str | None) -> dict:
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

# ---------- scatter plot helpers ----------
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
    plt.scatter(xv, yv, s=18, alpha=0.75)
    try:
        m, b = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        plt.plot(xs, m * xs + b)
    except Exception:
        pass
    if label_topk > 0:
        resid = np.abs(yv - (m * xv + b)) if 'm' in locals() else np.abs(yv - np.median(yv))
        idx = np.argsort(resid)[-label_topk:]
        for i in idx:
            plt.annotate(labs[i], (xv[i], yv[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout()
    plt.savefig(figdir / fname, dpi=150)
    plt.close()

def scatter_by_modeltype(figdir: Path, joined, metric_col, error_col, model_type_col="model_type"):
    import seaborn as sns
    plt.figure()
    sns.scatterplot(data=joined, x=metric_col, y=error_col, hue=model_type_col, s=30, alpha=0.75)
    plt.xlabel(metric_col); plt.ylabel(error_col)
    plt.title(f"{error_col} vs {metric_col} by model type")
    plt.tight_layout()
    plt.savefig(figdir / f"scatter_{error_col}_vs_{metric_col}_by_modeltype.png", dpi=200)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default="metrics_summary_wide.csv")
    ap.add_argument("--results-csv", default="merged_gift_results.csv")
    ap.add_argument("--error-metrics", default="sMAPE[0.5]")
    ap.add_argument("--metrics-cols", default="omega,spectral_entropy,permen,wavelet_entropy,apen,sampen,lle")
    ap.add_argument("--within-domain", action="store_true")
    ap.add_argument("--by-model", action="store_true")
    ap.add_argument("--aliases-json", default=None)
    ap.add_argument("--modeltype_json", default="model_types.json", help="JSON mapping of model -> model_type")
    ap.add_argument("--figdir", default="corr_out/figures")
    ap.add_argument("--outdir", default="corr_out")
    args = ap.parse_args()

    outdir = Path(args.outdir); figdir = Path(args.figdir)
    ensure_dirs(outdir, figdir)

    aliases = load_aliases(args.aliases_json)
    model_types = load_model_types(args.modeltype_json)

    # ----- load metrics -----
    mdf = pd.read_csv(args.metrics_csv)
    if "dataset_base" not in mdf.columns:
        if "dataset_path" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_path"].apply(lambda p: canon(Path(p).parent.name))
        elif "dataset_label" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_label"].apply(lambda s: canon(s.split("/")[0]))
        else:
            raise ValueError("metrics CSV missing dataset_base and dataset_path/label")
    mdf["dataset_base"] = mdf["dataset_base"].apply(lambda x: aliases.get(x, x))

    want_metrics = [c.strip() for c in args.metrics_cols.split(",") if c.strip()]
    have_metrics = [c for c in want_metrics if c in mdf.columns]
    met_agg = mdf.groupby("dataset_base", as_index=False)[have_metrics].mean(numeric_only=True)
    if "domain" in mdf.columns:
        dom_map = (mdf.groupby("dataset_base", as_index=True)["domain"]
                    .agg(first_nonnull)
                    .rename("domain")
                    .reset_index())
        met_agg = met_agg.merge(dom_map, on="dataset_base", how="left")

    # ----- load results -----
    rdf = pd.read_csv(args.results_csv)
    rdf.columns = [c.replace("eval_metrics/", "") for c in rdf.columns]
    if "dataset" not in rdf.columns:
        rdf["dataset"] = rdf[[c for c in rdf.columns if "dataset" in c.lower()][0]]
    rdf["dataset_base"] = rdf["dataset"].apply(dataset_base_from_results).apply(lambda x: aliases.get(x, x))
    err_cols_req = [c.strip() for c in args.error_metrics.split(",") if c.strip()]
    err_cols = [c for c in err_cols_req if c in rdf.columns]
    for c in err_cols: rdf[c] = pd.to_numeric(rdf[c], errors="coerce")

    # ----- attach model types -----
    if "model" in rdf.columns:
        rdf["model_type"] = rdf["model"].map(lambda m: model_types.get(m, "unknown"))
    else:
        rdf["model_type"] = "unknown"
    

    # ----- report unmapped models (unknown type) -----
    unk = rdf[rdf["model_type"] == "unknown"].copy()
    if not unk.empty:
        # Aggregate useful stats per unmapped model
        unk_summary = (
            unk.groupby("model")
            .agg(
                n_rows=("model", "size"),
                n_datasets=("dataset_base", "nunique"),
                example_datasets=("dataset_base", lambda s: ", ".join(sorted(set(s))[:6]))
            )
            .sort_values(["n_rows", "n_datasets"], ascending=False)
            .reset_index()
        )
        unk_csv = Path(args.outdir) / "unmapped_models.csv"
        unk_summary.to_csv(unk_csv, index=False)

        # JSON skeleton to help you fill in model_types quickly
        skeleton = {m: "TBD" for m in sorted(unk["model"].unique())}
        skeleton_json = Path(args.outdir) / "model_types_skeleton.json"
        with open(skeleton_json, "w", encoding="utf-8") as f:
            json.dump(skeleton, f, indent=2, ensure_ascii=False)

        print("\n[WARN] Some models have no assigned type (model_type='unknown').")
        print(f"       -> Summary written to: {unk_csv}")
        print(f"       -> Fill in types here and merge into your mapping: {skeleton_json}")
        print("       Top unmapped models:")
        for _, row in unk_summary.head(10).iterrows():
            print(f"          - {row['model']}  | rows={row['n_rows']}  "
                f"| datasets={row['n_datasets']}  | ex: {row['example_datasets']}")
    else:
        print("\n[OK] All models had a mapped model_type.")

    # ----- correlations overall -----
    agg_strategies = {
        "mean": lambda df: df.groupby("dataset_base", as_index=False)[err_cols].mean(numeric_only=True),
        "median": lambda df: df.groupby("dataset_base", as_index=False)[err_cols].median(numeric_only=True),
        "best": lambda df: df.groupby("dataset_base", as_index=False)[err_cols].min(numeric_only=True),
    }

    corrs, joined_tables = [], {}
    for name, fn in agg_strategies.items():
        err_agg = fn(rdf)
        joined = met_agg.merge(err_agg, on="dataset_base", how="inner")
        joined_tables[name] = joined.copy()
        for m in have_metrics:
            for e in err_cols:
                x, y = joined[m], joined[e]
                mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 4: continue
                pear = joined.loc[mask, [m, e]].corr("pearson").iloc[0,1]
                spear = joined.loc[mask, [m, e]].corr("spearman").iloc[0,1]
                corrs.append({"aggregation": name, "metric": m, "error": e,
                              "pearson": pear, "spearman": spear, "n": int(mask.sum())})
    pd.DataFrame(corrs).to_csv(outdir / "corr_overall.csv", index=False)

    # ----- correlations by model type -----
    if "model_type" in rdf.columns:
        rows = []
        for mtype, sub in rdf.groupby("model_type"):
            err_agg = sub.groupby("dataset_base", as_index=False)[err_cols].mean(numeric_only=True)
            joined = met_agg.merge(err_agg, on="dataset_base", how="inner")
            for m in have_metrics:
                for e in err_cols:
                    x, y = joined[m], joined[e]
                    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 4: continue
                    pear = joined.loc[mask, [m, e]].corr("pearson").iloc[0,1]
                    spear = joined.loc[mask, [m, e]].corr("spearman").iloc[0,1]
                    rows.append({"model_type": mtype, "metric": m, "error": e,
                                 "pearson": pear, "spearman": spear, "n": int(mask.sum())})
        pd.DataFrame(rows).to_csv(outdir / "corr_by_modeltype.csv", index=False)

    # ----- slope analysis for omega vs sMAPE -----
    if "omega" in met_agg.columns and "sMAPE[0.5]" in err_cols:
        slopes = []
        for mtype, sub in rdf.groupby("model_type"):
            err_agg = sub.groupby("dataset_base", as_index=False)["sMAPE[0.5]"].mean(numeric_only=True)
            joined = met_agg.merge(err_agg, on="dataset_base", how="inner")
            if joined["omega"].nunique() < 4: continue
            m, b = np.polyfit(joined["omega"], joined["sMAPE[0.5]"], 1)
            slopes.append({"model_type": mtype, "slope": m, "intercept": b})
        pd.DataFrame(slopes).to_csv(outdir / "omega_slope_by_modeltype.csv", index=False)

   # pick the sMAPE column automatically
    smape_candidates = [c for c in err_cols if c.lower().startswith("smape")]
    if "omega" in met_agg.columns and smape_candidates and "model_type" in rdf.columns:
        smape_col = smape_candidates[0]  # e.g., "sMAPE[0.5]"

        # aggregate error by (dataset_base, model_type)
        err_agg_mt = (rdf.groupby(["dataset_base", "model_type"], as_index=False)[smape_col]
                        .mean(numeric_only=True))

        # join with metrics (omega, etc.)
        joined_mt = met_agg.merge(err_agg_mt, on="dataset_base", how="inner")

        # sanitize: rename target to 'y' so the formula is simple
        joined_mt = joined_mt.rename(columns={smape_col: "y"})

        # drop NAs / require enough variety
        mask = joined_mt["omega"].notna() & joined_mt["y"].notna() & joined_mt["model_type"].notna()
        df_ols = joined_mt.loc[mask, ["y", "omega", "model_type"]].copy()

        if df_ols["omega"].nunique() >= 4 and df_ols["model_type"].nunique() >= 2:
            # treat model_type as categorical with C()
            fit = smf.ols("y ~ omega * C(model_type)", data=df_ols).fit()
            #print("\n=== OLS: y ~ omega * C(model_type) ===")
            #print(fit.summary())
            fit_summary = fit.summary().as_text()
            outpath = Path(args.outdir) / "ols_omega_by_modeltype.txt"
            with open(outpath, "w") as f:
                f.write(fit_summary)
            print(f"[OK] Saved OLS summary to {outpath}")
            pd.DataFrame(fit.params, columns=["coef"]).to_csv(Path(args.outdir) / "ols_coefficients.csv")

    print(f"\n[OK] wrote correlations & model-type analyses to {outdir} and plots to {figdir}")

if __name__ == "__main__":
    main()
