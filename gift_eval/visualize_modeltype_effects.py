#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize how model type moderates the relationship between Ω (predictability) and error.

Figures written to: <outdir>/figures/
Also saves: OLS summary + coefficient table

Usage:
  python visualize_modeltype_effects.py \
    --metrics-csv metrics_summary_wide.csv \
    --results-csv merged_gift_results.csv \
    --modeltype-json model_types.json \
    --outdir corr_out --figdir corr_out/figures
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from collections import defaultdict

# ----------------- helpers -----------------
def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def canon(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def dataset_base_from_results(name: str) -> str:
    return canon(str(name).split("/")[0])

def load_json(path: str | None) -> dict:
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8"))

def first_nonnull(s, default=""):
    for v in s:
        if pd.notna(v) and v != "":
            return v
    return default

def pick_smape_column(columns):
    # Prefer sMAPE[0.5], otherwise first that startswith 'sMAPE'
    if "sMAPE[0.5]" in columns: return "sMAPE[0.5]"
    for c in columns:
        if c.lower().startswith("smape"): return c
    raise ValueError("No sMAPE-like column found in results CSV.")

# Robust quick linear fit helper
def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2: return np.nan, np.nan
    m, b = np.polyfit(x[mask], y[mask], 1)
    return float(m), float(b)

def bootstrap_slope(x, y, B=2000, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return np.nan, (np.nan, np.nan)
    samples = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m, _ = fit_line(x[idx], y[idx])
        if np.isfinite(m): samples.append(m)
    if not samples:
        return np.nan, (np.nan, np.nan)
    arr = np.sort(np.array(samples))
    return float(arr.mean()), (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
import re
import matplotlib.ticker as mticker

def _parse_pair(s: str) -> tuple[str,str]:
    """Accept 'A:B', 'A,B', or 'A->B' (case/space tolerant)."""
    sep = ":" if ":" in s else ("," if "," in s else "->" if "->" in s else None)
    if not sep:
        raise ValueError(f'Could not parse pair "{s}". Use "A:B", "A,B", or "A->B".')
    a, b = s.split(sep, 1)
    return a.strip(), b.strip()

def _bin_edges_from_quantiles(omega: pd.Series, bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(omega.dropna().to_numpy(float), qs)
    # de-duplicate edges (rare when data are discrete)
    edges = np.unique(edges)
    # need at least 3 edges (=> >=2 bins) to be useful; caller can check length
    return edges

def _agg_type_by_bin(df: pd.DataFrame, mt: str, edges: np.ndarray) -> pd.DataFrame:
    """Per-bin stats for a single model_type using within-bin *mean Ω* (not bin mid)."""
    g = df[df["model_type"] == mt].copy()
    if g.empty:
        return pd.DataFrame(columns=["omega_bin","omega_mean","yA","nA"])
    g["omega_bin"] = pd.cut(g["omega"], bins=edges, include_lowest=True, labels=False)
    out = (g.dropna(subset=["omega_bin"])
             .groupby("omega_bin", as_index=False)
             .agg(yA=("y","mean"),
                  nA=("y","count"),
                  omega_mean=("omega","mean")))
    # keep tidy order & names
    return out[["omega_bin","omega_mean","yA","nA"]]

def relative_gain_by_omega(joined: pd.DataFrame,
                           typeA: str,
                           typeB: str,
                           rel_bins: int = 6) -> pd.DataFrame:
    """
    Return per-Ω-bin relative gain of B over A:
        rel_gain_pct = 100 * (yA - yB) / yA
    where y is sMAPE (lower is better).
    """
    if rel_bins < 2:
        rel_bins = 2
    edges = _bin_edges_from_quantiles(joined["omega"], rel_bins)
    if len(edges) < 3:
        return pd.DataFrame(columns=["omega_mid","rel_gain_pct","nA","nB"])

    A = _agg_type_by_bin(joined, typeA, edges)
    B = _agg_type_by_bin(joined, typeB, edges)
    if A.empty or B.empty:
        return pd.DataFrame(columns=["omega_mean","rel_gain_pct","nA","nB"])

    M = pd.merge(A, B.rename(columns={"yA":"yB","nA":"nB","omega_mean":"omega_mean_B"}),
                on=["omega_bin"], how="inner")
    if M.empty:
        return pd.DataFrame(columns=["omega_mean","rel_gain_pct","nA","nB"])

    # Count-weighted within-bin mean Ω over A and B
    num = M["nA"] * M["omega_mean"] + M["nB"] * M["omega_mean_B"]
    den = (M["nA"] + M["nB"]).replace(0, np.nan)
    M["omega_mean_w"] = num / den

    denom = M["yA"].replace(0, np.nan)  # guard tiny/zero sMAPE
    M["rel_gain_pct"] = 100.0 * (M["yA"] - M["yB"]) / denom

    keep = ["omega_bin","omega_mean_w","rel_gain_pct","nA","nB"]
    return M[keep].dropna().sort_values("omega_bin").rename(
        columns={"omega_mean_w":"omega_mean"}
    ).reset_index(drop=True)


def plot_relative_gain(df_pairs: pd.DataFrame,
                       typeA: str,
                       typeB: str,
                       out_png: Path) -> None:
    """Simple scatter + 0% line. Positive = B has lower error than A."""
    if df_pairs.empty:
        print(f"[rel] No overlapping Ω-bins for {typeA} vs {typeB}; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300, layout="constrained")
    ax.scatter(df_pairs["omega_mean"], df_pairs["rel_gain_pct"], s=160, alpha=0.85, edgecolors="none")
    ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7, label="No gain (0%)")

    # aesthetics
    ax.set_xlabel("Spectral predictability (Ω) — within-bin mean")
    ax.set_ylabel(f"Relative gain of {typeB} over {typeA}")
    ax.set_title(f"Error Increase Δ (%): {typeA} → {typeB}")
    ax.minorticks_off()
    ax.grid(False, which="minor")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.legend(frameon=False, loc="best")

    # tight bounds (light padding)
    def _span(a, pad=0.06):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0: return (0.0, 1.0)
        lo, hi = float(a.min()), float(a.max())
        padv = pad * max(hi - lo, 1e-9)
        return (lo - padv, hi + padv)

    xlo, xhi = _span(df_pairs["omega_mean"], pad=0.06)
    xlo = max(0.0, xlo)  # Ω is nonnegative
    ylo, yhi = _span(df_pairs["rel_gain_pct"], pad=0.06)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default="metrics_summary_wide.csv")
    ap.add_argument("--results-csv", default="merged_gift_results.csv")
    ap.add_argument("--modeltype-json", dest="modeltype_json", default="model_types.json")
    ap.add_argument("--aliases-json", dest="aliases_json", default=None)
    ap.add_argument("--outdir", default="corr_out")
    ap.add_argument("--figdir", default="corr_out/figures")
    ap.add_argument("--bins", type=int, default=6, help="# quantile bins for binned curves")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstraps for slope CIs")
    ap.add_argument(
        "--rel-pairs",
        nargs="+",
        default=[],
        help='One or more model-type pairs to compare, formatted like "zero-shot:statistical". '
            'Spaces are allowed; you can also use "," or "->" as the separator.'
    )
    ap.add_argument(
        "--rel-bins",
        type=int,
        default=6,
        help="Number of Ω quantile bins for relative-gain curve."
    )

    args = ap.parse_args()

    outdir = Path(args.outdir); figdir = Path(args.figdir)
    ensure_dirs(outdir, figdir)

    aliases = load_json(args.aliases_json)
    model_types = load_json(args.modeltype_json)

    # ---------- load metrics ----------
    mdf = pd.read_csv(args.metrics_csv)
    if "dataset_base" not in mdf.columns:
        if "dataset_path" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_path"].astype(str).apply(lambda p: canon(Path(p).parent.name))
        elif "dataset_label" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_label"].astype(str).apply(lambda s: canon(s.split("/")[0]))
        else:
            raise ValueError("metrics CSV missing dataset_base and dataset_path/dataset_label to infer it.")
    mdf["dataset_base"] = mdf["dataset_base"].apply(lambda x: aliases.get(x, x))
    if "omega" not in mdf.columns:
        raise ValueError("metrics CSV must include 'omega'")

    met_agg = (mdf.groupby("dataset_base", as_index=False)[["omega"]]
                 .mean(numeric_only=True))
    # carry domain if present
    if "domain" in mdf.columns:
        dom_map = (mdf.groupby("dataset_base", as_index=True)["domain"]
                     .agg(first_nonnull).rename("domain").reset_index())
        met_agg = met_agg.merge(dom_map, on="dataset_base", how="left")

    # ---------- load results ----------
    rdf = pd.read_csv(args.results_csv)
    rdf.columns = [c.replace("eval_metrics/", "") for c in rdf.columns]
    if "dataset" not in rdf.columns:
        cand = [c for c in rdf.columns if "dataset" in c.lower()]
        if not cand: raise ValueError("results CSV missing 'dataset' column.")
        rdf["dataset"] = rdf[cand[0]]
    rdf["dataset_base"] = rdf["dataset"].astype(str).apply(dataset_base_from_results).apply(lambda x: aliases.get(x, x))
    if "model" not in rdf.columns:
        raise ValueError("results CSV must include 'model' column.")

    smape_col = pick_smape_column(rdf.columns)
    rdf[smape_col] = pd.to_numeric(rdf[smape_col], errors="coerce")
    rdf["model_type"] = rdf["model"].map(lambda m: model_types.get(m, "unknown"))

    # Aggregate error by (dataset_base, model_type)
    err_agg = (rdf.groupby(["dataset_base", "model_type"], as_index=False)[smape_col]
                 .mean(numeric_only=True)
                 .rename(columns={smape_col: "y"}))

    # Join with omega
    joined = met_agg.merge(err_agg, on="dataset_base", how="inner")
    # Remove rows with missing values
    joined = joined[np.isfinite(joined["omega"]) & np.isfinite(joined["y"])]

    if not args.rel_pairs:
        # canonical names you use in model_types.json (case-insensitive match)
        WANT = ["pretrained", "statistical", "deep-learning", "zero-shot"]
        present = {mt.lower(): mt for mt in joined["model_type"].dropna().unique()}
        have = [present[w] for w in WANT if w in present]

        # Desired default comparisons:
        # pretrained vs (statistical, deep-learning, zero-shot)
        # zero-shot  vs (statistical, deep-learning, pretrained)
        def _mk(a, b_list): return [f"{a}:{b}" for b in b_list if b in have and a in have and b != a]

        args.rel_pairs = (
            _mk(present.get("pretrained",""), ["statistical","deep-learning","zero-shot"]) +
            _mk(present.get("zero-shot",""),  ["statistical","deep-learning","pretrained"])
        )

    # ----------------- 1) Interaction lines (Ω vs sMAPE by model_type) -----------------
    plt.figure(figsize=(6.4, 4.5))
    colors = {}
    for i, (mt, g) in enumerate(sorted(joined.groupby("model_type"), key=lambda kv: kv[0])):
        if len(g) < 3: 
            continue
        m, b = fit_line(g["omega"], g["y"])
        xs = np.linspace(g["omega"].min(), g["omega"].max(), 100)
        ys = m * xs + b
        colors[mt] = colors.get(mt, None)  # let matplotlib assign
        plt.scatter(g["omega"], g["y"], s=16, alpha=0.45, label=f"{mt} (n={len(g)})")
        plt.plot(xs, ys)
    plt.xlabel("Spectral predictability (Ω)")
    plt.ylabel("sMAPE")
    plt.title("Ω vs sMAPE by model type (mean over datasets)")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(figdir / "omega_vs_smape_by_modeltype_lines.png", dpi=300)
    plt.close()

    # ----------------- 2) Forest plot of per-type slope with bootstrap CIs -----------------
    rows = []
    for mt, g in joined.groupby("model_type"):
        if g["omega"].nunique() < 3 or len(g) < 4:
            continue
        mean_slope, (lo, hi) = bootstrap_slope(g["omega"].values, g["y"].values, B=args.bootstrap)
        rows.append({"model_type": mt, "slope": mean_slope, "lo": lo, "hi": hi, "n": len(g)})
    slope_df = pd.DataFrame(rows).sort_values("slope")
    slope_df.to_csv(outdir / "slopes_bootstrap_by_modeltype.csv", index=False)

    if not slope_df.empty:
        plt.figure(figsize=(6.0, 0.45 * len(slope_df) + 1.5))
        y_pos = np.arange(len(slope_df))
        plt.hlines(y_pos, slope_df["lo"], slope_df["hi"])
        plt.plot(slope_df["slope"], y_pos, "o")
        plt.yticks(y_pos, [f'{mt} (n={n})' for mt, n in zip(slope_df["model_type"], slope_df["n"])])
        plt.axvline(0.0, ls="--", lw=1)
        plt.xlabel("Slope of sMAPE vs Ω  (lower is better)")
        plt.title("Per-model-type slope with 95% bootstrap CI")
        plt.tight_layout()
        plt.savefig(figdir / "forest_slope_by_modeltype.png", dpi=300)
        plt.close()

    # ----------------- 3) OLS interaction forest (coefficients with CI) -----------------
    # Fit: y ~ omega * C(model_type)
    df_ols = joined[["y", "omega", "model_type"]].dropna().copy()
    if df_ols["omega"].nunique() >= 4 and df_ols["model_type"].nunique() >= 2:
        fit = smf.ols("y ~ omega * C(model_type)", data=df_ols).fit()
        # Save text summary + params
        (Path(outdir) / "ols_omega_by_modeltype.txt").write_text(fit.summary().as_text(), encoding="utf-8")
        coef = fit.params.rename("coef").to_frame()
        ci = fit.conf_int().rename(columns={0: "lo", 1: "hi"})
        tbl = coef.join(ci)
        tbl.to_csv(outdir / "ols_coefficients.csv")

        # Plot coefficients except Intercept to declutter
        plot_tbl = tbl.drop(index=["Intercept"], errors="ignore").copy()
        plot_tbl = plot_tbl.sort_values("coef")
        plt.figure(figsize=(6.2, 0.45 * len(plot_tbl) + 1.5))
        y_pos = np.arange(len(plot_tbl))
        plt.hlines(y_pos, plot_tbl["lo"], plot_tbl["hi"])
        plt.plot(plot_tbl["coef"], y_pos, "o")
        plt.axvline(0.0, ls="--", lw=1)
        plt.yticks(y_pos, plot_tbl.index)
        plt.xlabel("OLS coefficient (95% CI)")
        plt.title("OLS: y ~ omega * C(model_type)")
        plt.tight_layout()
        plt.savefig(figdir / "ols_coef_forest.png", dpi=300)
        plt.close()

    # ----------------- 4) Binned curves: quantile-Ω vs mean sMAPE per model_type -----------------
    bins = args.bins
    if bins >= 3:
        # global quantile bins on omega
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(joined["omega"].dropna(), qs)
        # guard for duplicate edges (rare but possible)
        edges = np.unique(edges)
        if len(edges) >= 4:
            bin_labels = [f"Q{i}" for i in range(1, len(edges))]
            binned = []
            for mt, g in joined.groupby("model_type"):
                g = g.copy()
                g["omega_bin"] = pd.cut(g["omega"], bins=edges, include_lowest=True, labels=False)
                agg = (g.dropna(subset=["omega_bin"])
                    .groupby("omega_bin", as_index=False)
                    .agg(mean=("y","mean"),
                        count=("y","count"),
                        std=("y","std"),
                        omega_mean=("omega","mean")))
                
                if len(agg):
                    agg["model_type"] = mt
                    agg["omega_mid"] = [np.mean(edges[i:i+2]) for i in agg["omega_bin"]] #remove this for omega mean?
                    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
                    binned.append(agg[["model_type","omega_bin","omega_mean","mean","se","count"]])

            if binned:
                bdf = pd.concat(binned, ignore_index=True)
                plt.figure(figsize=(6.4, 4.5))
                for mt, g in bdf.groupby("model_type"):
                    plt.errorbar(g["omega_mean"], g["mean"], yerr=g["se"],
                                 marker="o", linestyle="-", capsize=3, label=f"{mt}")
                plt.xlabel("Spectral predictability (Ω) — within-bin mean")
                plt.ylabel("Mean sMAPE (±1 SE)")
                plt.title("Binned trend of sMAPE vs Ω by model type")
                plt.legend(frameon=False, ncol=2)
                plt.tight_layout()
                plt.savefig(figdir / "binned_smape_vs_omega_by_modeltype.png", dpi=300)
                plt.close()

    print(f"[OK] Wrote figures to {figdir} and tables to {outdir}")

        # ----------------- 5) Relative-gain curves: any pair of model_types over Ω -----------------
    if args.rel_pairs:
        # 'joined' is already (dataset_base, omega, model_type, y)
        # ensure we only keep rows with finite values
        rel_joined = joined[np.isfinite(joined["omega"]) & np.isfinite(joined["y"])].copy()

        for spec in args.rel_pairs:
            try:
                A, B = _parse_pair(spec)
            except ValueError as e:
                print(f"[rel] {e}")
                continue

            # compute per-Ω-bin relative gain (% lower sMAPE) of B over A
            RG = relative_gain_by_omega(rel_joined, A, B, rel_bins=args.rel_bins)
            if RG.empty:
                print(f"[rel] No bins for pair {A} vs {B} (maybe one type missing).")
                continue

            # save CSV
            safeA = re.sub(r"[^A-Za-z0-9]+", "", A)
            safeB = re.sub(r"[^A-Za-z0-9]+", "", B)
            csv_path = outdir / f"RELGAIN_{safeA}_to_{safeB}_vs_Omega_sMAPE.csv"
            RG.to_csv(csv_path, index=False)
            print(f"[rel] wrote {csv_path}")

            # plot
            fig_path = Path(args.figdir) / f"RELGAIN_{safeA}_to_{safeB}_vs_Omega_sMAPE.png"
            plot_relative_gain(RG, A, B, fig_path)
            print(f"[rel] wrote {fig_path}")

if __name__ == "__main__":
    main()
