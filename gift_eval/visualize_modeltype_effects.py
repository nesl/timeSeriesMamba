#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize how model type moderates the relationship between Ω (predictability) and error.

All figures saved as PDF to: <figdir>/pdf/
Also writes: correlation CSVs + OLS outputs

Usage:
  python visualize_modeltype_effects.py \
    --metrics_csv metrics_summary_wide.csv \
    --results_csv merged_gift_results.csv \
    --modeltype_json model_types.json \
    --outdir corr_out --figdir corr_out/figures
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.formula.api as smf

# ----------------- ONE global style (affects ALL plots) -----------------
mpl.rcParams.update({
    # Size & layout
    "figure.figsize": (6.6, 4.2),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # Fonts & weights (bold everywhere)
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",

    # Sizes
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,

    # Lines & markers
    "lines.linewidth": 3.0,
    "lines.markersize": 8,

    # Spines & ticks
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 5,
    "ytick.major.size": 5,

    "axes.grid": False,
})
AXIS_FONTSIZE = 20
# ----------------- helpers -----------------
def ensure_dirs(*paths: Path):
    for p in paths: p.mkdir(parents=True, exist_ok=True)

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
    if "sMAPE[0.5]" in columns: return "sMAPE[0.5]"
    for c in columns:
        if c.lower().startswith("smape"): return c
    raise ValueError("No sMAPE-like column found in results CSV.")

def fit_line(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2: return np.nan, np.nan
    m_, b_ = np.polyfit(x[m], y[m], 1)
    return float(m_), float(b_)

def bootstrap_slope(x, y, B=2000, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 4: return np.nan, (np.nan, np.nan)
    samples = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m_, _ = fit_line(x[idx], y[idx])
        if np.isfinite(m_): samples.append(m_)
    if not samples: return np.nan, (np.nan, np.nan)
    arr = np.sort(np.array(samples))
    return float(arr.mean()), (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

def _parse_pair(s: str) -> tuple[str,str]:
    sep = ":" if ":" in s else ("," if "," in s else "->" if "->" in s else None)
    if not sep:
        raise ValueError(f'Could not parse pair "{s}". Use "A:B", "A,B", or "A->B".')
    a, b = s.split(sep, 1)
    return a.strip(), b.strip()

def _bin_edges_from_quantiles(omega: pd.Series, bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(omega.dropna().to_numpy(float), qs)
    return np.unique(edges)

def _agg_type_by_bin(df: pd.DataFrame, mt: str, edges: np.ndarray) -> pd.DataFrame:
    g = df[df["model_type"] == mt].copy()
    if g.empty: return pd.DataFrame(columns=["omega_bin","omega_mean","yA","nA"])
    g["omega_bin"] = pd.cut(g["omega"], bins=edges, include_lowest=True, labels=False)
    out = (g.dropna(subset=["omega_bin"])
             .groupby("omega_bin", as_index=False)
             .agg(yA=("y","mean"),
                  nA=("y","count"),
                  omega_mean=("omega","mean")))
    return out[["omega_bin","omega_mean","yA","nA"]]

def relative_gain_by_omega(joined: pd.DataFrame, typeA: str, typeB: str, rel_bins: int = 6) -> pd.DataFrame:
    if rel_bins < 2: rel_bins = 2
    edges = _bin_edges_from_quantiles(joined["omega"], rel_bins)
    if len(edges) < 3: return pd.DataFrame(columns=["omega_mid","rel_gain_pct","nA","nB"])
    A = _agg_type_by_bin(joined, typeA, edges)
    B = _agg_type_by_bin(joined, typeB, edges)
    if A.empty or B.empty: return pd.DataFrame(columns=["omega_mean","rel_gain_pct","nA","nB"])
    M = pd.merge(A, B.rename(columns={"yA":"yB","nA":"nB","omega_mean":"omega_mean_B"}),
                 on=["omega_bin"], how="inner")
    if M.empty: return pd.DataFrame(columns=["omega_mean","rel_gain_pct","nA","nB"])
    num = M["nA"] * M["omega_mean"] + M["nB"] * M["omega_mean_B"]
    den = (M["nA"] + M["nB"]).replace(0, np.nan)
    M["omega_mean_w"] = num / den
    denom = M["yA"].replace(0, np.nan)
    M["rel_gain_pct"] = 100.0 * (M["yA"] - M["yB"]) / denom
    keep = ["omega_bin","omega_mean_w","rel_gain_pct","nA","nB"]
    return (M[keep].dropna().sort_values("omega_bin")
              .rename(columns={"omega_mean_w":"omega_mean"})
              .reset_index(drop=True))

def _corrs(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
    try:
        from scipy import stats
        pearson_r, pearson_p   = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
    except Exception:
        if len(x) >= 2:
            pearson_r = float(np.corrcoef(x, y)[0,1])
        try:
            spearman_rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
        except Exception:
            pass
    return pearson_r, pearson_p, spearman_rho, spearman_p

def savefig_pdf(fig: plt.Figure, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.pdf"
    fig.savefig(path, format="pdf")
    print(f"[savefig] wrote {path}")

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default="metrics_summary_wide.csv")
    ap.add_argument("--results_csv", default="merged_gift_results.csv")
    ap.add_argument("--modeltype_json", dest="modeltype_json", default="model_types.json")
    ap.add_argument("--aliases_json", dest="aliases_json", default=None)
    ap.add_argument("--outdir", default="corr_out")
    ap.add_argument("--figdir", default="corr_out/figures")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--plot_modeltypes", nargs="+", default=None)
    ap.add_argument("--rel_pairs", nargs="+", default=[])
    ap.add_argument("--rel_bins", type=int, default=6)
    ap.add_argument("--granularity", choices=["base", "label"], default="base",
                    help="Use 'label' to keep LOOP_SEATTLE/H separate.")
    ap.add_argument("--debug_keys", action="store_true")
    ap.add_argument("--colors_json", default=None,
                    help='JSON mapping of model_type -> color, e.g. {"pretrained":"#1f77b4"}')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    pdfdir = figdir / "pdf"
    ensure_dirs(outdir, figdir, pdfdir)

    aliases = load_json(args.aliases_json)
    model_types = load_json(args.modeltype_json)
    colors_map = load_json(args.colors_json) if args.colors_json else {}

    ''' Test the plotting
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1])
    ax.set_xlabel("Spectral plredictability (Ω)", fontsize=20)
    ax.set_ylabel("sMAPE", fontsize=20)
    fig.savefig("test_axes_font.pdf")

    exit()
    '''
    def first_two_parts(s: str) -> str:
        parts = str(s).split("/")
        return canon("/".join(parts[:2]))

    # ---------- load metrics ----------
    mdf = pd.read_csv(args.metrics_csv)
    if "dataset_label" in mdf.columns:
        mdf["dataset_label_canon"] = mdf["dataset_label"].astype(str).apply(first_two_parts)
    else:
        mdf["dataset_label_canon"] = mdf.get("dataset_path", mdf.index.astype(str)).astype(str).apply(first_two_parts)

    if "dataset_base" not in mdf.columns:
        if "dataset_path" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_path"].astype(str).apply(lambda p: canon(Path(p).parent.name))
        elif "dataset_label" in mdf.columns:
            mdf["dataset_base"] = mdf["dataset_label"].astype(str).apply(lambda s: canon(s.split("/")[0]))
        else:
            raise ValueError("metrics CSV missing dataset_base and dataset_path/dataset_label to infer it.")
    mdf["dataset_base"] = mdf["dataset_base"].apply(lambda x: aliases.get(x, x))
    mdf["dataset_id"] = mdf["dataset_label_canon"] if args.granularity == "label" else mdf["dataset_base"]

    if "omega" not in mdf.columns:
        raise ValueError("metrics CSV must include 'omega'")

    met_agg = (mdf.groupby("dataset_id", as_index=False)[["omega"]].mean(numeric_only=True))
    if "domain" in mdf.columns:
        dom_map = (mdf.groupby("dataset_id", as_index=True)["domain"].agg(first_nonnull).rename("domain").reset_index())
        met_agg = met_agg.merge(dom_map, on="dataset_id", how="left")

    # ---------- load results ----------
    rdf = pd.read_csv(args.results_csv)
    rdf.columns = [c.replace("eval_metrics/", "") for c in rdf.columns]
    if "dataset" not in rdf.columns:
        cand = [c for c in rdf.columns if "dataset" in c.lower()]
        if not cand: raise ValueError("results CSV missing 'dataset' column.")
        rdf["dataset"] = rdf[cand[0]]

    rdf["dataset_full_canon"] = rdf["dataset"].astype(str).apply(first_two_parts)
    rdf["dataset_base"] = rdf["dataset"].astype(str).apply(dataset_base_from_results).apply(lambda x: aliases.get(x, x))
    rdf["dataset_id"] = rdf["dataset_full_canon"] if args.granularity == "label" else rdf["dataset_base"]

    if "model" not in rdf.columns:
        raise ValueError("results CSV must include 'model' column.")
    smape_col = pick_smape_column(rdf.columns)
    rdf[smape_col] = pd.to_numeric(rdf[smape_col], errors="coerce")
    rdf["model_type"] = rdf["model"].map(lambda m: model_types.get(m, "unknown"))

    err_agg = (rdf.groupby(["dataset_id", "model_type"], as_index=False)[smape_col]
                 .mean(numeric_only=True).rename(columns={smape_col: "y"}))

    joined = met_agg.merge(err_agg, on="dataset_id", how="inner")
    joined = joined[np.isfinite(joined["omega"]) & np.isfinite(joined["y"])]

    # ----------------- RAW unbinned scatter (all points; colored by model_type) -----------------
    raw = joined.copy()
    if not raw.empty:
        pr, pp, sr, sp = _corrs(raw["omega"].values, raw["y"].values)
        # PRINT nice summary for paper
        print(f"[corr-overall] n={len(raw)} | Pearson r={pr:.4f}, p={pp if np.isfinite(pp) else 'NA'} | "
              f"Spearman ρ={sr:.4f}, p={sp if np.isfinite(sp) else 'NA'}")
        # Save CSV too
        pd.DataFrame([{
            "n_points": len(raw), "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp
        }]).to_csv(outdir / "correlation_overall.csv", index=False)

        # Global fit
        m, b = fit_line(raw["omega"].values, raw["y"].values) if len(raw) >= 2 else (np.nan, np.nan)

        fig, ax = plt.subplots()
        for mt, g in raw.groupby("model_type"):
            c = colors_map.get(str(mt), None) or None
            ax.scatter(g["omega"], g["y"], alpha=0.65, label=f"{mt} (n={len(g)})", color=c)
        if np.isfinite(m) and np.isfinite(b):
            xs = np.linspace(raw["omega"].min(), raw["omega"].max(), 200)
            ax.plot(xs, m * xs + b, alpha=0.95, color="black")
        ax.set_xlabel("Spectral predictability (Ω)", fontsize=AXIS_FONTSIZE, fontweight="bold")
        ax.set_ylabel("sMAPE",fontsize=AXIS_FONTSIZE, fontweight="bold")
        ax.set_title("Ω vs sMAPE — Colored by Model Type")
        ax.legend(frameon=False, ncol=2)
        smape_used = "sMAPE[0.5]" if "sMAPE[0.5]" in rdf.columns else pick_smape_column(rdf.columns)
        savefig_pdf(fig, pdfdir, f"scatter_omega_vs_{smape_used.replace('/', '_')}")
        plt.close(fig)
    else:
        print("[scatter-all] empty after filtering finite omega/y; nothing to plot.")

    # ----------------- GROUPED SCATTERS (no top-5 labels) -----------------
    def _fit_line_and_residuals(df):
        x = df["omega"].to_numpy(float); y = df["y"].to_numpy(float)
        if len(df) >= 2: m, b = fit_line(x, y)
        else: m, b = np.nan, np.nan
        resid = np.abs(y - (m * x + b)) if np.isfinite(m) and np.isfinite(b) else np.full_like(y, np.nan, float)
        return m, b, resid

    def _plot_group(ax, data, title, color=None, label_outliers=False):
        pr, pp, sr, sp = _corrs(data["omega"].values, data["y"].values)
        print(f"[corr-{title}] n={len(data)} | Pearson r={pr:.4f}, p={pp if np.isfinite(pp) else 'NA'} | "
              f"Spearman ρ={sr:.4f}, p={sp if np.isfinite(sp) else 'NA'}")
        m, b, resid = _fit_line_and_residuals(data)
        ax.scatter(data["omega"], data["y"], alpha=0.70, color=color)
        if np.isfinite(m) and np.isfinite(b):
            xs = np.linspace(np.nanmin(data["omega"]), np.nanmax(data["omega"]), 200)
            ax.plot(xs, m * xs + b, alpha=0.95, color=(color if color else "black"))
        if label_outliers:
            # optional: label furthest 5 points (OFF for requested graph; leave available)
            d = data.copy(); d["resid"] = resid
            top5 = d.sort_values("resid", ascending=False).head(5)
            for _, r in top5.iterrows():
                ax.annotate(r["dataset_id"], (r["omega"], r["y"]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.set_xlabel("Spectral predictability (Ω)",fontsize=AXIS_FONTSIZE, fontweight="bold")
        ax.set_ylabel("sMAPE",fontsize=AXIS_FONTSIZE, fontweight="bold")
        ax.set_title(title)
        return {"pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp,
                "m": m, "b": b}

    # (A) Per-model_type (points are dataset_id means) — NO labels
    per_type_stats = []
    for mt, g in joined.groupby("model_type", dropna=True):
        g = g[np.isfinite(g["omega"]) & np.isfinite(g["y"])].copy()
        if g.empty: continue
        c = colors_map.get(str(mt), None) or None
        fig, ax = plt.subplots()
        stats = _plot_group(ax, g, title=f"{mt}", color=c, label_outliers=False)
        savefig_pdf(fig, pdfdir, f"scatter_grouped_omega_vs_sMAPE_by_{str(mt).replace(' ','_')}_nolabels")
        plt.close(fig)
        per_type_stats.append({"model_type": mt, **stats})
    if per_type_stats:
        pd.DataFrame(per_type_stats).to_csv(outdir / "correlation_by_model_type.csv", index=False)

    # (B) Collapsed: one point per dataset_id (mean over model_types) — NO labels
    collapsed = (joined.groupby("dataset_id", as_index=False)
                 .agg(omega=("omega","mean"), y=("y","mean")))
    if not collapsed.empty:
        fig, ax = plt.subplots()
        stats_all = _plot_group(ax, collapsed, title="sMAPE vs. Ω (dataset-level means)", color=None, label_outliers=False)
        savefig_pdf(fig, pdfdir, "scatter_grouped_omega_vs_sMAPE_ALLTYPES_nolabels")
        plt.close(fig)
        pd.DataFrame([stats_all]).to_csv(outdir / "correlation_alltypes_grouped.csv", index=False)

    # ----------------- Defaults for rel-gain pairs if none provided -----------------
    if not args.rel_pairs:
        WANT = ["pretrained", "statistical", "deep-learning", "zero-shot"]
        present = {mt.lower(): mt for mt in joined["model_type"].dropna().unique()}
        have = [present[w] for w in WANT if w in present]
        def _mk(a, b_list): return [f"{a}:{b}" for b in b_list if b in have and a in have and b != a]
        args.rel_pairs = (
            _mk(present.get("pretrained",""), ["statistical","deep-learning","zero-shot"]) +
            _mk(present.get("zero-shot",""),  ["statistical","deep-learning","pretrained"])
        )

    # ----------------- Interaction lines (Ω vs sMAPE by model_type) -----------------
    fig, ax = plt.subplots()
    for mt, g in sorted(joined.groupby("model_type"), key=lambda kv: kv[0]):
        if len(g) < 3: continue
        m, b = fit_line(g["omega"], g["y"])
        xs = np.linspace(g["omega"].min(), g["omega"].max(), 100)
        ax.scatter(g["omega"], g["y"], alpha=0.45, label=f"{mt} (n={len(g)})")
        ax.plot(xs, m * xs + b)
    ax.set_xlabel("Spectral predictability (Ω)",fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("sMAPE",fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title("Ω vs sMAPE by model type (mean over datasets)")
    ax.legend(frameon=False, ncol=2)
    savefig_pdf(fig, pdfdir, "omega_vs_smape_by_modeltype_lines")
    plt.close(fig)

    # ----------------- Forest plot of slopes with bootstrap CIs -----------------
    rows = []
    for mt, g in joined.groupby("model_type"):
        if g["omega"].nunique() < 3 or len(g) < 4: continue
        mean_slope, (lo, hi) = bootstrap_slope(g["omega"].values, g["y"].values, B=args.bootstrap)
        rows.append({"model_type": mt, "slope": mean_slope, "lo": lo, "hi": hi, "n": len(g)})
    slope_df = pd.DataFrame(rows).sort_values("slope")
    slope_df.to_csv(outdir / "slopes_bootstrap_by_modeltype.csv", index=False)
    if not slope_df.empty:
        fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(slope_df) + 1.5))
        y_pos = np.arange(len(slope_df))
        ax.hlines(y_pos, slope_df["lo"], slope_df["hi"])
        ax.plot(slope_df["slope"], y_pos, "o")
        ax.set_yticks(y_pos, [f'{mt} (n={n})' for mt, n in zip(slope_df["model_type"], slope_df["n"])])
        ax.axvline(0.0, ls="--", lw=1)
        ax.set_xlabel("Slope of sMAPE vs Ω",fontsize=AXIS_FONTSIZE, fontweight="bold")
        ax.set_title("Per-model-type slope with CI",fontsize=AXIS_FONTSIZE, fontweight="bold")
        savefig_pdf(fig, pdfdir, "forest_slope_by_modeltype")
        plt.close(fig)

    # ----------------- OLS interaction forest (coefficients with CI) -----------------
    df_ols = joined[["y", "omega", "model_type"]].dropna().copy()
    if df_ols["omega"].nunique() >= 4 and df_ols["model_type"].nunique() >= 2:
        fit = smf.ols("y ~ omega * C(model_type)", data=df_ols).fit()
        (outdir / "ols_omega_by_modeltype.txt").write_text(fit.summary().as_text(), encoding="utf-8")
        tbl = (fit.params.rename("coef").to_frame()
               .join(fit.conf_int().rename(columns={0:"lo",1:"hi"})))
        tbl.to_csv(outdir / "ols_coefficients.csv")
        plot_tbl = tbl.drop(index=["Intercept"], errors="ignore").sort_values("coef")
        fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(plot_tbl) + 1.5))
        y_pos = np.arange(len(plot_tbl))
        ax.hlines(y_pos, plot_tbl["lo"], plot_tbl["hi"])
        ax.plot(plot_tbl["coef"], y_pos, "o")
        ax.axvline(0.0, ls="--", lw=1)
        ax.set_yticks(y_pos, plot_tbl.index)
        ax.set_xlabel("OLS coefficient (95% CI)",fontsize=AXIS_FONTSIZE, fontweight="bold")
        ax.set_title("OLS: y ~ omega * C(model_type)",fontsize=AXIS_FONTSIZE, fontweight="bold")
        savefig_pdf(fig, pdfdir, "ols_coef_forest")
        plt.close(fig)

    # ----------------- Binned curves -----------------
    bins = args.bins
    if bins >= 3:
        edges = np.unique(np.quantile(joined["omega"].dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) >= 4:
            binned = []
            present_map = {mt.lower(): mt for mt in joined["model_type"].dropna().unique()}
            def _resolve_modeltypes(requested):
                if not requested: return None
                keep = [present_map[r.strip().lower()] for r in requested if r.strip().lower() in present_map]
                return set(keep) if keep else set()
            keep_types = _resolve_modeltypes(args.plot_modeltypes)

            for mt, g in joined.groupby("model_type"):
                if keep_types is not None and mt not in keep_types: continue
                g = g.copy()
                g["omega_bin"] = pd.cut(g["omega"], bins=edges, include_lowest=True, labels=False)
                agg = (g.dropna(subset=["omega_bin"])
                        .groupby("omega_bin", as_index=False)
                        .agg(mean=("y","mean"), count=("y","count"), std=("y","std"),
                             omega_mean=("omega","mean")))
                if len(agg):
                    agg["model_type"] = mt
                    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
                    binned.append(agg[["model_type","omega_bin","omega_mean","mean","se","count"]])
            if binned:
                bdf = pd.concat(binned, ignore_index=True)
                fig, ax = plt.subplots()
                for mt, g in bdf.groupby("model_type"):
                    ax.errorbar(g["omega_mean"], g["mean"], yerr=g["se"],
                                marker="o", linestyle="-", capsize=3, label=f"{mt}")
                ax.set_xlabel("Spectral predictability (Ω)",fontsize=AXIS_FONTSIZE, fontweight="bold")
                ax.set_ylabel("Mean sMAPE (±1 SE)",fontsize=AXIS_FONTSIZE, fontweight="bold")
                ax.set_title("Binned trend of sMAPE vs Ω by model type" + (" (filtered)" if keep_types else ""))
                ax.legend(frameon=False, ncol=2)
                savefig_pdf(fig, pdfdir, "binned_smape_vs_omega_by_modeltype")
                plt.close(fig)

    # ----------------- Relative-gain curves -----------------
    if not args.rel_pairs:
        # no defaults here; keep empty unless user asks
        pass
    else:
        rel_joined = joined.copy()
        for spec in args.rel_pairs:
            try:
                A, B = _parse_pair(spec)
            except ValueError as e:
                print(f"[rel] {e}"); continue
            RG = relative_gain_by_omega(rel_joined, A, B, rel_bins=args.rel_bins)
            if RG.empty:
                print(f"[rel] No bins for pair {A} vs {B} (maybe one type missing)."); continue
            safeA = re.sub(r"[^A-Za-z0-9]+", "", A)
            safeB = re.sub(r"[^A-Za-z0-9]+", "", B)
            RG.to_csv(outdir / f"RELGAIN_{safeA}_to_{safeB}_vs_Omega_sMAPE.csv", index=False)
            fig, ax = plt.subplots()
            ax.scatter(RG["omega_mean"], RG["rel_gain_pct"], s=160, alpha=0.85, edgecolors="none")
            ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7, label="No gain (0%)")
            ax.set_xlabel("Spectral predictability (Ω)",fontsize=AXIS_FONTSIZE, fontweight="bold")
            ax.set_ylabel(f"Relative Error Gain",fontsize=AXIS_FONTSIZE, fontweight="bold")
            ax.set_title(f"Relative Error Gain Δ (%): {A} → {B}")
            ax.minorticks_off(); ax.grid(False, which="minor")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            ax.legend(frameon=False, loc="best")
            savefig_pdf(fig, pdfdir, f"RELGAIN_{safeA}_to_{safeB}_vs_Omega_sMAPE")
            plt.close(fig)

    print(f"[OK] Wrote PDFs to {pdfdir} and tables to {outdir}")

if __name__ == "__main__":
    main()
