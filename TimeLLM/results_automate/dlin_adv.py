#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse
from glob import glob
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Defaults & styling ----
# --- Debug helper ---
DEBUG = False
def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


DEFAULT_LOG_DIRS = [
    "../results/pems_eval/",
    "../results/fitbit_eval/",
    "../results/carbon_eval/",
    "../results/spectralUniTest/",
    "../results/uniSynth_eval/",
]
DEFAULT_OUT_DIR = "./dlin"

DOMAIN_NAME_FIX = {
    "spectralUniTest": "CarbonCast",
    "carbon_eval": "CarbonCast",
    "pems_eval": "PEMS",
    "fitbit_eval": "Fitbit",
    "uniSynth_eval": "UniSynth",
}
DOMAINS_CANON = ("CarbonCast", "PEMS", "Fitbit", "UniSynth")

MODEL_COLOR = {
    "Language Pretrained": "tab:blue",
    "Random Init": "tab:orange",
    "DLinear": "tab:green",
    "GPT2": "tab:blue",
}
MODEL_MARKER = {
    "Language Pretrained": "^",
    "Random Init": "^",
    "DLinear": "o",
    "GPT2": "s",
}

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": "--",
    "grid.alpha": 0.30,
    "legend.frameon": False,
})

# ---- Parsing helpers ----
NUM_PAT = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
SMAPE_PATS = [re.compile(r"\bsmape(_loss)?\b", re.I), re.compile(r"\bsym(_)?mape\b", re.I)]

def norm_key(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())
# Treat non-alphanumeric characters (including underscore) as delimiters
def _token_regex(choices: List[str]) -> re.Pattern:
    # (?<![A-Za-z0-9]) ensures left boundary is start or non-alnum
    # (?![A-Za-z0-9]) ensures right boundary is end or non-alnum
    inner = "|".join(map(re.escape, choices))
    return re.compile(rf"(?<![A-Za-z0-9])({inner})(?![A-Za-z0-9])", re.IGNORECASE)

def parse_run_summary(text: str) -> Dict[str, float]:
    metrics = {}
    in_block = False
    for line in text.splitlines():
        if "wandb: Run summary:" in line:
            in_block = True; continue
        if not in_block: continue
        m = re.match(r"^\s*wandb:\s+(.+?)\s+(" + NUM_PAT + r")\s*$", line)
        if m:
            key, val = norm_key(m.group(1)), float(m.group(2))
            metrics[key] = val
    return metrics

def choose_metric(metrics: Dict[str, Any], patterns: List[re.Pattern]) -> Optional[float]:
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            if any(pat.search(k) for pat in patterns): return float(v)
    return None

def infer_model(basename: str) -> Optional[str]:
    b = basename.lower(); bU = basename.upper()
    # DLinear
    if "dlin" in b:
        dprint("[model] DLinear:", basename)
        return "DLinear"
    # GPT2
    if "mgpt2" in b or re.search(r"\bgpt2\b", b):
        dprint("[model] GPT2:", basename)
        return "GPT2"
    # LLaMA3.2 variants. Your earlier files sometimes use mLLAMA3.2_* with r0/r1.
    # We require r0 (pretrained) or r1 (random) to avoid ambiguity; log if missing.
    if "llama3.2" in b:
        if "_R0_" in bU:
            dprint("[model] Language Pretrained (LLAMA3.2 r0):", basename)
            return "Language Pretrained"
        if "_R1_" in bU:
            dprint("[model] Random Init (LLAMA3.2 r1):", basename)
            return "Random Init"
        dprint("[skip][model] LLAMA3.2 found but no _r0_/_r1_ flag:", basename)
        return None
    # fallback tokens
    if "language_pretrained" in b:
        dprint("[model] Language Pretrained (explicit token):", basename)
        return "Language Pretrained"
    if "random_init" in b or "rand_init" in b:
        dprint("[model] Random Init (explicit token):", basename)
        return "Random Init"
    dprint("[skip][model] Unrecognized model:", basename)
    return None

def domain_name_from_dir(dir_path: str) -> str:
    base = os.path.basename(os.path.normpath(dir_path))
    return DOMAIN_NAME_FIX.get(base, base)

def infer_domain(basename: str, fallback_dir_name: str) -> str:
    b = basename.lower()
    if "fitbit" in b: return "Fitbit"
    if "pems" in b: return "PEMS"
    if "unisynth" in b or "unisynthpsd" in b: return "UniSynth"
    if "carbon" in b or "spectralunitest" in b: return "CarbonCast"
    return DOMAIN_NAME_FIX.get(fallback_dir_name, fallback_dir_name)

# ---- Pair-key logic (domain-aware) ----
REGIONS = ["CISO","NYISO","MISO","PJM","ERCOT","ISONE","SPP","BPA","IESO","CAISO","CENACE"]
SOURCES = ["solar","wind","hydro","nuclear","coal","nat_gas","oil","other","thermal"]

_level_pat = re.compile(r"(high|low|medium|mid(?:dle)?)", re.IGNORECASE)

def _norm_level(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower()
    if s.startswith("mid"): return "medium"
    return s

# precompile with underscore-safe boundaries
_REGION_PAT = _token_regex(REGIONS)
_SOURCE_PAT = _token_regex(SOURCES)

def _find_region(b: str) -> Optional[str]:
    # 1) explicit region tokens (underscore-safe)
    m = _REGION_PAT.search(b)
    if m:
        reg = m.group(0).upper()
        dprint(f"[region] token match: '{reg}' in '{b}'")
        return reg

    # 2) EVAL<REGION> pattern (e.g., EVALPJM)
    m2 = re.search(r"(?<![A-Za-z0-9])EVAL([A-Za-z0-9]+)(?![A-Za-z0-9])", b, re.IGNORECASE)
    if m2:
        reg = m2.group(1).upper()
        dprint(f"[region] EVAL* match: '{reg}' in '{b}'")
        return reg

    # 3) FINAL fallback: split tokens and look for a REGIONS member
    toks = re.split(r"[^A-Za-z0-9]+", b.upper())
    for t in toks:
        if t in REGIONS:
            dprint(f"[region] split fallback: '{t}' in '{b}'")
            return t

    dprint(f"[region] none in '{b}'")
    return None


def _find_source(b: str) -> Optional[str]:
    # 1) explicit source tokens (underscore-safe; handles 'nat_gas', 'solar', etc.)
    m = _SOURCE_PAT.search(b)
    if m:
        src = m.group(0).lower()
        dprint(f"[source] token match: '{src}' in '{b}'")
        return src

    # 2) 'carbon<source>' (e.g., carbonsolar, carbonnat_gas)
    m2 = re.search(r"(?<![A-Za-z0-9])carbon([a-z_]+)(?![A-Za-z0-9])", b, re.IGNORECASE)
    if m2:
        cand = m2.group(1).lower()
        for s in SOURCES:
            if cand.startswith(s):
                dprint(f"[source] carbon* fallback: '{s}' (from '{cand}') in '{b}'")
                return s

    # 3) generic carbon[-_]*<src>
    m3 = re.search(r"(?<![A-Za-z0-9])carbon(?:[-_]?)([a-z]+)(?![A-Za-z0-9])", b, re.IGNORECASE)
    if m3:
        cand = m3.group(1).lower()
        if cand in SOURCES:
            dprint(f"[source] carbon[-_]* fallback: '{cand}' in '{b}'")
            return cand

    # 4) FINAL fallback: naive substring check (safe because we know exact source names)
    bl = b.lower()
    for s in SOURCES:
        if s in bl:
            dprint(f"[source] substring fallback: '{s}' in '{b}'")
            return s

    dprint(f"[source] none in '{b}'")
    return None


def _find_fitbit_signal(b: str) -> Optional[str]:
    m = re.search(r"\b(hr|steps|sleep|temp|resp|spo2)\b", b, re.IGNORECASE)
    if m: return m.group(1).lower()
    if "fitbit_hr" in b.lower() or "hrdlin" in b.lower(): return "hr"
    return None

def _find_level(b: str) -> Optional[str]:
    m = _level_pat.search(b)
    if m: return _norm_level(m.group(1))
    m2 = re.search(r"EVAL(high|low|medium|mid(?:dle)?)", b, re.IGNORECASE)
    if m2: return _norm_level(m2.group(1))
    m3 = re.search(r"testing_(high|low|medium|mid(?:dle)?)", b, re.IGNORECASE)
    if m3: return _norm_level(m3.group(1))
    return None


# Treat non-alphanumeric characters (including underscore) as delimiters
_BOUNDARY = r"(?<![A-Za-z0-9])"   # left boundary: start or non-alnum
_RBOUND  = r"(?![A-Za-z0-9])"     # right boundary: end or non-alnum

def _find_unisynth_h(b: str) -> Optional[str]:
    # h### like "..._h800_..." (underscore-safe)
    m = re.search(rf"{_BOUNDARY}h(\d{{2,4}}){_RBOUND}", b, re.IGNORECASE)
    if m:
        h = f"h{m.group(1)}"
        dprint(f"[unisynth] matched 'h###': {h} in '{b}'")
        return h
    # fallback: EVAL### like "..._EVAL200_..."
    m2 = re.search(rf"{_BOUNDARY}eval(\d{{2,4}}){_RBOUND}", b, re.IGNORECASE)
    if m2:
        h = f"h{m2.group(1)}"
        dprint(f"[unisynth] matched 'EVAL###' → {h} in '{b}'")
        return h
    dprint(f"[unisynth] no h/EVAL token in '{b}'")
    return None


def extract_task_key(basename: str, domain: str) -> Optional[str]:
    b = re.sub(r"\.txt$", "", basename, flags=re.IGNORECASE)
    dom = domain.lower()
    if dom in ("carboncast","spectralunitest","carbon_eval"):
        region = _find_region(b); source = _find_source(b)
        if region and source: 
            print("region: ", region)
            print("source: ", source)
            return f"{region.lower()}__{source}"
        if region: return region.lower()
        if source: return source
        return None
    if dom == "fitbit":
        sig = _find_fitbit_signal(b) or "hr"
        lvl = _find_level(b); return f"{sig}__{lvl}" if lvl else sig
    if dom == "pems":
        lvl = _find_level(b); return lvl or "default"
    if dom == "unisynth":
        h = _find_unisynth_h(b); return h or "default"
    return None

def make_pair_key(basename: str, domain: str) -> str:
    tk = extract_task_key(basename, domain)
    if tk: return tk
    b = re.sub(r"\.txt$", "", basename, flags=re.I)
    b = re.sub(r"(dlin|gpt2|llama3\.?2|language_pretrained|random_init)", "", b, flags=re.I)
    b = re.sub(r"[_\-](seed|s|initseed|init|i)\d+", "", b, flags=re.I)
    b = re.sub(r"__+", "_", b)
    return b.lower().strip("_")

# ---- Loading ----
def load_logs(log_dirs: List[str]) -> pd.DataFrame:
    rows = []
    skipped = {
        "no_model": [],
        "no_read": [],
        "no_smape": [],
        "no_pair_key": [],
    }
    kept = 0

    for d in log_dirs:
        base_dir_name = os.path.basename(os.path.normpath(d))
        fallback_domain = DOMAIN_NAME_FIX.get(base_dir_name, base_dir_name)
        files = sorted(glob(os.path.join(d, "**", "*.txt"), recursive=True))
        dprint(f"[scan] Dir={d}  files={len(files)}")
        for path in files:
            bn = os.path.basename(path)

            # 1) model
            model = infer_model(bn)
            if model is None:
                skipped["no_model"].append(bn)
                continue

            # 2) read file
            try:
                with open(path, "r", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                dprint(f"[skip][read] {bn} error: {e}")
                skipped["no_read"].append(bn)
                continue

            # 3) parse smape
            metrics = parse_run_summary(text)
            smape = choose_metric(metrics, SMAPE_PATS)
            if smape is None or not np.isfinite(smape):
                dprint(f"[skip][smape] {bn} (no sMAPE metric found)")
                skipped["no_smape"].append(bn)
                continue

            # 4) domain + pair key (verbose)
            domain = infer_domain(bn, fallback_domain)
            pk = make_pair_key(bn, domain)
            if not pk:
                dprint(f"[skip][pair_key] {bn} (domain={domain})")
                skipped["no_pair_key"].append(bn)
                continue

            rows.append(dict(domain=domain, model=model, pair_key=pk, smape=float(smape), filename=bn))
            kept += 1
            dprint(f"[keep] {bn}  → domain={domain}  model={model}  key={pk}  sMAPE={smape:.4g}")

    # --- summary of skips ---
    print("\n=== LOAD SUMMARY ===")
    print(f"kept rows: {kept}")
    total_skipped = sum(len(v) for v in skipped.values())
    print(f"skipped total: {total_skipped}")
    for reason, lst in skipped.items():
        print(f"  {reason}: {len(lst)}")
        if DEBUG and lst:
            for ex in lst[:8]:
                print("    •", ex)
            if len(lst) > 8:
                print(f"    ... (+{len(lst)-8} more)")

    return pd.DataFrame(rows)


# ---- Easiness & stats ----
def build_easiness_table(df: pd.DataFrame, ref: str="mean") -> pd.DataFrame:
    """
    One row per (domain, pair_key) with:
      other_smape = aggregate(non-DLinear sMAPE)
      dlinear_smape = aggregate(DLinear sMAPE)
      delta_smape = other_smape - dlinear_smape
    """
    cols = ["domain","pair_key","other_smape","dlinear_smape","delta_smape","n_other","n_dlinear"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for (dom, key), g in df.groupby(["domain","pair_key"]):
        dlin = g[g["model"]=="DLinear"]
        other = g[g["model"]!="DLinear"]
        if dlin.empty or other.empty:
            continue

        d_smape = dlin["smape"].mean()
        if ref == "best":
            o_smape = other["smape"].min()
        elif ref == "median":
            o_smape = other["smape"].median()
        else:
            o_smape = other["smape"].mean()

        if not (np.isfinite(d_smape) and np.isfinite(o_smape)):
            continue

        rows.append(dict(domain=dom,
                         pair_key=key,
                         other_smape=float(o_smape),
                         dlinear_smape=float(d_smape),
                         delta_smape=float(o_smape - d_smape),
                         n_other=len(other),
                         n_dlinear=len(dlin)))
    return pd.DataFrame(rows)

def theilsen(x,y):
    x,y = np.asarray(x,float),np.asarray(y,float)
    ok = np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    if x.size<2: return np.nan,np.nan
    slopes=[]
    for i in range(len(x)-1):
        dx=x[i+1:]-x[i]; v=dx!=0
        if np.any(v): slopes.extend(((y[i+1:][v]-y[i])/dx[v]).tolist())
    if not slopes: return np.nan,np.nan
    s=float(np.median(slopes))
    b=float(np.median(y - s*x))
    return s,b

def spearman_rho(x,y):
    try:
        from scipy import stats as ss
        r,_=ss.spearmanr(x,y,nan_policy="omit"); return float(r)
    except Exception:
        return np.nan

# ---- Plot ----
def plot_adv(df_dom: pd.DataFrame, out_png: str, domain: str):
    # X = OtherModels sMAPE, Y = (OtherModels − DLinear) sMAPE
    x = df_dom["other_smape"].to_numpy(float)
    y = df_dom["delta_smape"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    fig, ax = plt.subplots(figsize=(6.2,4.0), dpi=300, layout="constrained")
    if x.size == 0:
        ax.set_title(f"DLinear improvement vs OtherModels — {domain}")
        ax.set_xlabel("OtherModels mean sMAPE"); ax.set_ylabel("Δ sMAPE (Other − DLinear)")
        ax.text(0.5,0.5,"No pairs",ha="center",va="center",transform=ax.transAxes)
        fig.savefig(out_png,bbox_inches="tight"); plt.close(fig)
        print("Saved", out_png)
        return dict(domain=domain, n=0, spearman_rho=np.nan)

    ax.scatter(x,y,s=32,alpha=0.85,edgecolors="none",label="pairs")
    ax.axhline(0.0,color="gray",linestyle="--",linewidth=1.0)

    # Spearman correlation: does DLinear advantage grow with task difficulty?
    try:
        from scipy import stats as ss
        rho, _ = ss.spearmanr(x,y,nan_policy="omit")
    except Exception:
        rho = np.nan

    ax.set_xlabel("OtherModels mean sMAPE", labelpad=2)
    ax.set_ylabel("Δ sMAPE (Other − DLinear)", labelpad=2)
    ax.set_title(f"DLinear advantage vs task difficulty — {domain}", pad=2)
    ax.legend(loc="best"); ax.grid(True,linestyle="--",alpha=0.30)

    fig.savefig(out_png,bbox_inches="tight"); plt.close(fig)
    print("Saved", out_png)
    return dict(domain=domain, n=int(x.size), spearman_rho=rho)

def plot_adv_all(E: pd.DataFrame, out_png: str):
    """
    Pooled plot across all domains:
      X = OtherModels mean sMAPE
      Y = (OtherModels − DLinear) sMAPE
      color by domain
    """
    if E.empty:
        print("[info] No data for pooled plot.")
        return {"n": 0, "spearman_rho": np.nan}

    # keep only finite
    M = E[np.isfinite(E["other_smape"]) & np.isfinite(E["delta_smape"])].copy()
    if M.empty:
        print("[info] No finite rows for pooled plot.")
        return {"n": 0, "spearman_rho": np.nan}

    x = M["other_smape"].to_numpy(float)
    y = M["delta_smape"].to_numpy(float)

    # Spearman (overall)
    try:
        from scipy import stats as ss
        rho, _ = ss.spearmanr(x, y, nan_policy="omit")
    except Exception:
        rho = np.nan

    # color by domain
    doms = sorted(M["domain"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_for = {d: cmap(i % 10) for i, d in enumerate(doms)}

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300, layout="constrained")

    for d in doms:
        g = M[M["domain"] == d]
        ax.scatter(
            g["other_smape"].to_numpy(float),
            g["delta_smape"].to_numpy(float),
            s=30, alpha=0.9, edgecolors="none", label=d, color=color_for[d]
        )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, label="Δ=0")

    # Optional robust fit (commented, enable if desired)
    # s, b = theilsen(x, y)
    # if np.isfinite(s):
    #     xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    #     ys = b + s * xs
    #     ax.plot(xs, ys, linewidth=1.6, label=f"Theil–Sen slope={s:.3g}")

    slope, intercept = np.polyfit(x, y, 1)

    xs = np.linspace(x.min(), x.max(), 200)
    ys = intercept + slope * xs
    ax.plot(xs, ys, color="black", lw=1.8, label=f"OLS slope={slope:.2f}")

    ax.set_xlabel("TSFM mean sMAPE", labelpad=2)
    ax.set_ylabel("Δ sMAPE (TSFM − DLinear)", labelpad=2)
    ax.set_title(f"DLinear advantage vs TSFMs — ρ={rho:.2f}", pad=2)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.30)

    
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_png)

    
    return {"n": int(len(M)), "spearman_rho": float(rho) if np.isfinite(rho) else np.nan}

# ---- Main ----
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log-dirs", type=str, default=",".join(DEFAULT_LOG_DIRS),
                    help="Comma-separated dirs with *.txt logs.")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--easy-ref", type=str, default="mean", choices=["mean","best","median"],
                    help="How to aggregate non-DLinear baselines.")
    ap.add_argument("--debug", action="store_true", help="Verbose file-by-file diagnostics")
               
    args=ap.parse_args()
    global DEBUG
    DEBUG = bool(args.debug)
    # Robustly parse log_dirs (avoid bool/None issues)
    log_dirs = []
    if isinstance(args.log_dirs, str) and args.log_dirs.strip():
        log_dirs = [d.strip() for d in args.log_dirs.split(",") if d.strip()]
    else:
        log_dirs = DEFAULT_LOG_DIRS[:]

    os.makedirs(args.out_dir, exist_ok=True)

    df=load_logs(log_dirs)
    print("\n=== PARSED ROWS ===")
    print("[debug] parsed rows:", len(df))
    if not df.empty:
        print("[debug] domains:", sorted(df["domain"].unique().tolist()))
        print("[debug] models:", sorted(df["model"].unique().tolist()))
        print("\n[debug] domain × model counts:")
        print(df.pivot_table(index="domain", columns="model", values="pair_key", aggfunc="count", fill_value=0))

        # Inspect keys present per domain/model (helps see where pairing fails)
        for dom in sorted(df["domain"].unique()):
            print(f"\n[debug] sample keys — {dom}:")
            ex = (df[df["domain"]==dom]
                .groupby(["model","pair_key"])
                .size()
                .reset_index(name="n")
                .sort_values("n", ascending=False)
                .head(12))
            print(ex.to_string(index=False))

        if df.empty:
            print("[info] No logs parsed. Check --log-dirs.")
            print("[debug] attempted dirs:", log_dirs)
            return
        if "domain" not in df.columns or "pair_key" not in df.columns:
            print("[fatal] parsed DataFrame missing required columns.")
            print(df.head(3))
            return

    # Quick debug summary
    print("[debug] parsed rows:", len(df))
    print("[debug] domains:", sorted(df["domain"].unique().tolist()))
    print("[debug] models:", sorted(df["model"].unique().tolist()))

    # Build (baseline vs gain) table
    E=build_easiness_table(df,ref=args.easy_ref)
    pairs_csv=os.path.join(args.out_dir,"advantage_pairs.csv")
    E.to_csv(pairs_csv,index=False)
    print(f"[write] {pairs_csv}  ({len(E)} rows)")

    # --- pooled (all domains) plot ---
    pooled_png = os.path.join(args.out_dir, "ALL_advantage.png")
    pooled_stats = plot_adv_all(E, pooled_png)

    # (optional) append pooled row to stats csv
    # we’ll write it after per-domain stats are collected below

    if E.empty:
        print("[info] No (DLinear, baseline) pairs found after pairing. Check filename patterns.")
        return

    # Per-domain plots + stats
    stats = []
    for dom,g in E.groupby("domain",sort=False):
        if g.empty: continue
        out=os.path.join(args.out_dir,f"{dom}_advantage.png")
        st=plot_adv(g,out,dom); stats.append(st)

    # append pooled
    if pooled_stats and pooled_stats.get("n", 0) > 0:
        stats.append(dict(domain="ALL", **{k:v for k,v in pooled_stats.items() if k!="n"}))

    stats_csv=os.path.join(args.out_dir,"advantage_stats.csv")
    pd.DataFrame(stats).to_csv(stats_csv,index=False)
    print(f"[write] {stats_csv}")

    

if __name__=="__main__":
    main()
