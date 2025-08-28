#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

# -----------------------
# Config (kept consistent with your pipeline)
# -----------------------
INPUT_DIR   = Path("..")
OUT_DIR     = Path(".")
HELDOUT_DIR = OUT_DIR / "heldout"

TRAINVAL_CSV         = OUT_DIR / "train_val.csv"              # not required here, just for reference
BOUNDARY_JSON        = OUT_DIR / "train_val_boundaries.json"  # read-only
HELDOUT_JSON         = OUT_DIR / "heldout_meta.json"          # read & append

SOURCES     = ['coal','nat_gas','nuclear','oil','hydro','solar','wind','other']
COLUMNS_TO_KEEP = ['date'] + SOURCES

HOURS_PER_DAY = 24
SEGMENT_TRAIN_HOURS = 12 * 7 * HOURS_PER_DAY   # 12 weeks (same as pipeline)
SEGMENT_VAL_HOURS   =  3 * 7 * HOURS_PER_DAY   # 3 weeks (not needed here)
TRAIN_ROWS = SEGMENT_TRAIN_HOURS

# How many NEW heldouts to add
NUM_NEW_HELDOUTS = 5

# -----------------------
# Helpers (mirrors your pipeline)
# -----------------------
def is_clean_input_file(name: str) -> bool:
    return ("clean" in name) and name.endswith(".csv") and ("combined" not in name) and ("heldout" not in name)

def load_region_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df[COLUMNS_TO_KEEP].copy()

def spectral_entropy(x: np.ndarray) -> float:
    """
    Spectral entropy in [0,1] — same recipe you used:
    - z-score
    - real FFT power spectrum (drop DC)
    - Shannon entropy / log(N_bins)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 16:
        return 0.0
    mu, sd = x.mean(), x.std()
    if sd <= 1e-12:
        return 0.0
    x = (x - mu) / sd
    psd = np.abs(np.fft.rfft(x))**2
    psd = psd[1:]  # drop DC
    s = psd.sum()
    if s <= 1e-20:
        return 0.0
    p = np.clip(psd / s, 1e-20, 1.0)
    H = -(p * np.log(p)).sum()
    Hmax = np.log(p.size) if p.size > 0 else 1.0
    return float(H / Hmax)

def _max_consecutive_true(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    m = mask.astype(np.uint8)
    diffs = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    runs = ends - starts
    return int(runs.max()) if runs.size else 0

def mostly_zero(series: pd.Series,
                abs_eps: float = 1e-9,
                max_nonzero_frac: float = 0.01,
                max_std: float = 1e-6,
                zero_run_len: int = 300) -> bool:
    s = pd.to_numeric(series, errors='coerce')
    s = s[np.isfinite(s)]
    if s.empty:
        return True
    near_zero = (np.abs(s.values) <= abs_eps)
    if _max_consecutive_true(near_zero) >= zero_run_len:
        return True
    nonzero_frac = (~near_zero).mean()
    if nonzero_frac <= max_nonzero_frac:
        return True
    if s.std() <= max_std:
        return True
    return False

def pick_uniform_by_sorted(n, k):
    if k <= 0: return []
    if n <= k: return list(range(n))
    pos = np.rint(np.linspace(0, n-1, k)).astype(int)
    uniq, seen = [], set()
    for i in pos:
        if i not in seen:
            uniq.append(i); seen.add(i)
    j = 0
    step = max(1, n // k)
    while len(uniq) < k and j < n:
        if j not in seen:
            uniq.append(j); seen.add(j)
        j += step
    return uniq[:k]

# -----------------------
# Load existing “used” regions
# -----------------------
used_regions = set()

# 1) Already-held-out regions
if HELDOUT_JSON.exists():
    with open(HELDOUT_JSON, "r") as f:
        held = json.load(f)
    for item in held.get("heldout", []):
        # item["region"] looks like "REGION::source"
        used_regions.add(item["region"])

# 2) Regions used inside train_val (boundaries)
if BOUNDARY_JSON.exists():
    with open(BOUNDARY_JSON, "r") as f:
        bounds = json.load(f)
    for r in bounds.get("regions", []):
        used_regions.add(r)

print(f"[info] Regions already used (heldout/trainval): {len(used_regions)}")

# -----------------------
# Build fresh candidate list (excluding used)
# -----------------------
all_files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and is_clean_input_file(p.name)])
candidates = []  # dicts: {region, se, full}

for p in all_files:
    region_name = p.stem.replace("_clean", "")
    df = load_region_csv(p)

    for src in SOURCES:
        tag = f"{region_name}::{src}"
        if tag in used_regions:
            continue

        full = df[['date', src]].rename(columns={src: 'value'}).copy()

        # Skip dead series based on FULL series (heldout uses the full 5y)
        if mostly_zero(full['value']):
            continue

        # Compute SE on train window to be consistent with your pipeline
        seg_train = df.iloc[:TRAIN_ROWS][src].to_numpy()
        if len(seg_train) < TRAIN_ROWS:
            continue
        se_score = spectral_entropy(seg_train)

        candidates.append({
            "region": tag,
            "se": float(se_score),
            "full": full[['date','value']].copy()
        })

if not candidates:
    raise SystemExit("No remaining candidates. Everything is already used or filtered as zero/short.")

# -----------------------
# Pick 5 diverse regions across the SE spectrum
# -----------------------
candidates_sorted = sorted(candidates, key=lambda d: d["se"])
idxs = pick_uniform_by_sorted(len(candidates_sorted), NUM_NEW_HELDOUTS)
picked = [candidates_sorted[i] for i in idxs]

print("[info] Selected new regions:")
for c in picked:
    print("  -", c["region"], f"(SE={c['se']:.3f})")

# -----------------------
# Write their full 5y heldout CSVs and append to heldout_meta.json
# -----------------------
HELDOUT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing meta to append to (if present)
heldout_meta = {"percentiles": [], "heldout": []}
if HELDOUT_JSON.exists():
    with open(HELDOUT_JSON, "r") as f:
        heldout_meta = json.load(f)

# Compute percentiles of SE among *all remaining* for labeling
se_vals = np.array([c["se"] for c in candidates_sorted], dtype=float)
def se_percentile(v):
    # Percentile in the *remaining* pool
    return int(round(100.0 * (se_vals <= v).mean()))

new_entries = []
for c in picked:
    region_tag = c["region"].replace("::", "__")
    pct = se_percentile(c["se"])
    out_csv = HELDOUT_DIR / f"{region_tag}__p{pct:02d}_extra.csv"
    c["full"].to_csv(out_csv, index=False)

    entry = {
        "percentile": pct,
        "region": c["region"],
        "se": c["se"],
        "rows": int(len(c["full"])),
        "csv": str(out_csv)
    }
    heldout_meta["heldout"].append(entry)
    new_entries.append(entry)

# Optionally store the set of percentiles we’ve used historically
# (we won’t overwrite existing “percentiles” array; just keep as-is)
with open(HELDOUT_JSON, "w") as f:
    json.dump(heldout_meta, f, indent=2)

print(f"[OK] wrote {len(new_entries)} new heldouts to {HELDOUT_DIR}")
for e in new_entries:
    print(f"    • {e['region']}  SE={e['se']:.3f}  p≈{e['percentile']:02d}  → {e['csv']}")
