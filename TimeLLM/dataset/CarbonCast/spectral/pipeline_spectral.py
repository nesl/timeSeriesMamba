#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import json

# -----------------------
# Config
# -----------------------
INPUT_DIR   = Path("..")
OUT_DIR     = Path(".")

SOURCES     = ['coal','nat_gas','nuclear','oil','hydro','solar','wind','other']
COLUMNS_TO_KEEP = ['date'] + SOURCES

HOURS_PER_DAY = 24
SEGMENT_TRAIN_HOURS = 12 * 7 * HOURS_PER_DAY   # 12 weeks
SEGMENT_VAL_HOURS   =  3 * 7 * HOURS_PER_DAY   # 3 weeks

TRAIN_ROWS = SEGMENT_TRAIN_HOURS
VAL_ROWS   = SEGMENT_VAL_HOURS

# Keep trainval at ~30k rows. Budget applies ONLY to train_val.csv, not heldouts.
TRAINVAL_TOKEN_BUDGET   = 40_000
MAX_SEGMENTS_PER_SOURCE = None
DOWNSAMPLE_FACTOR       = 1        # for train/val only; heldout is full length by design

TRAINVAL_CSV   = OUT_DIR / "train_val.csv"               # date,value,region
TRAINVAL_NOREG = OUT_DIR / "train_val_no_region.csv"     # date,value
BOUNDARY_JSON  = OUT_DIR / "train_val_boundaries.json"

HELDOUT_DIR    = OUT_DIR / "heldout"
HELDOUT_JSON   = OUT_DIR / "heldout_meta.json"
HELDOUT_PCTS   = [5, 25, 50, 75, 95]  # percentile targets

# -----------------------
# Helpers
# -----------------------
def is_clean_input_file(name: str) -> bool:
    return ("clean" in name) and name.endswith(".csv") and ("combined" not in name) and ("heldout" not in name)

def load_region_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df[COLUMNS_TO_KEEP].copy()

def downsample(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    if factor and factor > 1:
        return df.iloc[::factor].reset_index(drop=True)
    return df

def spectral_entropy(x: np.ndarray) -> float:
    """
    Spectral entropy in [0,1].
    - z-score normalize
    - real FFT power spectrum
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

def nearest_indices_to_percentiles(values, percentiles):
    vals = np.array(values, dtype=float)
    order = np.argsort(vals)
    sorted_vals = vals[order]
    chosen, used = [], set()
    for q in percentiles:
        target = np.percentile(sorted_vals, q)
        idx = int(np.argmin(np.abs(sorted_vals - target)))
        # walk if collision
        L = R = idx
        while order[idx] in used:
            if L > 0 and order[L-1] not in used:
                L -= 1; idx = L
            elif R < len(sorted_vals)-1 and order[R+1] not in used:
                R += 1; idx = R
            else:
                break
        gi = order[idx]
        if gi not in used:
            chosen.append(gi); used.add(gi)
    return chosen

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


def _max_consecutive_true(mask: np.ndarray) -> int:
    """Return the maximum run length of True in a 1D boolean mask."""
    if mask.size == 0:
        return 0
    m = mask.astype(np.uint8)
    # Pad with 0s to catch edges; rising edges==1, falling edges==-1
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
    """
    True if the entire 5-year series is effectively zero/silent.
    Triggers if:
      - there exists a near-zero run of length >= zero_run_len, OR
      - fraction of non-zeros <= max_nonzero_frac, OR
      - std dev <= max_std
    """
    s = pd.to_numeric(series, errors='coerce')
    s = s[np.isfinite(s)]
    if s.empty:
        return True

    near_zero = (np.abs(s.values) <= abs_eps)

    # NEW: long consecutive near-zero run
    if _max_consecutive_true(near_zero) >= zero_run_len:
        return True

    # Mostly zeros overall
    nonzero_frac = (~near_zero).mean()
    if nonzero_frac <= max_nonzero_frac:
        return True

    # Vanishing variance
    if s.std() <= max_std:
        return True

    return False

# -----------------------
# 1) Collect candidates with SE (train-window) and keep full 5y series for heldout
# -----------------------
all_files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and is_clean_input_file(p.name)])
candidates = []  # dicts: {region, source, se, seg (train+val, ds), full (full 5y, no ds)}

for p in all_files:
    region_name = p.stem.replace("_clean", "")
    df = load_region_csv(p)

    for src in SOURCES:
        full = df[['date', src]].rename(columns={src: 'value'}).copy()

        # Skip dead series (full-length test uses this, so test it on the full 5y)
        if mostly_zero(full['value']):
            continue

        # Build train/val windows for SE + trainval candidates
        seg_train = df.iloc[:TRAIN_ROWS][['date', src]].copy()
        seg_val   = df.iloc[TRAIN_ROWS:TRAIN_ROWS+VAL_ROWS][['date', src]].copy()
        if len(seg_train) < TRAIN_ROWS or len(seg_val) < VAL_ROWS:
            continue

        se_score = spectral_entropy(seg_train[src].to_numpy())

        seg = pd.concat([
            seg_train.rename(columns={src: 'value'}),
            seg_val.rename(columns={src: 'value'}),
        ], ignore_index=True)
        seg = downsample(seg, DOWNSAMPLE_FACTOR)
        seg['region'] = f"{region_name}::{src}"
        seg = seg[['date', 'value', 'region']]

        candidates.append({
            'region': f"{region_name}::{src}",
            'source': src,
            'se': se_score,
            'seg': seg,               # 12w+3w (maybe downsampled) for train/val
            'full': full[['date','value']].copy()  # full 5y for heldout (no ds, no region)
        })

if MAX_SEGMENTS_PER_SOURCE is not None and candidates:
    by_src = {}
    for c in candidates:
        by_src.setdefault(c['source'], []).append(c)
    filtered = []
    for src, lst in by_src.items():
        lst_sorted = sorted(lst, key=lambda d: d['se'])
        keep_idx = pick_uniform_by_sorted(len(lst_sorted), min(MAX_SEGMENTS_PER_SOURCE, len(lst_sorted)))
        filtered.extend([lst_sorted[i] for i in keep_idx])
    candidates = filtered

if not candidates:
    raise SystemExit("No usable candidates (after zero-series filter). Check inputs/thresholds.")

# -----------------------
# 2) HELDOUT by SE percentiles (FULL 5y files, split-free)
# -----------------------
se_all = [c['se'] for c in candidates]
heldout_idx = set(nearest_indices_to_percentiles(se_all, HELDOUT_PCTS))
heldout = [candidates[i] for i in sorted(heldout_idx)]

# -----------------------
# 3) Train/val: compute N from budget; pick uniformly across SE (excluding heldouts)
# -----------------------
remaining = [c for i, c in enumerate(candidates) if i not in heldout_idx]
if not remaining:
    raise SystemExit("All candidates went to heldout; relax HELDOUT_PCTS or add more data.")

seg_len = len(remaining[0]['seg'])
budget = None if TRAINVAL_TOKEN_BUDGET is None else int(TRAINVAL_TOKEN_BUDGET)
N = len(remaining) if budget is None else max(1, budget // seg_len)
N = min(N, len(remaining))

remaining_sorted = sorted(remaining, key=lambda d: d['se'])
pick_idx = pick_uniform_by_sorted(len(remaining_sorted), N)
picked = [remaining_sorted[i] for i in pick_idx]

# Re-check budget (should be consistent if seg lengths match)
tokens_used = sum(len(c['seg']) for c in picked)
if budget is not None and tokens_used > budget:
    acc, trimmed = 0, []
    for c in picked:
        if acc + len(c['seg']) <= budget:
            trimmed.append(c)
            acc += len(c['seg'])
        else:
            break
    picked, tokens_used = trimmed, acc
    if not picked:
        raise SystemExit("Budget too small for even one segment. Increase budget or downsample more.")

# -----------------------
# 4) Write train/val outputs
# -----------------------
cursor = 0
bound_meta = []
tv_frames = []
for item in picked:
    seg = item['seg']
    start, end = cursor, cursor + len(seg) - 1
    bound_meta.append([start, end, item['region']])
    tv_frames.append(seg)
    cursor = end + 1

train_val_df = pd.concat(tv_frames, ignore_index=True)
train_val_df.to_csv(TRAINVAL_CSV, index=False)
train_val_df[['date','value']].to_csv(TRAINVAL_NOREG, index=False)

with open(BOUNDARY_JSON, "w") as f:
    json.dump({
        "boundaries": [[b[0], b[1]] for b in bound_meta],
        "regions": [b[2] for b in bound_meta]
    }, f, indent=2)

# -----------------------
# 5) Write HELDOUT: full 5 years, filename reflects percentile, CSV has only date,value
# -----------------------
HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
nearest_map = dict(zip(HELDOUT_PCTS, nearest_indices_to_percentiles(se_all, HELDOUT_PCTS)))
heldout_meta = []

for pct in HELDOUT_PCTS:
    idx = nearest_map[pct]
    c = candidates[idx]
    tag = c['region'].replace("::", "__")
    out_csv = HELDOUT_DIR / f"{tag}__p{int(pct):02d}.csv"
    # Full-length series, no region/split, no downsample
    c['full'][['date','value']].to_csv(out_csv, index=False)
    heldout_meta.append({
        "percentile": pct,
        "region": c['region'],
        "se": float(c['se']),
        "rows": int(len(c['full'])),
        "csv": str(out_csv)
    })

with open(HELDOUT_JSON, "w") as f:
    json.dump({
        "percentiles": HELDOUT_PCTS,
        "heldout": heldout_meta
    }, f, indent=2)

# -----------------------
# 6) Logs
# -----------------------
print(f"[OK] train_val ▶ {TRAINVAL_CSV} rows={len(train_val_df):,}  (budget={budget if budget is not None else 'unlimited'}, seg_len={seg_len})")
print(f"[OK] boundaries ▶ {BOUNDARY_JSON}")
print(f"[OK] no_region ▶ {TRAINVAL_NOREG}")
print(f"[OK] heldout ▶ {HELDOUT_DIR} (k={len(heldout_meta)}; full 5y each)")
print("[SE] train/val SE range:", f"{min([c['se'] for c in picked]):.3f} .. {max([c['se'] for c in picked]):.3f}")
