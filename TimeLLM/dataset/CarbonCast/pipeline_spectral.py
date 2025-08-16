#!/usr/bin/env python3
import os, json, math
import numpy as np
import pandas as pd

# -----------------------
# Config (adjust as needed)
# -----------------------
input_dir   = '.'
output_csv  = 'spectral/combined_years.csv'
output_json = 'spectral/combined_boundaries.json'

base_rows  = 8766            # ~1 year hourly
train_rows = base_rows * 4   # 4y train
val_rows   = base_rows       # 1y val
test_rows  = base_rows       # use 1y test slice for SE/eval bins

columns_to_keep = ['date','coal','nat_gas','nuclear','oil','hydro','solar','wind','other']

# Pool we will evaluate on (frozen model) and from which we pick SE percentiles
holdout_files = ["CISO_clean.csv", "TVA_clean.csv", "FR_clean.csv", "PJM_clean.csv"]

# How many series to keep around each percentile (pick nearest k)
k_per_bin = 3
percentiles = [10, 50, 90]

# Where to write selected per-series CSVs for evaluation
eval_dir = 'eval_bins'
os.makedirs(eval_dir, exist_ok=True)

# -----------------------
# 1) Build train+val exactly like your current script
# -----------------------
all_train_val, train, val = [], [], []
boundaries, region_names = [], []
current_start = 0

def valid_clean_csv(name: str) -> bool:
    return (
        ('clean' in name) and name.endswith('.csv') and
        ('heldout' not in name) and ('combined' not in name) and
        (name != os.path.basename(output_csv))
    )

for fname in sorted(os.listdir(input_dir)):
    if not valid_clean_csv(fname):
        continue
    if any(bad in fname for bad in holdout_files):
        # skip heldout pool for training
        continue

    df = pd.read_csv(os.path.join(input_dir, fname))
    if any(col not in df for col in columns_to_keep):
        print(f"[SKIP] {fname} missing required columns")
        continue

    region = fname.replace('.csv','')
    df = df[columns_to_keep].copy()

    # 4y train
    tr = df.iloc[:train_rows].copy()
    tr['region'] = region
    train.append(tr)

    # 1y val immediately after
    vl = df.iloc[train_rows:train_rows+val_rows].copy()
    vl['region'] = region
    val.append(vl)

# Save combined (train+val) and boundaries
all_train_val = train + val
combined_df = pd.concat(all_train_val, ignore_index=True)

boundaries, region_names, current_start = [], [], 0
for df_slice in all_train_val:
    n = len(df_slice)
    boundaries.append([current_start, current_start + n - 1])
    region_names.append(df_slice['region'].iloc[0])
    current_start += n

with open(output_json, 'w') as f:
    json.dump({"boundaries": boundaries, "regions": region_names}, f, indent=2)

combined_df.to_csv(output_csv, index=False)
combined_df.drop(columns=['region']).to_csv(f'{output_csv}_no_region.csv', index=False)

print(f"train+val ▶ {output_csv} ({combined_df.shape})")
print(f"metadata ▶ {output_json}")

# -----------------------
# 2) Spectral Entropy util (periodogram-based; no SciPy needed)
# -----------------------
def spectral_entropy(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Periodogram-based spectral entropy in [0,1].
    - Detrend by removing mean
    - Real FFT power spectrum
    - Normalize to pmf over positive frequencies
    """
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    # zero-fill NaNs if any remain after simple fill below
    x = np.nan_to_num(x, nan=0.0)

    # real FFT power spectrum
    X = np.fft.rfft(x)
    psd = (X * np.conj(X)).real
    # drop DC if you want to reduce trend impact; keep it but avoid NaNs
    psd = np.maximum(psd, 0.0)

    # pmf
    p = psd / (psd.sum() + eps)
    # avoid log(0)
    p = np.clip(p, eps, 1.0)
    H = -(p * np.log(p)).sum()
    H_norm = H / math.log(len(p))  # normalize
    # guard numerical drift
    return float(np.clip(H_norm, 0.0, 1.0))

# -----------------------
# 3) Build SE catalog over holdout pool
# -----------------------
records = []
for fname in holdout_files:
    path = os.path.join(input_dir, fname)
    if not os.path.exists(path):
        print(f"[WARN] Holdout file missing: {fname}")
        continue

    df = pd.read_csv(path)
    if any(col not in df for col in columns_to_keep):
        print(f"[SKIP] {fname} missing required columns")
        continue

    # sort and coerce dates defensively
    df = df[columns_to_keep].copy()
    # df['date'] = pd.to_datetime(df['date'], errors='coerce')  # keep as-is if your loader wants raw strings
    # standardize test slice: next chunk after 4y train + 1y val
    start = train_rows + val_rows
    end   = start + test_rows
    test = df.iloc[start:end].copy()

    # Fallback if not enough rows: use the last 'test_rows' rows available
    if len(test) < test_rows:
        test = df.tail(test_rows).copy()

    # Simple imputation to avoid NaNs in entropy
    test = test.fillna(method='ffill').fillna(method='bfill')

    region = fname.replace('.csv','')

    for source in ['coal','nat_gas','nuclear','oil','hydro','solar','wind','other']:
        series = test[source].values
        if np.all(np.isnan(series)) or len(series) < 64:
            continue
        se = spectral_entropy(series)
        records.append({
            'region': region,
            'file': fname,
            'source': source,
            'se': se,
            'start_idx': start,
            'end_idx': start + len(test) - 1
        })

se_df = pd.DataFrame.from_records(records)
if se_df.empty:
    raise SystemExit("No SE records computed; check holdout_files and columns.")

se_df = se_df.sort_values('se').reset_index(drop=True)
se_df.to_csv(os.path.join(eval_dir, 'se_catalog.csv'), index=False)
print(f"[INFO] SE catalog saved → {os.path.join(eval_dir, 'se_catalog.csv')}  (n={len(se_df)})")

# -----------------------
# 4) Select near-percentile series (k_per_bin per percentile)
# -----------------------
def pick_nearest(df, percentile, k):
    target = np.percentile(df['se'].values, percentile)
    df = df.assign(dist=(df['se'] - target).abs())
    picked = df.nsmallest(k, 'dist').copy()
    picked['percentile'] = percentile
    picked['target_se'] = target
    return picked

picked_list = []
for p in percentiles:
    picked_list.append(pick_nearest(se_df, p, k_per_bin))
picked_df = pd.concat(picked_list, ignore_index=True)

# Deduplicate in case of overlaps between percentiles
picked_df = picked_df.drop_duplicates(subset=['region','source']).reset_index(drop=True)
picked_df = picked_df.sort_values(['percentile','se']).reset_index(drop=True)

picked_manifest = os.path.join(eval_dir, 'manifest_entropy_tiers.csv')
picked_df.to_csv(picked_manifest, index=False)
print(f"[INFO] Manifest saved → {picked_manifest}")

# -----------------------
# 5) Materialize per-series heldout CSVs for evaluation (univariate)
#    Format: keep 'date' + chosen target column; name encodes region/source/percentile
# -----------------------
made = []
for _, row in picked_df.iterrows():
    fname, region, source, p = row['file'], row['region'], row['source'], int(row['percentile'])
    path = os.path.join(input_dir, fname)
    df = pd.read_csv(path)[['date', source]].copy()

    # Use the same test slice indices chosen above; fallback to tail if needed
    start, end = int(row['start_idx']), int(row['end_idx'])
    if end >= len(df) or start < 0 or (end - start + 1) < test_rows:
        df_slice = df.tail(test_rows).copy()
    else:
        df_slice = df.iloc[start:end+1].copy()

    out_name = f'heldout_{region}_{source}_p{p}.csv'
    out_path = os.path.join(eval_dir, out_name)
    df_slice.to_csv(out_path, index=False)
    made.append(out_name)

print(f"[INFO] Wrote {len(made)} per-series heldout CSVs to {eval_dir}")
