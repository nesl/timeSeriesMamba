import os
import math
import pandas as pd
import json

input_dir = '.'
output_csv = 'AFNPST.csv'
output_json = 'AFNPST_boundaries.json'

columns_to_keep = ['date','coal','nat_gas','nuclear','oil','hydro','solar','wind','other']

# choose regions explicitly; leave empty to auto-pick first N from dir
selected_region_files = ["AZPS_clean.csv","FR_clean.csv","NYIS_clean.csv","PJM_clean.csv","SWPP_clean.csv","TVA_clean.csv"]  # e.g., ["CISO_clean.csv","PJM_clean.csv","TVA_clean.csv"]
N = 0

holdout_files = ["CISO_clean.csv"]  # exclude these if you’re doing heldout elsewhere

base_rows_per_year = 8766
total_years = 5
total_rows = base_rows_per_year * total_years
train_total_rows = base_rows_per_year * 4
val_total_rows = base_rows_per_year * 1

def even_split(total, n):
    q, r = divmod(total, n)
    return [q + 1 if i < r else q for i in range(n)]

def proportional_train_alloc(totals, desired_train_total):
    raw = [t * 0.8 for t in totals]
    base = [math.floor(x) for x in raw]
    diff = desired_train_total - sum(base)
    if diff > 0:
        frac = [(raw[i] - base[i], i) for i in range(len(totals))]
        frac.sort(reverse=True)
        for k in range(diff):
            base[frac[k][1]] += 1
    return base

eligible = []
for fname in sorted(os.listdir(input_dir)):
    if ('clean' in fname and fname.endswith('.csv') 
        and 'heldout' not in fname 
        and 'combined' not in fname 
        and fname != os.path.basename(output_csv)):
        eligible.append(fname)

if selected_region_files:
    files = [f for f in selected_region_files if f in eligible]
else:
    files = [f for f in eligible if f not in holdout_files][:N]

if len(files) == 0:
    raise RuntimeError("No region files selected.")
N = len(files)

per_region_totals = even_split(total_rows, N)
per_region_train = proportional_train_alloc(per_region_totals, train_total_rows)
per_region_val = [per_region_totals[i] - per_region_train[i] for i in range(N)]

train_slices = []
val_slices = []
boundaries = []
region_names = []
current_start = 0

for i, fname in enumerate(files):
    df = pd.read_csv(os.path.join(input_dir, fname))
    if any(col not in df for col in columns_to_keep):
        raise ValueError(f"{fname} missing required columns.")
    region = fname.replace('.csv','')
    df = df[columns_to_keep].copy()

    need_total = per_region_totals[i]
    if len(df) < need_total:
        raise ValueError(f"{fname} has only {len(df)} rows; need {need_total}.")

    tr_n = per_region_train[i]
    vl_n = per_region_val[i]

    tr = df.iloc[:tr_n].copy()
    tr['region'] = region
    train_slices.append(tr)
    boundaries.append([current_start, current_start + tr_n - 1])
    region_names.append(region)
    current_start += tr_n

    vl = df.iloc[tr_n:tr_n + vl_n].copy()
    vl['region'] = region
    val_slices.append(vl)
    boundaries.append([current_start, current_start + vl_n - 1])
    region_names.append(region)
    current_start += vl_n

all_train_val = train_slices + val_slices
combined_df = pd.concat(all_train_val, ignore_index=True)

with open(output_json, 'w') as f:
    json.dump({
        "boundaries": boundaries,
        "regions": region_names
    }, f, indent=2)

combined_df.to_csv(output_csv, index=False)
no_region_df = combined_df.drop(columns=['region'])
no_region_df.to_csv(f'{output_csv}_no_region.csv', index=False)

print(f"Selected regions: {files}")
print(f"Per-region totals: {per_region_totals}")
print(f"Per-region train:  {per_region_train}  (sum={sum(per_region_train)})")
print(f"Per-region val:    {per_region_val}    (sum={sum(per_region_val)})")
print(f"train+val ▶ {output_csv} {combined_df.shape}")
print(f"no_region ▶ {output_csv}_no_region.csv {no_region_df.shape}")
print(f"metadata ▶ {output_json}")
