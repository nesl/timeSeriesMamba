import os
import pandas as pd
import json

input_dir = '.'
output_csv = 'combined3_clean_years.csv'
output_json = 'combined3_clean_boundaries.json'
#heldout_csv = 'heldout_CISO.csv'

base_rows = 8766
train_rows = base_rows * 4
val_rows   = base_rows

columns_to_keep = ['date','coal','nat_gas','nuclear','oil','hydro','solar','wind','other']
#holdout_fname  = 'CISO_clean.csv'
holdout_files = ["CISO_clean.csv", "TVA_clean.csv", "FR_clean.csv", "PJM_clean.csv"]
#holdout_files = ["CISO_clean.csv"]

all_train_val = []
train = []
val = []
boundaries    = []
region_names  = []
current_start = 0

# 1) split train/val on all regions except CISO
for fname in sorted(os.listdir(input_dir)):
    # retain exclusion logic for only clean CSVs, no heldout or combined files
    if ('clean' in fname and fname.endswith('.csv') 
        and 'heldout' not in fname 
        and 'combined' not in fname 
        and fname != os.path.basename(output_csv)):
        # skip the holdout region file
        #if fname == holdout_fname:
        #    continue
        if any(bad in fname for bad in holdout_files):
            continue

        df = pd.read_csv(os.path.join(input_dir, fname))
        if any(col not in df for col in columns_to_keep):
            print(f"[SKIP] {fname} missing cols")
            continue

        region = fname.replace('.csv','')
        df = df[columns_to_keep].copy()
        
        # train slice
        tr = df.iloc[:train_rows].copy()
        tr['region'] = region
        train.append(tr)
        start = current_start
        end   = start + len(tr) - 1
        #boundaries.append([start, end])
        #region_names.append(region)
        current_start = end + 1

        # val slice
        vl = df.iloc[train_rows:train_rows+val_rows].copy()
        vl['region'] = region
        val.append(vl)
        start = current_start
        end   = start + len(vl) - 1
        #boundaries.append([start, end])
        #region_names.append(region)
        current_start = end + 1

# 2) save train+val
all_train_val = train + val
combined_df = pd.concat(all_train_val, ignore_index=True)

boundaries = []
region_names = []
current_start = 0

for df_slice in all_train_val:
    n = len(df_slice)
    boundaries.append([current_start, current_start + n - 1])
    region_names.append(df_slice['region'].iloc[0])
    current_start += n

with open(output_json, 'w') as f:
    json.dump({
        "boundaries": boundaries,
        "regions": region_names
    }, f, indent=2)
    
combined_df.to_csv(output_csv, index=False)
no_region_df = combined_df.drop(columns=['region'])
# 3) load entire CISO as heldout
#hold_df = pd.read_csv(os.path.join(input_dir, holdout_fname))[columns_to_keep].copy()
#hold_df['region'] = 'CISO'
#hold_df.to_csv(heldout_csv, index=False)

print(f"train+val ▶ {output_csv} ({combined_df.shape})")
#print(f"heldout  ▶ {heldout_csv} ({hold_df.shape})")
no_region_df.to_csv(f'{output_csv}_no_region.csv', index=False)

print(f"metadata ▶ {output_json}")
