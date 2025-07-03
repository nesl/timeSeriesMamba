import os
import pandas as pd
import json

input_dir = '.'
output_csv = 'combined_clean_49000.csv'
output_json = 'combined_clean_boundaries.json'
rows_per_file = 7000

# Columns you want to keep from input files
columns_to_keep = ['date', 'coal', 'nat_gas', 'nuclear', 'oil', 'hydro', 'solar', 'wind', 'other']

all_dfs = []
boundaries = []
region_names = []
current_start = 0

for fname in sorted(os.listdir(input_dir)):
    if 'clean' in fname and fname.endswith('.csv') and fname != os.path.basename(output_csv):
        path = os.path.join(input_dir, fname)
        df = pd.read_csv(path)

        missing = [col for col in columns_to_keep if col not in df.columns]
        if missing:
            print(f"[SKIPPED] {fname} is missing columns: {missing}")
            continue

        df_trimmed = df[columns_to_keep].iloc[:rows_per_file].copy()
        df_trimmed['region'] = fname.replace('.csv', '')
        all_dfs.append(df_trimmed)

        start = current_start
        end = current_start + len(df_trimmed) - 1
        boundaries.append([start, end])
        region_names.append(fname.replace('.csv', ''))
        current_start = end + 1

# Combine all and save
combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
combined_df.to_csv(output_csv, index=False)

with open(output_json, 'w') as f:
    json.dump({"boundaries": boundaries, "regions": region_names}, f, indent=2)

print(f"Saved combined CSV: {output_csv}, shape: {combined_df.shape}")
print(f"Saved boundary file: {output_json}")
