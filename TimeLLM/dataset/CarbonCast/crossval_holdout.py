import pandas as pd
import json

# Input files
combined_csv = 'combined3_clean_years.csv'
boundary_file = 'combined3_clean_boundaries.json'

# Load data
df = pd.read_csv(combined_csv)
with open(boundary_file, 'r') as f:
    meta = json.load(f)

boundaries = meta['boundaries']
region_names = meta['regions']

# Generate splits
for i, (start, end) in enumerate(boundaries):
    held_out_region = region_names[i]
    mask = df.index < start
    mask |= df.index > end
    split_df = df[mask].drop(columns=['region'])

    output_file = f'combined3_{held_out_region}.csv'
    split_df.to_csv(output_file, index=False)
    print(f"Saved split with held-out region '{held_out_region}': {output_file}, shape: {split_df.shape}")
