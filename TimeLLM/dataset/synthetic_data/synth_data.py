import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# ------------------------------
# Parameters
# ------------------------------
duration_years = 5  # Duration of the dataset in years
train_regions = [1, 2, 3, 4, 5, 6]  # Regions for train+val
train_frac = 0.8  # Fraction of data for training
start_date = datetime(2024, 1, 1, 0, 0, 0)  # Start date

# ------------------------------
# Fixed settings
# ------------------------------
test_region = 7  # Test region
regions = {
    1: {'freq': [0.1, 0.2, 1.0, 2.0], 'noise': 0.05},
    2: {'freq': [0.11, 0.21, 1.1, 2.1], 'noise': 0.1},
    3: {'freq': [0.12, 0.22, 1.2, 2.2], 'noise': 0.15},
    4: {'freq': [0.13, 0.23, 1.3, 2.3], 'noise': 0.2},
    5: {'freq': [0.14, 0.24, 1.4, 2.4], 'noise': 0.25},
    6: {'freq': [0.15, 0.25, 1.5, 2.5], 'noise': 0.3},
    7: {'freq': [0.2, 0.3, 2.0, 3.0], 'noise': 0.35}
}

# ------------------------------
# Helper functions
# ------------------------------
def generate_time_series(t, frequencies, noise_std, seed):
    np.random.seed(seed)
    amplitudes = [1 / f for f in frequencies]
    phases = np.random.uniform(0, 2 * np.pi, len(frequencies))
    y = np.zeros_like(t)
    for a, f, p in zip(amplitudes, frequencies, phases):
        y += a * np.sin(2 * np.pi * f * t + p)
    y += np.random.normal(0, noise_std, t.shape)
    return y

def generate_timestamps(start_date, num_points, freq='H'):
    return [start_date + timedelta(hours=i) for i in range(num_points)]

# ------------------------------
# Generate dataset
# ------------------------------
num_points = int(duration_years * 365.25 * 24)  # Total hours
train_split = int(train_frac * num_points)  # Train points per region
val_split = num_points - train_split  # Val points per region
t = np.linspace(0, duration_years, num_points)
timestamps = generate_timestamps(start_date, num_points)

# Generate train and val data
train_data = []
val_data = []
for region_id in train_regions:
    ts = generate_time_series(t, regions[region_id]['freq'], regions[region_id]['noise'], seed=region_id)
    train_ts = ts[:train_split]
    val_ts = ts[train_split:]
    train_df = pd.DataFrame({
        'date': timestamps[:train_split],
        'synth': train_ts
    })
    val_df = pd.DataFrame({
        'date': timestamps[train_split:],
        'synth': val_ts
    })
    train_data.append(train_df)
    val_data.append(val_df)

# Concatenate train and val
train_df_concat = pd.concat(train_data, ignore_index=True)
val_df_concat = pd.concat(val_data, ignore_index=True)
train_val_df = pd.concat([train_df_concat, val_df_concat], ignore_index=True)

# Generate test data
test_ts = generate_time_series(t, regions[test_region]['freq'], regions[test_region]['noise'], seed=test_region)
test_df = pd.DataFrame({
    'date': timestamps,
    'synth': test_ts
})

# ------------------------------
# Create boundaries
# ------------------------------
boundaries = []
regions_list = []
for k, region_id in enumerate(train_regions):
    start_train = k * train_split
    end_train = start_train + train_split - 1
    boundaries.append([start_train, end_train])
    regions_list.append(f'Region {region_id}')
val_start_offset = len(train_regions) * train_split
for k, region_id in enumerate(train_regions):
    start_val = val_start_offset + k * val_split
    end_val = start_val + val_split - 1
    boundaries.append([start_val, end_val])
    regions_list.append(f'Region {region_id}')

boundaries_dict = {
    'boundaries': boundaries,
    'regions': regions_list
}

# ------------------------------
# Save files
# ------------------------------
#if not os.path.exists('synthetic_data'):
#    os.makedirs('synthetic_data')
train_val_df.to_csv('train_val.csv', index=False)
test_df.to_csv('test.csv', index=False)
with open('train_boundaries.json', 'w') as f:
    json.dump(boundaries_dict, f, indent=2)

print(f"Generated train_val.csv and test.csv with {duration_years} years of data.")
print(f"train_boundaries.json reflects 6 train + 6 val regions.")