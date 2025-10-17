import os
import json
import pandas as pd

SERIES_DIR = "series"
RESULTS_DIR = "git_repo/results"
GIT_REPO = "git_repo"

def load_series(dataset_name):
    # Depending on format (e.g. JSON, Parquet, Arrow), adapt
    path = os.path.join(SERIES_DIR, f"{dataset_name}.json")
    return pd.read_json(path)  # or pd.read_parquet, etc.

def find_result_csvs():
    # scan RESULTS_DIR for *.csv
    csvs = []
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return csvs

def load_results():
    """Load all evaluation results into a big DataFrame with metadata."""
    frames = []
    for fpath in find_result_csvs():
        df = pd.read_csv(fpath)
        # You’ll need to inspect column names: they might have
        # dataset, model, horizon, split, MSE, MAE, etc.
        # Also derive metadata (e.g. domain, frequency) from fpath or dataset name
        df["source_csv"] = fpath
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def normalize_results(df):
    # Example normalizations:
    # rename columns to canonical names, e.g. ‘dataset’, ‘model’, ‘horizon’, ‘split’, 'mse', 'mae'
    mapping = {
       "MSE": "mse",
       "MAE": "mae",
       # etc
    }
    df = df.rename(columns=mapping)
    # filter / drop incomplete rows
    df = df.dropna(subset=["mse", "mae"])
    return df
