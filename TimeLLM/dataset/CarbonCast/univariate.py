import pandas as pd
import os

# Load the full dataset
#df = pd.read_csv("heldout_CISO_combined_clean_years_no_region.csv")
df = pd.read_csv("CISO_clean.csv")
# Make output directory
os.makedirs("univariate", exist_ok=True)

# Assume the first column is time or index — adjust if needed
time_col = df.columns[0]
features = df.columns[1:]

for feature in features:
    univariate_df = df[[time_col, feature]]
    output_path = f"univariate/CISO_{feature}.csv"
    univariate_df.to_csv(output_path, index=False)
