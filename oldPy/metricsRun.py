#TIME TO ANALYZE DATA

import pandas as pd
import numpy as np
from utils.metrics import metric

file = '/home/oliver/Desktop/Time-LLM/valiResults/llamaTest3epoch6layersTimeLLM-llamaTest3epoch6layers'
# Read the CSV file into a pandas DataFrame for predictions
predictions_df = pd.read_csv(file+'/forecasts.csv')

# Convert DataFrame to a NumPy array and discard the first row
predictions_array = predictions_df.iloc[1:, 1:].to_numpy().astype(float)

# Read the CSV file into a pandas DataFrame for test set
test_df = pd.read_csv(file+'/trues.csv')

# Convert DataFrame to a NumPy array and discard the first row
test_array = test_df.iloc[1:, 1:].to_numpy().astype(float)

metrics = metric(predictions_array, test_array)
print(metrics)