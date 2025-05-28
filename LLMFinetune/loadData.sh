#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Usage check
: '
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <csv_path> <json_path> <seq_len> <pred_len> <target> <col_percent> <percent>"
    exit 1
fi
'
# Assign arguments to variables
CSV_PATH="/home/nesl/oliver/timeSeriesMamba/TimeLLM/dataset/synth/combo1.csv"
JSON_PATH="jsons/combo1.json"
SEQ_LEN=512
PRED_LEN=96
TARGET="OT"
COL_PERCENT=100
PERCENT=100

# Run the Python script
python dataloader.py "$CSV_PATH" "$JSON_PATH" "$SEQ_LEN" "$PRED_LEN" "$TARGET" "$COL_PERCENT" "$PERCENT"

echo "JSON file successfully created at $JSON_PATH"
