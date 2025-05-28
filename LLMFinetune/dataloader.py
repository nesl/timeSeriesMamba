import csv
import json
import argparse
import pandas as pd

def read_csv_data(file_path, target, col_percent, percent):
    df_raw = pd.read_csv(file_path)
    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove('date')
    num_cols_to_keep = int(len(cols) * (col_percent / 100))
    cols = cols[:num_cols_to_keep]
    df_raw = df_raw[['date'] + cols + [target]]

    if percent < 100:
        num_rows_to_keep = int(len(df_raw) * (percent / 100))
        df_raw = df_raw.iloc[:num_rows_to_keep]
    
    return df_raw.drop(columns=['date']).round(3).values.tolist()

def window_to_string(window):
    return "\n".join([", ".join(map(str, row)) for row in window])

def create_dialogues(data, seq_len, pred_len):
    stride = seq_len+pred_len 
    dialogues = []
    for i in range(0, min(32 * stride, len(data) - seq_len - pred_len + 1), stride):
        user_prompt = data[i:i + seq_len]
        assistant_response = data[i + seq_len:i + seq_len + pred_len]
        dialogues.append({
            "user": window_to_string(user_prompt),
            "assistant": window_to_string(assistant_response)
        })
    return dialogues

def write_json(output_path, dialogues):
    with open(output_path, "w") as f:
        json.dump(dialogues, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", default="/home/nesl/oliver/timeSeriesMamba/TimeLLM/dataset/synth/combo1.csv", help="Path to input CSV file")
    parser.add_argument("json_path", default="test_output", help="Path to output JSON file")
    parser.add_argument("seq_len", default=512, type=int, help="Number of rows for user prompt")
    parser.add_argument("pred_len", default=96,type=int, help="Number of rows for assistant response")
    parser.add_argument("--target", default="OT", help="Target column name")
    parser.add_argument("--col_percent", default=100,type=int, help="Percentage of columns to keep")
    parser.add_argument("--percent", default=100, type=int, help="Percentage of rows to keep")
    parser.add_argument("--train_percent", default=100, type=int, help="Percentage of training rows to keep")
    parser.add_argument("--dsampfactor",default=1, type=int, help="downsampling factor")
    args = parser.parse_args()

    data = read_csv_data(args.csv_path, args.target, args.col_percent, args.percent)
    dialogues = create_dialogues(data, args.seq_len, args.pred_len)
    write_json(args.json_path, dialogues)

if __name__ == "__main__":
    main()
