#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quickly inspect the structure of a HuggingFace dataset on disk (.arrow format).

Usage:
  python inspect_arrow_dataset.py path/to/dataset

It will print:
  - Whether the path is a dataset or a dict of splits
  - The available split names (if any)
  - Number of rows per split
  - Example fields and types
  - Example row content (first item)
"""

from pathlib import Path
from datasets import load_from_disk

def inspect_dataset(ds_path: str):
    ds_path = Path(ds_path).expanduser().resolve()
    print(f"\n=== Inspecting {ds_path} ===")

    obj = load_from_disk(str(ds_path))

    if isinstance(obj, dict):
        print(f"Found dataset dict with splits: {list(obj.keys())}\n")
        for split_name, split in obj.items():
            print(f"--- Split: {split_name} ---")
            print(f"Num rows: {len(split)}")
            print(f"Columns: {split.column_names}")
            print("Example row:")
            print(split[0])
            print()
    else:
        print("Found single dataset (no explicit splits)")
        print(f"Num rows: {len(obj)}")
        print(f"Columns: {obj.column_names}")
        #print("Example row:")
        #print(obj[0])
        print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inspect_arrow_dataset.py path/to/dataset")
        sys.exit(1)
    inspect_dataset(sys.argv[1])
