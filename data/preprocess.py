# data/preprocess.py
"""
preprocess.py
-------------
Reads a raw OHLC mineral-price CSV (Symbol,Date,Open,High,Low,Close or any
variant) and writes a clean Date,Open,High,Low,Close CSV used by all training
scripts.

Usage
-----
    python data/preprocess.py --input data/raw/lithium.csv \
                               --out   data/processed/lithium_processed.csv
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.data_utils import load_market_csv


def preprocess(input_path: str, output_path: str) -> None:
    df = load_market_csv(input_path)

    # Keep only OHLC + Date; drop Symbol if present
    keep = ["Date", "Open", "High", "Low", "Close"]
    df = df[[c for c in keep if c in df.columns]].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Preprocessed {len(df)} rows  →  {output_path}")
    print(f"     Date range: {df['Date'].min().date()} … {df['Date'].max().date()}")
    print(f"     Columns   : {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess raw mineral OHLC CSV to a clean Date,Open,High,Low,Close file."
    )
    ap.add_argument("--input", required=True, help="Path to raw CSV (OHLC or Date/Close).")
    ap.add_argument("--out",   required=True, help="Output processed CSV path.")
    args = ap.parse_args()
    preprocess(args.input, args.out)


if __name__ == "__main__":
    main()
