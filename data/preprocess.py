import argparse
import os
import pandas as pd

def _read_csv_flexible(path: str) -> pd.DataFrame:
    """Read a CSV exported from various sources (2-col Date/Close or OHLC) robustly.

    Handles:
    - UTF-8 / UTF-16 (common for some exports)
    - comma, semicolon, tab delimiters
    - 'all-in-one-column' case where the header is 'Symbol,Date,Open,...'
    """
    encodings = ["utf-8", "utf-16", "utf-8-sig", "latin1"]
    seps = [",", ";", "\t", "|"]

    last_err = None
    for enc in encodings:
        # 1) Try standard delimiters
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                # If it parsed into multiple cols, accept
                if df.shape[1] > 1:
                    return df
                # If a single-column but header contains delimiters, we'll handle below
                return df
            except Exception as e:
                last_err = e

        # 2) Let pandas infer delimiter (python engine)
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to read CSV: {path}. Last error: {last_err}")

def _split_single_column(df: pd.DataFrame) -> pd.DataFrame:
    """If the CSV landed in one column, attempt to split by delimiter."""
    if df.shape[1] != 1:
        return df

    col0 = str(df.columns[0])
    # Heuristic: header string contains common delimiters
    delim = None
    for d in [",", ";", "\t", "|"]:
        if d in col0:
            delim = d
            break

    if delim is None:
        # Might be data without proper header; try splitting rows using comma
        sample = str(df.iloc[0, 0]) if len(df) else ""
        for d in [",", ";", "\t", "|"]:
            if d in sample:
                delim = d
                break

    if delim is None:
        return df

    # Split header into new columns
    new_cols = [c.strip() for c in col0.split(delim)]
    # Split each row
    parts = df.iloc[:, 0].astype(str).str.split(delim, expand=True)
    # If the header row got duplicated as first data row, drop it
    if len(parts) and all(str(parts.iloc[0, i]).strip() == new_cols[i] for i in range(min(len(new_cols), parts.shape[1]))):
        parts = parts.iloc[1:].reset_index(drop=True)

    # Align number of columns
    if parts.shape[1] >= len(new_cols):
        parts = parts.iloc[:, :len(new_cols)]
    else:
        # pad missing
        for _ in range(len(new_cols) - parts.shape[1]):
            parts[parts.shape[1]] = None

    parts.columns = new_cols
    return parts

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_column(df: pd.DataFrame, candidates) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # partial match (e.g., "Close " or "Close (USD)")
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None

def preprocess(input_path: str, output_path: str):
    df = _read_csv_flexible(input_path)
    df = _normalize_columns(df)
    df = _split_single_column(df)
    df = _normalize_columns(df)

    # Accept either a 2-col (Date/Close) or OHLC-style (Symbol,Date,Open,High,Low,Close)
    date_col = _find_column(df, ["Date", "date", "Time", "time", "Timestamp", "timestamp"])
    close_col = _find_column(df, ["Close", "close", "Price", "price", "Settlement", "settlement", "Settle", "settle"])

    if date_col is None or close_col is None:
        cols = ", ".join(df.columns)
        raise ValueError(
            "Could not find required columns. "
            f"Need a date-like column and a close/price column. Found columns: {cols}"
        )

    out = df[[date_col, close_col]].copy()
    out.columns = ["Date", "Close"]

    # Parse and clean
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

def main():
    ap = argparse.ArgumentParser(description="Preprocess mineral price CSV to Date,Close format.")
    ap.add_argument("--input", required=True, help="Raw CSV path (Date+Close or OHLC export).")
    ap.add_argument("--out", required=True, help="Output processed CSV (Date,Close).")
    args = ap.parse_args()
    preprocess(args.input, args.out)
    print(f"[OK] Wrote processed CSV to: {args.out}")

if __name__ == "__main__":
    main()
