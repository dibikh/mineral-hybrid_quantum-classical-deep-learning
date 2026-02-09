# Data

Raw mineral price data are **not redistributed** in this repository due to licensing/terms of use.

## Expected input format

Place your raw file here (example):
- `data/raw/lithium.csv`

Your file can be **either**:

### Option A — 2-column series (recommended)
Columns (case-insensitive):
- `Date`
- `Close` (or `Price` / `Settlement`)

### Option B — OHLC export (also supported)
Columns (case-insensitive), e.g.:
- `Symbol, Date, Open, High, Low, Close`

Only `Date` and `Close` are used; the other columns are ignored.

**Delimiter / encoding notes**
- UTF-8, UTF-8-SIG, and UTF-16 are supported.
- Comma, semicolon, and tab-delimited exports are supported.
- If Excel shows everything in **one column** (e.g., `Symbol,Date,Open,High,Low,Close` in cell A1), that is a *display/import* issue in Excel; the Python preprocessing script can still parse the file.  
  (In Excel: Data → Text to Columns → Delimited → Comma.)

## Preprocessing

Run:
```bash
python data/preprocess.py --input data/raw/lithium.csv --out data/processed/lithium_processed.csv
```

This produces a cleaned, sorted, numeric time series (`Date,Close`) used by the experiments.


