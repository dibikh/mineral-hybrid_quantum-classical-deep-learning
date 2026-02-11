# Regime-Aware Mineral Price Prediction (Hybrid Quantum–Classical Deep Learning)

This repository provides **data preprocessing**, **training**, **inference**, and **reproducibility scripts** for:
- **BiLSTM** (classical baseline)
- **QBiLSTM** (quantum-gated recurrent model)
- **Hybrid** (gated fusion of BiLSTM + QBiLSTM)



## Quickstart

### 1) Create the environment
**Conda 
```bash
conda env create -f environment.yml
conda activate mineral-hybrid
```

### 2) Data
Raw price data are **not redistributed** (see `data/README.md`).
Put your CSV in:
```
data/raw/lithium.csv
```
Expected columns (case-insensitive):
- Option A: `Date`, `Close` (or `Price` / `Settlement`)
- Option B: an OHLC export such as `Symbol, Date, Open, High, Low, Close` (only `Date` and `Close` are used)

UTF-8/UTF-16 are supported. Comma/semicolon/tab-delimited exports are supported.

Preprocess:
```bash
python data/preprocess.py --input data/raw/lithium.csv --out data/processed/lithium_processed.csv
```

### 3) Run the full pipeline (train → infer → table → figures)
```bash
bash scripts/run_all.sh
```

Outputs are written to `outputs/`:
- `outputs/predictions_test.csv`
- `outputs/table1_metrics.csv` (+ `outputs/table1_metrics.tex`)
- `outputs/figures/*.png` 

## Reproducibility
All scripts accept `--seed` (default: 42) and set seeds for Python/NumPy/TensorFlow/PyTorch.

## Data & Code Availability statement (paste into the manuscript)
See `DATA_CODE_AVAILABILITY.md`.



