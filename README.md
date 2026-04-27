# Regime-Aware Mineral Price Prediction  
## Hybrid Quantum–Classical Deep Learning (v1.1)

This repository provides **data preprocessing**, **training**, **inference**, and **reproducibility scripts** for:

| Model | Framework | Input |
|-------|-----------|-------|
| **BiLSTM** | TensorFlow / Keras | Multivariate |
| **QBiLSTM** | PyTorch + Qiskit | Multivariate |
| **Hybrid** (gated BiLSTM + QBiLSTM) | PyTorch + Qiskit | Multivariate |

Scripts reproduce **Table 1** and **Figs. 2–11** from the paper.

---

## What's new in v1.1

- **Multivariate inputs** — all three models now consume 10 features per time step:  
  `Open, High, Low, Close, HL_range, OC_change, log_return, close_diff, ma_5, std_5`
- **Chronological 3-way split** — 60 % train / 10 % val / 30 % test (previously 70 / 30)
- **Separate x / y scalers** — input features and target are scaled independently with `MinMaxScaler` fitted on the training set only
- **Robust OHLC loader** — handles UTF-8/UTF-16, comma/semicolon/tab delimiters, and single-column pasted CSVs
- **BiLSTM epochs** increased to 30 (with early stopping, patience = 3)

---

## Quickstart

### 1) Create the environment

```bash
conda env create -f environment.yml
conda activate mineral-hybrid
```

### 2) Data

Raw price data are **not redistributed** (see `data/README.md`).  
Place your OHLC export in:

```
data/raw/lithium.csv
```

Expected columns (case-insensitive, any order):

```
Symbol, Date, Open, High, Low, Close
```

UTF-8/UTF-16 and comma/semicolon/tab-delimited exports are all supported.

Preprocess:

```bash
python data/preprocess.py --input data/raw/lithium.csv \
                           --out   data/processed/lithium_processed.csv
```

### 3) Run the full pipeline

```bash
bash scripts/run_all.sh
```

Outputs are written to `outputs/`:

```
outputs/
  models/           bilstm.keras  qbilstm.pt  hybrid_gated.pt
  artifacts/        scaler.json
  predictions_test.csv
  table1_metrics.csv
  table1_metrics.tex
  figures/          Fig2_*.png … Fig11_*.png
```

---

## Feature engineering

| Feature | Formula |
|---------|---------|
| `HL_range` | `(High − Low) / (|Close| + ε)` |
| `OC_change` | `(Close − Open) / (|Open| + ε)` |
| `log_return` | `ln(Close_t / Close_{t−1})` |
| `close_diff` | `Close_t − Close_{t−1}` |
| `ma_5` | 5-day rolling mean of Close |
| `std_5` | 5-day rolling std of Close |

---

## Configuration

All hyper-parameters live in `configs/lithium.yaml`.  
Key settings:

```yaml
data:
  window:      175   # look-back steps
  train_ratio: 0.60
  val_ratio:   0.10
  use_engineered_features: true
```

---

## Reproducibility

All scripts accept `--seed` (default: 42) and fix seeds for Python / NumPy / TensorFlow / PyTorch.

---

## Citation

If you use this code, please cite it (see `CITATION.cff`).

## Data & Code Availability

See `DATA_CODE_AVAILABILITY.md`.
