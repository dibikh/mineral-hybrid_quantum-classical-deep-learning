#!/usr/bin/env bash
set -e

# 1) Preprocess (expects data/raw/lithium.csv)
python data/preprocess.py --input data/raw/lithium.csv --out data/processed/lithium_processed.csv

# 2) Train models
python src/train_bilstm.py --config configs/lithium.yaml
python src/train_qbilstm.py --config configs/lithium.yaml
python src/train_hybrid.py --config configs/lithium.yaml

# 3) Inference + Table 1 metrics
python src/inference.py --config configs/lithium.yaml

# 4) Paper figures only (Figs. 2–11)
python scripts/make_figures.py --pred outputs/predictions_test.csv --raw_processed data/processed/lithium_processed.csv --window 175 --train_frac 0.7 --outdir outputs/figures

# 5) Table 1 (LaTeX)
python scripts/make_table1.py --out_csv outputs/table1_metrics.csv --out_tex outputs/table1_metrics.tex

echo "[DONE] Outputs are in ./outputs"
