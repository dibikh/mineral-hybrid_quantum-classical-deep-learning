#!/usr/bin/env bash
# run_all.sh
# Reproduces all outputs: Table 1 + Figs. 2-11
# Usage: bash scripts/run_all.sh
set -e

# 1) Preprocess raw OHLC CSV  →  data/processed/lithium_processed.csv
python data/preprocess.py \
    --input data/raw/lithium.csv \
    --out   data/processed/lithium_processed.csv

# 2) Train models (multivariate: Open,High,Low,Close + 6 engineered features)
python src/train_bilstm.py   --config configs/lithium.yaml
python src/train_qbilstm.py  --config configs/lithium.yaml
python src/train_hybrid.py   --config configs/lithium.yaml

# 3) Inference + Table-1 metrics CSV
python src/inference.py --config configs/lithium.yaml

# 4) Paper figures (Figs. 2-11)
python scripts/make_figures.py \
    --pred           outputs/predictions_test.csv \
    --raw_processed  data/processed/lithium_processed.csv \
    --window         175 \
    --train_frac     0.60 \
    --outdir         outputs/figures

# 5) Table 1  (LaTeX)
python scripts/make_table1.py \
    --out_csv outputs/table1_metrics.csv \
    --out_tex outputs/table1_metrics.tex

echo "[DONE] All outputs written to ./outputs"
