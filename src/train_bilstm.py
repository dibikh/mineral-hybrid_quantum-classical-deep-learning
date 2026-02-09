import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, os, json
import numpy as np
import pandas as pd
import yaml

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.utils_seed import set_all_seeds
from src.data_utils import load_series, scale_close, make_sequences, train_test_split_seq
from src.models_bilstm_keras import build_bilstm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lithium.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_all_seeds(args.seed)

    df = load_series(cfg["data"]["processed_csv"])
    scaler, scaled = scale_close(df)
    X, y = make_sequences(scaled, cfg["data"]["window"])
    Xtr, Xte, ytr, yte, split = train_test_split_seq(X, y, cfg["data"]["train_frac"])

    model = build_bilstm(cfg["data"]["window"], units=cfg["bilstm"]["units"])
    model.compile(optimizer=Adam(learning_rate=cfg["bilstm"]["lr"]), loss="mse")

    model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=cfg["bilstm"]["epochs"],
        batch_size=cfg["bilstm"]["batch_size"],
        verbose=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=cfg["bilstm"]["patience"], restore_best_weights=True)]
    )

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    model_path = os.path.join(cfg["paths"]["models_dir"], "bilstm.keras")
    model.save(model_path)

    # Save scaler params for inverse transform
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    scaler_path = os.path.join(cfg["paths"]["artifacts_dir"], "scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({"min_": scaler.min_.tolist(), "scale_": scaler.scale_.tolist()}, f)

    print(f"[OK] Saved BiLSTM model to {model_path}")
    print(f"[OK] Saved scaler to {scaler_path}")

if __name__ == "__main__":
    main()