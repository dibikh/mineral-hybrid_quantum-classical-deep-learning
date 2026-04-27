# src/train_bilstm.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import yaml

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.utils_seed import set_all_seeds
from src.data_utils import (
    load_market_csv, add_engineered_features,
    fit_scalers, create_multivariate_sequences, chronological_split,
)
from src.models_bilstm_keras import build_bilstm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lithium.yaml")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_all_seeds(args.seed)

    # ── data ──────────────────────────────────────────────────────
    df = load_market_csv(cfg["data"]["raw_csv"])

    if cfg["data"].get("use_engineered_features", True):
        df = add_engineered_features(df)
        feature_cols = cfg["data"].get("feature_cols", [
            "Open", "High", "Low", "Close",
            "HL_range", "OC_change", "log_return", "close_diff", "ma_5", "std_5"
        ])
    else:
        feature_cols = ["Open", "High", "Low", "Close"]

    target_col  = "Close"
    window      = cfg["data"]["window"]
    train_ratio = cfg["data"]["train_ratio"]
    val_ratio   = cfg["data"]["val_ratio"]

    N             = len(df)
    train_end_row = int(N * train_ratio)
    val_end_row   = int(N * (train_ratio + val_ratio))

    if train_end_row <= window:
        raise ValueError(f"Training portion too small for WINDOW={window}.")

    x_scaler, y_scaler, X_all_scaled, y_all_scaled = fit_scalers(
        df, train_end_row, feature_cols, target_col
    )

    X, y, y_dates, target_indices = create_multivariate_sequences(
        X_all_scaled, y_all_scaled, df["Date"], window
    )

    (X_train, y_train, _,
     X_val,   y_val,   _,
     X_test,  y_test,  _) = chronological_split(
        X, y, y_dates, target_indices, train_end_row, val_end_row
    )

    n_features = X.shape[2]
    print(f"[BiLSTM] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}  n_features={n_features}")

    # ── model ─────────────────────────────────────────────────────
    model = build_bilstm(
        window=window,
        n_features=n_features,
        units=cfg["bilstm"]["units"]
    )
    model.compile(optimizer=Adam(learning_rate=cfg["bilstm"]["lr"]), loss="mse")
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["bilstm"]["epochs"],
        batch_size=cfg["bilstm"]["batch_size"],
        verbose=1,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=cfg["bilstm"]["patience"],
            restore_best_weights=True
        )]
    )

    # ── save ──────────────────────────────────────────────────────
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    model_path = os.path.join(cfg["paths"]["models_dir"], "bilstm.keras")
    model.save(model_path)

    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    scaler_path = os.path.join(cfg["paths"]["artifacts_dir"], "scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({
            "x_min_":   x_scaler.min_.tolist(),
            "x_scale_": x_scaler.scale_.tolist(),
            "y_min_":   y_scaler.min_.tolist(),
            "y_scale_": y_scaler.scale_.tolist(),
            "feature_cols": feature_cols,
            "n_features":   n_features,
            "window":       window,
        }, f, indent=2)

    print(f"[OK] BiLSTM model  → {model_path}")
    print(f"[OK] Scaler params → {scaler_path}")


if __name__ == "__main__":
    main()
