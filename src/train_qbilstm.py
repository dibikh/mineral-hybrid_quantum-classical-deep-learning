# src/train_qbilstm.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
import yaml
import torch
from torch import nn

from src.utils_seed import set_all_seeds
from src.data_utils import (
    load_market_csv, add_engineered_features,
    fit_scalers, create_multivariate_sequences, chronological_split,
)
from src.models_qbilstm_torch import QBiLSTMNetwork


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
    print(f"[QBiLSTM] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}  n_features={n_features}")

    # ── model ─────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    model = QBiLSTMNetwork(
        input_dim=n_features,
        hidden_dim=cfg["qbilstm"]["hidden_dim"],
        time_steps=window,
        num_qubits=cfg["qbilstm"]["num_qubits"],
        quantum_depth=cfg["qbilstm"]["quantum_depth"],
        out_dim=1
    ).to(device)

    opt     = torch.optim.Adam(model.parameters(), lr=cfg["qbilstm"]["lr"])
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(x_train, y_train_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["qbilstm"]["batch_size"], shuffle=True)

    model.train()
    t0 = time.time()
    for ep in range(cfg["qbilstm"]["epochs"]):
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        print(f"[QBiLSTM] Epoch {ep+1}/{cfg['qbilstm']['epochs']}  loss={ep_loss/len(dl):.6f}")
    t1 = time.time()

    # ── save ──────────────────────────────────────────────────────
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    model_path = os.path.join(cfg["paths"]["models_dir"], "qbilstm.pt")
    torch.save(model.state_dict(), model_path)

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

    print(f"[OK] QBiLSTM model  → {model_path}")
    print(f"[OK] Training time  : {t1-t0:.2f} s")


if __name__ == "__main__":
    main()
