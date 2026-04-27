# src/inference.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import pandas as pd
import yaml
import torch

from tensorflow.keras.models import load_model

from src.utils_seed import set_all_seeds
from src.data_utils import (
    load_market_csv, add_engineered_features,
    fit_scalers, create_multivariate_sequences, chronological_split,
)
from src.models_qbilstm_torch import QBiLSTMNetwork
from src.models_hybrid_gated import ClassicalBiLSTM, GatedHybrid
from src.metrics import rmse, mae, mape, smape, r2


def _load_scaler_params(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d


def _inverse_y(y_scaled, params):
    """Inverse-transform scaled Close predictions back to price space."""
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y_min   = np.array(params["y_min_"]).reshape(1, -1)
    y_scale = np.array(params["y_scale_"]).reshape(1, -1)
    # sklearn MinMaxScaler: X_orig = X_scaled / scale_ + min_   (note: min_ already includes data_min)
    # Actually inverse: X = X_scaled * (data_max - data_min) + data_min  =>  (X_scaled - min_) / scale_
    # But sklearn stores differently; safest to reconstruct scaler and call inverse_transform.
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    sc.min_   = y_min.reshape(-1)
    sc.scale_ = y_scale.reshape(-1)
    sc.data_min_  = (0 - sc.min_) / sc.scale_
    sc.data_max_  = (1 - sc.min_) / sc.scale_
    sc.data_range_ = sc.data_max_ - sc.data_min_
    sc.feature_range = (0, 1)
    sc.n_features_in_ = 1
    sc.n_samples_seen_ = 1
    return sc.inverse_transform(y_scaled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lithium.yaml")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_all_seeds(args.seed)

    # ── data (same pipeline as training) ──────────────────────────
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

    (_, _, _,
     _, _, _,
     X_test, y_test, dates_test) = chronological_split(
        X, y, y_dates, target_indices, train_end_row, val_end_row
    )

    n_features = X.shape[2]

    # ── load saved scaler params ───────────────────────────────────
    params = _load_scaler_params(
        os.path.join(cfg["paths"]["artifacts_dir"], "scaler.json")
    )

    # ── BiLSTM (Keras) ─────────────────────────────────────────────
    bilstm = load_model(os.path.join(cfg["paths"]["models_dir"], "bilstm.keras"))
    bilstm_pred_s = bilstm.predict(X_test, verbose=0).reshape(-1, 1)

    # ── QBiLSTM (PyTorch) ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qbilstm = QBiLSTMNetwork(
        input_dim=n_features,
        hidden_dim=cfg["qbilstm"]["hidden_dim"],
        time_steps=window,
        num_qubits=cfg["qbilstm"]["num_qubits"],
        quantum_depth=cfg["qbilstm"]["quantum_depth"],
        out_dim=1
    ).to(device)
    qbilstm.load_state_dict(
        torch.load(os.path.join(cfg["paths"]["models_dir"], "qbilstm.pt"), map_location=device)
    )
    qbilstm.eval()
    with torch.no_grad():
        q_pred_s = qbilstm(
            torch.tensor(X_test, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # ── Hybrid ─────────────────────────────────────────────────────
    classical = ClassicalBiLSTM(
        input_dim=n_features,
        hidden_dim=cfg["hybrid"]["classical_hidden_dim"],
        num_layers=cfg["hybrid"]["classical_num_layers"]
    ).to(device)
    quantum = QBiLSTMNetwork(
        input_dim=n_features,
        hidden_dim=cfg["hybrid"]["quantum_hidden_dim"],
        time_steps=window,
        num_qubits=cfg["hybrid"]["num_qubits"],
        quantum_depth=cfg["hybrid"]["quantum_depth"],
        out_dim=1
    ).to(device)
    hybrid = GatedHybrid(classical, quantum, gate_hidden=cfg["hybrid"]["gate_hidden"]).to(device)
    hybrid.load_state_dict(
        torch.load(os.path.join(cfg["paths"]["models_dir"], "hybrid_gated.pt"), map_location=device)
    )
    hybrid.eval()
    Xte_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        h_pred_s, alpha_c, alpha_q = hybrid(Xte_t, Xte_t, return_alpha=True)
        h_pred_s = h_pred_s.cpu().numpy()
        alpha_c  = alpha_c.cpu().numpy().reshape(-1)
        alpha_q  = alpha_q.cpu().numpy().reshape(-1)

    # ── inverse transform → price space ───────────────────────────
    y_true_price   = y_scaler.inverse_transform(y_test).reshape(-1)
    bilstm_price   = y_scaler.inverse_transform(bilstm_pred_s).reshape(-1)
    q_price        = y_scaler.inverse_transform(q_pred_s).reshape(-1)
    h_price        = y_scaler.inverse_transform(h_pred_s).reshape(-1)

    # ── save predictions ──────────────────────────────────────────
    os.makedirs(cfg["paths"]["outputs_dir"], exist_ok=True)
    out_pred = os.path.join(cfg["paths"]["outputs_dir"], "predictions_test.csv")
    pd.DataFrame({
        "Date":                pd.to_datetime(dates_test[:len(y_true_price)]),
        "TruePrice":           y_true_price,
        "BiLSTM_Pred":         bilstm_price[:len(y_true_price)],
        "QBiLSTM_Pred":        q_price[:len(y_true_price)],
        "Hybrid_Pred":         h_price[:len(y_true_price)],
        "TrueScaled":          y_test.reshape(-1)[:len(y_true_price)],
        "BiLSTM_Pred_Scaled":  bilstm_pred_s.reshape(-1)[:len(y_true_price)],
        "QBiLSTM_Pred_Scaled": q_pred_s.reshape(-1)[:len(y_true_price)],
        "Hybrid_Pred_Scaled":  h_pred_s.reshape(-1)[:len(y_true_price)],
        "alpha_c":             alpha_c[:len(y_true_price)],
        "alpha_q":             alpha_q[:len(y_true_price)],
    }).to_csv(out_pred, index=False)
    print(f"[OK] Wrote {out_pred}")

    # ── metrics (Table 1) ─────────────────────────────────────────
    out_metrics = os.path.join(cfg["paths"]["outputs_dir"], "table1_metrics.csv")
    rows = []
    for name, pred in [
        ("BiLSTM", bilstm_price),
        ("QBiLSTM", q_price),
        ("Hybrid",  h_price),
    ]:
        rows.append({
            "Model":   name,
            "RMSE":    rmse(y_true_price, pred),
            "MAE":     mae(y_true_price, pred),
            "MAPE%":   mape(y_true_price, pred),
            "SMAPE%":  smape(y_true_price, pred),
            "R2":      r2(y_true_price, pred),
        })
    pd.DataFrame(rows).to_csv(out_metrics, index=False)
    print(f"[OK] Wrote {out_metrics}")


if __name__ == "__main__":
    main()
