import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, os, json
import numpy as np
import pandas as pd
import yaml
import torch

from tensorflow.keras.models import load_model

from src.utils_seed import set_all_seeds
from src.data_utils import load_series, scale_close, make_sequences, train_test_split_seq, test_dates
from src.models_qbilstm_torch import QBiLSTMNetwork
from src.models_hybrid_gated import ClassicalBiLSTM, GatedHybrid
from src.metrics import rmse, mae, mape, smape, r2

def _load_scaler_params(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    min_ = np.array(d["min_"]).reshape(1, -1)
    scale_ = np.array(d["scale_"]).reshape(1, -1)
    return min_, scale_

def _inverse_minmax(y_scaled, min_, scale_):
    # sklearn MinMaxScaler inverse: X = (X_scaled - min_) / scale_
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return (y_scaled - min_) / scale_

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
    dates_te = test_dates(df, cfg["data"]["window"], split)

    # load scaler params (use the ones saved during training for strict reproducibility)
    min_, scale_ = _load_scaler_params(os.path.join(cfg["paths"]["artifacts_dir"], "scaler.json"))

    # BiLSTM (Keras)
    bilstm = load_model(os.path.join(cfg["paths"]["models_dir"], "bilstm.keras"))
    bilstm_pred_s = bilstm.predict(Xte, verbose=0).reshape(-1, 1)

    # QBiLSTM (Torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qbilstm = QBiLSTMNetwork(
        input_dim=1,
        hidden_dim=cfg["qbilstm"]["hidden_dim"],
        time_steps=cfg["data"]["window"],
        num_qubits=cfg["qbilstm"]["num_qubits"],
        quantum_depth=cfg["qbilstm"]["quantum_depth"],
        out_dim=1
    ).to(device)
    qbilstm.load_state_dict(torch.load(os.path.join(cfg["paths"]["models_dir"], "qbilstm.pt"), map_location=device))
    qbilstm.eval()
    with torch.no_grad():
        q_pred_s = qbilstm(torch.tensor(Xte, dtype=torch.float32).to(device)).cpu().numpy()

    # Hybrid
    classical = ClassicalBiLSTM(
        input_dim=1,
        hidden_dim=cfg["hybrid"]["classical_hidden_dim"],
        num_layers=cfg["hybrid"]["classical_num_layers"]
    ).to(device)
    quantum = QBiLSTMNetwork(
        input_dim=1,
        hidden_dim=cfg["hybrid"]["quantum_hidden_dim"],
        time_steps=cfg["data"]["window"],
        num_qubits=cfg["hybrid"]["num_qubits"],
        quantum_depth=cfg["hybrid"]["quantum_depth"],
        out_dim=1
    ).to(device)
    hybrid = GatedHybrid(classical, quantum, gate_hidden=cfg["hybrid"]["gate_hidden"]).to(device)
    hybrid.load_state_dict(torch.load(os.path.join(cfg["paths"]["models_dir"], "hybrid_gated.pt"), map_location=device))
    hybrid.eval()
    with torch.no_grad():
        h_pred_s, alpha_c, alpha_q = hybrid(
            torch.tensor(Xte, dtype=torch.float32).to(device),
            torch.tensor(Xte, dtype=torch.float32).to(device),
            return_alpha=True
        )
        h_pred_s = h_pred_s.cpu().numpy()
        alpha_c = alpha_c.cpu().numpy().reshape(-1)
        alpha_q = alpha_q.cpu().numpy().reshape(-1)

    # Inverse to price space
    y_true_price = _inverse_minmax(yte, min_, scale_).reshape(-1)
    bilstm_price = _inverse_minmax(bilstm_pred_s, min_, scale_).reshape(-1)
    q_price      = _inverse_minmax(q_pred_s,      min_, scale_).reshape(-1)
    h_price      = _inverse_minmax(h_pred_s,      min_, scale_).reshape(-1)

    # Save predictions
    os.makedirs(cfg["paths"]["outputs_dir"], exist_ok=True)
    out_pred = os.path.join(cfg["paths"]["outputs_dir"], "predictions_test.csv")
    pd.DataFrame({
        "Date": pd.to_datetime(dates_te[:len(y_true_price)]),
        # Price space (for Figs. 2–5 and error plots)
        "TruePrice": y_true_price,
        "BiLSTM_Pred": bilstm_price[:len(y_true_price)],
        "QBiLSTM_Pred": q_price[:len(y_true_price)],
        "Hybrid_Pred": h_price[:len(y_true_price)],
        # Scaled space (for parity + residual diagnostics figures)
        "TrueScaled": yte.reshape(-1)[:len(y_true_price)],
        "BiLSTM_Pred_Scaled": bilstm_pred_s.reshape(-1)[:len(y_true_price)],
        "QBiLSTM_Pred_Scaled": q_pred_s.reshape(-1)[:len(y_true_price)],
        "Hybrid_Pred_Scaled": h_pred_s.reshape(-1)[:len(y_true_price)],
        # Hybrid gate weights
        "alpha_c": alpha_c[:len(y_true_price)],
        "alpha_q": alpha_q[:len(y_true_price)],
    }).to_csv(out_pred, index=False)
    print(f"[OK] Wrote {out_pred}")

    # Save metrics used in Table 1
    out_metrics = os.path.join(cfg["paths"]["outputs_dir"], "table1_metrics.csv")
    rows = []
    for name, pred in [
        ("BiLSTM", bilstm_price),
        ("QBiLSTM", q_price),
        ("Hybrid", h_price),
    ]:
        rows.append({
            "Model": name,
            "RMSE": rmse(y_true_price, pred),
            "MAE": mae(y_true_price, pred),
            "MAPE%": mape(y_true_price, pred),
            "SMAPE%": smape(y_true_price, pred),
            "R2": r2(y_true_price, pred),
        })
    pd.DataFrame(rows).to_csv(out_metrics, index=False)
    print(f"[OK] Wrote {out_metrics}")

if __name__ == "__main__":
    main()