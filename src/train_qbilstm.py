import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, os, json, time
import numpy as np
import pandas as pd
import yaml
import torch
from torch import nn

from src.utils_seed import set_all_seeds
from src.data_utils import load_series, scale_close, make_sequences, train_test_split_seq
from src.models_qbilstm_torch import QBiLSTMNetwork

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(Xtr, dtype=torch.float32).to(device)
    y_train = torch.tensor(ytr, dtype=torch.float32).to(device)
    x_test  = torch.tensor(Xte, dtype=torch.float32).to(device)

    model = QBiLSTMNetwork(
        input_dim=1,
        hidden_dim=cfg["qbilstm"]["hidden_dim"],
        time_steps=cfg["data"]["window"],
        num_qubits=cfg["qbilstm"]["num_qubits"],
        quantum_depth=cfg["qbilstm"]["quantum_depth"],
        out_dim=1
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["qbilstm"]["lr"])
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(x_train, y_train)
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
        print(f"[QBiLSTM] Epoch {ep+1}/{cfg['qbilstm']['epochs']} loss={ep_loss/len(dl):.6f}")
    t1 = time.time()

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    model_path = os.path.join(cfg["paths"]["models_dir"], "qbilstm.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler params (same format as in train_bilstm.py)
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    scaler_path = os.path.join(cfg["paths"]["artifacts_dir"], "scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({"min_": scaler.min_.tolist(), "scale_": scaler.scale_.tolist()}, f)

    print(f"[OK] Saved QBiLSTM model to {model_path}")
    print(f"[OK] Training time: {t1-t0:.2f} s")

if __name__ == "__main__":
    main()