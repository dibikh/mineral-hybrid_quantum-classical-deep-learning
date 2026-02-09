import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, os, time, json
import numpy as np
import yaml
import torch
from torch import nn

from src.utils_seed import set_all_seeds
from src.data_utils import load_series, scale_close, make_sequences, train_test_split_seq
from src.models_qbilstm_torch import QBiLSTMNetwork
from src.models_hybrid_gated import ClassicalBiLSTM, GatedHybrid

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
    xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
    xte = torch.tensor(Xte, dtype=torch.float32).to(device)
    ytr = torch.tensor(ytr, dtype=torch.float32).to(device)

    # classical branch
    classical = ClassicalBiLSTM(
        input_dim=1,
        hidden_dim=cfg["hybrid"]["classical_hidden_dim"],
        num_layers=cfg["hybrid"]["classical_num_layers"]
    ).to(device)

    # quantum branch
    quantum = QBiLSTMNetwork(
        input_dim=1,
        hidden_dim=cfg["hybrid"]["quantum_hidden_dim"],
        time_steps=cfg["data"]["window"],
        num_qubits=cfg["hybrid"]["num_qubits"],
        quantum_depth=cfg["hybrid"]["quantum_depth"],
        out_dim=1
    ).to(device)

    model = GatedHybrid(classical, quantum, gate_hidden=cfg["hybrid"]["gate_hidden"]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["hybrid"]["lr"])
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(xtr, xtr, ytr)  # x_classical, x_quantum, y
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["hybrid"]["batch_size"], shuffle=True)

    model.train()
    t0 = time.time()
    for ep in range(cfg["hybrid"]["epochs"]):
        ep_loss = 0.0
        for xb_c, xb_q, yb in dl:
            opt.zero_grad()
            pred = model(xb_c, xb_q)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        print(f"[Hybrid] Epoch {ep+1}/{cfg['hybrid']['epochs']} loss={ep_loss/len(dl):.6f}")
    t1 = time.time()

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    model_path = os.path.join(cfg["paths"]["models_dir"], "hybrid_gated.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler params
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    scaler_path = os.path.join(cfg["paths"]["artifacts_dir"], "scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({"min_": scaler.min_.tolist(), "scale_": scaler.scale_.tolist()}, f)

    print(f"[OK] Saved Hybrid model to {model_path}")
    print(f"[OK] Training time: {t1-t0:.2f} s")

if __name__ == "__main__":
    main()