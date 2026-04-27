import torch
from torch import nn

class ClassicalBiLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        return self.fc(out)  # (B,1)

class GatedHybrid(nn.Module):
    """Gated fusion y_hat = alpha_c*y_c + alpha_q*y_q, with alpha = softmax(gate([y_c,y_q])).

    This provides the per-sample expert weights used in Figs. 1 & diagnostics (if reported).
    """
    def __init__(self, classical_branch: nn.Module, quantum_branch: nn.Module, gate_hidden=32):
        super().__init__()
        self.classical_branch = classical_branch
        self.quantum_branch = quantum_branch
        self.gate_net = nn.Sequential(
            nn.Linear(2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 2)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_c, x_q, return_alpha: bool = False):
        y_c = self.classical_branch(x_c)  # (B,1)
        y_q = self.quantum_branch(x_q)    # (B,1)
        gate_in = torch.cat([y_c, y_q], dim=1)  # (B,2)
        w = self.softmax(self.gate_net(gate_in))  # (B,2)
        alpha_c = w[:, 0:1]
        alpha_q = w[:, 1:2]
        y_hat = alpha_c * y_c + alpha_q * y_q
        if return_alpha:
            return y_hat, alpha_c, alpha_q
        return y_hat
