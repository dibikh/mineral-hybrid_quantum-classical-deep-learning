import torch
from torch import nn

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.connectors import TorchConnector

class QLSTMCell(nn.Module):
    """Quantum-gated LSTM cell (as used for the paper figures).

    Encodes concat([x_t, h_{t-1}]) to rotation angles -> VQC -> per-qubit Z expectations.
    Those quantum features are mapped to i/f/o/g gates and used in a standard LSTM update.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_qubits: int = 3, quantum_depth: int = 1):
        super().__init__()
        assert num_qubits >= 2, "num_qubits must be >= 2"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits

        self.enc = nn.Linear(input_dim + hidden_dim, num_qubits)

        self.params_enc = ParameterVector('enc', num_qubits)
        self.params_var = ParameterVector('var', num_qubits * quantum_depth)

        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.rx(self.params_enc[i], i)
        w = 0
        for _ in range(quantum_depth):
            for i in range(num_qubits):
                qc.ry(self.params_var[w], i)
                w += 1
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)

        observables = []
        for i in range(num_qubits):
            pauli = "I"*i + "Z" + "I"*(num_qubits - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli, 1.0)]))

        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=self.params_enc,
            weight_params=self.params_var,
            estimator=estimator
        )
        init_w = 0.05 * torch.randn(qnn.num_weights)
        self.vqc = TorchConnector(qnn, initial_weights=init_w)

        self.W_i = nn.Linear(num_qubits, hidden_dim)
        self.W_f = nn.Linear(num_qubits, hidden_dim)
        self.W_o = nn.Linear(num_qubits, hidden_dim)
        self.W_g = nn.Linear(num_qubits, hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t: torch.Tensor, hx):
        h_prev, c_prev = hx
        z = torch.cat([x_t, h_prev], dim=1)
        angles = self.enc(z)
        q_t = self.vqc(angles)

        i_t = self.sigmoid(self.W_i(q_t))
        f_t = self.sigmoid(self.W_f(q_t))
        o_t = self.sigmoid(self.W_o(q_t))
        g_t = self.tanh(self.W_g(q_t))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * self.tanh(c_t)
        return h_t, c_t

class QLSTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, time_steps: int,
                 num_qubits: int = 3, quantum_depth: int = 1, out_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.cell = QLSTMCell(input_dim, hidden_dim, num_qubits, quantum_depth)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        batch = x.size(0)
        dev = x.device
        h_t = torch.zeros(batch, self.hidden_dim, device=dev)
        c_t = torch.zeros(batch, self.hidden_dim, device=dev)
        for t in range(self.time_steps):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, (h_t, c_t))
        return self.fc_out(h_t), h_t

class QBiLSTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, time_steps: int,
                 num_qubits: int = 3, quantum_depth: int = 1, out_dim: int = 1):
        super().__init__()
        self.forward_net  = QLSTMNetwork(input_dim, hidden_dim, time_steps, num_qubits, quantum_depth, out_dim=out_dim)
        self.backward_net = QLSTMNetwork(input_dim, hidden_dim, time_steps, num_qubits, quantum_depth, out_dim=out_dim)
        self.fc_fusion = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        _, h_f = self.forward_net(x)
        x_rev = torch.flip(x, dims=[1])
        _, h_b = self.backward_net(x_rev)
        h_cat = torch.cat([h_f, h_b], dim=1)
        return self.fc_fusion(h_cat)
