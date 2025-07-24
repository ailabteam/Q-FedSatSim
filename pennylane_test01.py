import pennylane as qml
from pennylane import numpy as np
import torch

# Use lightning.qubit simulator (can be replaced with default.qubit if needed)
dev = qml.device("default.qubit", wires=3)

# Define the teleportation circuit
@qml.qnode(dev, interface="torch")
def teleportation(alpha, beta):
    # Chuẩn bị trạng thái đầu vào |ψ⟩ = α|0⟩ + β|1⟩ trên qubit 0
    qml.StatePrep([alpha, beta], wires=0)

    # Entangle qubit 1 và 2 (tạo cặp EPR)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])

    # Alice: teleportation gates
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

    # Đo qubit 0 và 1
    m0 = qml.measure(wires=0)
    m1 = qml.measure(wires=1)

    # Bob: điều chỉnh qubit 2 tùy theo kết quả đo
    qml.cond(m1, qml.PauliX)(wires=2)
    qml.cond(m0, qml.PauliZ)(wires=2)

    return qml.state()

# Example: state |ψ⟩ = 0.6|0⟩ + 0.8|1⟩
alpha = torch.tensor(0.6, dtype=torch.complex64)
beta = torch.tensor(0.8, dtype=torch.complex64)
psi_state = teleportation(alpha, beta)

# Extract the qubit 2 state from full system (qubit 2 is last 2 elements)
teleported_state = psi_state[-2:]
print("Teleported state on Bob's qubit:")
print(teleported_state)

# Compare with original
target_state = torch.tensor([alpha, beta])
print("\nOriginal state:")
print(target_state)

# Fidelity (dot product squared)
fidelity = torch.abs(torch.dot(torch.conj(teleported_state), target_state)) ** 2
print("\nFidelity:", fidelity.item())

