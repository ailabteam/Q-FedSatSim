:q
import pennylane as qml
from pennylane import numpy as np
import torch

# Device với 3 qubit
dev = qml.device("default.qubit", wires=3)

# Step 1: Alice thực hiện phép đo xác suất qubit 0,1
@qml.qnode(dev, interface="torch")
def alice_measurement(alpha, beta):
    qml.StatePreparation(np.array([alpha.item(), beta.item()]), wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    return qml.probs(wires=[0, 1])

# Step 2: Bob điều chỉnh lại qubit 2 theo kết quả đo m0, m1
@qml.qnode(dev, interface="torch")
def bob_reconstruct(alpha, beta, m0, m1):
    qml.StatePreparation(np.array([alpha.item(), beta.item()]), wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

    if m1 == 1:
        qml.PauliX(wires=2)
    if m0 == 1:
        qml.PauliZ(wires=2)
    
    return qml.state()

# ==== Chạy ví dụ ====
alpha = torch.tensor(0.6, dtype=torch.float64)
beta = torch.tensor(0.8, dtype=torch.float64)

# Alice đo qubit 0 và 1
probs = alice_measurement(alpha, beta)
probs_np = probs.detach().numpy()
probs_np /= probs_np.sum()

outcomes = [(0,0), (0,1), (1,0), (1,1)]
measured_index = np.random.choice(4, p=probs_np)
m0, m1 = outcomes[measured_index]

print(f"Xác suất đo (normalized): {probs_np}")
print(f"[Alice đo] → Kết quả đo: m0={m0}, m1={m1}")

# Bob tái tạo trạng thái
full_state = bob_reconstruct(alpha, beta, m0, m1).detach().numpy()

# Lấy trạng thái riêng của qubit 2
reduced_state = qml.math.reduce_state(full_state, wires=[2], wire_order=[0,1,2])

# Chuyển sang torch tensor
teleported_state = torch.tensor(reduced_state, dtype=torch.complex128)

# Trạng thái gốc (target)
target_state = torch.tensor([alpha, beta], dtype=torch.complex128)

print(f"\n[Bob nhận] → Trạng thái qubit 2 (teleported): {teleported_state}")
print(f"[Mục tiêu] → Trạng thái ban đầu:              {target_state}")

# Tính fidelity: |⟨ψ_target|ψ_teleported⟩|^2
fidelity = torch.abs(torch.dot(torch.conj(target_state), teleported_state))**2
print(f"\n🎯 Fidelity giữa trạng thái Bob và ban đầu: {fidelity.item():.6f}")

