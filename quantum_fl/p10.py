import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os

print("Bắt đầu bài toán QAOA (Objective Function mới)...")
if not os.path.exists("figures"): os.makedirs("figures")

# --- Bài toán và Hamiltonian ---
data_packets = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
n_qubits = len(data_packets)
dev = qml.device("default.qubit", wires=n_qubits)
cost_h = qml.Hamiltonian(data_packets.tolist(), [qml.PauliZ(i) for i in range(n_qubits)])
mixer_h = qml.Hamiltonian([1] * n_qubits, [qml.PauliX(i) for i in range(n_qubits)])

# --- Mạch QAOA ---
p = 4
@qml.qnode(dev, interface="torch")
def qaoa_circuit(params):
    gammas, betas = params[0], params[1]
    for i in range(n_qubits): qml.Hadamard(wires=i)
    for i in range(p):
        qml.ApproxTimeEvolution(cost_h, gammas[i], 1)
        qml.ApproxTimeEvolution(mixer_h, betas[i], 1)
    return qml.probs(wires=range(n_qubits))

# --- Hàm tính cost cổ điển ---
def precompute_classical_costs(data, n_qubits):
    costs = torch.zeros(2**n_qubits)
    for i in range(2**n_qubits):
        binary_str = format(i, f'0{n_qubits}b')
        sum_A, sum_B = 0, 0
        for k in range(n_qubits):
            if binary_str[k] == '0':
                sum_A += data[k]
            else:
                sum_B += data[k]
        costs[i] = (sum_A - sum_B)**2
    return costs
classical_costs = precompute_classical_costs(data_packets, n_qubits)
print("Đã tính toán trước cost cổ điển cho mọi giải pháp.")

# --- Vòng lặp Tối ưu ---
params = torch.tensor(np.random.uniform(0, np.pi, (2, p)), requires_grad=True, dtype=torch.float32)
# SỬA LỖI Ở ĐÂY: Đặt params vào trong một list
optimizer = optim.Adam([params], lr=0.05)
epochs = 150
cost_history = []

print("\nBắt đầu vòng lặp QAOA...")
for epoch in range(epochs):
    optimizer.zero_grad()
    probs = qaoa_circuit(params)
    cost = torch.sum(probs * classical_costs)
    cost.backward()
    optimizer.step()
    cost_history.append(cost.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Expected Cost: {cost.item():.4f}")

print("\n--- Huấn luyện hoàn tất! ---")

# --- Diễn giải kết quả ---
solution_probs = qaoa_circuit(params).detach()
print(f"\nPhân phối xác suất của các giải pháp (top 5):")
top_indices = torch.topk(solution_probs, k=5).indices.numpy()
top_probs = torch.topk(solution_probs, k=5).values.numpy()
for i in range(len(top_indices)):
    sol_idx, sol_prob = top_indices[i], top_probs[i]
    binary_str = format(sol_idx, f'0{n_qubits}b')
    print(f"  - Chuỗi bit: {binary_str}, Xác suất: {sol_prob:.3f}, Cost cổ điển: {classical_costs[sol_idx].item():.1f}")

best_solution_index = top_indices[0]
best_solution_binary = format(best_solution_index, f'0{n_qubits}b')
print(f"\nGiải pháp có xác suất cao nhất: {best_solution_binary}")

set_A, set_B = [], []; sum_A, sum_B = 0, 0
for i in range(n_qubits):
    if best_solution_binary[i] == '0':
        set_A.append(data_packets[i].item()); sum_A += data_packets[i].item()
    else:
        set_B.append(data_packets[i].item()); sum_B += data_packets[i].item()
print(f"\nPhân chia theo giải pháp tốt nhất:")
print(f"  - Vệ tinh A: {set_A} (Tổng: {sum_A:.1f})")
print(f"  - Vệ tinh B: {set_B} (Tổng: {sum_B:.1f})")

plt.figure(figsize=(10, 6)); plt.plot(cost_history)
plt.title("QAOA (New Objective Function)")
plt.xlabel("Optimization Step"); plt.ylabel("Expected Classical Cost")
plt.grid(True); figure_path = "figures/qaoa_new_objective.png"
plt.savefig(figure_path); print(f"\nBiểu đồ hội tụ đã được lưu tại: {figure_path}"); plt.close()
