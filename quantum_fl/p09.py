import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os

print("Bắt đầu bài toán QAOA (Tối ưu hóa cuối cùng)...")
if not os.path.exists("figures"): os.makedirs("figures")

# --- Bài toán và Hamiltonian (giữ nguyên) ---
data_packets = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
n_qubits = len(data_packets)
dev = qml.device("default.qubit", wires=n_qubits)
cost_h = qml.Hamiltonian(data_packets.tolist(), [qml.PauliZ(i) for i in range(n_qubits)])
mixer_h = qml.Hamiltonian([1] * n_qubits, [qml.PauliX(i) for i in range(n_qubits)])

# --- Mạch QAOA (CẢI TIẾN) ---
p = 6 # Tăng p lên một chút nữa
@qml.qnode(dev, interface="torch")
def qaoa_circuit(params):
    gammas, betas = params[0], params[1]
    for i in range(n_qubits): qml.Hadamard(wires=i)
    for i in range(p):
        qml.ApproxTimeEvolution(cost_h, gammas[i], 1)
        qml.ApproxTimeEvolution(mixer_h, betas[i], 1)
    return qml.expval(cost_h)

# --- Vòng lặp Tối ưu (CẢI TIẾN) ---
params = torch.tensor(np.random.uniform(0, np.pi, (2, p)), requires_grad=True, dtype=torch.float32)
# Sử dụng Adagrad, một optimizer cẩn thận hơn
optimizer = optim.Adagrad([params], lr=0.1) 
epochs = 250 # Tăng thêm epochs
cost_history = []

print("\nBắt đầu vòng lặp QAOA với Adagrad...")
for epoch in range(epochs):
    optimizer.zero_grad()
    expval_h = qaoa_circuit(params)
    cost = expval_h**2
    cost.backward()
    optimizer.step()
    cost_history.append(cost.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Cost: {cost.item():.4f}")

print("\n--- Huấn luyện hoàn tất! ---")

# --- Diễn giải kết quả ---
@qml.qnode(dev)
def get_solution_distribution(params):
    gammas, betas = params[0], params[1]
    for i in range(n_qubits): qml.Hadamard(wires=i)
    for i in range(p):
        qml.ApproxTimeEvolution(cost_h, gammas[i], 1)
        qml.ApproxTimeEvolution(mixer_h, betas[i], 1)
    return qml.probs(wires=range(n_qubits))

solution_probs = get_solution_distribution(params.detach())
print(f"\nPhân phối xác suất của các giải pháp (top 5):")
top_indices = torch.topk(solution_probs, k=5).indices.numpy()
top_probs = torch.topk(solution_probs, k=5).values.detach().numpy()

for i in range(len(top_indices)):
    sol_idx, sol_prob = top_indices[i], top_probs[i]
    binary_str = format(sol_idx, f'0{n_qubits}b')
    print(f"  - Chuỗi bit: {binary_str}, Xác suất: {sol_prob:.3f}")

best_solution_index = top_indices[0]
best_solution_binary = format(best_solution_index, f'0{n_qubits}b')
print(f"\nGiải pháp tối ưu nhất được tìm thấy (chuỗi bit): {best_solution_binary}")

set_A, set_B = [], []; sum_A, sum_B = 0, 0
for i in range(n_qubits):
    if best_solution_binary[i] == '0':
        set_A.append(data_packets[i].item()); sum_A += data_packets[i].item()
    else:
        set_B.append(data_packets[i].item()); sum_B += data_packets[i].item()

print(f"\nPhân chia tối ưu:")
print(f"  - Vệ tinh A nhận các gói: {set_A} (Tổng: {sum_A:.1f})")
print(f"  - Vệ tinh B nhận các gói: {set_B} (Tổng: {sum_B:.1f})")
print(f"  - Chênh lệch bình phương thực tế: {(sum_A - sum_B)**2:.4f}")

plt.figure(figsize=(10, 6)); plt.plot(cost_history)
plt.title("QAOA for Satellite Load Balancing (Final Attempt)")
plt.xlabel("Optimization Step"); plt.ylabel("Cost")
plt.grid(True); figure_path = "figures/qaoa_load_balancing_final.png"
plt.savefig(figure_path); print(f"\nBiểu đồ hội tụ đã được lưu tại: {figure_path}"); plt.close()
