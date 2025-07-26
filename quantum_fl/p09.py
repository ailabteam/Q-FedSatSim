import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os

print("Bắt đầu bài toán VQE cho Cân bằng tải Vệ tinh (Phân chia Tập hợp)...")

if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Bước 1: Định nghĩa Bài toán ---
data_packets = torch.tensor([10.0, 20.0, 30.0, 40.0])
n_qubits = len(data_packets)
dev = qml.device("default.qubit", wires=n_qubits)

# --- Bước 2: Xây dựng Cost Hamiltonian (đã sửa) ---
cost_coeffs = []
cost_obs = []

# Lặp qua tất cả các cặp (i, j) để xây dựng H^2
for i in range(n_qubits):
    for j in range(n_qubits):
        # Khi i == j, Z_i^2 = I. Thuật ngữ này là một hằng số và không ảnh hưởng
        # đến việc tìm trạng thái cơ bản, nhưng chúng ta có thể thêm vào để
        # giá trị cost có ý nghĩa vật lý hơn. Tuy nhiên, để đơn giản, ta có thể bỏ qua.
        # Ở đây ta sẽ thêm vào cho đầy đủ.
        if i == j:
            # s_i^2 * Z_i^2 = s_i^2 * I
            # PennyLane xử lý hằng số này trong Hamiltonian
            pass # Tạm thời bỏ qua các hằng số vì chúng chỉ dịch chuyển toàn bộ phổ năng lượng
        else:
            # s_i * s_j * Z_i * Z_j
            # Chỉ cần thêm một lần cho mỗi cặp (ví dụ i < j)
            if i < j:
                cost_coeffs.append(2 * data_packets[i] * data_packets[j])
                cost_obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

# Thêm các hằng số từ s_i^2 * I
# Tổng của các hằng số này là sum(s_i^2)
constant_term = torch.sum(data_packets**2)
# Chúng ta có thể thêm nó vào sau, hoặc để Hamiltonian chỉ chứa các phần tương tác.
# Việc tối ưu sẽ không bị ảnh hưởng.

cost_h = qml.Hamiltonian(cost_coeffs, cost_obs)
print("Cost Hamiltonian đã được tạo.")

# --- Bước 3: Thiết kế Mạch Ansatz (giữ nguyên) ---
@qml.qnode(dev, interface="torch")
def vqe_circuit(params):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(cost_h)

# --- Bước 4: Vòng lặp Tối ưu VQE ---
n_layers = 2
params = torch.tensor(np.pi * np.random.rand(n_layers, n_qubits, 3), requires_grad=True, dtype=torch.float32)
optimizer = optim.AdamW([params], lr=0.1)
epochs = 150
cost_history = []

print("\nBắt đầu vòng lặp VQE...")
for epoch in range(epochs):
    optimizer.zero_grad()
    # Cost được tính từ H^2 đã khai triển
    cost = vqe_circuit(params)
    cost.backward()
    optimizer.step()
    
    # Để hiển thị đúng giá trị (Sum_A - Sum_B)^2, ta cộng lại hằng số đã bỏ qua
    cost_with_constant = cost.item() + constant_term.item()
    cost_history.append(cost_with_constant)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Cost (chênh lệch bình phương): {cost_with_constant:.4f}")
        
print("\n--- Huấn luyện hoàn tất! ---")
final_cost = cost_history[-1]
print(f"Giá trị cost tối thiểu tìm được: {final_cost:.4f}")

# --- Bước 5: Diễn giải Kết quả (giữ nguyên) ---
@qml.qnode(dev)
def get_solution_distribution(params):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

solution_probs = get_solution_distribution(params.detach().numpy())
best_solution_index = np.argmax(solution_probs)
best_solution_binary = format(best_solution_index, f'0{n_qubits}b')

print(f"\nPhân phối xác suất của các giải pháp:\n{np.round(solution_probs, 3)}")
print(f"\nGiải pháp tối ưu nhất được tìm thấy (chuỗi bit): {best_solution_binary}")

set_A, set_B = [], []
sum_A, sum_B = 0, 0
for i in range(n_qubits):
    if best_solution_binary[i] == '0':
        set_A.append(data_packets[i].item())
        sum_A += data_packets[i].item()
    else:
        set_B.append(data_packets[i].item())
        sum_B += data_packets[i].item()

print(f"\nPhân chia tối ưu:")
print(f"  - Vệ tinh A nhận các gói: {set_A} (Tổng: {sum_A:.1f})")
print(f"  - Vệ tinh B nhận các gói: {set_B} (Tổng: {sum_B:.1f})")
print(f"  - Chênh lệch bình phương thực tế: {(sum_A - sum_B)**2:.4f}")

# --- Trực quan hóa ---
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.title("VQE for Satellite Load Balancing (Number Partitioning)")
plt.xlabel("Optimization Step (Epoch)")
plt.ylabel("Cost (Squared Difference)")
plt.grid(True)
figure_path = "figures/vqe_load_balancing.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ hội tụ đã được lưu tại: {figure_path}")
plt.close()
