import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

# --- Thiết lập ---
torch.manual_seed(42)
np.random.seed(42)
print("Bắt đầu Bài toán 4: Quantum Autoencoder (QAE) với Hàm Loss Chính xác...")
if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Các tham số ---
n_total_qubits = 3
n_trash_qubits = 2
n_latent_qubits = n_total_qubits - n_trash_qubits
wires_total = list(range(n_total_qubits))
wires_trash = list(range(n_latent_qubits, n_total_qubits))

# --- Thiết kế Mạch Lượng tử ---
dev = qml.device("default.qubit", wires=n_total_qubits)

def encoder_circuit(params):
    qml.StronglyEntanglingLayers(params, wires=wires_total)

def prepare_input_state():
    fixed_params = np.array([
        [[4.526488, 5.583498, 5.43899], [2.63677, 1.33475, 4.74246], [4.17412, 1.41163, 2.11245]],
        [[5.13433, 0.29851, 0.44822], [5.12282, 4.25203, 3.29322], [4.13386, 4.41525, 3.49479]],
    ])
    qml.StronglyEntanglingLayers(fixed_params, wires=wires_total)

# --- Tính toán trạng thái đầu vào một lần duy nhất ---
@qml.qnode(dev)
def get_input_state_vector():
    prepare_input_state()
    return qml.state()
input_state_vector = torch.tensor(get_input_state_vector(), dtype=torch.cdouble)

# --- QNode mới cho việc huấn luyện, chỉ chạy Encoder ---
@qml.qnode(dev, interface="torch")
def encoding_qnode(encoder_params):
    qml.StatePrep(input_state_vector, wires=wires_total)
    encoder_circuit(encoder_params)
    # Trả về mật độ ma trận rút gọn của các qubit rác
    return qml.density_matrix(wires=wires_trash)

# --- Vòng lặp Huấn luyện với Hàm Loss mới ---
n_layers_encoder = 4 # Tăng độ sâu của encoder
params = torch.tensor(0.1 * np.random.randn(n_layers_encoder, n_total_qubits, 3), requires_grad=True, dtype=torch.float64)
optimizer = optim.Adam([params], lr=0.05) # Giảm learning rate một chút

epochs = 150 # Tăng epochs
cost_history = []

print("\nBắt đầu huấn luyện với hàm loss mới...")
start_time = time.time()

# Trạng thái mục tiêu cho các qubit rác: |0...0>
trash_target_state = torch.zeros(2**n_trash_qubits, dtype=torch.cdouble)
trash_target_state[0] = 1.0

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Lấy mật độ ma trận của các qubit rác sau khi mã hóa
    trash_density_matrix = encoding_qnode(params)
    
    # Tính fidelity giữa trạng thái của các qubit rác và trạng thái |0...0>
    # Fidelity(rho, |psi><psi|) = <psi| rho |psi>
    # Với |psi> = |0...0>, <psi| = [1, 0,...], |psi> = [1, 0,...]^T
    # Kết quả là phần tử trên cùng bên trái của mật độ ma trận, rho[0,0]
    fidelity_trash = torch.real(trash_density_matrix[0, 0])
    
    cost = 1 - fidelity_trash
    cost_history.append(cost.item())
    
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Cost (1 - Trash Fidelity): {cost.item():.6f}")

total_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất! Tổng thời gian: {total_time:.2f}s")


# --- Đánh giá kết quả ---
print("\nĐánh giá hiệu quả nén:")
# Lấy xác suất của các qubit rác
@qml.qnode(dev, interface="torch")
def compression_eval_qnode(encoder_params):
    qml.StatePrep(input_state_vector, wires=wires_total)
    encoder_circuit(encoder_params)
    return qml.probs(wires=wires_trash)

trash_probs = compression_eval_qnode(params.detach())
prob_trash_is_zero = trash_probs[0].item()
print(f"Xác suất các qubit rác ở trạng thái |00>: {prob_trash_is_zero:.6f}")

# --- Trực quan hóa ---
plt.figure(figsize=(10, 5))
plt.plot(cost_history)
plt.title("Training Cost (1 - Fidelity on Trash Qubits)")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)
figure_path = "figures/qae_cost_corrected.png"
plt.savefig(figure_path)
print(f"Biểu đồ cost đã được lưu tại: {figure_path}")
plt.close()
