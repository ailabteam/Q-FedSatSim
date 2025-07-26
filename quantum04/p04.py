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
print("Bắt đầu Bài toán 4: Quantum Autoencoder (QAE)...")
if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Các tham số ---
n_total_qubits = 3
n_trash_qubits = 2
n_latent_qubits = n_total_qubits - n_trash_qubits

wires_total = list(range(n_total_qubits))
wires_trash = list(range(n_latent_qubits, n_total_qubits))

# --- Thiết kế Mạch Lượng tử ---
# SỬA LỖI Ở ĐÂY: Sử dụng device tiêu chuẩn
dev = qml.device("default.qubit", wires=n_total_qubits)

def encoder_circuit(params):
    qml.StronglyEntanglingLayers(params, wires=wires_total)

def prepare_input_state():
    # Sử dụng các tham số cố định để trạng thái đầu vào luôn giống nhau
    # Điều này giúp việc tối ưu hóa dễ dàng hơn
    fixed_params = np.array([
        [[4.526488, 5.583498, 5.43899], [2.63677, 1.33475, 4.74246], [4.17412, 1.41163, 2.11245]],
        [[5.13433, 0.29851, 0.44822], [5.12282, 4.25203, 3.29322], [4.13386, 4.41525, 3.49479]],
    ])
    qml.StronglyEntanglingLayers(fixed_params, wires=wires_total)

# --- Xây dựng QNode cho việc huấn luyện ---
@qml.qnode(dev, interface="torch")
def autoencoder_qnode(encoder_params):
    # Tạo một "scope" riêng cho prepare_input_state để PennyLane hiểu
    # rằng phần này không có tham số cần tính đạo hàm.
    with qml.tape.QuantumTape() as tape:
        prepare_input_state()
    qml.apply(tape.operations) # Áp dụng các cổng đã tạo

    encoder_circuit(encoder_params)
    qml.adjoint(encoder_circuit)(encoder_params)
    
    return qml.state()

# --- QNode để kiểm tra việc nén ---
@qml.qnode(dev, interface="torch")
def compression_qnode(encoder_params):
    with qml.tape.QuantumTape() as tape:
        prepare_input_state()
    qml.apply(tape.operations)

    encoder_circuit(encoder_params)
    return qml.probs(wires=wires_trash)

# --- Vòng lặp Huấn luyện ---
n_layers_encoder = 3
# Khởi tạo tham số với dtype của PyTorch
params = torch.tensor(0.1 * np.random.randn(n_layers_encoder, n_total_qubits, 3), requires_grad=True, dtype=torch.float64)
optimizer = optim.Adam([params], lr=0.1)

epochs = 100
cost_history = []

print("\nBắt đầu huấn luyện Quantum Autoencoder...")
start_time = time.time()

# --- Tính toán trạng thái đầu vào một lần duy nhất ---
@qml.qnode(dev)
def get_input_state_vector():
    prepare_input_state()
    return qml.state()

input_state_vector = torch.tensor(get_input_state_vector(), dtype=torch.cfloat)

@qml.qnode(dev, interface="torch")
def full_circuit_qnode(encoder_params):
    # Áp dụng các cổng đã tạo
    qml.QubitStateVector(input_state_vector, wires=wires_total)
    encoder_circuit(encoder_params)
    qml.adjoint(encoder_circuit)(encoder_params)
    return qml.state()


for epoch in range(epochs):
    optimizer.zero_grad()
    
    output_state = full_circuit_qnode(params)
    
    # Fidelity giữa output_state và input_state_vector
    fidelity = torch.abs(torch.vdot(output_state, input_state_vector))**2
    
    cost = 1 - fidelity
    cost_history.append(cost.item())
    
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Cost (1 - Fidelity): {cost.item():.6f}")

total_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất! Tổng thời gian: {total_time:.2f}s")


# --- Đánh giá kết quả ---
print("\nĐánh giá hiệu quả nén:")
# Cần một qnode riêng để đánh giá trên input_state_vector
@qml.qnode(dev, interface="torch")
def compression_eval_qnode(encoder_params):
    qml.QubitStateVector(input_state_vector, wires=wires_total)
    encoder_circuit(encoder_params)
    return qml.probs(wires=wires_trash)

trash_probs = compression_eval_qnode(params)
prob_trash_is_zero = trash_probs[0].item()
print(f"Xác suất các qubit rác ở trạng thái |00>: {prob_trash_is_zero:.6f}")

# --- Trực quan hóa ---
plt.figure(figsize=(10, 5))
plt.plot(cost_history)
plt.title("Training Cost (1 - Fidelity)")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)
figure_path = "figures/qae_cost.png"
plt.savefig(figure_path)
print(f"Biểu đồ cost đã được lưu tại: {figure_path}")
plt.close()
