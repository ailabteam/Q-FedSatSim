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
n_total_qubits = 3  # Tổng số qubit
n_trash_qubits = 2  # Số qubit sẽ bị loại bỏ (nén vào)
n_latent_qubits = n_total_qubits - n_trash_qubits # Số qubit lưu trữ thông tin nén

wires_total = list(range(n_total_qubits))
wires_trash = list(range(n_latent_qubits, n_total_qubits)) # Các qubit [1, 2]

# --- Thiết kế Mạch Lượng tử ---
# Sử dụng simulator "default.qubit.torch" để tích hợp tốt hơn và tính toán state vector
dev = qml.device("default.qubit.torch", wires=n_total_qubits)

def encoder_circuit(params):
    """Phần mã hóa của autoencoder."""
    # params là các trọng số có thể huấn luyện
    # Cấu trúc này có thể thay đổi, ở đây ta dùng một cấu trúc đơn giản
    qml.StronglyEntanglingLayers(params, wires=wires_total)

# Đây không phải là một qnode, mà là một hàm tạo mạch bình thường.
# Chúng ta sẽ xây dựng qnode hoàn chỉnh bên dưới.

# --- Hàm chuẩn bị dữ liệu đầu vào ---
def prepare_input_state():
    """Tạo một trạng thái 3-qubit ngẫu nhiên để nén."""
    # params ngẫu nhiên cho một mạch tạo trạng thái
    random_params = np.random.uniform(0, 2 * np.pi, (2, n_total_qubits, 3))
    qml.StronglyEntanglingLayers(random_params, wires=wires_total)

# --- Xây dựng QNode cho việc huấn luyện ---
@qml.qnode(dev, interface="torch")
def autoencoder_qnode(encoder_params):
    """
    QNode hoàn chỉnh: Chuẩn bị -> Mã hóa -> Giải mã.
    Nó sẽ trả về state vector cuối cùng để tính fidelity.
    """
    # 1. Chuẩn bị trạng thái đầu vào (cố định trong mỗi lần chạy qnode)
    prepare_input_state()
    
    # 2. Chạy bộ mã hóa
    encoder_circuit(encoder_params)
    
    # 3. Chạy bộ giải mã, là nghịch đảo của bộ mã hóa
    # qml.adjoint áp dụng phép toán liên hợp Hermite, tức là nghịch đảo cho cổng Unitary
    qml.adjoint(encoder_circuit)(encoder_params)
    
    # Trả về trạng thái của toàn bộ hệ thống
    return qml.state()

# --- QNode để kiểm tra việc nén ---
@qml.qnode(dev, interface="torch")
def compression_qnode(encoder_params):
    """
    QNode chỉ chạy Encoder để kiểm tra xác suất các qubit rác có phải là |0>
    """
    prepare_input_state()
    encoder_circuit(encoder_params)
    # Trả về xác suất của các qubit rác
    return qml.probs(wires=wires_trash)

# --- Vòng lặp Huấn luyện ---
# Khởi tạo các tham số cho Encoder
n_layers_encoder = 3
params = torch.tensor(0.1 * np.random.randn(n_layers_encoder, n_total_qubits, 3), requires_grad=True)
optimizer = optim.Adam([params], lr=0.1)

epochs = 100
cost_history = []

print("\nBắt đầu huấn luyện Quantum Autoencoder...")
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Trạng thái mục tiêu là trạng thái |000...>, state vector của nó là [1, 0, 0, ...]
    target_state = torch.zeros(2**n_total_qubits)
    target_state[0] = 1.0
    
    # Lấy state vector đầu ra từ autoencoder
    output_state = autoencoder_qnode(params)
    
    # Tính fidelity giữa đầu ra và trạng thái |000...>
    # Fidelity của state_a và state_b là |<state_a | state_b>|^2
    # Vì target_state là [1, 0, ...], dot product chính là phần tử đầu tiên của output_state
    fidelity = torch.abs(output_state[0])**2
    
    # Mục tiêu là tối thiểu hóa 1 - fidelity
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
# Lấy xác suất của các qubit rác sau khi nén
trash_probs = compression_qnode(params)
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
