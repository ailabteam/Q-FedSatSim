import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

print("Bắt đầu bài toán QNN cho phân loại XOR (Phiên bản cải tiến)...")

# Tạo thư mục để lưu ảnh nếu chưa có
if not os.path.exists("figures"):
    os.makedirs("figures")

# Bước 1: Chuẩn bị dữ liệu XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
Y = Y.view(-1, 1)

# Bước 2: Thiết kế Mạch Lượng tử (Không thay đổi)
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# CẢI TIẾN 1: Tăng năng lực mô hình bằng StronglyEntanglingLayers
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # Sử dụng tầng lượng tử mạnh hơn
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Bước 3: Xây dựng Mô hình Lượng tử với PyTorch
class QuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        # StronglyEntanglingLayers yêu cầu shape (n_layers, n_qubits, 3)
        n_layers = 3 # Tăng số lớp để mô hình mạnh hơn
        weight_shape = (n_layers, n_qubits, 3)
        q_params = torch.randn(weight_shape, dtype=torch.float32) * 0.01
        self.q_weights = nn.Parameter(q_params)

    def forward(self, x):
        # CẢI TIẾN 2: Xử lý toàn bộ batch cùng lúc, không cần vòng lặp for
        # Mạch lượng tử sẽ tự động chạy cho từng mẫu trong batch x
        q_out = quantum_circuit(x, self.q_weights)
        
        # Output của q_out là 1D tensor, cần reshape thành (batch_size, 1)
        # và co giãn về [0, 1]
        return (q_out.unsqueeze(1) + 1) / 2

# Bước 4: Huấn luyện Mô hình
model = QuantumNet()
# CẢI TIẾN 3: Giảm learning rate và tăng epochs
optimizer = optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

epochs = 150 # Tăng số epoch để có thời gian hội tụ
loss_history = []

print("\nBắt đầu huấn luyện...")
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    
    loss = loss_fn(predictions.float(), Y.float())
    loss_history.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Huấn luyện hoàn tất!")

# Bước 5: Đánh giá kết quả
print("\nĐánh giá mô hình sau khi huấn luyện:")
with torch.no_grad():
    predictions_final = model(X)
    predicted_labels = torch.round(predictions_final)

    correct_predictions = 0
    for i in range(len(X)):
        is_correct = "Đúng" if int(predicted_labels[i].item()) == int(Y[i].item()) else "Sai"
        if is_correct == "Đúng":
            correct_predictions += 1
        print(f"Input: {X[i].tolist()}, Target: {int(Y[i].item())}, Predicted: {int(predicted_labels[i].item())}, Probability: {predictions_final[i].item():.3f} -> {is_correct}")
    
    accuracy = correct_predictions / len(X) * 100
    print(f"\nĐộ chính xác: {accuracy:.2f}%")


# Bước 6: Trực quan hóa và lưu loss
figure_path = "figures/qnn_xor_loss_improved.png"
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Training Loss History (Improved Model)")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.grid(True)
plt.savefig(figure_path)
print(f"\nBiểu đồ loss đã được lưu tại: {figure_path}")
plt.close()
