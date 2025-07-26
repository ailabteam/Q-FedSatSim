import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Vẫn giữ dòng này như một thói quen tốt để đảm bảo các thành phần PyTorch nhất quán
torch.set_default_dtype(torch.float32)

print("Bắt đầu bài toán QNN cho phân loại XOR (Phiên bản Hybrid Tối ưu)...")

# Tạo thư mục
if not os.path.exists("figures"):
    os.makedirs("figures")

# Dữ liệu
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
Y = Y.view(-1, 1)

# Mạch lượng tử
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Mô hình Hybrid
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Phần lượng tử
        n_layers = 3
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
        
        # Phần cổ điển
        self.classical_layer = nn.Linear(1, 1)

    def forward(self, x):
        q_out_raw = quantum_circuit(x, self.q_weights)
        q_out_reshaped = q_out_raw.unsqueeze(1)
        
        # --- GIẢI PHÁP CUỐI CÙNG Ở ĐÂY ---
        # Ép kiểu tường minh đầu ra của mạch lượng tử về float32
        logits = self.classical_layer(q_out_reshaped.float())
        
        return logits

# Huấn luyện mô hình
model = HybridModel()
optimizer = optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()

epochs = 150
loss_history = []

print("\nBắt đầu huấn luyện...")
for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(X)
    
    loss = loss_fn(logits, Y)
    loss_history.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Huấn luyện hoàn tất!")

# Đánh giá kết quả
print("\nĐánh giá mô hình sau khi huấn luyện:")
with torch.no_grad():
    final_logits = model(X)
    final_probs = torch.sigmoid(final_logits)
    predicted_labels = torch.round(final_probs)

    correct_predictions = 0
    for i in range(len(X)):
        is_correct = "Đúng" if int(predicted_labels[i].item()) == int(Y[i].item()) else "Sai"
        if is_correct == "Đúng":
            correct_predictions += 1
        print(f"Input: {X[i].tolist()}, Target: {int(Y[i].item())}, Predicted: {int(predicted_labels[i].item())}, Probability: {final_probs[i].item():.4f} -> {is_correct}")
    
    accuracy = correct_predictions / len(X) * 100
    print(f"\nĐộ chính xác: {accuracy:.2f}%")

# Lưu biểu đồ
figure_path = "figures/qnn_xor_hybrid_optimal.png"
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Training Loss History (Optimal Hybrid Model)")
plt.xlabel("Epoch")
plt.ylabel("BCEWithLogits Loss")
plt.grid(True)
plt.savefig(figure_path)
print(f"\nBiểu đồ loss đã được lưu tại: {figure_path}")
plt.close()
