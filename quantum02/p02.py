import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

# --- GIẢI PHÁP CUỐI CÙNG: CHUYỂN TOÀN BỘ SANG FLOAT64 ---
torch.set_default_dtype(torch.float64)

print("Bắt đầu Bài toán 2 (Phiên bản Float64): Phân loại MNIST...")

# --- Thiết lập ---
if not os.path.exists("figures"):
    os.makedirs("figures")
n_qubits, batch_size, n_train_samples, n_test_samples = 4, 32, 500, 200

# --- Tải và xử lý dữ liệu với dtype=float64 ---
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

def filter_and_process_data(dataset, n_samples):
    mask = (dataset.targets == 0) | (dataset.targets == 1)
    # Chuyển data sang float64 ngay từ đầu
    data = dataset.data[mask].double()[:n_samples]
    targets = dataset.targets[mask].double()[:n_samples]
    return data.reshape(len(data), -1) / 255.0, targets

print("Đang xử lý dữ liệu huấn luyện...")
train_data_flat, train_targets = filter_and_process_data(train_dataset, n_train_samples)
print("Đang xử lý dữ liệu kiểm thử...")
test_data_flat, test_targets = filter_and_process_data(test_dataset, n_test_samples)

pca = PCA(n_components=n_qubits)
train_data_pca = pca.fit_transform(train_data_flat.numpy()) # PCA cần numpy array
test_data_pca = pca.transform(test_data_flat.numpy())

# Tạo DataLoader với tensor float64
train_loader = DataLoader(TensorDataset(torch.tensor(train_data_pca), train_targets.view(-1, 1)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(test_data_pca), test_targets.view(-1, 1)), batch_size=batch_size, shuffle=False)
print(f"Đã chuẩn bị xong dữ liệu: {len(train_loader.dataset)} mẫu huấn luyện, {len(test_loader.dataset)} mẫu kiểm thử.")

# --- Mô hình Hybrid (tự động dùng float64) ---
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        n_layers = 3
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
        self.classical_layer = nn.Linear(n_qubits, 1)

    def forward(self, x):
        q_out_list = quantum_circuit(x, self.q_weights)
        # torch.stack sẽ tự động tạo tensor float64
        q_out_tensor = torch.stack(q_out_list, dim=1)
        # Không cần ép kiểu nữa, cả hai đều là float64
        logits = self.classical_layer(q_out_tensor)
        return logits

# --- Huấn luyện và Đánh giá ---
model = HybridModel(n_qubits=n_qubits)
optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCEWithLogitsLoss()

epochs = 15
loss_history, accuracy_history = [], []

print("\nBắt đầu huấn luyện...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = loss_fn(logits, batch_Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            logits = model(batch_X)
            predicted = torch.round(torch.sigmoid(logits))
            total += batch_Y.size(0)
            correct += (predicted == batch_Y).sum().item()
            
    avg_accuracy = 100 * correct / total
    accuracy_history.append(avg_accuracy)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.2f}%")

print("Huấn luyện hoàn tất!")

# --- Trực quan hóa ---
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color=color)
ax1.plot(loss_history, color=color, marker='o'); ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(accuracy_history, color=color, marker='x'); ax2.tick_params(axis='y', labelcolor=color)
fig.suptitle('Training (Float64 Version)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
figure_path = "figures/qnn_mnist_training_float64.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ huấn luyện đã được lưu tại: {figure_path}")
plt.close()
