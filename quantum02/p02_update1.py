import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Thiết lập môi trường Float64 ---
torch.set_default_dtype(torch.float64)

print("Bắt đầu Bài toán 5 (Ổn định hóa): Hybrid CNN-QNN 60/40...")

# --- Thiết lập ---
if not os.path.exists("figures"):
    os.makedirs("figures")
n_qubits = 4
batch_size = 32

# --- Tải và Xử lý Dữ liệu ---
full_train_dataset_raw = torchvision.datasets.MNIST(root="./data", train=True, download=True)
full_test_dataset_raw = torchvision.datasets.MNIST(root="./data", train=False, download=True)
def get_all_zeros_and_ones(dataset):
    mask = (dataset.targets == 0) | (dataset.targets == 1)
    return dataset.data[mask], dataset.targets[mask]
train_data, train_targets = get_all_zeros_and_ones(full_train_dataset_raw)
test_data, test_targets = get_all_zeros_and_ones(full_test_dataset_raw)
all_data = torch.cat([train_data, test_data], dim=0)
all_targets = torch.cat([train_targets, test_targets], dim=0)
print(f"Tổng số mẫu '0' và '1' trong MNIST: {len(all_data)}")

class FullMNISTDataset(Dataset):
    def __init__(self, data, targets):
        self.data = (data.double() / 255.0).unsqueeze(1)
        self.targets = targets.double()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].view(-1)
full_dataset = FullMNISTDataset(all_data, all_targets)
train_size = int(0.6 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
print(f"Đã chia dữ liệu: {len(train_dataset)} mẫu huấn luyện, {len(test_dataset)} mẫu kiểm thử.")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Mạch Lượng tử và Mô hình (Giữ nguyên) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
class HybridCNNQNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, n_qubits)
        )
        n_layers = 3
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
        self.classical_layer = nn.Linear(n_qubits, 1)
    def forward(self, x):
        features = self.cnn_feature_extractor(x)
        scaled_features = torch.pi * torch.sigmoid(features)
        q_out_list = quantum_circuit(scaled_features, self.q_weights)
        q_out_tensor = torch.stack(q_out_list, dim=1)
        logits = self.classical_layer(q_out_tensor)
        return logits

# --- Huấn luyện và Đánh giá ---
model = HybridCNNQNN(n_qubits=n_qubits)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()
epochs = 10
loss_history, accuracy_history = [], []

print("\nBắt đầu huấn luyện với Gradient Clipping...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = loss_fn(logits, batch_Y)
        loss.backward()
        
        # THÊM DÒNG NÀY ĐỂ ỔN ĐỊNH HÓA
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
# ... (phần vẽ biểu đồ giữ nguyên)
color = 'tab:red'
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color=color)
ax1.plot(loss_history, color=color, marker='o'); ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(accuracy_history, color=color, marker='x'); ax2.tick_params(axis='y', labelcolor=color)
fig.suptitle('Training (CNN-QNN, 60/40 Split with Grad-Clip)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
figure_path = "figures/cnn_qnn_mnist_60_40_stable.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ huấn luyện đã được lưu tại: {figure_path}")
plt.close()
