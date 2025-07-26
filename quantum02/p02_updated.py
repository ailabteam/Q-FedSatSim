import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import os

# --- Thiết lập môi trường Float64 để nhất quán ---
torch.set_default_dtype(torch.float64)

print("Bắt đầu Bài toán 5: Hybrid CNN-QNN để phân loại MNIST...")

# --- Thiết lập ---
if not os.path.exists("figures"):
    os.makedirs("figures")
n_qubits = 4          # Số đặc trưng CNN sẽ xuất ra, và là số qubit QNN sẽ nhận vào
batch_size = 32
n_train_samples = 1000 # Tăng số mẫu để CNN học tốt hơn
n_test_samples = 400

# --- Tải và Xử lý Dữ liệu (Không dùng PCA) ---
class FilteredMNIST(Dataset):
    def __init__(self, mnist_dataset, n_samples=None):
        mask = (mnist_dataset.targets == 0) | (mnist_dataset.targets == 1)
        self.data = mnist_dataset.data[mask].double()
        self.targets = mnist_dataset.targets[mask].double()
        
        if n_samples:
            self.data = self.data[:n_samples]
            self.targets = self.targets[:n_samples]
            
        # Chuẩn hóa và thêm chiều kênh (channel dimension)
        self.data = (self.data / 255.0).unsqueeze(1) # Shape: (N, 1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].view(-1)

# Tải bộ MNIST gốc
full_train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
full_test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)

# Tạo dataset đã lọc
train_dataset = FilteredMNIST(full_train_dataset, n_samples=n_train_samples)
test_dataset = FilteredMNIST(full_test_dataset, n_samples=n_test_samples)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Đã chuẩn bị xong dữ liệu: {len(train_dataset)} mẫu huấn luyện, {len(test_dataset)} mẫu kiểm thử.")
print(f"Kích thước mỗi ảnh: {train_dataset.data.shape[1:]}")

# --- Mạch Lượng tử (giữ nguyên) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Mô hình Hybrid CNN-QNN ---
class HybridCNNQNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        # Phần 1: CNN Feature Extractor
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), # 1 kênh vào, 6 kênh ra, kernel 5x5 -> output: (N, 6, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # -> output: (N, 6, 12, 12)
            nn.Conv2d(6, 16, kernel_size=5),# -> output: (N, 16, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # -> output: (N, 16, 4, 4)
            nn.Flatten(),                   # -> output: (N, 16*4*4 = 256)
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, n_qubits)        # Đầu ra có n_qubits chiều
        )
        
        # Phần 2: Quantum Classifier
        n_layers = 3
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
        self.classical_layer = nn.Linear(n_qubits, 1)

    def forward(self, x):
        # 1. Trích xuất đặc trưng bằng CNN
        features = self.cnn_feature_extractor(x)
        
        # 2. Đưa đặc trưng vào mạch lượng tử
        # Cần scale đặc trưng về khoảng [0, pi] để AngleEmbedding hoạt động tốt
        scaled_features = torch.pi * torch.sigmoid(features)
        
        q_out_list = quantum_circuit(scaled_features, self.q_weights)
        q_out_tensor = torch.stack(q_out_list, dim=1)
        
        # 3. Phân loại cuối cùng
        logits = self.classical_layer(q_out_tensor)
        return logits

# --- Huấn luyện và Đánh giá ---
model = HybridCNNQNN(n_qubits=n_qubits)
optimizer = optim.Adam(model.parameters(), lr=0.01) # Giảm lr một chút cho ổn định
loss_fn = nn.BCEWithLogitsLoss()

epochs = 10
loss_history, accuracy_history = [], []

print("\nBắt đầu huấn luyện mô hình CNN-QNN...")
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
# ... (phần vẽ biểu đồ giữ nguyên)
color = 'tab:red'
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color=color)
ax1.plot(loss_history, color=color, marker='o'); ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(accuracy_history, color=color, marker='x'); ax2.tick_params(axis='y', labelcolor=color)
fig.suptitle('Training (Hybrid CNN-QNN Model)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
figure_path = "figures/cnn_qnn_mnist_training.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ huấn luyện đã được lưu tại: {figure_path}")
plt.close()
