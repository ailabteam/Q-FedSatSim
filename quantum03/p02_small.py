import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import matplotlib.pyplot as plt
import os
import time

# --- Thiết lập ---
torch.set_default_dtype(torch.float32)
print("Bắt đầu Bài toán MNIST 10 lớp (Kiến trúc Cải tiến)...")
if not os.path.exists("figures"):
    os.makedirs("figures")

# CẬP NHẬT: Giảm số qubit để tránh Barren Plateaus
n_qubits = 4
batch_size = 64 # Tăng lại batch size
n_train_samples = 4000
n_test_samples = 1000
epochs = 20

# --- Dữ liệu (giữ nguyên) ---
train_dataset_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
train_subset = Subset(train_dataset_full, range(n_train_samples))
test_subset = Subset(test_dataset_full, range(n_test_samples))
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
print(f"Đã chuẩn bị xong dữ liệu: {len(train_loader.dataset)} mẫu huấn luyện, {len(test_loader.dataset)} mẫu kiểm thử.")

# --- Mạch Lượng tử (4 qubits) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Mô hình Hybrid (Kiến trúc Cải tiến) ---
class HybridModelImproved(nn.Module):
    def __init__(self, n_qubits, n_classes=10):
        super().__init__()
        # Phần 1: CNN Feature Extractor
        self.cnn_part = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            # Tầng cổ điển NÉN đặc trưng xuống n_qubits chiều
            nn.Linear(120, n_qubits)
        )
        
        # Phần 2: Quantum Processor
        n_layers = 2
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
        
        # Phần 3: Tầng cổ điển MỞ RỘNG để phân loại
        self.classical_classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        # 1. Nén đặc trưng bằng các tầng cổ điển
        features = self.cnn_part(x)
        
        # 2. Xử lý bằng mạch lượng tử
        scaled_features = torch.pi * torch.sigmoid(features)
        q_out_list = quantum_circuit(scaled_features.double(), self.q_weights.double())
        q_out = torch.stack(q_out_list, dim=1).float()
        
        # 3. Phân loại cuối cùng
        logits = self.classical_classifier(q_out)
        return logits

# --- Huấn luyện ---
model = HybridModelImproved(n_qubits=n_qubits)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("\nBắt đầu huấn luyện mô hình cải tiến...")
start_time = time.time()

for epoch in range(epochs):
    # ... Vòng lặp huấn luyện và đánh giá giữ nguyên ...
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            total += batch_Y.size(0)
            correct += (predicted == batch_Y).sum().item()
    avg_accuracy = 100 * correct / total
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.2f}%, Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất! Tổng thời gian: {total_time:.2f}s")
