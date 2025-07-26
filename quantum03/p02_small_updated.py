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
print("Bắt đầu Bài toán MNIST 10 lớp (Kiến trúc cuối cùng: Data Re-uploading)...")
if not os.path.exists("figures"):
    os.makedirs("figures")

n_qubits = 4
n_layers = 4 # Số lần nạp lại dữ liệu (và số lớp trọng số)
batch_size = 64
n_train_samples = 4000
n_test_samples = 1000
epochs = 20
learning_rate = 0.005 # Giảm learning rate một chút để ổn định hơn

# --- Dữ liệu (giữ nguyên) ---
train_dataset_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
train_subset = Subset(train_dataset_full, range(n_train_samples))
test_subset = Subset(test_dataset_full, range(n_test_samples))
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
print(f"Đã chuẩn bị xong dữ liệu: {len(train_loader.dataset)} mẫu huấn luyện, {len(test_loader.dataset)} mẫu kiểm thử.")

# --- Mạch Lượng tử (CẬP NHẬT: Data Re-uploading) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit_reuploading(inputs, weights):
    # weights giờ có shape (n_layers, n_qubits)
    for l in range(len(weights)):
        # Lớp trọng số biến phân
        qml.StronglyEntanglingLayers(weights[l], wires=range(n_qubits))
        # Lớp nạp lại dữ liệu
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Mô hình Hybrid (CẬP NHẬT để dùng mạch mới) ---
class HybridModelFinal(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes=10):
        super().__init__()
        self.cnn_part = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, n_qubits)
        )
        
        # CẬP NHẬT: Trọng số lượng tử cho Data Re-uploading
        # Mỗi lớp trong StronglyEntanglingLayers cần 3 tham số/qubit
        weight_shape = (n_layers, 1, n_qubits, 3) 
        # Khởi tạo trọng số cho từng lớp một cách cẩn thận
        self.q_weights = nn.Parameter(0.01 * torch.randn(weight_shape))
        
        self.classical_classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        features = self.cnn_part(x)
        scaled_features = torch.pi * torch.sigmoid(features)
        
        # Gọi mạch lượng tử mới
        q_out_list = quantum_circuit_reuploading(scaled_features.double(), self.q_weights.double())
        q_out = torch.stack(q_out_list, dim=1).float()
        
        logits = self.classical_classifier(q_out)
        return logits

# --- Huấn luyện ---
model = HybridModelFinal(n_qubits=n_qubits, n_layers=n_layers)
# Sử dụng optimizer và learning rate khác
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

print("\nBắt đầu huấn luyện mô hình cuối cùng...")
start_time = time.time()

for epoch in range(epochs):
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
