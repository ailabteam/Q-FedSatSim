import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import matplotlib.pyplot as plt
import os
import time
from copy import deepcopy

# --- Thiết lập ---
torch.manual_seed(42)
torch.set_default_dtype(torch.float32)
print("Bắt đầu Bài toán: Federated Quantum-CNN trên MNIST 10 lớp...")

if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Các tham số cho FQML ---
NUM_CLIENTS = 10
COMMUNICATION_ROUNDS = 15 # Số vòng giao tiếp giữa server và client
LOCAL_EPOCHS = 3          # Số epoch huấn luyện tại mỗi client trong một vòng
BATCH_SIZE = 32
LEARNING_RATE = 0.005

# --- Các tham số cho mô hình ---
N_QUBITS = 4
Q_LAYERS = 3

# --- Bước 1: Chuẩn bị Dữ liệu Phân tán ---
def get_mnist_dataloaders(num_clients):
    """Chia MNIST thành num_clients phần không chồng chéo nhau."""
    # Tải toàn bộ tập huấn luyện
    full_train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    
    # Tính kích thước cho mỗi client
    subset_size = len(full_train_dataset) // num_clients
    lengths = [subset_size] * num_clients
    
    # Chia dataset
    subsets = random_split(full_train_dataset, lengths)
    
    # Tạo DataLoader cho mỗi client
    client_dataloaders = [DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True) for subset in subsets]
    
    # Tạo một tập test trung tâm để đánh giá mô hình toàn cục
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    return client_dataloaders, test_loader

client_loaders, test_loader = get_mnist_dataloaders(NUM_CLIENTS)
print(f"Đã tạo dữ liệu cho {len(client_loaders)} clients và 1 test loader trung tâm.")

# --- Bước 2: Định nghĩa Mô hình Lai (Tái sử dụng kiến trúc tốt nhất) ---
dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit_reuploading(inputs, weights):
    for l in range(len(weights)):
        qml.StronglyEntanglingLayers(weights[l], wires=range(N_QUBITS))
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class HybridModelFinal(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=Q_LAYERS, n_classes=10):
        super().__init__()
        self.cnn_part = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Flatten(), nn.Linear(256, 120), nn.ReLU(), nn.Linear(120, n_qubits)
        )
        weight_shape = (n_layers, 1, n_qubits, 3)
        self.q_weights = nn.Parameter(0.01 * torch.randn(weight_shape))
        self.classical_classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        features = self.cnn_part(x)
        scaled_features = torch.pi * torch.sigmoid(features)
        q_out_list = quantum_circuit_reuploading(scaled_features.double(), self.q_weights.double())
        q_out = torch.stack(q_out_list, dim=1).float()
        logits = self.classical_classifier(q_out)
        return logits

# --- Bước 3: Định nghĩa Logic của Client và Server ---
def client_update(model, dataloader, local_epochs):
    """Huấn luyện mô hình trên dữ liệu của client."""
    local_model = deepcopy(model) # Tạo bản sao để huấn luyện
    local_model.train()
    optimizer = optim.AdamW(local_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(local_epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = local_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return local_model.state_dict()

def federated_average(local_state_dicts):
    """Tính trung bình các tham số từ các client."""
    global_state_dict = deepcopy(local_state_dicts[0])
    
    # Cộng dồn tất cả các tham số
    for key in global_state_dict.keys():
        for i in range(1, len(local_state_dicts)):
            global_state_dict[key] += local_state_dicts[i][key]
        
        # Lấy trung bình
        global_state_dict[key] = torch.div(global_state_dict[key], len(local_state_dicts))
        
    return global_state_dict

def evaluate_model(model, dataloader):
    """Đánh giá mô hình trên tập test."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# --- Bước 4: Chạy Vòng lặp Học Liên kết ---
global_model = HybridModelFinal()
accuracy_history = []

print("\nBắt đầu quá trình học liên kết lượng tử...")
start_time = time.time()

for round_num in range(COMMUNICATION_ROUNDS):
    print(f"\n--- Vòng Giao tiếp {round_num + 1}/{COMMUNICATION_ROUNDS} ---")
    
    local_state_dicts = []
    
    # Giai đoạn huấn luyện cục bộ
    for client_id in range(NUM_CLIENTS):
        print(f"  > Client {client_id+1} đang huấn luyện...")
        local_state_dict = client_update(global_model, client_loaders[client_id], LOCAL_EPOCHS)
        local_state_dicts.append(local_state_dict)
        
    # Giai đoạn tổng hợp tại server
    print("  > Server đang tổng hợp các bản cập nhật...")
    global_state_dict = federated_average(local_state_dicts)
    global_model.load_state_dict(global_state_dict)
    
    # Đánh giá mô hình toàn cục
    test_accuracy = evaluate_model(global_model, test_loader)
    accuracy_history.append(test_accuracy)
    print(f"  > Độ chính xác của Mô hình Toàn cục: {test_accuracy:.2f}%")
    
total_time = time.time() - start_time
print(f"\nQuá trình học liên kết hoàn tất! Tổng thời gian: {total_time/60:.2f} phút")

# --- Bước 5: Trực quan hóa ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, COMMUNICATION_ROUNDS + 1), accuracy_history, marker='o')
plt.title("Federated Quantum Learning Accuracy Over Communication Rounds")
plt.xlabel("Communication Round")
plt.ylabel("Global Model Test Accuracy (%)")
plt.xticks(range(1, COMMUNICATION_ROUNDS + 1))
plt.grid(True)
figure_path = "figures/fqml_accuracy.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ huấn luyện đã được lưu tại: {figure_path}")
plt.close()
