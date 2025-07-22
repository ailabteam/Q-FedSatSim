import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt

# Try importing SimulatedAnnealingSampler
try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from neal import SimulatedAnnealingSampler  # Fallback to dwave-neal
    print("Using neal.SimulatedAnnealingSampler as fallback")

# Thiết lập hạt giống ngẫu nhiên để tái tạo kết quả
np.random.seed(42)
torch.manual_seed(42)

# Tham số mô phỏng GEO
N = 100  # Số client
k = 10   # Số client được chọn mỗi vòng
num_rounds = 20  # Giảm số vòng để chạy nhanh hơn trên Colab
lambda_ = 0.5    # Hệ số cân bằng latency và accuracy
gamma = 1.0      # Hệ số phạt ràng buộc k
geo_latency_range = (480, 560)  # Độ trễ GEO (ms, tổng uplink + downlink)

# Tạo dữ liệu độ trễ và đóng góp độ chính xác giả lập
d_i = np.random.uniform(geo_latency_range[0], geo_latency_range[1], N)  # Latency (ms)
a_i = np.random.uniform(0.7, 0.95, N)  # Accuracy contribution (giả lập)

# Chuẩn bị dữ liệu MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Phân chia dữ liệu cho các client (IID)
num_samples_per_client = len(mnist_train) // N
client_data = [Subset(mnist_train, range(i * num_samples_per_client, (i + 1) * num_samples_per_client)) for i in range(N)]
client_loaders = [DataLoader(data, batch_size=32, shuffle=True) for data in client_data]

# Định nghĩa mô hình CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hàm huấn luyện mô hình trên client
def train_client(model, data_loader, epochs=5):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# Hàm đánh giá mô hình toàn cục
def evaluate_global_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Hàm FedAvg
def fedavg(global_model, selected_clients, client_loaders):
    client_models = []
    for client_id in selected_clients:
        client_model = CNN()
        client_model.load_state_dict(global_model.state_dict())
        client_model = train_client(client_model, client_loaders[client_id])
        client_models.append(client_model)
    
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.mean(torch.stack([cm.state_dict()[key] for cm in client_models]), dim=0)
    global_model.load_state_dict(global_state)
    return global_model

# Hàm chọn client bằng Simulated Annealing
def select_clients_sa(d_i, a_i, k, lambda_=0.5, gamma=1.0):
    Q = {}
    for i in range(N):
        Q[(i, i)] = d_i[i] - lambda_ * a_i[i] + gamma * (1 - 2 * k)
    for i in range(N):
        for j in range(iಮ
        Q[(i, j)] = 2 * gamma
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=1000)
    best_solution = response.first.sample
    return [i for i, val in best_solution.items() if val == 1]

# Hàm chọn client bằng Greedy
def select_clients_greedy(d_i, k):
    return np.argsort(d_i)[:k]

# Hàm chọn client ngẫu nhiên
def select_clients_random(N, k):
    return np.random.choice(N, k, replace=False)

# Hàm tính độ công bằng
def calculate_fairness(selection_counts):
    return np.std(list(selection_counts.values()))

# Hàm chạy thực nghiệm cho một phương pháp
def run_experiment(method, d_i, a_i, k, client_loaders, test_loader):
    global_model = CNN()
    latency_history = []
    accuracy_history = []
    selection_counts = {i: 0 for i in range(N)}
    
    for round in range(num_rounds):
        # Chọn client
        if method == "SA":
            selected_clients = select_clients_sa(d_i, a_i, k, lambda_, gamma)
        elif method == "Greedy":
            selected_clients = select_clients_greedy(d_i, k)
        elif method == "Random":
            selected_clients = select_clients_random(N, k)
        
        # Cập nhật tần suất chọn client
        for client_id in selected_clients:
            selection_counts[client_id] += 1
        
        # Tính độ trễ
        round_latency = sum(d_i[client_id] for client_id in selected_clients)
        latency_history.append(round_latency)
        
        # Cập nhật mô hình toàn cục
        global_model = fedavg(global_model, selected_clients, client_loaders)
        
        # Đánh giá độ chính xác
        accuracy = evaluate_global_model(global_model, test_loader)
        accuracy_history.append(accuracy)
        
        # Kiểm tra hội tụ (độ chính xác > 90%)
        if accuracy > 90:
            convergence_rounds = round + 1
            break
    else:
        convergence_rounds = num_rounds
    
    fairness = calculate_fairness(selection_counts)
    return np.mean(latency_history), np.mean(accuracy_history), convergence_rounds, fairness

# Chạy thực nghiệm cho tất cả phương pháp
methods = ["SA", "Greedy", "Random"]
results = {"Method": [], "Latency (s)": [], "Accuracy (%)": [], "Convergence Rounds": [], "Fairness": []}

for method in methods:
    latency, accuracy, rounds, fairness = run_experiment(method, d_i, a_i, k, client_loaders, test_loader)
    results["Method"].append(method)
    results["Latency (s)"].append(latency / 1000)  # Chuyển sang giây
    results["Accuracy (%)"].append(accuracy)
    results["Convergence Rounds"].append(rounds)
    results["Fairness"].append(fairness)

# Hiển thị kết quả
df = pd.DataFrame(results)
print("\nPerformance Comparison:")
print(df)
