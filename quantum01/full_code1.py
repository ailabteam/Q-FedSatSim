import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt
import dimod

# Thiết lập device CUDA
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using torch device: {device}")

# Tạo folder lưu kết quả
os.makedirs("./results", exist_ok=True)

# Thiết lập hạt giống
np.random.seed(42)
torch.manual_seed(42)

# Tham số mô phỏng
N = 100  # số client
k = 10   # số client chọn mỗi vòng
num_rounds = 3  # giảm round để nhanh
lambda_ = 0.5
gamma = 1.0
geo_latency_range = (480, 560)

# Dữ liệu latency và accuracy giả lập
d_i = np.random.uniform(geo_latency_range[0], geo_latency_range[1], N)
a_i = np.random.uniform(0.7, 0.95, N)

# Chuẩn bị MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Phân chia dữ liệu IID cho client
num_samples_per_client = len(mnist_train) // N
client_data = [Subset(mnist_train, range(i * num_samples_per_client, (i + 1) * num_samples_per_client)) for i in range(N)]
client_loaders = [DataLoader(data, batch_size=32, shuffle=True) for data in client_data]

# Mô hình CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
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

# Huấn luyện client
def train_client(model, data_loader, epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# Đánh giá global model
def evaluate_global_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# FedAvg aggregation
def fedavg(global_model, selected_clients, client_loaders):
    if len(selected_clients) == 0:
        print("Warning: No clients selected, skipping FedAvg")
        return global_model

    client_models = []
    for cid in selected_clients:
        client_model = CNN().to(device)
        client_model.load_state_dict(global_model.state_dict())
        client_model = train_client(client_model, client_loaders[cid])
        client_models.append(client_model)

    global_state = global_model.state_dict()
    for key in global_state:
        stacked = torch.stack([cm.state_dict()[key].to(device) for cm in client_models], dim=0)
        global_state[key] = torch.mean(stacked, dim=0)
    global_model.load_state_dict(global_state)
    return global_model

# Chọn client bằng Simulated Annealing với dimod
def select_clients_sa(d_i, a_i, k, lambda_=0.5, gamma=1.0):
    print("Selecting clients using Simulated Annealing...")
    start = time.time()
    Q = {}
    for i in range(N):
        Q[(i, i)] = d_i[i] - lambda_ * a_i[i] + gamma * (1 - 2 * k)
    for i in range(N):
        for j in range(i + 1, N):
            Q[(i, j)] = 2 * gamma

    sampler = dimod.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=100)
    elapsed = time.time() - start
    print(f"  SA selection done in {elapsed:.2f} seconds")

    best_solution = response.first.sample
    selected = [i for i, val in best_solution.items() if val == 1]

    if len(selected) == 0:
        print("Warning: SA returned empty selection, fallback to random selection")
        selected = np.random.choice(N, k, replace=False).tolist()

    print(f"  Selected clients (SA): {selected}")
    return selected

# Chọn client Greedy theo latency thấp nhất
def select_clients_greedy(d_i, k):
    return np.argsort(d_i)[:k]

# Chọn client ngẫu nhiên
def select_clients_random(N, k):
    return np.random.choice(N, k, replace=False)

# Độ công bằng (std của số lần chọn)
def calculate_fairness(selection_counts):
    return np.std(list(selection_counts.values()))

# Chạy thực nghiệm 1 phương pháp
def run_experiment(method, d_i, a_i, k, client_loaders, test_loader):
    print(f"\n=== Running experiment: {method} ===")
    global_model = CNN().to(device)
    latency_history = []
    accuracy_history = []
    selection_counts = {i: 0 for i in range(N)}

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")

        if method == "SA":
            selected_clients = select_clients_sa(d_i, a_i, k, lambda_, gamma)
        elif method == "Greedy":
            selected_clients = select_clients_greedy(d_i, k).tolist()
        elif method == "Random":
            selected_clients = select_clients_random(N, k).tolist()
        else:
            raise ValueError(f"Unknown method {method}")

        for cid in selected_clients:
            selection_counts[cid] += 1

        round_latency = sum(d_i[cid] for cid in selected_clients)
        latency_history.append(round_latency)

        global_model = fedavg(global_model, selected_clients, client_loaders)

        accuracy = evaluate_global_model(global_model, test_loader)
        accuracy_history.append(accuracy)
        print(f"  Accuracy after round {round+1}: {accuracy:.2f}%")
        print(f"  Latency this round: {round_latency:.2f} ms")

        if accuracy > 90:
            print(f"Accuracy threshold reached at round {round+1}")
            break

    fairness = calculate_fairness(selection_counts)
    convergence_rounds = round + 1
    return np.mean(latency_history), np.mean(accuracy_history), convergence_rounds, fairness

# Main
methods = ["SA", "Greedy", "Random"]
results = {"Method": [], "Latency (s)": [], "Accuracy (%)": [], "Convergence Rounds": [], "Fairness": []}

print("Loading MNIST dataset...")
# Dataset đã tải ở trên rồi, nên không cần tải lại

for method in methods:
    latency, accuracy, rounds, fairness = run_experiment(method, d_i, a_i, k, client_loaders, test_loader)
    results["Method"].append(method)
    results["Latency (s)"].append(latency / 1000)  # ms to s
    results["Accuracy (%)"].append(accuracy)
    results["Convergence Rounds"].append(rounds)
    results["Fairness"].append(fairness)

df = pd.DataFrame(results)
print("\nPerformance Comparison:")
print(df)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.bar(df["Method"], df["Latency (s)"])
plt.title("Latency Comparison")
plt.ylabel("Latency (s)")
plt.savefig("./results/latency_comparison.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df["Method"], df["Accuracy (%)"])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.savefig("./results/accuracy_comparison.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df["Method"], df["Convergence Rounds"])
plt.title("Convergence Rounds Comparison")
plt.ylabel("Convergence Rounds")
plt.savefig("./results/convergence_rounds_comparison.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df["Method"], df["Fairness"])
plt.title("Fairness Comparison")
plt.ylabel("Fairness (Std of Selection Counts)")
plt.savefig("./results/fairness_comparison.png")
plt.show()

