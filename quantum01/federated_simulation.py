import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt
import time

try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from neal import SimulatedAnnealingSampler
    print("Using neal.SimulatedAnnealingSampler as fallback")

# Tạo thư mục lưu kết quả
os.makedirs("results", exist_ok=True)

# Thiết lập hạt giống ngẫu nhiên
np.random.seed(42)
torch.manual_seed(42)

# Tham số mô phỏng
N = 100
k = 10
num_rounds = 20
lambda_ = 0.5
gamma = 1.0
geo_latency_range = (480, 560)

d_i = np.random.uniform(*geo_latency_range, N)
a_i = np.random.uniform(0.7, 0.95, N)

# Tải dữ liệu MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

num_samples_per_client = len(mnist_train) // N
client_data = [Subset(mnist_train, range(i * num_samples_per_client, (i + 1) * num_samples_per_client)) for i in range(N)]
client_loaders = [DataLoader(data, batch_size=32, shuffle=True) for data in client_data]

# CNN Model
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

def select_clients_sa(d_i, a_i, k, lambda_=0.5, gamma=1.0):
    Q = {}
    for i in range(N):
        Q[(i, i)] = d_i[i] - lambda_ * a_i[i] + gamma * (1 - 2 * k)
    for i in range(N):
        for j in range(i+1, N):
            Q[(i, j)] = 2 * gamma
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=1000)
    best_solution = response.first.sample
    return [i for i, val in best_solution.items() if val == 1]

def select_clients_greedy(d_i, k):
    return np.argsort(d_i)[:k]

def select_clients_random(N, k):
    return np.random.choice(N, k, replace=False)

def calculate_fairness(selection_counts):
    return np.std(list(selection_counts.values()))

def run_experiment(method, d_i, a_i, k, client_loaders, test_loader):
    global_model = CNN()
    latency_history = []
    accuracy_history = []
    selection_counts = {i: 0 for i in range(N)}
    start = time.time()

    for round in range(num_rounds):
        if method == "SA":
            selected_clients = select_clients_sa(d_i, a_i, k, lambda_, gamma)
        elif method == "Greedy":
            selected_clients = select_clients_greedy(d_i, k)
        elif method == "Random":
            selected_clients = select_clients_random(N, k)

        for client_id in selected_clients:
            selection_counts[client_id] += 1

        round_latency = sum(d_i[client_id] for client_id in selected_clients)
        latency_history.append(round_latency)

        global_model = fedavg(global_model, selected_clients, client_loaders)
        accuracy = evaluate_global_model(global_model, test_loader)
        accuracy_history.append(accuracy)

        if accuracy > 90:
            convergence_rounds = round + 1
            break
    else:
        convergence_rounds = num_rounds

    elapsed_time = time.time() - start
    fairness = calculate_fairness(selection_counts)
    return np.mean(latency_history), np.mean(accuracy_history), convergence_rounds, fairness, elapsed_time

# Run all methods
methods = ["SA", "Greedy", "Random"]
results = {"Method": [], "Latency (s)": [], "Accuracy (%)": [], "Convergence Rounds": [], "Fairness": [], "Time (s)": []}

for method in methods:
    latency, accuracy, rounds, fairness, elapsed = run_experiment(method, d_i, a_i, k, client_loaders, test_loader)
    results["Method"].append(method)
    results["Latency (s)"].append(latency / 1000)
    results["Accuracy (%)"].append(accuracy)
    results["Convergence Rounds"].append(rounds)
    results["Fairness"].append(fairness)
    results["Time (s)"].append(elapsed)

df = pd.DataFrame(results)
print("\n=== Performance Comparison ===")
print(df)

# Plot results
df.set_index("Method", inplace=True)

for metric in ["Latency (s)", "Accuracy (%)", "Convergence Rounds", "Fairness"]:
    plt.figure(figsize=(8, 5))
    df[metric].plot(kind="bar", title=metric)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"./results/{metric.replace(' ', '_').lower()}.png")
    plt.close()

