import torch
import numpy as np
import dimod
from dimod import ExactSolver

# Sử dụng GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hyperparameters
NUM_CLIENTS = 10
K = 3  # Số client được chọn mỗi vòng
ROUNDS = 3
MAX_BANDWIDTH = 10  # Mbps
MAX_DELAY = 5       # s

# Tạo ngẫu nhiên bandwidth và delay
np.random.seed(42)
bandwidths = np.random.uniform(1, MAX_BANDWIDTH, NUM_CLIENTS)
delays = np.random.uniform(0.5, MAX_DELAY, NUM_CLIENTS)
print(f"Bandwidths (Mbps): {np.round(bandwidths, 2)}")
print(f"Delays (s): {np.round(delays, 3)}")

# Chuyển về torch tensors
bandwidths = torch.tensor(bandwidths, dtype=torch.float32, device=device)
delays = torch.tensor(delays, dtype=torch.float32, device=device)

# Định nghĩa QUBO matrix
def build_qubo(bandwidths, delays, alpha=0.7, beta=0.3):
    n = len(bandwidths)
    Q = {}

    norm_bandwidths = bandwidths / MAX_BANDWIDTH
    norm_delays = delays / MAX_DELAY

    for i in range(n):
        for j in range(n):
            if i == j:
                # Objective: maximize bandwidth, minimize delay
                Q[(i, i)] = -alpha * norm_bandwidths[i].item() + beta * norm_delays[i].item()
            else:
                Q[(i, j)] = 2.0  # penalty to avoid selecting more than K clients
    return Q

# Giải QUBO bằng ExactSolver
def solve_qubo(Q, k):
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    bqm.add_linear_equality_constraint(
        [(i, 1.0) for i in range(NUM_CLIENTS)],
        lagrange_multiplier=5.0,
        constant=-k
    )

    sampler = ExactSolver()
    response = sampler.sample(bqm)

    best = response.first.sample
    selected = [i for i, val in best.items() if val == 1]
    return selected

# Main loop
for rnd in range(ROUNDS):
    print(f"\n🌀 Round {rnd + 1}")
    Q = build_qubo(bandwidths, delays)
    selected_clients = solve_qubo(Q, k=K)
    print(f"Selected clients: {selected_clients}")

print("\n✅ Training Done.")

