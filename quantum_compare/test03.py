import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

# --- THƯ VIỆN BỔ SUNG ---
import pulp
import pandas as pd

print("Bắt đầu so sánh 3 phương pháp cho bài toán Max-Cut...")
if not os.path.exists("figures"): os.makedirs("figures")

# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA BÀI TOÁN
# ==============================================================================
n_nodes = 4
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

def evaluate_solution(bitstring, edges):
    """Tính số cạnh bị cắt cho một chuỗi bit."""
    cut_size = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cut_size += 1
    return cut_size

# --- Xây dựng Ising Hamiltonian cho Max-Cut ---
# H = sum_{i,j in E} 0.5 * (I - Z_i Z_j)
# Để tối đa hóa số cạnh cắt, ta cần TỐI THIỂU HÓA H' = sum_{i,j in E} Z_i Z_j
cost_h = qml.Hamiltonian(
    [1] * len(edges), 
    [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in edges]
)
print("Ising Hamiltonian cho Max-Cut đã được tạo.")

# ==============================================================================
# PHƯƠNG PHÁP 1: HYBRID QUANTUM-CLASSICAL (QAOA)
# ==============================================================================
def solve_with_qaoa(p=3, epochs=100, lr=0.1):
    print("\n--- Bắt đầu giải bằng QAOA ---")
    dev = qml.device("default.qubit", wires=n_nodes)
    mixer_h = qml.Hamiltonian([1]*n_nodes, [qml.PauliX(i) for i in range(n_nodes)])
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qaoa_circuit(params):
        gammas, betas = params[0], params[1]
        for i in range(n_nodes): qml.Hadamard(wires=i)
        for i in range(p):
            qml.exp(cost_h, -1j * gammas[i])
            qml.exp(mixer_h, -1j * betas[i])
        return qml.expval(cost_h)

    params = torch.tensor(np.random.uniform(0, np.pi, (2, p)), requires_grad=True, dtype=torch.float32)
    optimizer = optim.Adam([params], lr=lr)
    
    start_time = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        cost = qaoa_circuit(params)
        cost.backward()
        optimizer.step()
    end_time = time.time()
    
    @qml.qnode(dev)
    def get_probs(params):
        gammas, betas = params[0], params[1]
        for i in range(n_nodes): qml.Hadamard(wires=i)
        for i in range(p):
            qml.exp(cost_h, -1j * gammas[i])
            qml.exp(mixer_h, -1j * betas[i])
        return qml.probs(wires=range(n_nodes))

    probs = get_probs(params.detach().numpy())
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_nodes}b')
    
    value_found = evaluate_solution(best_bitstring, edges)
    return value_found, end_time - start_time, best_bitstring

# ==============================================================================
# PHƯƠNG PHÁP 2: QUANTUM-INSPIRED (SIMULATED ANNEALING)
# ==============================================================================
def solve_with_sa(n_iterations=5000):
    print("\n--- Bắt đầu giải bằng Simulated Annealing ---")
    
    def cost_function_sa(bitstring_array):
        # SA tối thiểu hóa, Max-Cut tối đa hóa, nên ta dùng dấu trừ
        return -evaluate_solution("".join(map(str, bitstring_array)), edges)

    temp_schedule = np.geomspace(10.0, 0.01, n_iterations)
    
    start_time = time.time()
    current_solution = np.random.randint(0, 2, n_nodes)
    current_cost = cost_function_sa(current_solution)
    best_solution, best_cost = current_solution, current_cost
    
    for temp in temp_schedule:
        neighbor = np.copy(current_solution)
        flip_idx = np.random.randint(0, n_nodes)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        neighbor_cost = cost_function_sa(neighbor)
        
        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temp):
            current_solution, current_cost = neighbor, neighbor_cost
        if current_cost < best_cost:
            best_solution, best_cost = current_solution, current_cost
            
    end_time = time.time()
    
    best_bitstring = "".join(map(str, best_solution))
    value_found = evaluate_solution(best_bitstring, edges)
    return value_found, end_time - start_time, best_bitstring

# ==============================================================================
# PHƯƠNG PHÁP 3: CLASSICAL SOLVER (PULP)
# ==============================================================================
def solve_with_classical():
    print("\n--- Bắt đầu giải bằng Classical Solver (PuLP) ---")
    start_time = time.time()
    
    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)
    # x_i = 1 nếu node i ở nhóm 1, 0 nếu ở nhóm 0
    x = pulp.LpVariable.dicts("x", range(n_nodes), cat='Binary')
    # y_ij = 1 nếu cạnh (i,j) bị cắt
    y = pulp.LpVariable.dicts("y", edges, cat='Binary')
    
    # Hàm mục tiêu
    prob += pulp.lpSum(y)
    
    # Ràng buộc
    for i, j in edges:
        prob += y[(i, j)] <= x[i] + x[j]
        prob += y[(i, j)] <= 2 - (x[i] + x[j])
        
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    end_time = time.time()
    
    best_bitstring = "".join([str(int(x[i].varValue)) for i in range(n_nodes)])
    value_found = evaluate_solution(best_bitstring, edges)
    return value_found, end_time - start_time, best_bitstring

# ==============================================================================
# CHẠY SO SÁNH VÀ TỔNG HỢP KẾT QUẢ
# ==============================================================================
# Chạy Classical Solver để tìm giá trị tối ưu tuyệt đối
optimal_value, classical_time, classical_solution = solve_with_classical()

# Chạy các thuật toán heuristic nhiều lần để đánh giá độ tin cậy
N_RUNS = 20
qaoa_results = []
sa_results = []
print(f"\n--- Chạy {N_RUNS} lần cho SA và QAOA để lấy thống kê ---")
for i in range(N_RUNS):
    print(f"  Run {i+1}/{N_RUNS}...")
    qaoa_val, qaoa_t, _ = solve_with_qaoa(p=3, epochs=100, lr=0.1)
    sa_val, sa_t, _ = solve_with_sa(n_iterations=1000) # Giảm iter để chạy nhanh hơn
    qaoa_results.append({"value": qaoa_val, "time": qaoa_t})
    sa_results.append({"value": sa_val, "time": sa_t})

# --- Tổng hợp Metrics ---
results = {
    "Classical (PuLP)": {
        "Value Found": optimal_value,
        "Execution Time (s)": classical_time,
        "Approximation Ratio": optimal_value / optimal_value,
        "Prob. of Finding Opt. (%)": 100.0
    },
    "Quantum-Inspired (SA)": {
        "Value Found": np.mean([r["value"] for r in sa_results]),
        "Execution Time (s)": np.mean([r["time"] for r in sa_results]),
        "Approximation Ratio": np.mean([r["value"] for r in sa_results]) / optimal_value,
        "Prob. of Finding Opt. (%)": 100 * np.sum(np.array([r["value"] for r in sa_results]) == optimal_value) / N_RUNS
    },
    "Hybrid (QAOA)": {
        "Value Found": np.mean([r["value"] for r in qaoa_results]),
        "Execution Time (s)": np.mean([r["time"] for r in qaoa_results]),
        "Approximation Ratio": np.mean([r["value"] for r in qaoa_results]) / optimal_value,
        "Prob. of Finding Opt. (%)": 100 * np.sum(np.array([r["value"] for r in qaoa_results]) == optimal_value) / N_RUNS
    }
}

# --- Hiển thị Bảng Kết quả ---
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results = df_results[["Value Found", "Approximation Ratio", "Prob. of Finding Opt. (%)", "Execution Time (s)"]]

print("\n\n" + "="*80)
print("BẢNG SO SÁNH KẾT QUẢ CUỐI CÙNG")
print("="*80)
print(df_results.to_string(formatters={
    'Value Found': '{:.2f}'.format,
    'Approximation Ratio': '{:.3f}'.format,
    'Prob. of Finding Opt. (%)': '{:.1f}'.format,
    'Execution Time (s)': '{:.4f}'.format
}))
print("="*80)
