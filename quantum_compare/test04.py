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
import networkx as nx

print("Bắt đầu so sánh 3 phương pháp cho bài toán Max-Cut 10-đỉnh...")
if not os.path.exists("figures"): os.makedirs("figures")

# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA BÀI TOÁN (NÂNG CẤP)
# ==============================================================================
n_nodes = 10
# Tạo một đồ thị ngẫu nhiên nhưng có thể tái tạo (reproducible)
G = nx.erdos_renyi_graph(n=n_nodes, p=0.6, seed=42)
edges = list(G.edges)
print(f"Đã tạo đồ thị ngẫu nhiên với {n_nodes} đỉnh và {len(edges)} cạnh.")

# Vẽ và lưu đồ thị để trực quan hóa
plt.figure(figsize=(6,6))
nx.draw(G, with_labels=True, font_weight='bold', node_color='skyblue', node_size=700)
plt.title("Đồ thị Max-Cut 10-đỉnh")
plt.savefig("figures/maxcut_graph_10_nodes.png")
plt.close()

def evaluate_solution(bitstring, edges):
    """Tính số cạnh bị cắt cho một chuỗi bit."""
    cut_size = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cut_size += 1
    return cut_size

# --- Xây dựng Ising Hamiltonian cho Max-Cut ---
cost_h = qml.Hamiltonian(
    [1] * len(edges), 
    [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in edges]
)
print("Ising Hamiltonian cho Max-Cut đã được tạo.")

# ==============================================================================
# PHƯƠNG PHÁP 1: HYBRID QUANTUM-CLASSICAL (QAOA)
# ==============================================================================
def solve_with_qaoa(p=6, epochs=250, lr=0.05):
    print("\n--- Bắt đầu giải bằng QAOA (sẽ rất chậm) ---")
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
    print("  QAOA đang huấn luyện...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = qaoa_circuit(params)
        cost.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"    QAOA Epoch {epoch+1}/{epochs}, Cost: {cost.item():.2f}")
    
    end_time = time.time()
    
    @qml.qnode(dev)
    def get_probs(params):
        gammas, betas = params[0], params[1]
        for i in range(n_nodes): qml.Hadamard(wires=i)
        for i in range(p):
            qml.exp(cost_h, -1j * gammas[i])
            qml.exp(mixer_h, -1j * betas[i])
        return qml.probs(wires=range(n_nodes))

    print("  QAOA đang tính xác suất cuối cùng...")
    probs = get_probs(params.detach().numpy())
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_nodes}b')
    
    value_found = evaluate_solution(best_bitstring, edges)
    return value_found, end_time - start_time, best_bitstring

# ==============================================================================
# PHƯƠNG PHÁP 2: QUANTUM-INSPIRED (SIMULATED ANNEALING)
# ==============================================================================
def solve_with_sa(n_iterations=100000): # Tăng iter cho bài toán khó hơn
    print("\n--- Bắt đầu giải bằng Simulated Annealing ---")
    
    def cost_function_sa(bitstring_array):
        return -evaluate_solution("".join(map(str, bitstring_array)), edges)

    temp_schedule = np.geomspace(20.0, 0.01, n_iterations)
    
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
    
    prob = pulp.LpProblem("MaxCut_10_Nodes", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n_nodes), cat='Binary')
    y = pulp.LpVariable.dicts("y", edges, cat='Binary')
    prob += pulp.lpSum(y)
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
# Do QAOA rất chậm, chúng ta chỉ chạy mỗi phương pháp 1 LẦN
# Trong paper, bạn có thể chạy SA nhiều lần và báo cáo kết quả trung bình
# và chạy QAOA 1 lần như một ví dụ minh họa.
all_results = {}

# --- Chạy Classical ---
val_c, time_c, sol_c = solve_with_classical()
optimal_value = val_c
all_results["Classical (PuLP)"] = {"Value Found": val_c, "Time (s)": time_c, "Solution": sol_c}

# --- Chạy SA ---
val_sa, time_sa, sol_sa = solve_with_sa()
all_results["Quantum-Inspired (SA)"] = {"Value Found": val_sa, "Time (s)": time_sa, "Solution": sol_sa}

# --- Chạy QAOA ---
val_qaoa, time_qaoa, sol_qaoa = solve_with_qaoa()
all_results["Hybrid (QAOA)"] = {"Value Found": val_qaoa, "Time (s)": time_qaoa, "Solution": sol_qaoa}

# --- Tổng hợp Metrics ---
for method, res in all_results.items():
    res["Approximation Ratio"] = res["Value Found"] / optimal_value

# --- Hiển thị Bảng Kết quả ---
df_results = pd.DataFrame.from_dict(all_results, orient='index')
df_results = df_results[["Value Found", "Approximation Ratio", "Time (s)", "Solution"]]

print("\n\n" + "="*80)
print(f"BẢNG SO SÁNH KẾT QUẢ CUỐI CÙNG (Max-Cut {n_nodes} đỉnh)")
print(f"(Giá trị tối ưu tuyệt đối: {optimal_value})")
print("="*80)
print(df_results.to_string(formatters={
    'Approximation Ratio': '{:.3f}'.format,
    'Time (s)': '{:.4f}'.format
}))
print("="*80)
