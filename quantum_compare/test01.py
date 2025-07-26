import pennylane as qml
# SỬA Ở ĐÂY: Dùng numpy gốc
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

# --- THƯ VIỆN BỔ SUNG ---
import pulp

print("Bắt đầu bài toán Lập lịch Tải dữ liệu Vệ tinh (Knapsack)...")
if not os.path.exists("figures"): os.makedirs("figures")

# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA BÀI TOÁN VÀ MÔ HÌNH HÓA
# ==============================================================================
values = np.array([70, 20, 15, 85, 30, 55, 65, 40])
weights = np.array([31, 10, 8, 42, 16, 25, 33, 22])
capacity = 100
n_items = len(values)
A = 1
B = max(values)

def build_qubo_hamiltonian(values, weights, capacity, A, B):
    # Các phép tính ở đây giờ dùng numpy gốc
    coeffs, obs = [], []
    for i in range(n_items):
        coeff_z_i = 0.5 * A * values[i] - B * weights[i] * capacity + 0.5 * B * weights[i]**2
        for j in range(n_items):
            if i != j:
                coeff_z_i += 0.5 * B * weights[i] * weights[j]
        coeffs.append(coeff_z_i)
        obs.append(qml.PauliZ(i))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            coeff_z_i_z_j = 0.5 * B * weights[i] * weights[j]
            coeffs.append(coeff_z_i_z_j)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    # Chuyển đổi coeffs thành list số float thông thường, an toàn nhất
    return qml.Hamiltonian(list(coeffs), obs)

cost_h = build_qubo_hamiltonian(values, weights, capacity, A, B)
print("Ising Hamiltonian cho bài toán Knapsack đã được tạo.")

def evaluate_solution(bitstring, values, weights, capacity):
    selected_items = [i for i, bit in enumerate(bitstring) if bit == '1']
    total_value = np.sum(values[selected_items])
    total_weight = np.sum(weights[selected_items])
    is_valid = total_weight <= capacity
    return total_value, total_weight, is_valid

# ==============================================================================
# PHƯƠNG PHÁP 1: HYBRID QUANTUM-CLASSICAL (QAOA)
# ==============================================================================
def solve_with_qaoa():
    print("\n--- Bắt đầu giải bằng QAOA ---")
    dev = qml.device("default.qubit", wires=n_items)
    p = 5
    
    mixer_h = qml.Hamiltonian([1]*n_items, [qml.PauliX(i) for i in range(n_items)])
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qaoa_circuit(params):
        gammas, betas = params[0], params[1]
        for i in range(n_items): qml.Hadamard(wires=i)
        for i in range(p):
            qml.exp(cost_h, -1j * gammas[i])
            qml.exp(mixer_h, -1j * betas[i])
        return qml.expval(cost_h)

    # params là torch.tensor, đây là phần duy nhất cần tính đạo hàm
    params = torch.tensor(np.random.uniform(0, np.pi, (2, p)), requires_grad=True, dtype=torch.float32)
    optimizer = optim.Adam([params], lr=0.05)
    epochs = 200
    
    start_time = time.time()
    print("  QAOA đang chạy... (quá trình này có thể mất vài phút)")
    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = qaoa_circuit(params)
        cost.backward()
        optimizer.step()
        if (epoch + 1) % 40 == 0:
            print(f"    QAOA Epoch {epoch+1}/{epochs}, Cost: {cost.item():.2f}")
    
    end_time = time.time()
    
    @qml.qnode(dev)
    def get_probs(params):
        gammas, betas = params[0], params[1]
        for i in range(n_items): qml.Hadamard(wires=i)
        for i in range(p):
            qml.exp(cost_h, -1j * gammas[i])
            qml.exp(mixer_h, -1j * betas[i])
        return qml.probs(wires=range(n_items))

    probs = get_probs(params.detach().numpy())
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_items}b')
    
    val, w, is_valid = evaluate_solution(best_bitstring, values, weights, capacity)
    print(f"QAOA Hoàn tất sau {end_time - start_time:.2f}s")
    print(f"Giải pháp: {best_bitstring}, Giá trị: {val}, Trọng lượng: {w} (Ràng buộc: {'OK' if is_valid else 'VI PHẠM!'})")
    
# ... (Phần code của SA và Classical Solver giữ nguyên) ...
# ==============================================================================
# PHƯƠ-NG PHÁP 2: QUANTUM-INSPIRED (SIMULATED ANNEALING)
# ==============================================================================
def solve_with_sa():
    print("\n--- Bắt đầu giải bằng Simulated Annealing ---")
    def cost_function_sa(bitstring_array):
        objective = -A * np.dot(values, bitstring_array)
        constraint_val = np.dot(weights, bitstring_array) - capacity
        penalty = B * (max(0, constraint_val))**2
        return objective + penalty
    n_iterations = 20000; temp_schedule = np.geomspace(50.0, 0.01, n_iterations)
    start_time = time.time()
    current_solution = np.random.randint(0, 2, n_items)
    current_cost = cost_function_sa(current_solution)
    best_solution, best_cost = current_solution, current_cost
    for temp in temp_schedule:
        neighbor = np.copy(current_solution)
        flip_idx = np.random.randint(0, n_items)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        neighbor_cost = cost_function_sa(neighbor)
        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temp):
            current_solution, current_cost = neighbor, neighbor_cost
        if current_cost < best_cost:
            best_solution, best_cost = current_solution, current_cost
    end_time = time.time()
    best_bitstring = "".join(map(str, best_solution))
    val, w, is_valid = evaluate_solution(best_bitstring, values, weights, capacity)
    print(f"SA Hoàn tất sau {end_time - start_time:.2f}s")
    print(f"Giải pháp: {best_bitstring}, Giá trị: {val}, Trọng lượng: {w} (Ràng buộc: {'OK' if is_valid else 'VI PHẠM!'})")

# ==============================================================================
# PHƯƠ-NG PHÁP 3: CLASSICAL SOLVER (PULP)
# ==============================================================================
def solve_with_classical():
    print("\n--- Bắt đầu giải bằng Classical Solver (PuLP) ---")
    start_time = time.time()
    prob = pulp.LpProblem("SatelliteKnapsack", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("item", range(n_items), cat='Binary')
    prob += pulp.lpSum([values[i] * x[i] for i in range(n_items)])
    prob += pulp.lpSum([weights[i] * x[i] for i in range(n_items)]) <= capacity
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    end_time = time.time()
    best_bitstring = "".join([str(int(x[i].varValue)) for i in range(n_items)])
    val, w, is_valid = evaluate_solution(best_bitstring, values, weights, capacity)
    print(f"Classical Solver Hoàn tất sau {end_time - start_time:.4f}s")
    print(f"Giải pháp TỐI ƯU: {best_bitstring}, Giá trị: {val}, Trọng lượng: {w} (Ràng buộc: {'OK' if is_valid else 'VI PHẠM!'})")

# ==============================================================================
# CHẠY TẤT CẢ
# ==============================================================================
solve_with_qaoa()
solve_with_sa()
solve_with_classical()
