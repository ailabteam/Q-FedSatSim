import numpy as np
import matplotlib.pyplot as plt
import os
import time

print("Bắt đầu bài toán Cân bằng tải Vệ tinh với Quantum-Inspired Evolutionary Algorithm (QIEA)...")

if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Bước 1: Định nghĩa Bài toán và Hàm Cost ---
data_packets = np.array([10.0, 20.0, 30.0, 40.0])
n_vars = len(data_packets)

def calculate_cost(spins):
    """Tính cost cổ điển (chênh lệch bình phương) cho một cấu hình spin."""
    sum1 = np.sum(data_packets[spins == 1])
    sum2 = np.sum(data_packets[spins == -1])
    return (sum1 - sum2)**2

# --- Bước 2: Thuật toán QIEA ---
def qiea(cost_function, n_vars, pop_size, n_generations, rotation_angle):
    """
    Tối ưu hóa một hàm cost bằng QIEA.
    pop_size: Kích thước quần thể.
    rotation_angle: Góc quay delta_theta, kiểm soát tốc độ hội tụ.
    """
    # 1. Khởi tạo quần thể Q-bit ở trạng thái chồng chập đều
    # Shape: (pop_size, n_vars, 2) -> (alpha, beta)
    q_population = np.full((pop_size, n_vars, 2), 1 / np.sqrt(2))
    
    # Lưu trữ giải pháp tốt nhất toàn cục
    best_overall_solution = None
    best_overall_cost = float('inf')
    cost_history = []

    # 2. Vòng lặp qua các thế hệ
    for gen in range(n_generations):
        # 2a. Đo lường: Tạo ra các giải pháp cổ điển từ quần thể Q-bit
        classical_population = np.zeros((pop_size, n_vars))
        probs_alpha = q_population[:, :, 0]**2
        
        # Lấy mẫu ngẫu nhiên dựa trên xác suất
        rands = np.random.rand(pop_size, n_vars)
        classical_population[rands < probs_alpha] = 1
        classical_population[rands >= probs_alpha] = -1

        # 2b. Đánh giá
        costs = np.array([cost_function(ind) for ind in classical_population])
        
        # 2c. Lưu trữ giải pháp tốt nhất
        current_best_idx = np.argmin(costs)
        current_best_cost = costs[current_best_idx]
        current_best_solution = classical_population[current_best_idx]
        
        if current_best_cost < best_overall_cost:
            best_overall_cost = current_best_cost
            best_overall_solution = current_best_solution
            
        cost_history.append(best_overall_cost)

        # 2d. Cập nhật quần thể Q-bit
        # Tất cả các cá thể sẽ được "kéo" về phía giải pháp tốt nhất của thế hệ này
        target_solution = current_best_solution
        
        # Tạo ma trận quay
        # Quay về phía |0> (spin +1)
        rot_plus = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                             [np.sin(rotation_angle), np.cos(rotation_angle)]])
        # Quay về phía |1> (spin -1)
        rot_minus = np.array([[np.cos(-rotation_angle), -np.sin(-rotation_angle)],
                              [np.sin(-rotation_angle), np.cos(-rotation_angle)]])

        for i in range(pop_size):
            for j in range(n_vars):
                if target_solution[j] == 1: # Nếu bit mục tiêu là +1
                    q_population[i, j, :] = rot_plus @ q_population[i, j, :]
                else: # Nếu bit mục tiêu là -1
                    q_population[i, j, :] = rot_minus @ q_population[i, j, :]
                    
    return best_overall_solution, best_overall_cost, cost_history

# --- Bước 3: Chạy Thuật toán ---
# Siêu tham số cho QIEA
pop_size = 20
n_generations = 100
rotation_angle = 0.05 * np.pi # Một góc quay nhỏ để tránh hội tụ quá sớm

print("\nBắt đầu quá trình QIEA...")
start_time = time.time()
best_solution, best_cost, cost_history = qiea(
    calculate_cost, n_vars, pop_size, n_generations, rotation_angle
)
end_time = time.time()
print(f"QIEA hoàn tất sau {end_time - start_time:.4f} giây.")

# --- Bước 4: Diễn giải Kết quả ---
print(f"\nGiải pháp tối ưu nhất được tìm thấy (chuỗi spin): {best_solution.astype(int)}")
print(f"Cost tối thiểu tìm được (chênh lệch bình phương): {best_cost:.4f}")

set_A, set_B = [], []
sum_A, sum_B = 0, 0
for i in range(n_vars):
    if best_solution[i] == 1:
        set_A.append(data_packets[i])
        sum_A += data_packets[i]
    else:
        set_B.append(data_packets[i])
        sum_B += data_packets[i]
        
print(f"\nPhân chia tối ưu:")
print(f"  - Vệ tinh A nhận các gói: {set_A} (Tổng: {sum_A:.1f})")
print(f"  - Vệ tinh B nhận các gói: {set_B} (Tổng: {sum_B:.1f})")

# --- Trực quan hóa ---
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.title("QIEA for Satellite Load Balancing")
plt.xlabel("Generation")
plt.ylabel("Best Cost Found (Squared Difference)")
plt.grid(True)
figure_path = "figures/qiea_load_balancing.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ hội tụ đã được lưu tại: {figure_path}")
plt.close()
