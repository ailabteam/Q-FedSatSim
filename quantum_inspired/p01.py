import numpy as np
import matplotlib.pyplot as plt
import os
import time

print("Bắt đầu bài toán Cân bằng tải Vệ tinh với Simulated Annealing (SA)...")
if not os.path.exists("figures"): os.makedirs("figures")

# --- Bước 1: Định nghĩa Bài toán và Hàm Cost ---
data_packets = np.array([10.0, 20.0, 30.0, 40.0])
n_vars = len(data_packets)

def calculate_cost(spins):
    """Tính cost cổ điển (chênh lệch bình phương) cho một cấu hình spin."""
    sum1 = np.sum(data_packets[spins == 1])
    sum2 = np.sum(data_packets[spins == -1])
    return (sum1 - sum2)**2

# --- Bước 2: Thuật toán Simulated Annealing ---
def simulated_annealing(cost_function, n_vars, n_iterations, temp_schedule):
    """
    Tối ưu hóa một hàm cost bằng Simulated Annealing.
    """
    # Khởi tạo một giải pháp ngẫu nhiên
    current_solution = np.random.choice([-1, 1], size=n_vars)
    current_cost = cost_function(current_solution)
    
    best_solution = np.copy(current_solution)
    best_cost = current_cost
    
    cost_history = []
    
    # Vòng lặp "ủ"
    for i, temp in enumerate(temp_schedule):
        # Tạo một giải pháp lân cận bằng cách lật ngẫu nhiên một spin
        neighbor_solution = np.copy(current_solution)
        flip_idx = np.random.randint(0, n_vars)
        neighbor_solution[flip_idx] *= -1
        
        neighbor_cost = cost_function(neighbor_solution)
        
        # Tính toán sự thay đổi cost
        cost_diff = neighbor_cost - current_cost
        
        # Quy tắc chấp nhận Metropolis
        # Chấp nhận nếu giải pháp mới tốt hơn, hoặc
        # chấp nhận một giải pháp tồi hơn với một xác suất nhất định
        if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temp):
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            
        # Cập nhật giải pháp tốt nhất đã từng thấy
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
            
        cost_history.append(current_cost)
        
    return best_solution, best_cost, cost_history

# --- Bước 3: Chạy Thuật toán ---
# Siêu tham số cho SA
n_iterations = 5000
# Lịch trình nhiệt độ: bắt đầu "nóng", kết thúc "lạnh"
temp_schedule = np.geomspace(5000, 0.01, n_iterations)

print("\nBắt đầu quá trình SA...")
start_time = time.time()
best_solution, best_cost, cost_history = simulated_annealing(
    calculate_cost, n_vars, n_iterations, temp_schedule
)
end_time = time.time()
print(f"SA hoàn tất sau {end_time - start_time:.4f} giây.")

# --- Bước 4: Diễn giải Kết quả ---
print(f"\nGiải pháp tối ưu nhất được tìm thấy (chuỗi spin): {best_solution}")
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
plt.title("Simulated Annealing for Satellite Load Balancing")
plt.xlabel("Iteration")
plt.ylabel("Current Cost (Squared Difference)")
plt.grid(True)
figure_path = "figures/sa_load_balancing.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ hội tụ đã được lưu tại: {figure_path}")
plt.close()
