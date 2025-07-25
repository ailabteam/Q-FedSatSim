:q
import numpy as np
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler

# ========================
# Step 1: Mô hình hóa input
# ========================

num_tasks = 6       # Số yêu cầu truyền thông (e.g., truyền dữ liệu giữa trạm mặt đất)
num_slots = 4       # Số slot khả dụng trên vệ tinh

np.random.seed(42)
priority = np.random.randint(10, 100, size=num_tasks)  # Mức độ ưu tiên của từng yêu cầu
slot_assignment = np.random.randint(0, num_slots, size=num_tasks)  # Gán mỗi yêu cầu vào 1 slot khả dụng

print("Priority:", priority)
print("Slot assignment:", slot_assignment)

# ========================
# Step 2: Tạo biến nhị phân
# ========================
# x_i = 1 nếu chọn task i, 0 nếu bỏ

linear = {}
quadratic = {}

# Hàm mục tiêu: Maximize tổng ưu tiên ⇒ Minimize -priority
for i in range(num_tasks):
    linear[f'x{i}'] = -priority[i]

# Ràng buộc 1: mỗi slot chỉ có tối đa 1 task
slot_to_tasks = {}
for i in range(num_tasks):
    s = slot_assignment[i]
    if s not in slot_to_tasks:
        slot_to_tasks[s] = []
    slot_to_tasks[s].append(i)

# Penalty nếu hai task cùng slot được chọn
penalty = 100  # phạt cao
for task_list in slot_to_tasks.values():
    for i in range(len(task_list)):
        for j in range(i+1, len(task_list)):
            a, b = task_list[i], task_list[j]
            quadratic[(f'x{a}', f'x{b}')] = penalty

# ========================
# Step 3: Tạo BQM và giải bằng SA
# ========================

bqm = BinaryQuadraticModel(linear, quadratic, 0.0, vartype='BINARY')
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)
best = sampleset.first.sample

# ========================
# Step 4: Hiển thị kết quả
# ========================

selected_tasks = [int(k[1:]) for k, v in best.items() if v == 1]
selected_priority = sum(priority[i] for i in selected_tasks)

print("\nSelected tasks:", selected_tasks)
print("Total priority:", selected_priority)
print("Conflicts (violations):", any(
    sum(1 for i in selected_tasks if slot_assignment[i] == s) > 1
    for s in range(num_slots)
))

