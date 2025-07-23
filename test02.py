import numpy as np
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler

# ========================
# Step 1: Input
# ========================

num_tasks = 6
num_slots = 4
np.random.seed(42)

priority = np.random.randint(10, 100, size=num_tasks)
delay = np.random.randint(50, 200, size=num_tasks)  # delay in ms
slot_assignment = np.random.randint(0, num_slots, size=num_tasks)

T_max = 400  # Tổng độ trễ tối đa (ms)

print("Priority:", priority)
print("Delay:", delay)
print("Slot assignment:", slot_assignment)

# ========================
# Step 2: Biến nhị phân
# ========================

linear = {}
quadratic = {}
penalty_slot = 200  # phạt nếu conflict slot
penalty_delay = 1   # hệ số penalty cho vi phạm tổng delay

# Mục tiêu: maximize priority → minimize -priority
for i in range(num_tasks):
    linear[f'x{i}'] = -priority[i]

# Ràng buộc 1: Không conflict slot
slot_to_tasks = {}
for i in range(num_tasks):
    s = slot_assignment[i]
    slot_to_tasks.setdefault(s, []).append(i)

for task_list in slot_to_tasks.values():
    for i in range(len(task_list)):
        for j in range(i+1, len(task_list)):
            a, b = task_list[i], task_list[j]
            quadratic[(f'x{a}', f'x{b}')] = penalty_slot

# Ràng buộc 2: Tổng độ trễ ≤ T_max
# Ta thêm penalty nếu delay quá lớn: penalty_delay * (total_delay - T_max)^2
# Expand: sum_i sum_j x_i x_j * delay[i]*delay[j] - 2*T_max*sum x_i*delay[i]

# Phần x_i * delay[i] → linear
for i in range(num_tasks):
    linear[f'x{i}'] += penalty_delay * (delay[i]**2 - 2 * T_max * delay[i])

# Phần x_i x_j * delay[i] * delay[j] → quadratic
for i in range(num_tasks):
    for j in range(i+1, num_tasks):
        quadratic[(f'x{i}', f'x{j}')] = quadratic.get((f'x{i}', f'x{j}'), 0) + \
                                        2 * penalty_delay * delay[i] * delay[j]

# Offset hằng số (T_max^2)
offset = penalty_delay * (T_max**2)

# ========================
# Step 3: Solve BQM
# ========================

bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype='BINARY')
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)
best = sampleset.first.sample

# ========================
# Step 4: Output
# ========================

selected_tasks = [int(k[1:]) for k, v in best.items() if v == 1]
total_priority = sum(priority[i] for i in selected_tasks)
total_delay = sum(delay[i] for i in selected_tasks)

conflicts = any(
    sum(1 for i in selected_tasks if slot_assignment[i] == s) > 1
    for s in range(num_slots)
)

print("\n--- Result ---")
print("Selected tasks:", selected_tasks)
print("Total priority:", total_priority)
print("Total delay:", total_delay, "ms")
print("Conflict violation:", conflicts)
print("Delay violation:", total_delay > T_max)

