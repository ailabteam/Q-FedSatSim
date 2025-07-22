import torch
import neal
import dimod
import numpy as np
import time

# Chọn GPU cuda:1 nếu có
torch_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using torch device: {torch_device}")

# === Config
N = 4  # Số quân hậu
USE_LEAP = False  # Khi có token D-Wave thì đổi lại thành True

# === Tạo BQM cho bài toán N-Queens
def build_n_queens_bqm(N):
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    # Mỗi hàng phải có đúng 1 quân hậu
    for i in range(N):
        row_vars = [f"x_{i}_{j}" for j in range(N)]
        for var in row_vars:
            bqm.add_variable(var, -2.0)
        for j in range(N):
            for k in range(j+1, N):
                bqm.add_interaction(row_vars[j], row_vars[k], 2.0)

    # Mỗi cột phải có đúng 1 quân hậu
    for j in range(N):
        col_vars = [f"x_{i}_{j}" for i in range(N)]
        for i in range(N):
            for k in range(i+1, N):
                bqm.add_interaction(col_vars[i], col_vars[k], 2.0)

    # Ràng buộc đường chéo (main + phụ)
    for i in range(N):
        for j in range(N):
            var1 = f"x_{i}_{j}"
            for k in range(1, N):
                if i+k < N and j+k < N:
                    var2 = f"x_{i+k}_{j+k}"
                    bqm.add_interaction(var1, var2, 2.0)
                if i+k < N and j-k >= 0:
                    var2 = f"x_{i+k}_{j-k}"
                    bqm.add_interaction(var1, var2, 2.0)
    return bqm

# === In kết quả ma trận
def print_solution(sample, N):
    board = [["." for _ in range(N)] for _ in range(N)]
    for var, val in sample.items():
        if val == 1:
            _, i, j = var.split("_")
            board[int(i)][int(j)] = "Q"
    for row in board:
        print(" ".join(row))

# === Main
def main():
    bqm = build_n_queens_bqm(N)

    if USE_LEAP:
        # Sẽ bật khi bạn có API token của D-Wave
        from dwave.system import LeapHybridSampler
        sampler = LeapHybridSampler()
    else:
        # Tạm dùng neal (Simulated Annealing)
        sampler = neal.SimulatedAnnealingSampler()

    start_time = time.time()
    sampleset = sampler.sample(bqm, num_reads=100)
    elapsed_time = time.time() - start_time

    best_sample = sampleset.first.sample
    energy = sampleset.first.energy

    print("=== N-Queens Example ===")
    print_solution(best_sample, N)
    print(f"Energy: {energy}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    main()

