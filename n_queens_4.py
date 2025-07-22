import dimod
import time

def n_queens_example(N=4):
    Q = {}
    # Ràng buộc mỗi hàng có đúng 1 quân hậu
    for i in range(N):
        for j1 in range(N):
            for j2 in range(j1 + 1, N):
                Q[(i * N + j1, i * N + j2)] = Q.get((i * N + j1, i * N + j2), 0) + 2
        for j in range(N):
            Q[(i * N + j, i * N + j)] = Q.get((i * N + j, i * N + j), 0) - 2

    # Ràng buộc mỗi cột có đúng 1 quân hậu
    for j in range(N):
        for i1 in range(N):
            for i2 in range(i1 + 1, N):
                Q[(i1 * N + j, i2 * N + j)] = Q.get((i1 * N + j, i2 * N + j), 0) + 2
        for i in range(N):
            Q[(i * N + j, i * N + j)] = Q.get((i * N + j, i * N + j), 0) - 2

    sampler = dimod.SimulatedAnnealingSampler()

    start = time.time()
    sampleset = sampler.sample_qubo(Q, num_reads=100, num_sweeps=100)
    end = time.time()

    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    # Chuyển nghiệm sang dạng bàn cờ
    board = [["." for _ in range(N)] for _ in range(N)]
    for key, val in best_sample.items():
        if val == 1:
            row = key // N
            col = key % N
            board[row][col] = "Q"

    print("=== N-Queens Example ===")
    for row in board:
        print(" ".join(row))
    print("Energy:", best_energy)
    print(f"Elapsed time: {end - start:.4f} seconds")

if __name__ == "__main__":
    n_queens_example()

