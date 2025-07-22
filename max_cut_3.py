import dimod
import time

def max_cut_example():
    # Đồ thị đơn giản 3 đỉnh, cạnh trọng số 1
    edges = [(0, 1), (1, 2), (0, 2)]
    w = 1
    Q = {}
    for i, j in edges:
        Q[(i, i)] = Q.get((i, i), 0) - w
        Q[(j, j)] = Q.get((j, j), 0) - w
        Q[(i, j)] = Q.get((i, j), 0) + 2 * w

    sampler = dimod.SimulatedAnnealingSampler()

    start = time.time()
    sampleset = sampler.sample_qubo(Q, num_reads=100, num_sweeps=100)
    end = time.time()

    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    print("=== Max-Cut Example ===")
    print("Best partition:", best_sample)
    print("Energy:", best_energy)
    print(f"Elapsed time: {end - start:.4f} seconds")

if __name__ == "__main__":
    max_cut_example()

