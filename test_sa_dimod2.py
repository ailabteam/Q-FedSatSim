import dimod
import time

def test_sa_sampler():
    sampler = dimod.SimulatedAnnealingSampler()
    Q = {
        (0, 0): -1,
        (1, 1): -1,
        (0, 1): 2
    }
    start = time.time()
    sampleset = sampler.sample_qubo(Q, num_reads=1000)
    end = time.time()

    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    print("Best sample found:", best_sample)
    print("Energy of best sample:", best_energy)
    print(f"Elapsed time: {end - start:.4f} seconds")

if __name__ == "__main__":
    test_sa_sampler()

