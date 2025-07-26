import dimod

def test_sa_sampler():
    sampler = dimod.SimulatedAnnealingSampler()
    Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
    sampleset = sampler.sample_qubo(Q, num_reads=1000, beta_range=(0.1, 10.0), num_sweeps=500)

    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    print("Best sample found:", best_sample)
    print("Energy of best sample:", best_energy)

if __name__ == "__main__":
    test_sa_sampler()

