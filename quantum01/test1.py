import dimod

sampler = dimod.SimulatedAnnealingSampler()

Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
sampleset = sampler.sample_qubo(Q, num_reads=10)

print(sampleset.first)

