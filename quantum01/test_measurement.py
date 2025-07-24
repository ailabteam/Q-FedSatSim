import pennylane as qml
from pennylane import numpy as np

# Device analytic mode (shots=None) để lấy state và probs
dev_analytic = qml.device("default.qubit", wires=1, shots=None)

@qml.qnode(dev_analytic)
def circuit_state_probs():
    qml.Hadamard(wires=0)
    return qml.probs(wires=0), qml.state()

probs, state = circuit_state_probs()
print("Probability (analytic):", probs)
print("Quantum state (analytic):", state)

# Device với shots để sample
dev_sample = qml.device("default.qubit", wires=1, shots=1000)

@qml.qnode(dev_sample)
def circuit_sample():
    qml.Hadamard(wires=0)
    return qml.sample(wires=0)

samples = circuit_sample()
print("Sampled measurements:", samples)

