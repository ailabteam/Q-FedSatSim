import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3)

def prepare_state(phi):
    qml.RY(phi, wires=0)

def teleportation_circuit(phi, m0, m1):
    prepare_state(phi)

    # EPR pair between qubit 1 (Alice) and 2 (Bob)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])

    # Bell measurement
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

    # Apply Pauli gates according to classical bits m0, m1
    if m1 == 1:
        qml.PauliX(wires=2)
    if m0 == 1:
        qml.PauliZ(wires=2)

@qml.qnode(dev)
def teleportation(phi, m0, m1):
    teleportation_circuit(phi, m0, m1)
    return qml.state()

def get_target_state(phi):
    return np.array([np.cos(phi/2), np.sin(phi/2)])

def fidelity_output(phi, m0, m1):
    state = teleportation(phi, m0, m1)
    # Lấy reduced state tại wire 2
    bob_state = state[-2:]
    target = get_target_state(phi)
    fidelity = np.abs(np.dot(np.conj(target), bob_state)) ** 2
    return fidelity

# Test all possible (m0, m1)
phi = np.pi / 4
print(f"Target state: RY({phi:.2f})")

for m0 in [0, 1]:
    for m1 in [0, 1]:
        fid = fidelity_output(phi, m0, m1)
        print(f"m0 = {m0}, m1 = {m1} => Fidelity: {fid:.4f}")

