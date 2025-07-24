import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit():
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    qml.StatePrep(state, wires=0)  # Sử dụng StatePrep thay cho QubitStateVector
    return qml.expval(qml.PauliX(0))

print("Expectation PauliX trên |+>: ", circuit())

