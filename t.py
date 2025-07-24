import pennylane as qml

dev = qml.device("lightning.gpu", wires=1)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

print("Expectation value:", circuit())

