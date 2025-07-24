import pennylane as qml
import torch

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, interface="torch")
def circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))

x = torch.tensor(0.5)
print("Output with Torch interface:", circuit(x))

