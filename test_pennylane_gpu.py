import pennylane as qml
from pennylane import numpy as np
import torch

print("Torch CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

device_name = "default.qubit.torch"
wires = 1

dev = qml.device(device_name, wires=wires)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

result = circuit()
print(f"Expectation value of PauliZ on Hadamard state: {result:.4f}")

