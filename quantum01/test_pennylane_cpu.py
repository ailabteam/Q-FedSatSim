import pennylane as qml
from pennylane import numpy as np

print(f"PennyLane version: {qml.__version__}")

# Khởi tạo device CPU với 1 qubit
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

print("Kết quả circuit trên CPU:", circuit())

