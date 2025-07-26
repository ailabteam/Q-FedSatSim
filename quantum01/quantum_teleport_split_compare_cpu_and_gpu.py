import pennylane as qml
from pennylane import numpy as np
import time
import torch

# Trạng thái đầu
initial_state = np.array([1/np.sqrt(1.2), 1/np.sqrt(3)], dtype=np.complex128)

# === SETUP 2 DEVICE: CPU and GPU ===
dev_cpu = qml.device('default.qubit', wires=3)
dev_gpu = qml.device('lightning.gpu', wires=3)  # Requires GPU

# === DEFINE CIRCUIT ===
@qml.qnode(dev_cpu)
def teleportation_cpu():
    # Encode input on qubit 0
    qml.QubitStateVector(initial_state, wires=0)

    # Entangle qubit 1 and 2 (EPR pair)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])

    # Bell measurement on qubit 0 and 1
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

    # Measure
    m0 = qml.measure(wires=0)
    m1 = qml.measure(wires=1)

    # Apply correction on qubit 2
    qml.ctrl(qml.PauliX, control=m1)(wires=2)
    qml.ctrl(qml.PauliZ, control=m0)(wires=2)

    return qml.density_matrix(wires=2)

@qml.qnode(dev_gpu)
def teleportation_gpu():
    qml.QubitStateVector(initial_state, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    m0 = qml.measure(wires=0)
    m1 = qml.measure(wires=1)
    qml.ctrl(qml.PauliX, control=m1)(wires=2)
    qml.ctrl(qml.PauliZ, control=m0)(wires=2)
    return qml.density_matrix(wires=2)

# === Measure Execution Time ===
start_cpu = time.time()
result_cpu = teleportation_cpu()
cpu_time = time.time() - start_cpu

start_gpu = time.time()
result_gpu = teleportation_gpu()
gpu_time = time.time() - start_gpu

# === Fidelity ===
def fidelity(rho, sigma):
    sqrt_rho = qml.math.linalg.sqrtm(rho)
    prod = qml.math.dot(sqrt_rho, qml.math.dot(sigma, sqrt_rho))
    return qml.math.real(qml.math.trace(qml.math.linalg.sqrtm(prod)))**2

target_dm = np.outer(initial_state, initial_state.conj())
fid_cpu = fidelity(target_dm, result_cpu)
fid_gpu = fidelity(target_dm, result_gpu)

# === Print result ===
print("✅ Fidelity CPU:", fid_cpu)
print("✅ Fidelity GPU:", fid_gpu)
print("⏱️  Time CPU: {:.6f}s".format(cpu_time))
print("⏱️  Time GPU: {:.6f}s".format(gpu_time))

if torch.cuda.is_available():
    print("📦 GPU Memory allocated: {:.2f} MB".format(torch.cuda.memory_allocated() / 1024 ** 2))
else:
    print("⚠️  GPU not available.")

