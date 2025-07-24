import pennylane as qml
from pennylane import numpy as np

# Device với 3 qubit
dev = qml.device("default.qubit", wires=3)

# Hàm chuẩn bị trạng thái bất kỳ (có thể chỉnh sửa)
def prepare_psi():
    qml.RY(np.pi / 4, wires=0)  # rotate trạng thái |0> thành trạng thái superposition

# Quantum Teleportation Circuit
@qml.qnode(dev)
def teleportation_circuit():
    # Step 1: chuẩn bị trạng thái ψ trên qubit 0
    prepare_psi()

    # Step 2: tạo EPR cặp giữa qubit 1 và 2
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])

    # Step 3: ghép qubit 0 (ψ) với EPR cặp và đo (Alice)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

    # Step 4: đo Alice qubit (classical bits)
    m0 = qml.measure(wires=0)
    m1 = qml.measure(wires=1)

    # Step 5: dùng kết quả đo để điều chỉnh Bob's qubit (qubit 2)
    qml.cond(m1, qml.PauliX)(wires=2)
    qml.cond(m0, qml.PauliZ)(wires=2)

    # Trả về trạng thái density matrix tại qubit Bob
    return qml.density_matrix(wires=2)

# Kiểm tra teleportation thành công không
@qml.qnode(dev)
def reference_state():
    prepare_psi()
    return qml.density_matrix(wires=0)

if __name__ == "__main__":
    rho_teleported = teleportation_circuit()
    rho_reference = reference_state()

    # So sánh fidelity
    fidelity = np.real(np.trace(np.dot(rho_reference.conj().T, rho_teleported)))

    print("📦 Trạng thái gốc (qubit 0):")
    print(np.round(rho_reference, 3))

    print("\n🚀 Trạng thái sau teleport (qubit 2):")
    print(np.round(rho_teleported, 3))

    print(f"\n✅ Fidelity: {fidelity:.4f}")

