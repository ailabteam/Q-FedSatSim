import pennylane as qml
from pennylane import numpy as np

# Device mô phỏng với 3 qubit (0,1: Alice; 2: Bob)
dev = qml.device("default.qubit", wires=3)

def apply_entanglement():
    """Tạo cặp EPR (Bell state) giữa Alice (wire 1) và Bob (wire 2)."""
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])

def encode_psi(theta):
    """Mã hóa trạng thái đầu vào psi = RY(theta)|0> lên qubit 0 (Alice)."""
    qml.RY(theta, wires=0)

def apply_teleportation():
    """Thực hiện thao tác teleportation: CNOT + H + đo + gửi kết quả."""
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    qml.measure(wires=0)
    qml.measure(wires=1)

@qml.qnode(dev)
def teleportation_circuit(theta):
    """Toàn bộ quá trình teleportation."""
    encode_psi(theta)
    apply_entanglement()
    apply_teleportation()
    return qml.probs(wires=[0, 1])

def get_measurement_result(theta):
    """Trích xuất kết quả đo ngẫu nhiên (m0, m1)."""
    probs = teleportation_circuit(theta)
    outcomes = np.random.choice(4, p=probs)
    m0 = (outcomes // 2) % 2
    m1 = outcomes % 2
    return m0, m1

@qml.qnode(dev)
def bob_correction(theta, m0, m1):
    """Bob thực hiện hiệu chỉnh để lấy lại trạng thái."""
    encode_psi(theta)
    apply_entanglement()
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    if m0 == 1:
        qml.PauliZ(wires=2)
    if m1 == 1:
        qml.PauliX(wires=2)
    return qml.state()

def fidelity_with_target(theta, state):
    """So sánh fidelity giữa trạng thái Bob nhận được và trạng thái gốc."""
    psi_target = np.array([
        np.cos(theta / 2),
        0,
        0,
        0,
        0,
        0,
        0,
        np.sin(theta / 2)
    ])
    return np.abs(np.vdot(psi_target, state)) ** 2

# MAIN
if __name__ == "__main__":
    theta = 0.79  # hoặc dùng np.pi / 4
    print(f"Target state: RY({theta:.2f})")

    for m0 in [0, 1]:
        for m1 in [0, 1]:
            state = bob_correction(theta, m0, m1)
            fid = fidelity_with_target(theta, state)
            print(f"m0 = {m0}, m1 = {m1} => Fidelity: {fid:.4f}")

