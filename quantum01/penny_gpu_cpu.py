import torch
import time

# Tạo trạng thái đầu: |ψ> = α|0⟩ + β|1⟩
def generate_random_qubit(device):
    theta = torch.rand(1, device=device) * torch.pi
    phi = torch.rand(1, device=device) * 2 * torch.pi
    state = torch.tensor([
        torch.cos(theta / 2),
        torch.exp(1j * phi) * torch.sin(theta / 2)
    ], device=device).reshape(2, 1)
    return state / torch.norm(state)

# Áp dụng cổng Hadamard
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))

# CNOT
CNOT = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.complex64)

def kronecker3(a, b, c):  # tensor product
    return torch.kron(torch.kron(a, b), c)

def teleport(device):
    # Tạo trạng thái gốc
    psi = generate_random_qubit(device)  # qubit 0
    init_state = torch.kron(torch.kron(psi, torch.tensor([[1.], [0.]], device=device)), torch.tensor([[1.], [0.]], device=device))

    # Cổng Hadamard lên qubit 1
    H1 = kronecker3(torch.eye(2, device=device), H.to(device), torch.eye(2, device=device))
    state1 = H1 @ init_state

    # CNOT giữa qubit 1 và 2
    CNOT_12 = kronecker3(torch.eye(2, device=device), CNOT.to(device))
    state2 = CNOT_12 @ state1

    # CNOT giữa qubit 0 và 1
    CNOT_01 = kronecker3(CNOT.to(device), torch.eye(2, device=device))
    state3 = CNOT_01 @ state2

    # Hadamard lên qubit 0
    H0 = kronecker3(H.to(device), torch.eye(4, device=device))
    final_state = H0 @ state3

    # Đo và chọn trạng thái qubit 2 dựa trên kết quả
    probs = torch.abs(final_state.squeeze()) ** 2
    index = torch.argmax(probs).item()
    bit0 = (index >> 2) & 1
    bit1 = (index >> 1) & 1

    # Tách qubit thứ 2
    teleport_state = final_state.reshape(2, 2, 2)[:, bit0, bit1]
    teleport_state = teleport_state / torch.norm(teleport_state)

    # So sánh fidelity
    fidelity = torch.abs(torch.dot(teleport_state.conj().flatten(), psi.flatten())) ** 2
    return fidelity.item()

# Chạy và đo thời gian
def run(device):
    start = time.time()
    fidelity = teleport(device)
    end = time.time()
    print(f"⚙️  Device: {device}")
    print(f"⏱️  Time: {end - start:.6f}s")
    print(f"✅ Fidelity: {fidelity:.4f}")
    print("")

if __name__ == "__main__":
    # CPU
    run(torch.device("cpu"))

    # GPU nếu có
    if torch.cuda.is_available():
        run(torch.device("cuda"))
    else:
        print("❌ GPU không khả dụng.")

