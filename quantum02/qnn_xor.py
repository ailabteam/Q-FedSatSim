import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt

# 1. Dữ liệu XOR, ép kiểu float64 (double precision)
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float64)

Y = torch.tensor([0, 1, 1, 0], dtype=torch.float64)

# Vẽ và lưu hình dữ liệu
plt.scatter(X[:, 0], X[:, 1], c=['blue' if y == 0 else 'red' for y in Y])
plt.title("XOR Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.savefig("xor_dataset.png")
plt.close()
print("✔️ Đã lưu hình ảnh dữ liệu XOR vào xor_dataset.png")

# 2. Thiết lập quantum device
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# 3. Định nghĩa quantum circuit (QNode)
@qml.qnode(dev, interface="torch")
def quantum_circuit(x, weights):
    # Encode input với rotation RY (pi * x)
    for i in range(n_qubits):
        qml.RY(torch.pi * x[i], wires=i)

    # Entanglement
    qml.CNOT(wires=[0, 1])

    # Tham số lượng tử - các rotation RY theo weights
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    # Output expectation value của PauliZ trên qubit 0
    return qml.expval(qml.PauliZ(0))

# 4. Định nghĩa mô hình QNN kết hợp torch.nn.Module
class QuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Khởi tạo tham số lượng tử với dtype=float64
        init_weights = 0.01 * torch.randn(n_qubits, dtype=torch.float64)
        self.q_weights = nn.Parameter(init_weights)

    def forward(self, x):
        batch_outputs = []
        for i in range(x.shape[0]):
            batch_outputs.append(quantum_circuit(x[i], self.q_weights))
        return torch.stack(batch_outputs)

# 5. Hàm huấn luyện
def train(model, X, Y, epochs=300, lr=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()  # dùng BCELoss nên output phải sigmoid

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)

        # Đưa output về [0,1] với sigmoid
        outputs = torch.sigmoid(outputs)

        loss = loss_fn(outputs, Y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    return model

# 6. Đánh giá
def evaluate(model, X, Y):
    with torch.no_grad():
        outputs = model(X)
        preds = torch.sigmoid(outputs) > 0.5
        correct = (preds.squeeze() == Y).sum().item()
        accuracy = correct / len(Y)
        print("\nDự đoán:")
        for i in range(len(X)):
            print(f"Input: {X[i].tolist()}, Predicted: {int(preds[i])}, True: {int(Y[i])}")
        print(f"\nĐộ chính xác: {accuracy*100:.2f}%")

# 7. Chạy toàn bộ
if __name__ == "__main__":
    model = QuantumNet()
    model = train(model, X, Y, epochs=300, lr=0.1)
    evaluate(model, X, Y)

