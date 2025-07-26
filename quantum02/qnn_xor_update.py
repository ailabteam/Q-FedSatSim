import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt

# 1. Dữ liệu XOR (float64)
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float64)

Y = torch.tensor([0, 1, 1, 0], dtype=torch.float64)

# Vẽ và lưu dữ liệu
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
depth = 5
dev = qml.device("default.qubit", wires=n_qubits)

# 3. QNode quantum circuit với BasicEntanglerLayers
@qml.qnode(dev, interface="torch")
def quantum_circuit(x, weights):
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# 4. Mạng Hybrid Quantum-Classical
class HybridQuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Khởi tạo weights trong [0, 2pi]
        init_weights = torch.rand((depth, n_qubits), dtype=torch.float64) * 2 * torch.pi
        self.q_weights = nn.Parameter(init_weights)
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        q_out = []
        for i in range(x.shape[0]):
            q_out.append(quantum_circuit(x[i], self.q_weights))
        q_out = torch.stack(q_out).unsqueeze(1)
        return torch.sigmoid(self.fc(q_out)).squeeze()

# 5. Huấn luyện
def train(model, X, Y, epochs=500, lr=0.05):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, Y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# 6. Đánh giá
def evaluate(model, X, Y):
    with torch.no_grad():
        outputs = model(X)
        preds = outputs > 0.5
        correct = (preds == Y).sum().item()
        accuracy = correct / len(Y)
        print("\nDự đoán:")
        for i in range(len(X)):
            print(f"Input: {X[i].tolist()}, Predicted: {int(preds[i])}, True: {int(Y[i])}")
        print(f"\nĐộ chính xác: {accuracy*100:.2f}%")

# 7. Chạy chính
if __name__ == "__main__":
    model = HybridQuantumNet()
    train(model, X, Y, epochs=500, lr=0.05)
    evaluate(model, X, Y)

