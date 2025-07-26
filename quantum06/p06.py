import pennylane as qml
from pennylane import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

print("Bắt đầu Bài toán 7: Quantum Support Vector Machine (QSVM)...")

if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Bước 1: Chuẩn bị Dữ liệu ---
n_samples = 100
X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.1, random_state=42)
# Chuyển nhãn 0 thành -1, theo yêu cầu của một số công thức SVM
y[y == 0] = -1

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Đã chuẩn bị {len(X_train)} mẫu huấn luyện và {len(X_test)} mẫu kiểm thử.")

# --- Bước 2: Thiết kế Hàm nhân Lượng tử (Quantum Kernel) ---
n_qubits = 2 # Số qubit bằng số chiều dữ liệu
dev = qml.device("default.qubit", wires=n_qubits)

# Định nghĩa một mạch mã hóa đặc trưng (feature map)
# Đây là mạch sẽ ánh xạ dữ liệu cổ điển vào không gian lượng tử
def feature_map(x):
    # Sử dụng một feature map phổ biến: ZZFeatureMap
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    for i in range(n_qubits):
        qml.RZ(2 * x[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.RZ(2 * (np.pi - x[0]) * (np.pi - x[1]), wires=1)
    qml.CNOT(wires=[0, 1])

# PennyLane cung cấp một lớp tiện ích qml.kernels để xây dựng kernel
# Nó tự động xử lý việc tính toán |<ψ(x_j)|ψ(x_i)>|^2
@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel_matrix(A, B):
    """Tính toán ma trận kernel giữa hai bộ dữ liệu A và B."""
    matrix = np.zeros((len(A), len(B)))
    for i, x1 in enumerate(A):
        for j, x2 in enumerate(B):
            # kernel_circuit trả về xác suất của các trạng thái cơ sở.
            # Phần tử đầu tiên (probs[0]) chính là |<0|U(x2)† U(x1)|0>|^2,
            # tương đương với |<ψ(x2)|ψ(x1)>|^2
            probs = kernel_circuit(x1, x2)
            matrix[i, j] = probs[0]
    return matrix

# --- Bước 3: Huấn luyện SVM với Kernel Lượng tử ---
print("\nĐang tính toán ma trận kernel lượng tử cho tập huấn luyện...")
# Tính ma trận kernel K(x_i, x_j) cho tất cả các cặp điểm trong tập huấn luyện
kernel_train = quantum_kernel_matrix(X_train, X_train)

print("Đang huấn luyện mô hình SVM...")
# Khởi tạo một SVM và chỉ định rằng chúng ta sẽ cung cấp một kernel đã được tính toán trước
# `kernel='precomputed'`
svm = SVC(kernel='precomputed')
svm.fit(kernel_train, y_train)

# --- Bước 4: Đánh giá mô hình ---
print("\nĐang tính toán ma trận kernel lượng tử cho tập kiểm thử...")
# Tính ma trận kernel giữa tập test và tập train (là các support vector)
# K(x_test, x_train)
kernel_test = quantum_kernel_matrix(X_test, X_train)

print("Đang dự đoán trên tập kiểm thử...")
y_pred = svm.predict(kernel_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Độ chính xác của QSVM trên tập kiểm thử: {accuracy * 100:.2f}%")

# --- Bước 5: Trực quan hóa ranh giới quyết định ---
print("Đang vẽ ranh giới quyết định...")
# Tạo một lưới các điểm để vẽ
h = 0.05
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Dự đoán trên từng điểm của lưới
grid_points = np.c_[xx.ravel(), yy.ravel()]
kernel_grid = quantum_kernel_matrix(grid_points, X_train)
Z = svm.predict(kernel_grid)
Z = Z.reshape(xx.shape)

# Vẽ biểu đồ
plt.figure(figsize=(8, 8))
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['red', 'blue']
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(cmap_bold), edgecolor='k', s=50, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(cmap_bold), marker='^', edgecolor='k', s=70, label='Test')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"QSVM Decision Boundary (Accuracy: {accuracy*100:.2f}%)")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.legend()
figure_path = "figures/qsvm_decision_boundary.png"
plt.savefig(figure_path)
print(f"Biểu đồ đã được lưu tại: {figure_path}")
plt.close()
