import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time

# --- THƯ VIỆN BỔ SUNG ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

print("Bắt đầu so sánh toàn diện: Classical vs. Quantum Classifiers...")
torch.manual_seed(42)

# ==============================================================================
# BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
data = load_breast_cancer(); X, y = data.data, data.target
y = y * 2 - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
N_QUBITS = 4
pca = PCA(n_components=N_QUBITS); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)
print(f"Dữ liệu đã được xử lý. Số chiều sau PCA: {N_QUBITS}")

# ==============================================================================
# ĐỊNH NGHĨA CÁC MÔ HÌNH
# ==============================================================================
def train_classical_svm(X_train, y_train):
    svm = SVC(kernel='rbf', C=1.0, gamma='scale'); svm.fit(X_train, y_train); return svm

class ClassicalMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, x): return self.net(x)

dev_qsvm = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev_qsvm)
def qsvm_kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))

def qsvm_kernel_matrix(A, B):
    matrix = np.zeros((len(A), len(B)))
    for i, x1 in enumerate(A):
        for j, x2 in enumerate(B):
            matrix[i, j] = qsvm_kernel_circuit(x1, x2)[0]
    return matrix

dev_qnn = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev_qnn, interface="torch", diff_method="backprop")
def qnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

class HybridQNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.q_params = nn.Parameter(0.1 * torch.randn(2, n_qubits, 3))
        # Thêm một lớp cổ điển để xử lý đầu ra, giúp ổn định và mạnh hơn
        self.classical_layer = nn.Linear(1, 1)

    def forward(self, x):
        # SỬA LỖI Ở ĐÂY
        # 1. Tạo list kết quả từ qnode
        results_list = [qnn_circuit(inp, self.q_params) for inp in x]
        
        # 2. Dùng torch.stack để gộp lại, bảo toàn gradient
        q_out = torch.stack(results_list).view(-1, 1)
        
        # 3. Đưa qua lớp cổ điển để có logit cuối cùng
        return self.classical_layer(q_out.float())


# ==============================================================================
# HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ==============================================================================
results = {}

def train_torch_model(model, X_train, y_train, epochs=100):
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            # Chuyển y từ {-1,1} về {0,1} cho loss
            loss = loss_fn(output, (target + 1) / 2)
            loss.backward()
            optimizer.step()
    return model

# --- Main Evaluation Loop ---
models_to_test = {
    "Classical SVM (RBF)": train_classical_svm,
    "Classical MLP": ClassicalMLP(X_train_pca.shape[1]),
    "QSVM": "precomputed",
    "Hybrid QNN": HybridQNN(N_QUBITS)
}

for name, model_or_func in models_to_test.items():
    print(f"\n--- Đang xử lý: {name} ---")
    start_time = time.time()
    
    if name == "Classical SVM (RBF)":
        model = model_or_func(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        params = "N/A"
    elif name == "Classical MLP":
        model = train_torch_model(model_or_func, X_train_pca, y_train)
        with torch.no_grad():
            logits = model(torch.tensor(X_test_pca, dtype=torch.float32))
            y_pred = torch.sign(logits).numpy().flatten()
        params = sum(p.numel() for p in model.parameters())
    elif name == "QSVM":
        kernel_train = qsvm_kernel_matrix(X_train_pca, X_train_pca)
        svm = SVC(kernel='precomputed').fit(kernel_train, y_train)
        kernel_test = qsvm_kernel_matrix(X_test_pca, X_train_pca)
        y_pred = svm.predict(kernel_test)
        params = "N/A (Kernel-based)"
    elif name == "Hybrid QNN":
        model = train_torch_model(model_or_func, X_train_pca, y_train)
        with torch.no_grad():
            logits = model(torch.tensor(X_test_pca, dtype=torch.float32))
            y_pred = torch.sign(logits).numpy().flatten()
        params = sum(p.numel() for p in model.parameters())
        
    end_time = time.time()
    
    # Ép kiểu y_test để tính metric
    y_test_numeric = np.array(y_test, dtype=float)
    y_pred_numeric = np.array(y_pred, dtype=float)

    results[name] = {
        "Accuracy": accuracy_score(y_test_numeric, y_pred_numeric),
        "Precision": precision_score(y_test_numeric, y_pred_numeric),
        "Recall": recall_score(y_test_numeric, y_pred_numeric),
        "F1-Score": f1_score(y_test_numeric, y_pred_numeric),
        "Train Time (s)": end_time - start_time,
        "Parameters": params,
        "Confusion Matrix": confusion_matrix(y_test_numeric, y_pred_numeric).ravel()
    }
# ==============================================================================
# HIỂN THỊ KẾT QUẢ
# ==============================================================================
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results = df_results[["F1-Score", "Accuracy", "Precision", "Recall", "Parameters", "Train Time (s)", "Confusion Matrix"]]

print("\n\n" + "="*80)
print("BẢNG SO SÁNH KẾT QUẢ CUỐI CÙNG")
print("="*80)
print(df_results.to_string(formatters={
    'F1-Score': '{:.4f}'.format, 'Accuracy': '{:.4f}'.format,
    'Precision': '{:.4f}'.format, 'Recall': '{:.4f}'.format,
    'Train Time (s)': '{:.2f}'.format
}))
print("\n*Chú thích Ma trận Nhầm lẫn: (TN, FP, FN, TP)")
print("="*80)
