import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --- Thiết lập ---
torch.manual_seed(42) # Đặt seed để có kết quả lặp lại
np.random.seed(42)
torch.set_default_dtype(torch.float32)

print("Bắt đầu Bài toán 3: Quantum GAN (QGAN)...")

if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Các tham số ---
n_qubits = 2         # Số qubit = số chiều của dữ liệu chúng ta muốn sinh ra (2D)
q_depth = 2          # Số lớp trong mạch lượng tử
batch_size = 128
epochs = 5000
lr_g = 0.02          # Learning rate cho Generator
lr_d = 0.02          # Learning rate cho Discriminator

# --- Dữ liệu thật: Phân phối Gaussian 2D ---
def get_real_samples(n_samples):
    # Tạo dữ liệu từ phân phối chuẩn đa biến
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[0.5, 0.1], [0.1, 0.5]])
    dist = torch.distributions.MultivariateNormal(mean, cov)
    return dist.sample((n_samples,))

# --- Phần 1: Generator Lượng tử (Quantum Generator) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(noise, weights):
    # noise có shape (n_qubits,)
    # weights có shape (q_depth, n_qubits, 3)
    qml.AngleEmbedding(noise, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Trả về 2 giá trị kỳ vọng, tương ứng với điểm dữ liệu 2D
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class Generator(nn.Module):
    def __init__(self, n_qubits, q_depth):
        super().__init__()
        self.q_weights = nn.Parameter(0.1 * torch.randn(q_depth, n_qubits, 3))

    def forward(self, n_samples):
        # Tạo nhiễu đầu vào
        noise = torch.rand(n_samples, n_qubits) * torch.pi
        
        # Tạo dữ liệu giả theo từng mẫu
        fake_samples = []
        for n in noise:
            # Ép kiểu để tương thích với qnode
            q_out = quantum_circuit(n.double(), self.q_weights.double())
            fake_samples.append(torch.tensor(q_out))
            
        return torch.stack(fake_samples).float()

# --- Phần 2: Discriminator Cổ điển (Classical Discriminator) ---
class Discriminator(nn.Module):
    def __init__(self, input_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output là một logit duy nhất
        )
    def forward(self, x):
        return self.net(x)

# --- Khởi tạo và Huấn luyện ---
generator = Generator(n_qubits, q_depth)
discriminator = Discriminator()

# Optimizers
opt_g = optim.Adam(generator.parameters(), lr=lr_g)
opt_d = optim.Adam(discriminator.parameters(), lr=lr_d)

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

print("\nBắt đầu huấn luyện QGAN...")
start_time = time.time()
d_losses, g_losses = [], []

for epoch in range(epochs):
    # --- Giai đoạn 1: Huấn luyện Discriminator ---
    opt_d.zero_grad()
    
    # 1.1 Loss trên dữ liệu thật
    real_samples = get_real_samples(batch_size)
    d_real_output = discriminator(real_samples)
    loss_d_real = loss_fn(d_real_output, torch.ones_like(d_real_output))
    
    # 1.2 Loss trên dữ liệu giả
    # .detach() rất quan trọng: nó ngăn gradient lan truyền vào Generator
    fake_samples = generator(batch_size).detach()
    d_fake_output = discriminator(fake_samples)
    loss_d_fake = loss_fn(d_fake_output, torch.zeros_like(d_fake_output))
    
    # 1.3 Cập nhật Discriminator
    loss_d = (loss_d_real + loss_d_fake) / 2
    loss_d.backward()
    opt_d.step()
    
    # --- Giai đoạn 2: Huấn luyện Generator ---
    opt_g.zero_grad()
    
    # 2.1 Tạo dữ liệu giả và xem Discriminator phán xét thế nào
    fake_samples_for_g = generator(batch_size)
    d_output_on_fake = discriminator(fake_samples_for_g)
    
    # 2.2 Loss của Generator: lừa Discriminator (làm nó nghĩ đây là hàng thật)
    loss_g = loss_fn(d_output_on_fake, torch.ones_like(d_output_on_fake))
    
    # 2.3 Cập nhật Generator
    loss_g.backward()
    opt_g.step()

    # --- Ghi log ---
    d_losses.append(loss_d.item())
    g_losses.append(loss_g.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")


total_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất! Tổng thời gian: {total_time:.2f}s")


# --- Trực quan hóa kết quả ---
# 1. Trực quan hóa loss
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title("Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
figure_path = "figures/qgan_loss.png"
plt.savefig(figure_path)
print(f"Biểu đồ loss đã được lưu tại: {figure_path}")
plt.close()

# 2. Trực quan hóa phân phối dữ liệu
with torch.no_grad():
    final_fake_samples = generator(500).numpy()
final_real_samples = get_real_samples(500).numpy()

plt.figure(figsize=(8, 8))
plt.scatter(final_real_samples[:, 0], final_real_samples[:, 1], c='blue', alpha=0.5, label='Real Data')
plt.scatter(final_fake_samples[:, 0], final_fake_samples[:, 1], c='red', alpha=0.5, label='Generated Data (Quantum)')
plt.title("Real vs. Generated Data Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
figure_path = "figures/qgan_distribution.png"
plt.savefig(figure_path)
print(f"Biểu đồ phân phối đã được lưu tại: {figure_path}")
plt.close()
