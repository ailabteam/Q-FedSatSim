import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from collections import deque

# --- Thiết lập ---
torch.manual_seed(42); np.random.seed(42)
print("Bắt đầu Bài toán 6 (Thuật toán Actor-Critic)...")
if not os.path.exists("figures"): os.makedirs("figures")

# --- Các tham số ---
n_qubits = 4
embedding_dim = 8 # Tăng embedding dim một chút
n_actions = 4
q_depth = 2
gamma = 0.99
lr_actor = 0.005  # Learning rate riêng
lr_critic = 0.01
epochs = 1500 # Giảm epochs vì Actor-Critic thường học nhanh hơn

# --- Môi trường ---
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
n_states = env.observation_space.n

# --- Mạch Lượng tử (giữ nguyên) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_processor(params, inputs):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Actor: Chính sách Lượng tử Lai ---
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Embedding(n_states, embedding_dim)
        self.linear_compress = nn.Linear(embedding_dim, n_qubits) # Tầng nén
        self.q_params = nn.Parameter(0.1 * torch.randn(q_depth, n_qubits, 3))
        self.action_head = nn.Linear(n_qubits, n_actions)

    def forward(self, state):
        embedded_state = self.state_encoder(state)
        compressed_state = torch.tanh(self.linear_compress(embedded_state)) # Thêm hàm kích hoạt
        q_in = compressed_state * np.pi / 2.0 # Scale về [-pi/2, pi/2]
        
        q_out_list = quantum_processor(self.q_params.double(), q_in.double())
        q_out = torch.stack(q_out_list, dim=-1).float()
        
        action_logits = self.action_head(q_out)
        return torch.softmax(action_logits, dim=-1)

# --- Critic: Mạng Cổ điển Đánh giá Trạng thái ---
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_states, 32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output là một giá trị V(s)
        )
    def forward(self, state):
        return self.net(state)

# --- Vòng lặp Huấn luyện Actor-Critic ---
actor = Actor()
critic = Critic()
opt_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
opt_critic = optim.AdamW(critic.parameters(), lr=lr_critic)
reward_history = deque(maxlen=100) # Lưu 100 phần thưởng gần nhất
avg_reward_history = []

print("\nBắt đầu huấn luyện Actor-Critic...")
for i in range(epochs):
    state_int, info = env.reset(seed=i)
    total_reward = 0
    
    while True:
        state_tensor = torch.tensor(state_int)
        
        # Actor quyết định hành động
        action_probs = actor(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        
        # Môi trường phản hồi
        next_state_int, reward, terminated, truncated, info = env.step(action)
        
        # Critic đánh giá
        state_value = critic(state_tensor)
        next_state_tensor = torch.tensor(next_state_int)
        with torch.no_grad(): # Không cần tính grad cho target
            next_state_value = critic(next_state_tensor)
        
        # Tính Advantage và target cho Critic
        if terminated or truncated:
            target_value = torch.tensor([reward])
            advantage = target_value - state_value
        else:
            target_value = reward + gamma * next_state_value
            advantage = target_value - state_value

        # Cập nhật Actor
        log_prob = torch.log(action_probs[action])
        actor_loss = -log_prob * advantage.detach() # detach() để advantage không ảnh hưởng đến grad của critic
        
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()
        
        # Cập nhật Critic
        critic_loss = nn.MSELoss()(state_value, target_value)
        
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
        
        state_int = next_state_int
        total_reward += reward
        
        if terminated or truncated:
            break
            
    reward_history.append(total_reward)
    avg_reward = sum(reward_history) / len(reward_history)
    avg_reward_history.append(avg_reward)
    
    if (i + 1) % 50 == 0:
        print(f"Epoch {i+1}/{epochs}, Average Reward (last 100): {avg_reward:.3f}")

print("\nHuấn luyện hoàn tất!")
env.close()

# --- Trực quan hóa ---
plt.figure(figsize=(12, 6))
plt.plot(avg_reward_history, color='red')
plt.title("Training Progress (Actor-Critic Agent)")
plt.xlabel("Episode")
plt.ylabel("Average Reward (over last 100 episodes)")
plt.grid(True)
figure_path = "figures/qrl_frozenlake_actor_critic.png"
plt.savefig(figure_path)
print(f"Biểu đồ huấn luyện đã được lưu tại: {figure_path}")
plt.close()
