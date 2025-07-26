import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# --- Thiết lập ---
torch.manual_seed(42)
np.random.seed(42)
print("Bắt đầu Bài toán 6: Quantum Reinforcement Learning (QRL) trên FrozenLake...")
if not os.path.exists("figures"):
    os.makedirs("figures")

# --- Các tham số ---
n_qubits = 4
n_actions = 4
n_states = 16
q_depth = 2
gamma = 0.99
lr = 0.01
epochs = 2000

# --- Môi trường FrozenLake ---
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

# --- Thiết kế Chính sách Lượng tử (Quantum Policy) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(params, state):
    qml.BasisEmbedding(state, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_actions)]

class QuantumPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_params = nn.Parameter(0.1 * torch.randn(q_depth, n_qubits, 3, dtype=torch.float64))

    def forward(self, state):
        state_binary = np.binary_repr(state, width=n_qubits)
        state_array = [int(x) for x in state_binary]
        
        # SỬA LỖI Ở ĐÂY:
        # 1. Mạch lượng tử trả về một list các tensor có grad_fn
        logits_list = quantum_circuit(self.q_params, state_array)
        
        # 2. Dùng torch.stack để gộp chúng lại, bảo toàn grad_fn
        return torch.stack(logits_list).float()

# --- Vòng lặp Huấn luyện ---
policy = QuantumPolicy()
optimizer = optim.Adam(policy.parameters(), lr=lr)
reward_history = []
print("\nBắt đầu huấn luyện tác nhân lượng tử...")

for i in range(epochs):
    state, info = env.reset(seed=i) # Thêm seed để có thể tái tạo
    episode_rewards = []
    log_action_probs = []
    
    while True:
        action_logits = policy(state)
        action_probs = torch.softmax(action_logits, dim=0)
        action = torch.multinomial(action_probs, 1).item()
        
        log_action_probs.append(torch.log(action_probs[action]))
        
        state, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)
        
        if terminated or truncated:
            break
            
    returns = []
    discounted_reward = 0
    for r in reversed(episode_rewards):
        discounted_reward = r + gamma * discounted_reward
        returns.insert(0, discounted_reward)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    policy_loss = []
    for log_prob, R in zip(log_action_probs, returns):
        policy_loss.append(-log_prob * R)
        
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
    
    total_reward = sum(episode_rewards)
    reward_history.append(total_reward)
    
    if (i + 1) % 100 == 0:
        avg_reward = sum(reward_history[-100:]) / 100
        print(f"Epoch {i+1}/{epochs}, Average Reward (last 100): {avg_reward:.2f}")

print("\nHuấn luyện hoàn tất!")
env.close()

# --- Trực quan hóa kết quả ---
plt.figure(figsize=(12, 6))
moving_avg = [np.mean(reward_history[max(0, i-100):i+1]) for i in range(len(reward_history))]
plt.plot(reward_history, label='Total Reward per Episode', alpha=0.3)
plt.plot(moving_avg, label='Moving Average (100 episodes)', color='red')
plt.title("Training Progress of Quantum Agent on FrozenLake")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
figure_path = "figures/qrl_frozenlake.png"
plt.savefig(figure_path)
print(f"Biểu đồ huấn luyện đã được lưu tại: {figure_path}")
plt.close()
