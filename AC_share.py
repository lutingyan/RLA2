import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
torch.autograd.set_detect_anomaly(True)
gamma = 0.99
lr_actor = 1e-4
lr_critic = 0.001  # Critic higher
hidden_dim = 128 
max_episodes = 2000
NUM_RUNS = 5

# 创建共享网络部分
class SharedNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

# Actor 网络
class Actor(nn.Module):
    def __init__(self, shared_network, action_dim):
        super().__init__()
        self.shared_network = shared_network
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.shared_network(x)
        return F.softmax(self.fc_out(x), dim=-1)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# Critic 网络
class Critic(nn.Module):
    def __init__(self, shared_network):
        super().__init__()
        self.shared_network = shared_network
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared_network(x)
        return self.fc_out(x)

def compute_returns(rewards, dones, gamma=0.99, n_steps=10):
    returns = np.zeros(len(rewards), dtype=np.float32)
    last_return = 0

    for t in reversed(range(len(rewards))):
        if t + n_steps < len(rewards):
            returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])
        else:
            returns[t] = rewards[t] + last_return * (1 - dones[t])
        last_return = returns[t]

    return torch.FloatTensor(returns)


# Actor-Critic 主函数
def train_actor_critic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建共享网络
    shared_network = SharedNetwork(state_dim, hidden_dim)

    # 创建 Actor 和 Critic
    actor = Actor(shared_network, action_dim)
    critic = Critic(shared_network)

    # 优化器参数分组
    optimizer = optim.Adam([
        {'params': shared_network.parameters(), 'lr': lr_critic},
        {'params': actor.fc_out.parameters(), 'lr': lr_actor},
        {'params': critic.fc_out.parameters(), 'lr': lr_critic}
    ])

    rewards_all = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        episode_data = []
        episode_rewards = []
        episode_steps = 0

        while not done:
            action, log_prob = actor.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = critic(state_tensor).item()
            episode_data.append((state, reward, value, log_prob, done))
            episode_rewards.append(reward)
            state = next_state
            episode_steps += 1

        total_reward = sum(episode_rewards)
        rewards_all.append(total_reward)

        states, rewards, values, log_probs, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        returns = compute_returns(rewards, dones, gamma)
        policy_loss = -(returns.detach() * torch.stack(log_probs)).mean()
            
        value_preds = critic(states).squeeze()
        value_loss = F.mse_loss(value_preds, returns)

        # 合并梯度更新
        optimizer.zero_grad()
        value_loss.backward()
        policy_loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {rewards_all[-1]:.1f}')

    return rewards_all

if __name__ == "__main__":
    all_rewards = []
    for run in range(NUM_RUNS):
        rewards = train_actor_critic(seed=run)
        all_rewards.append(rewards)
    avg_reward = np.nanmean(all_rewards, axis=0)
    std_reward = np.nanstd(all_rewards, axis=0)
    
    df = pd.DataFrame({
        'episode': np.arange(max_episodes),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
    })
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/ac_share_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))