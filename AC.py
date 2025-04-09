import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
# 设置环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
gamma = 0.99
n_steps = 5
lr_actor = 1e-4
lr_critic = 0.001
hidden_dim = 128
max_episodes = 2000
NUM_RUNS = 5

# 策略网络（Actor）—— 与 AC 保持一致结构
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# 值函数网络（Critic）
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 计算 n 步回报
def compute_returns(rewards, dones, gamma=0.99, n_steps=5):
    returns = np.zeros(len(rewards), dtype=np.float32)
    last_return = 0

    for t in reversed(range(len(rewards))):
        if t + n_steps < len(rewards):
            returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])
        else:
            returns[t] = rewards[t] + last_return * (1 - dones[t])
        last_return = returns[t]

    return torch.FloatTensor(returns)


# A2C 主训练函数
def run_a2c(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, hidden_dim)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    rewards_all = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        episode_data = []
        episode_rewards = []
        episode_steps = 0

        while not done:
            action, log_prob = actor.act(state)
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

        # 解包轨迹数据
        states, rewards, values, log_probs, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)

        # 计算回报
        returns = compute_returns(rewards, dones, gamma, n_steps)
        policy_loss = -(returns.detach() * torch.stack(log_probs)).mean()
        
        value_preds = critic(states).squeeze()
        value_loss = F.mse_loss(value_preds, returns)

        optimizer_actor.zero_grad()
        policy_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        value_loss.backward()
        optimizer_critic.step()

        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {rewards_all[-1]:.1f}')

    return rewards_all


# 可视化训练结果
if __name__ == "__main__":
    all_rewards = []
    for run in range(NUM_RUNS):
        rewards = run_a2c(seed=run)
        all_rewards.append(rewards)
    avg_reward = np.nanmean(all_rewards, axis=0)
    std_reward = np.nanstd(all_rewards, axis=0)
    
    df = pd.DataFrame({
        'episode': np.arange(max_episodes),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
    })
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/ac_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))