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

gamma = 0.99
lr_actor = 1e-4
lrs_critic = [0.001, 1e-4, 5e-4, 1e-5]    # Critic higher
hidden_dim = 128 
max_episodes = 2000
NUM_RUNS = 5

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

    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

# Actor-Critic 主函数
def train_actor_critic(lr_critic, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, hidden_dim)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    rewards_all = []

    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0

        while not done:
            action, log_prob = actor.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            reward_tensor = torch.tensor([reward], dtype=torch.float)

            target = reward_tensor + gamma * critic(next_state_tensor) * (1 - int(done))
            value = critic(state_tensor)
            td_error = target - value

            # 更新 Critic
            critic_loss = td_error.pow(2)
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # 更新 Actor
            actor_loss = -log_prob * td_error.detach()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            state = next_state
            total_reward += reward

        rewards_all.append(total_reward)
        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {rewards_all[-1]:.1f}')

    return rewards_all

if __name__ == "__main__":
    results = {}
    for lr_critic in lrs_critic:
        all_rewards = []
        for run in range(NUM_RUNS):
            rewards = train_actor_critic(lr_critic, seed=run)
            all_rewards.append(rewards)
        avg_reward = np.nanmean(all_rewards, axis=0)
        std_reward = np.nanstd(all_rewards, axis=0)
        
        results[lr_critic] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
        }


    dfs = []
    for lr in lrs_critic:
        df = pd.DataFrame({
            'lr': [lr] * max_episodes,
            'episode': np.arange(max_episodes),
            'avg_reward': results[lr]['avg_reward'],
            'std_reward': results[lr]['std_reward'],
        })
        dfs.append(df)

    final_df = pd.concat(dfs)
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/ac_lr_results.csv'
    final_df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(final_df['avg_reward'].agg(['mean', 'max']))