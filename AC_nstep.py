import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
from collections import deque

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

gamma = 0.99
n_steps = [1, 5, 10, 20]
lr_actor = 1e-4
lr_critic = 0.001
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

# Critic 
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


def train_actor_critic(n_step, seed=0):
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

        returns = compute_returns(rewards, dones, gamma, n_step)
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


if __name__ == "__main__":
    results = {}
    for n_step in n_steps:
        print(n_step)
        all_rewards = []
        for run in range(NUM_RUNS):
            rewards = train_actor_critic(n_step, seed=run)
            all_rewards.append(rewards)
        avg_reward = np.nanmean(all_rewards, axis=0)
        std_reward = np.nanstd(all_rewards, axis=0)
        results[n_step] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
        }
    
    dfs = []
    for n_step in n_steps:
        df = pd.DataFrame({
            'n_step': [n_step] * max_episodes,
            'episode': np.arange(max_episodes),
            'avg_reward': results[n_step]['avg_reward'],
            'std_reward': results[n_step]['std_reward'],
        })
        dfs.append(df)

    final_df = pd.concat(dfs)
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/ac_n_step_results.csv'
    final_df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(final_df['avg_reward'].agg(['mean', 'max']))