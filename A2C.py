import gymnasium as gym
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
gamma = 0.99
lr_actor = 1e-4
lr_critic = 1e-4 
hidden_dim = 128
max_episodes = 2000
NUM_RUNS = 5
n_steps = 5

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
        return action.item(), dist.log_prob(action)  # 移除熵

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

def compute_advantages_and_returns(rewards, values, dones, gamma=0.99, n_steps=5):
    advantages = []
    returns = []
    T = len(rewards)
    for t in range(T):
        R = 0
        for k in range(t, min(t + n_steps, T)):
            R += (gamma ** (k - t)) * rewards[k]
            if dones[k]:
                break
        if (t + n_steps < T) and not dones[t + n_steps]:
            R += (gamma ** n_steps) * values[t + n_steps]
        returns.append(R)
        advantages.append(R - values[t])
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)

def run_a2c(seed, clip=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, hidden_dim)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    rewards_all = []  # Make sure this stores the rewards for each run

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        states, rewards, log_probs, dones = [], [], [], []

        while not done:
            action, log_prob = actor.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            state = next_state

        # Store the rewards of the episode
        rewards_all.append(sum(rewards))  # Append total reward of this episode

        # 统一计算价值
        states_tensor = torch.FloatTensor(np.array(states))
        with torch.no_grad():
            values = critic(states_tensor).squeeze().numpy()

        # 计算优势
        advantages, returns = compute_advantages_and_returns(rewards, values, dones, gamma, n_steps)

        # 计算损失
        log_probs = torch.stack(log_probs)
        value_loss = F.mse_loss(returns, critic(states_tensor).squeeze())
        policy_loss = -(advantages * log_probs).mean()

        # 更新Actor
        optimizer_actor.zero_grad()
        policy_loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        optimizer_actor.step()

        # 更新Critic
        optimizer_critic.zero_grad()
        value_loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        optimizer_critic.step()

        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {sum(rewards):.1f}')

    return rewards_all 

if __name__ == "__main__":
    all_rewards = []
    for run in range(NUM_RUNS):
        rewards = run_a2c(seed=run, clip=False)
        all_rewards.append(rewards)
    all_rewards = np.array(all_rewards)
    avg_reward = np.nanmean(all_rewards, axis=0)
    std_reward = np.nanstd(all_rewards, axis=0)
    
    df = pd.DataFrame({
        'episode': np.arange(max_episodes),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
    })
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/a2c_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))