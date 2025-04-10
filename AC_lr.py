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

    def act(self, state):
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

def compute_returns(rewards, dones, values, gamma=0.99, n_steps=20):
    returns = np.zeros(len(rewards), dtype=np.float32)
    T = len(rewards)
    for t in range(T):
        R = 0
        step_count = 0
        for k in range(t, min(t + n_steps, T)):
            R += (gamma ** step_count) * rewards[k]
            step_count += 1
            if dones[k]: 
                break
        if (k < T - 1) and not dones[k]:
            R += (gamma ** step_count) * values[k + 1]
        returns[t] = R
    return torch.FloatTensor(returns)

def run_ac(lr_critic, seed):
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

        # Collecting trajectory data
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Ensure state is on the correct device
            action, log_prob = actor.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            value = critic(state_tensor).item()
            episode_data.append((state, reward, value, log_prob, done))
            episode_rewards.append(reward)
            state = next_state
            episode_steps += 1

        total_reward = sum(episode_rewards)
        rewards_all.append(total_reward)

        states, rewards, values, log_probs, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute n-step returns
        returns = compute_returns(rewards, dones, values, gamma)

        # Compute the policy loss
        policy_loss = -(returns.detach() * torch.stack(log_probs)).mean()

        # Critic's loss (value function loss)
        value_preds = critic(states).squeeze()
        value_loss = F.mse_loss(value_preds, returns)

        # Once the entire episode is completed, backpropagate and optimize
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
    all_results = []  # Create a list to store all results
    for lr in lrs_critic:
        print(lr)
        all_rewards = []
        for run in range(NUM_RUNS):
            rewards = run_ac(lr_critic=lr, seed=run)
            all_rewards.append(rewards)
        avg_reward = np.nanmean(all_rewards, axis=0)
        std_reward = np.nanstd(all_rewards, axis=0)
        
        df = pd.DataFrame({
            'lr': [lr] * max_episodes,
            'episode': np.arange(max_episodes),
            'avg_reward': avg_reward,
            'std_reward': std_reward,
        })
        all_results.append(df)  # Append each result to the list

    # Concatenate all results from different n_steps into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)

    # Save all results to a single CSV file
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/ac_lr_results.csv'
    final_df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(final_df.groupby('nstep')['avg_reward'].agg(['mean', 'max']))