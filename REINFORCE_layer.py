import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

fixed_lr = 1e-4
gamma = 0.99
hidden_dims = [[32, 32], [64, 64], [128, 128], [64, 64, 64]]
max_episodes = 200
NUM_RUNS = 5


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))  # Create layer for each hidden dimension
            layers.append(nn.ReLU())  # Add ReLU activation function
            prev_dim = dim  # Update previous dimension for the next layer
        
        layers.append(nn.Linear(prev_dim, action_dim))  # Output layer to map to action space
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def run_reinforce(hidden_dim, seed=0):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=fixed_lr)
    scores = []
    steps_per_episode = []
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        episode_data = {'log_probs': [], 'rewards': []}
        done = False
        steps = 0

        while not done:
            action, log_prob = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data['log_probs'].append(log_prob)
            episode_data['rewards'].append(reward)
            state = next_state
            steps += 1

        scores.append(sum(episode_data['rewards']))
        steps_per_episode.append(steps)

        returns = []
        R = 0
        for r in reversed(episode_data['rewards']):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        
        # returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        policy_loss = [-log_prob * R for log_prob, R in zip(episode_data['log_probs'], returns)]

        optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f'LR {fixed_lr} - Episode {episode}, Reward: {scores[-1]:.1f}')

    return scores, steps_per_episode


if __name__ == "__main__":
    results = {}

    for hidden_dim in hidden_dims:
        all_scores = []
        all_steps = []
        print(str(hidden_dim))
        for run in range(NUM_RUNS):
            scores, steps = run_reinforce(hidden_dim, seed=run)
            all_scores.append(scores)
            all_steps.append(steps)
        
        results[str(hidden_dim)] = {
            'avg_reward': np.nanmean(all_scores, axis=0),
            'std_reward': np.nanstd(all_scores, axis=0),
            'cum_steps': np.cumsum(all_steps[0])
        }

    dfs = []
    for hidden_dim in hidden_dims:
        df = pd.DataFrame({
            'hidden_dim': [str(hidden_dim)] * max_episodes,
            'episode': np.arange(max_episodes),
            'avg_reward': results[str(hidden_dim)]['avg_reward'],
            'std_reward': results[str(hidden_dim)]['std_reward'],
            'cum_steps': results[str(hidden_dim)]['cum_steps']
        })
        dfs.append(df)

    final_df = pd.concat(dfs)
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/reinforce_hidden_dim_results.csv'
    final_df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(final_df.groupby('hidden_dim')['avg_reward'].agg(['mean', 'max']))