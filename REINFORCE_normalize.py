import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

fixed_lr = 1e-4
gamma = 0.99
hidden_dim = 128
max_episodes = 2000
NUM_RUNS = 5


class PolicyNetwork(nn.Module):
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
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move state to the selected device
        with torch.no_grad():  
            probs = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), probs, action


def run_reinforce(seed=0):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)  # Move model to device
    optimizer = optim.Adam(policy.parameters(), lr=fixed_lr)
    scores = []
    steps_per_episode = []
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        episode_data = {'states': [], 'actions': [], 'probs': [], 'rewards': []}
        done = False
        steps = 0

        # Store trajectory (states, actions, probs, rewards)
        while not done:
            action, probs, action_tensor = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data['states'].append(state)
            episode_data['actions'].append(action_tensor)
            episode_data['probs'].append(probs)
            episode_data['rewards'].append(reward)
            state = next_state
            steps += 1

        # Compute returns (R_t) for each timestep
        returns = []
        rewards = episode_data['rewards']
        for t in range(len(rewards)):
            R = sum(gamma ** (k - t) * rewards[k] for k in range(t, len(rewards)))
            returns.append(R)

        # Normalize returns
        returns = np.array(returns)
        ret_mean = returns.mean()
        ret_std = returns.std()
        normalized_returns = (returns - ret_mean) / (ret_std + 1e-8)

        scores.append(sum(rewards))

        # Update the policy step-by-step
        for t in range(len(returns)):
            # Dynamic log_prob calculation
            state_t = torch.FloatTensor(episode_data['states'][t]).unsqueeze(0).to(device)  # Move state to device
            probs_t = policy.forward(state_t)
            dist_t = torch.distributions.Categorical(probs_t)
            log_prob_t = dist_t.log_prob(episode_data['actions'][t])

            R = normalized_returns[t]
            loss = -log_prob_t * (gamma ** t) * R

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {sum(rewards):.1f}')

    return scores, steps_per_episode


if __name__ == "__main__":
    all_scores = []
    all_steps = []

    for run in range(NUM_RUNS):
        scores, steps = run_reinforce(seed=run)
        all_scores.append(scores)
        all_steps.append(steps)

    # Pad or truncate all_runs to the same length
    max_len = max(len(run) for run in all_scores)  # Find the maximum length across runs

    # Ensure all runs have the same length by padding with NaN or truncating
    all_scores = [run + [np.nan] * (max_len - len(run)) for run in all_scores]
    all_steps = [run + [0] * (max_len - len(run)) for run in all_steps]

    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)

    # Calculate cumulative steps
    cum_steps = np.cumsum(all_steps[0])

    df = pd.DataFrame({
        'gamma': [gamma] * max_len,
        'episode': np.arange(max_len),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'cum_steps': cum_steps
    })

    os.makedirs('./results', exist_ok=True)
    csv_path = './results/reinforce_normalize_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))