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
hidden_dim = 128
max_episodes = 2000
NUM_RUNS = 5
minibatch_size = 32

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
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def run_reinforce(seed=0):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=fixed_lr)
    scores = []
    steps_per_episode = []
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    episode_data_batch = []

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

        episode_data_batch.append(episode_data)

        if len(episode_data_batch) >= minibatch_size:
            all_log_probs = []
            all_returns = []

            for ep_data in episode_data_batch:
                returns = []
                R = 0
                for r in reversed(ep_data['rewards']):
                    R = r + gamma * R
                    returns.insert(0, R)

                all_log_probs.extend(ep_data['log_probs'])
                all_returns.extend(returns)

            all_log_probs = torch.stack(all_log_probs)
            all_returns = torch.tensor(all_returns)

            # Normalize returns (advantage)
            returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-7)

            policy_loss = -all_log_probs * returns

            optimizer.zero_grad()
            policy_loss.sum().backward()
            optimizer.step()

            episode_data_batch = []  # Reset the batch after each update

        scores.append(sum(episode_data['rewards']))
        steps_per_episode.append(steps)

        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {scores[-1]:.1f}')

    return scores, steps_per_episode



if __name__ == "__main__":
    all_scores = []
    all_steps = []

    for run in range(NUM_RUNS):
        scores, steps = run_reinforce(seed=run)
        all_scores.append(scores)
        all_steps.append(steps)

    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)
    cum_steps = np.cumsum(all_steps[0])

    df = pd.DataFrame({
        'gamma': [gamma] * max_episodes,
        'episode': np.arange(max_episodes),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'cum_steps': cum_steps
    })

    os.makedirs('./results', exist_ok=True)
    csv_path = './results/reinforce_minibatch_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))