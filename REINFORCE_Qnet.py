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


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_reinforce(seed=0):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=fixed_lr)
    q_network = QNetwork(state_dim, action_dim, hidden_dim)
    q_optimizer = optim.Adam(q_network.parameters(), lr=fixed_lr)
    loss_fn = nn.MSELoss()
    scores = []
    steps_per_episode = []
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        episode_data = {'log_probs': [], 'rewards': [], 'states': [], 'actions': [], 'next_states': [], 'dones': []}
        done = False
        steps = 0

        while not done:
            action, log_prob = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data['log_probs'].append(log_prob)
            episode_data['rewards'].append(reward)
            episode_data['states'].append(state)
            episode_data['actions'].append(action)
            episode_data['next_states'].append(next_state)
            episode_data['dones'].append(done)
            state = next_state
            steps += 1

        scores.append(sum(episode_data['rewards']))
        steps_per_episode.append(steps)

        states = torch.FloatTensor(episode_data['states'])
        actions = torch.LongTensor(episode_data['actions']).unsqueeze(1)
        rewards = torch.FloatTensor(episode_data['rewards'])
        next_states = torch.FloatTensor(episode_data['next_states'])
        dones = torch.FloatTensor(episode_data['dones'])

        with torch.no_grad():
            next_q_values = q_network(next_states).max(1)[0]
            targets = rewards + gamma * next_q_values * (1 - dones)

        q_values = q_network(states).gather(1, actions).squeeze()
        q_loss = loss_fn(q_values, targets)

        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        returns = q_values.detach()
        policy_loss = [-log_prob * R for log_prob, R in zip(episode_data['log_probs'], returns)]

        optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        optimizer.step()

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
    csv_path = './results/reinforce_Qnet_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))