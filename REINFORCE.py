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


def run_reinforce(seed=0):
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

        optimizer.zero_grad() 
        
        R = 0
        policy_loss = 0
        t_all = 0
        
        # iterate reward reversely
        # global gamma
        for t in reversed(range(len(episode_data['rewards']))):
            r = episode_data['rewards'][t]
            log_prob = episode_data['log_probs'][t]
            
            R = r + gamma * R
            policy_loss += -log_prob * (gamma ** t_all) * R 
            t_all += 1

            policy_loss.backward(retain_graph=True)
            policy_loss = 0
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
    csv_path = './results/reinforce_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))