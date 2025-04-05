import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

gamma = 0.99
learning_rate = 1e-3  # ✅ 固定学习率
hidden_dim = 128
fc_layer_options = [1, 2, 3]  # ✅ 不同数量的全连接层
max_episodes = 2000
NUM_RUNS = 5


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_fc_layers):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_fc_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.model(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def run_reinforce(num_fc_layers):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, num_fc_layers)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    scores = []
    steps_per_episode = []

    for episode in range(max_episodes):
        state, _ = env.reset()
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

        # 计算returns
        returns = []
        R = 0
        for r in reversed(episode_data['rewards']):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        policy_loss = [-log_prob * R for log_prob, R in zip(episode_data['log_probs'], returns)]
        optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        optimizer.step()

        if episode % 100 == 0:
            reward = sum(episode_data['rewards'])
            print(f'Layers {num_fc_layers} - Episode {episode}, Reward: {reward:.1f}')

    return scores, steps_per_episode


if __name__ == "__main__":
    results = []

    for num_fc_layers in fc_layer_options:
        all_scores = []
        all_steps_per_episode = []

        for run in range(NUM_RUNS):
            scores, steps_per_episode = run_reinforce(num_fc_layers)
            all_scores.append(scores)
            all_steps_per_episode.append(steps_per_episode)

        all_scores = np.array(all_scores)
        all_steps_per_episode = np.array(all_steps_per_episode)

        average_per_step = np.nanmean(all_scores, axis=0)
        std_per_step = np.nanstd(all_scores, axis=0)
        cumulative_steps = np.cumsum(all_steps_per_episode[0])

        df = pd.DataFrame({
            'learning_rate': [f"{learning_rate:.0e}"] * len(average_per_step),
            'num_fc_layers': [num_fc_layers] * len(average_per_step),
            'episode': np.arange(len(average_per_step)),
            'average_reward': average_per_step,
            'std_reward': std_per_step,
            'cumulative_steps': cumulative_steps
        })
        results.append(df)

    final_df = pd.concat(results, ignore_index=True)

    os.makedirs('../results', exist_ok=True)
    csv_path = 'results/reinforce_fc_layer_ablation_fixed_lr.csv'
    final_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # 打印简要总结
    summary_df = final_df.groupby(['num_fc_layers']).agg({
        'average_reward': ['mean', 'std'],
        'cumulative_steps': 'max'
    })
    print("\nSummary of results:")
    print(summary_df)