import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd  # 新增导入pandas用于保存CSV
import os

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

gamma = 0.99
learning_rates = [1e-3, 1e-4, 1e-5]  # 三个不同的学习率
hidden_dim = 128
max_episodes = 2000
NUM_RUNS = 5


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # 2 hidden layers
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


def run_reinforce(lr):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scores = []
    steps_per_episode = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_data = {
            'log_probs': [],
            'rewards': []
        }
        done = False
        steps = 0

        while not done:
            action, log_prob = policy.act(state)
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = result
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
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        policy_loss = []
        for log_prob, R in zip(episode_data['log_probs'], returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        optimizer.step()

        if episode % 100 == 0:
            reward = sum(episode_data['rewards'])
            print(f'LR {lr:.0e} - Episode {episode}, Reward: {reward:.1f}')

    return scores, steps_per_episode


if __name__ == "__main__":
    results = {}

    for lr in learning_rates:
        all_scores = []
        all_steps_per_episode = []

        for run in range(NUM_RUNS):
            scores, steps_per_episode = run_reinforce(lr)
            all_scores.append(scores)
            all_steps_per_episode.append(steps_per_episode)

        all_scores = np.array(all_scores)
        all_steps_per_episode = np.array(all_steps_per_episode)

        average_per_step = np.nanmean(all_scores, axis=0)
        std_per_step = np.nanstd(all_scores, axis=0)
        cumulative_steps = np.cumsum(all_steps_per_episode[0])

        # 保存结果
        results[lr] = {
            'average_per_step': average_per_step,
            'cumulative_steps': cumulative_steps,
            'std_per_step': std_per_step
        }

    # 创建DataFrame保存结果
    dfs = []
    for lr in learning_rates:
        data = results[lr]
        df = pd.DataFrame({
            'learning_rate': [f"{lr:.0e}"] * len(data['average_per_step']),
            'episode': np.arange(len(data['average_per_step'])),
            'average_reward': data['average_per_step'],
            'std_reward': data['std_per_step'],
            'cumulative_steps': data['cumulative_steps']
        })
        dfs.append(df)

    # 合并所有学习率的数据
    final_df = pd.concat(dfs, ignore_index=True)

    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)

    # 保存到CSV文件
    csv_path = 'results/reinforce_resultslr.csv'
    final_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # 打印汇总信息
    summary_df = final_df.groupby('learning_rate').agg({
        'average_reward': ['mean', 'std'],
        'cumulative_steps': 'max'
    })
    print("\nSummary of results:")
    print(summary_df)