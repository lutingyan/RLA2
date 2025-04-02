import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
#111
# 创建环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数
gamma = 0.99
lr = 0.001
hidden_dim = 32
max_episodes = 2000


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        probs = self.forward(state)# 获取动作的概率分布
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()# 从概率分布中采样动作
        return action.item(), dist.log_prob(action)# 返回选中的动作及其 log 概率



def run_reinforce():
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scores = []
    for episode in range(max_episodes):
        # 处理新版gym的reset API变化
        state, _ = env.reset()
        episode_data = {
            'log_probs': [],
            'rewards': []
        }
        done = False
        while not done:
            action, log_prob = policy.act(state)
            # 处理新版gym的step API变化
            result = env.step(action)
            if len(result) == 4:  # 旧版gym
                next_state, reward, done, _ = result
            else:  # 新版gym
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            episode_data['log_probs'].append(log_prob)
            episode_data['rewards'].append(reward)
            state = next_state
        scores.append(sum(episode_data['rewards']))
        # 计算回报
        returns = []
        R = 0
        for r in reversed(episode_data['rewards']):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # 计算损失
        policy_loss = []
        for log_prob, R in zip(episode_data['log_probs'], returns):
            policy_loss.append(-log_prob * R)

        # 更新策略
        optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        optimizer.step()
        # 打印进度
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if episode >= 50 else np.mean(scores)
            print(f'Episode {episode}, Score: {scores[-1]}, Avg: {avg_score:.1f}')
    return scores


if __name__ == "__main__":
    scores = run_reinforce()
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Episode Score')
    window_size = 100
    plt.plot([np.mean(scores[max(0, i - window_size):i + 1]) for i in range(len(scores))],
             'r', label=f'{window_size}-episode Avg')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title('REINFORCE on CartPole-v1')
    plt.grid(True)
    plt.show()

