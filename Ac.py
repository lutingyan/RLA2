import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数优化
gamma = 0.99
lr_actor = 0.0003  # Actor学习率
lr_critic = 0.003  # Critic学习率更大
hidden_dim = 64  # 适当减小网络规模
max_episodes = 3000
clip_grad = 0.5  # 梯度裁剪阈值
entropy_coef = 0.01  # 熵正则化系数


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 初始化权重
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train():
    policy = PolicyNetwork()
    critic = ValueNetwork()
    opt_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    scores = []

    for episode in range(max_episodes):
        state = env.reset()[0]
        done = False
        ep_rewards = []
        ep_log_probs = []
        ep_states = []
        ep_dones = []
        ep_entropies = []

        # 1. 收集轨迹数据
        while not done:
            action, log_prob, entropy = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_states.append(state)
            ep_dones.append(done)
            ep_entropies.append(entropy)
            state = next_state

        # 2. 计算TD目标（关键修正点）
        states_tensor = torch.FloatTensor(np.array(ep_states))
        next_states = ep_states[1:] + [next_state]  # 正确的下一个状态序列
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.FloatTensor(ep_dones)

        values = critic(states_tensor).view(-1)
        next_values = critic(next_states_tensor).view(-1)
        next_values = next_values * (1 - dones_tensor)  # 终止状态处理

        # 3. 计算TD误差和优势
        td_targets = torch.FloatTensor(ep_rewards) + gamma * next_values
        advantages = td_targets - values.detach()

        # 4. 标准化优势函数（关键）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5. 计算损失（添加熵正则化）
        policy_loss = (-torch.stack(ep_log_probs) * advantages).mean()
        entropy_loss = -torch.stack(ep_entropies).mean()  # 最大化熵
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        critic_loss = F.mse_loss(values, td_targets.detach())

        # 6. 更新网络（带梯度裁剪）
        opt_actor.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        opt_actor.step()

        opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)
        opt_critic.step()

        # 记录结果
        scores.append(sum(ep_rewards))
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f"Episode {episode}, Score: {scores[-1]}, Avg: {avg_score:.1f}")
    return scores


if __name__ == "__main__":
    scores = train()
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