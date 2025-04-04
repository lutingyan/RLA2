'''
# ================== episode 实现 ==================
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# 设置环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数设置
gamma = 0.99  # 折扣因子
lr_actor = 0.0003  # Actor学习率
lr_critic = 0.0003  # Critic学习率
hidden_dim = 128  # 网络隐藏层维度
max_episodes = 2000  # 最大训练回合数
entropy_coef = 0.01  # 熵正则化系数
clip_grad = 1.0  # 梯度裁剪阈值
n_steps = 5  # n步回报的步数


# 定义策略网络（Actor）
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)  # 使用log_softmax
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        log_probs = self.forward(state)
        probs = torch.exp(log_probs)  # 转换回概率
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), log_probs[0, action.item()], dist.entropy()


# 定义值网络（Critic）
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 计算n步优势

def compute_advantages(rewards, values, dones, gamma=0.99, n_steps=5):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0
    # 反向计算优势
    for t in reversed(range(len(rewards))):
        if t + n_steps < len(rewards):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * (1 - dones[t]) * last_advantage
        else:
            delta = rewards[t] - values[t]
            advantages[t] = delta
        last_advantage = advantages[t]

    returns = advantages + values[:len(rewards)]
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)

# A2C算法
def run_a2c():
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    value_net = ValueNetwork(state_dim, hidden_dim)
    optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(value_net.parameters(), lr=lr_critic)
    # 学习率调度器
    scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=500, gamma=0.9)
    scheduler_critic = optim.lr_scheduler.StepLR(optimizer_critic, step_size=500, gamma=0.9)
    scores = []
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_data = []
        done = False
        # 1. 采样轨迹
        while not done:
            # 获取动作、log概率和熵
            action, log_prob, entropy = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 计算当前状态的值
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor).item()
            # 存储轨迹数据
            episode_data.append((state, action, reward, value, log_prob, entropy, done))
            episode_rewards.append(reward)
            state = next_state
        # 2. 处理轨迹数据
        states, actions, rewards, values, log_probs, entropies, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        # 3. 计算优势和回报
        advantages, returns = compute_advantages(rewards, values, dones, gamma, n_steps)

        # 4. 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5. 计算损失
        policy_loss = (-torch.stack(log_probs) * advantages).mean()
        entropy_loss = -torch.stack(entropies).mean()
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        # 值函数损失（带clip）
        values_pred = value_net(states).squeeze()
        value_loss = F.mse_loss(values_pred, returns)
        # 6. 更新网络
        optimizer_actor.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        optimizer_actor.step()
        optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
        optimizer_critic.step()
        # 更新学习率
        scheduler_actor.step()
        scheduler_critic.step()
        # 7. 记录结果
        total_reward = sum(episode_rewards)
        scores.append(total_reward)
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f'Episode {episode}, Score: {total_reward}, Avg: {avg_score:.1f}')
    return scores

if __name__ == "__main__":
    scores = run_a2c()
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

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# 设置环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数设置
gamma = 0.99  # 折扣因子
lr_actor = 0.0005  # Actor学习率
lr_critic = 0.0005  # Critic学习率
hidden_dim = 128  # 网络隐藏层维度
max_episodes = 2000  # 最大训练回合数
entropy_coef = 0.01  # 熵正则化系数
clip_grad = 1.0  # 梯度裁剪阈值
n_steps = 5  # n步回报的步数

# 定义策略网络（Actor）
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)  # 使用log_softmax

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        log_probs = self.forward(state)
        probs = torch.exp(log_probs)  # 转换回概率
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), log_probs[0, action.item()], dist.entropy()

# 定义值网络（Critic）
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 计算n步回报（Q_n）和优势
def compute_advantages(rewards, values, dones, gamma=0.99, n_steps=5):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    returns = np.zeros(len(rewards), dtype=np.float32)
    R = 0
    last_advantage = 0
    # 反向计算n步回报
    for t in reversed(range(len(rewards))):
        if t + n_steps < len(rewards):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * (1 - dones[t]) * last_advantage
        else:
            delta = rewards[t] - values[t]
            advantages[t] = delta
        last_advantage = advantages[t]
        returns[t] = advantages[t] + values[t]

    return torch.FloatTensor(advantages), torch.FloatTensor(returns)

# A2C算法
def run_a2c():
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    value_net = ValueNetwork(state_dim, hidden_dim)
    optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(value_net.parameters(), lr=lr_critic)

    scores = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_data = []
        done = False

        # 1. 采样轨迹
        while not done:
            # 获取动作、log概率和熵
            action, log_prob, entropy = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 计算当前状态的值
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor).item()
            # 存储轨迹数据
            episode_data.append((state, action, reward, value, log_prob, entropy, done))
            episode_rewards.append(reward)
            state = next_state

        # 2. 处理轨迹数据
        states, actions, rewards, values, log_probs, entropies, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)

        # 3. 计算优势和回报
        advantages, returns = compute_advantages(rewards, values, dones, gamma, n_steps)

        # 4. 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5. 计算损失
        policy_loss = (-torch.stack(log_probs) * advantages).mean()
        entropy_loss = -torch.stack(entropies).mean()
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        # 值函数损失（带clip）
        values_pred = value_net(states).squeeze()
        value_loss = F.mse_loss(values_pred, returns)

        # 6. 更新网络
        optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
        optimizer_critic.step()

        optimizer_actor.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        optimizer_actor.step()

        # 7. 记录结果
        total_reward = sum(episode_rewards)
        scores.append(total_reward)
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f'Episode {episode}, Score: {total_reward}, Avg: {avg_score:.1f}')
    return scores


if __name__ == "__main__":
    scores = run_a2c()
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

# ================== environment steps 实现 ==================
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# 设置环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数设置
gamma = 0.99  # 折扣因子
lr_actor = 0.0003  # Actor学习率
lr_critic = 0.0003  # Critic学习率
hidden_dim = 64  # 网络隐藏层维度
max_episodes = 2000  # 最大训练回合数
entropy_coef = 0.01  # 熵正则化系数
clip_grad = 1.0  # 梯度裁剪阈值
n_steps = 5  # n步回报的步数


# 定义策略网络（Actor）
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)  # 使用log_softmax

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        log_probs = self.forward(state)
        probs = torch.exp(log_probs)  # 转换回概率
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), log_probs[0, action.item()], dist.entropy()


# 定义值网络（Critic）
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 计算n步优势
def compute_advantages(rewards, values, dones, gamma=0.99, n_steps=5):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0
    # 反向计算优势
    for t in reversed(range(len(rewards))):
        if t + n_steps < len(rewards):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * (1 - dones[t]) * last_advantage
        else:
            delta = rewards[t] - values[t]
            advantages[t] = delta
        last_advantage = advantages[t]

    returns = advantages + values[:len(rewards)]
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


# A2C算法
def run_a2c():
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
    value_net = ValueNetwork(state_dim, hidden_dim)
    optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(value_net.parameters(), lr=lr_critic)
    # 学习率调度器
    scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=500, gamma=0.9)
    scheduler_critic = optim.lr_scheduler.StepLR(optimizer_critic, step_size=500, gamma=0.9)

    scores = []
    steps_per_episode = []  # 记录每个episode的步数
    total_steps = 0  # 总环境步数计数器

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_data = []
        done = False
        episode_steps = 0  # 当前episode的步数计数器

        # 1. 采样轨迹
        while not done:
            # 获取动作、log概率和熵
            action, log_prob, entropy = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 计算当前状态的值
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor).item()
            # 存储轨迹数据
            episode_data.append((state, action, reward, value, log_prob, entropy, done))
            episode_rewards.append(reward)
            state = next_state
            episode_steps += 1  # 增加步数计数

        # 更新总步数
        total_steps += episode_steps
        steps_per_episode.append(episode_steps)

        # 2. 处理轨迹数据
        states, actions, rewards, values, log_probs, entropies, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)

        # 3. 计算优势和回报
        advantages, returns = compute_advantages(rewards, values, dones, gamma, n_steps)

        # 4. 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5. 计算损失
        policy_loss = (-torch.stack(log_probs) * advantages).mean()
        entropy_loss = -torch.stack(entropies).mean()
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        # 值函数损失（带clip）
        values_pred = value_net(states).squeeze()
        value_loss = F.mse_loss(values_pred, returns)

        # 6. 更新网络
        optimizer_actor.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        optimizer_actor.step()
        optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
        optimizer_critic.step()

        # 更新学习率
        scheduler_actor.step()
        scheduler_critic.step()

        # 7. 记录结果
        total_reward = sum(episode_rewards)
        scores.append(total_reward)

        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f'Episode {episode}, Total Steps: {total_steps}, Score: {total_reward}, Avg: {avg_score:.1f}')

    return scores, steps_per_episode


if __name__ == "__main__":
    scores, steps_per_episode = run_a2c()
    cumulative_steps = np.cumsum(steps_per_episode)  # 计算累计步数

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_steps, scores, label='Episode Score')  # 使用累计步数作为x轴

    # 计算滑动平均（基于步数）
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
        plt.plot(cumulative_steps[window_size - 1:], moving_avg, 'r', label=f'{window_size}-episode Avg')

    plt.xlabel('Environment Steps')  # x轴标签改为环境步数
    plt.ylabel('Score')
    plt.legend()
    plt.title('A2C on CartPole-v1 (by Environment Steps)')
    plt.grid(True)
    plt.show()

'''

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数设置
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.003
hidden_dim = 64
max_episodes = 3000
entropy_coef = 0.01
clip_grad = 0.5
n_steps = 5


# 策略网络（Actor）—— 与 AC 保持一致结构
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
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


# 值函数网络（Critic）
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 计算 n 步优势
def compute_advantages(rewards, values, dones, gamma=0.99, n_steps=5):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    returns = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        if t + n_steps < len(rewards):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * (1 - dones[t]) * last_advantage
        else:
            delta = rewards[t] - values[t]
            advantages[t] = delta
        last_advantage = advantages[t]
        returns[t] = advantages[t] + values[t]

    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


# A2C 主训练函数
def run_a2c():
    policy = PolicyNetwork()
    value_net = ValueNetwork()
    optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(value_net.parameters(), lr=lr_critic)

    scores = []
    steps_per_episode = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        episode_data = []
        episode_rewards = []
        episode_steps = 0

        while not done:
            action, log_prob, entropy = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor).item()
            episode_data.append((state, reward, value, log_prob, entropy, done))
            episode_rewards.append(reward)
            state = next_state
            episode_steps += 1

        steps_per_episode.append(episode_steps)
        total_reward = sum(episode_rewards)
        scores.append(total_reward)

        # 解包轨迹数据
        states, rewards, values, log_probs, entropies, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)

        # 计算优势和回报
        advantages, returns = compute_advantages(rewards, values, dones, gamma, n_steps)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算损失
        policy_loss = (-torch.stack(log_probs) * advantages).mean()
        entropy_loss = -torch.stack(entropies).mean()
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        value_preds = value_net(states).squeeze()
        value_loss = F.mse_loss(value_preds, returns)

        # 更新 actor
        optimizer_actor.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        optimizer_actor.step()

        # 更新 critic
        optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
        optimizer_critic.step()

        # 打印进度
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f"Episode {episode}, Score: {total_reward}, Avg: {avg_score:.1f}")

    return scores, steps_per_episode


# 可视化训练结果
if __name__ == "__main__":
    scores, steps_per_episode = run_a2c()
    cumulative_steps = np.cumsum(steps_per_episode)

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_steps, scores, label='Episode Score')

    window_size = 100
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
        plt.plot(cumulative_steps[window_size - 1:], moving_avg, 'r', label=f'{window_size}-episode Avg')

    plt.xlabel('Environment Steps')
    plt.ylabel('Score')
    plt.legend()
    plt.title('A2C on CartPole-v1 (Same Policy Net as AC)')
    plt.grid(True)
    plt.show()
