import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# 设置环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数设置
gamma = 0.99  # 折扣因子
lr = 0.0005  # 学习率
hidden_dim = 128  # 隐藏层维度
max_episodes = 2000  # 最大训练回合数
batch_size = 64  # 每次更新时使用的样本批次大小
target_update_freq = 10  # 更新目标网络的频率
memory_size = 10000  # 经验回放池的大小
epsilon_start = 1.0  # epsilon的初始值（探索的概率）
epsilon_end = 0.1  # epsilon的最小值（利用的概率）
epsilon_decay = 0.995  # epsilon衰减的速度
clip_grad = 1.0  # 梯度裁剪的阈值


# 经验回放池（Experience Replay）
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# 定义Q网络（Q-Network）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # 初始化权重
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 输出每个动作的Q值


# DQN算法
def run_dqn():
    # 初始化Q网络和目标Q网络
    q_network = QNetwork(state_dim, action_dim, hidden_dim)
    target_q_network = QNetwork(state_dim, action_dim, hidden_dim)

    # 复制Q网络的参数到目标Q网络
    target_q_network.load_state_dict(q_network.state_dict())
    target_q_network.eval()  # 目标网络固定

    # 优化器
    optimizer = optim.Adam(q_network.parameters(), lr=lr)

    # 经验回放池
    memory = ReplayBuffer(memory_size)

    epsilon = epsilon_start
    scores = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        # 每个回合的采样
        while not done:
            # epsilon-greedy策略
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)  # 随机探索
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax(dim=1).item()  # 选择最大Q值对应的动作

            # 执行动作并获得反馈
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.push(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            # 更新epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_end)

            # 训练
            if memory.size() >= batch_size:
                # 从经验回放池中采样一个批次
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # 转换为Tensor
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones)

                # 计算Q网络的预测
                q_values = q_network(states_tensor)
                q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                # 计算目标Q值（使用目标Q网络）
                with torch.no_grad():
                    target_q_values = target_q_network(next_states_tensor)
                    target_q_value = rewards_tensor + gamma * target_q_values.max(dim=1)[0] * (1 - dones_tensor)

                # 计算损失
                loss = F.mse_loss(q_value, target_q_value)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), clip_grad)
                optimizer.step()

        # 更新目标Q网络
        if episode % target_update_freq == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        # 记录结果
        scores.append(total_reward)
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f'Episode {episode}, Score: {total_reward}, Avg: {avg_score:.1f}')

    return scores


if __name__ == "__main__":
    scores = run_dqn()
    # 绘制结果
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.6, label='Episode Score')
    plt.plot([np.mean(scores[max(0, i - 20):i + 1]) for i in range(len(scores))], 'r', label='20-episode Avg')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Performance on CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.show()
