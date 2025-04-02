import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# 公共环境设置
env_name = 'CartPole-v1'
state_dim = gym.make(env_name).observation_space.shape[0]
action_dim = gym.make(env_name).action_space.n


# ================== REINFORCE 实现 ==================
def run_reinforce():
    env = gym.make(env_name)
    gamma = 0.99
    lr = 0.001
    hidden_dim = 32
    max_episodes = 500

    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return F.softmax(self.fc2(x), dim=-1)

        def act(self, state):
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = self.forward(state)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scores = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_data = {'log_probs': [], 'rewards': []}
        done = False

        while not done:
            action, log_prob = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
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
        policy_loss = torch.stack([-log_prob * R for log_prob, R in
                                   zip(episode_data['log_probs'], returns)]).sum()

        # 更新策略
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if episode >= 50 else np.mean(scores)
            print(f'REINFORCE - Episode {episode}, Score: {scores[-1]}, Avg: {avg_score:.1f}')

    env.close()
    return scores


# ================== AC 实现 ==================
def run_ac():
    env = gym.make(env_name)
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.003
    hidden_dim = 64
    max_episodes = 500
    clip_grad = 0.5
    entropy_coef = 0.01

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
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), dist.entropy()

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

    policy = PolicyNetwork()
    critic = ValueNetwork()
    opt_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    scores = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        ep_rewards = []
        ep_log_probs = []
        ep_states = []
        ep_dones = []
        ep_entropies = []
        done = False

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

        # 计算TD目标
        states_tensor = torch.FloatTensor(np.array(ep_states))
        next_states = ep_states[1:] + [next_state]
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.FloatTensor(ep_dones)

        values = critic(states_tensor).view(-1)
        next_values = critic(next_states_tensor).view(-1)
        next_values = next_values * (1 - dones_tensor)

        # 计算TD误差和优势
        td_targets = torch.FloatTensor(ep_rewards) + gamma * next_values
        advantages = td_targets - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算损失
        policy_loss = (-torch.stack(ep_log_probs) * advantages).mean()
        entropy_loss = -torch.stack(ep_entropies).mean()
        total_policy_loss = policy_loss + entropy_coef * entropy_loss
        critic_loss = F.mse_loss(values, td_targets.detach())

        # 更新网络
        opt_actor.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        opt_actor.step()

        opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)
        opt_critic.step()

        scores.append(sum(ep_rewards))
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f'AC - Episode {episode}, Score: {scores[-1]}, Avg: {avg_score:.1f}')

    env.close()
    return scores


# ================== A2C 实现 ==================
def run_a2c():
    env = gym.make(env_name)
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.0003
    hidden_dim = 128
    max_episodes = 500
    entropy_coef = 0.01
    clip_grad = 1.0
    n_steps = 5

    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
            nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
            nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
            nn.init.orthogonal_(self.fc3.weight, gain=0.01)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return F.log_softmax(self.fc3(x), dim=-1)

        def act(self, state):
            state = torch.FloatTensor(state).unsqueeze(0)
            log_probs = self.forward(state)
            probs = torch.exp(log_probs)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), log_probs[0, action.item()], dist.entropy()

    class ValueNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
            nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
            nn.init.orthogonal_(self.fc3.weight, gain=1.0)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def compute_advantages(rewards, values, dones, gamma=0.99, n_steps=5):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_advantage = 0
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

    policy = PolicyNetwork()
    value_net = ValueNetwork()
    optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(value_net.parameters(), lr=lr_critic)
    scores = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_data = []
        done = False

        while not done:
            action, log_prob, entropy = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor).item()
            episode_data.append((state, action, reward, value, log_prob, entropy, done))
            episode_rewards.append(reward)
            state = next_state

        states, actions, rewards, values, log_probs, entropies, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)

        advantages, returns = compute_advantages(rewards, values, dones, gamma, n_steps)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = (-torch.stack(log_probs) * advantages).mean()
        entropy_loss = -torch.stack(entropies).mean()
        total_policy_loss = policy_loss + entropy_coef * entropy_loss
        values_pred = value_net(states).squeeze()
        value_loss = F.mse_loss(values_pred, returns)

        optimizer_actor.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
        optimizer_critic.step()

        scores.append(sum(episode_rewards))
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if episode >= 20 else np.mean(scores)
            print(f'A2C - Episode {episode}, Score: {scores[-1]}, Avg: {avg_score:.1f}')

    env.close()
    return scores


# ================== 并行运行和绘图 ==================
def run_all():
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_reinforce = executor.submit(run_reinforce)
        future_ac = executor.submit(run_ac)
        future_a2c = executor.submit(run_a2c)

        reinforce_scores = future_reinforce.result()
        ac_scores = future_ac.result()
        a2c_scores = future_a2c.result()

    # 确保所有结果长度一致（取最短的长度）
    min_len = min(len(reinforce_scores), len(ac_scores), len(a2c_scores))
    reinforce_scores = reinforce_scores[:min_len]
    ac_scores = ac_scores[:min_len]
    a2c_scores = a2c_scores[:min_len]

    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(reinforce_scores, alpha=0.6, label='REINFORCE')
    plt.plot(ac_scores, alpha=0.6, label='Actor-Critic')
    plt.plot(a2c_scores, alpha=0.6, label='A2C')

    # 绘制滑动平均
    window_size = 50
    plt.plot([np.mean(reinforce_scores[max(0, i - window_size):i + 1]) for i in range(min_len)],
             'b', label=f'REINFORCE Avg')
    plt.plot([np.mean(ac_scores[max(0, i - window_size):i + 1]) for i in range(min_len)],
             'g', label=f'AC Avg')
    plt.plot([np.mean(a2c_scores[max(0, i - window_size):i + 1]) for i in range(min_len)],
             'r', label=f'A2C Avg')

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Policy Gradient Algorithms Comparison on CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_all()