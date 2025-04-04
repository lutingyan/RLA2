import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import os
from collections import deque

# Create a folder to save learning curve plots
save_dir = "learning_curves"
os.makedirs(save_dir, exist_ok=True)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )
    def size(self):
        return len(self.buffer)

# Define Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, num_layers=2):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# Define DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=5e-4, gamma=0.98, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.999, hidden_size=128, num_layers=2, use_target_network=True, target_update_freq=200):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim, hidden_size, num_layers).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.use_target_network = use_target_network
        self.batch_size = 64  # Batch size for experience replay
        self.memory = ReplayBuffer(10000)  # Experience replay buffer

        if use_target_network:
            self.target_network = QNetwork(state_dim, action_dim, hidden_size, num_layers).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q_network(state_tensor)).item()
    def update(self):
        if self.memory.size() < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = (
            state.to(self.device),
            action.to(self.device),
            reward.to(self.device),
            next_state.to(self.device),
            done.to(self.device),
        )
        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        # **Use Double DQN to compute target Q values**
        with torch.no_grad():
            next_action = self.q_network(next_state).argmax(dim=1, keepdim=True)
            next_q_value = self.target_network(next_state).gather(1, next_action).squeeze(1)
            target_q_values = reward + (1 - done) * self.gamma * next_q_value
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    def update_target_network(self):
        if self.use_target_network:
            self.target_network.load_state_dict(self.q_network.state_dict())
# Train the agent
def train_agent(num_episodes=2000, target_update_freq=200):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, use_target_network=True, target_update_freq=target_update_freq)
    total_steps = 0
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward
            total_steps += 1  # Cumulative steps
        rewards.append((total_steps, episode_reward))
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Steps = {total_steps}, Reward = {episode_reward}")

    env.close()
    return rewards

# Run the experiment and save learning curves
def run_experiment():
    rewards = train_agent(num_episodes=2000)
    # Unpack steps and rewards
    total_steps, episode_rewards = zip(*rewards)
    # Choose smoothing method
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    window_size = 50
    smoothed_rewards = moving_average(episode_rewards, window_size)
    smoothed_steps = total_steps[len(total_steps) - len(smoothed_rewards):]
    plt.figure(figsize=(10, 6))
    plt.plot(total_steps, episode_rewards, label="Original Rewards", alpha=0.3)
    plt.plot(smoothed_steps, smoothed_rewards, label=f"Smoothed (Window={window_size})", linewidth=2)
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Reward")
    plt.title("DQN Learning Curve (Smoothed)")
    plt.legend()
    plt.grid()
    save_path = os.path.join(save_dir, "learning_curve_smoothed.png")
    plt.savefig(save_path)
    print(f"Saved smoothed learning curve plot: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_experiment()
