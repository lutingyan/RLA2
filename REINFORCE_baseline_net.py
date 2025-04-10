import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move state to the selected device
        with torch.no_grad():  
            probs = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), probs, action


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output single value, Q(s)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output Q value for the state


def run_reinforce_with_q_baseline(seed=0):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)  # Move policy model to device
    q_net = QNetwork(state_dim, hidden_dim).to(device)  # Q network for baseline
    optimizer_policy = optim.Adam(policy.parameters(), lr=fixed_lr)
    optimizer_q = optim.Adam(q_net.parameters(), lr=fixed_lr)
    scores = []
    steps_per_episode = []
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        episode_data = {'states': [], 'actions': [], 'probs': [], 'rewards': []}
        done = False
        steps = 0

        # Store trajectory (states, actions, probs, rewards)
        while not done:
            action, probs, action_tensor = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data['states'].append(state)
            episode_data['actions'].append(action_tensor)
            episode_data['probs'].append(probs)
            episode_data['rewards'].append(reward)
            state = next_state
            steps += 1

        scores.append(sum(episode_data['rewards']))

        # Compute the baseline using Q-network
        # Calculate Q-value for the initial state (using Q network)
        baseline = 0  # You can also use the average of Q values during the episode
        for t in range(len(episode_data['rewards'])):
            state_t = torch.FloatTensor(episode_data['states'][t]).unsqueeze(0).to(device)  # Move state to device
            baseline += q_net(state_t).item()  # Sum Q-values for each state
        baseline /= len(episode_data['rewards'])  # Average Q-value across the episode

        # Update the policy step-by-step using Advantage
        for t in range(len(episode_data['rewards'])):
            # Calculate the return R_t at time step t
            R = sum(gamma**(k-t) * episode_data['rewards'][k] for k in range(t, len(episode_data['rewards'])))

            # Advantage calculation using Q-network
            state_t = torch.FloatTensor(episode_data['states'][t]).unsqueeze(0).to(device)  # Move state to device
            Q_t = q_net(state_t).item()  # Get Q-value for the state
            A_t = R - Q_t  # Advantage function

            # Dynamic log_prob calculation
            probs_t = policy.forward(state_t)
            dist_t = torch.distributions.Categorical(probs_t)
            log_prob_t = dist_t.log_prob(episode_data['actions'][t])

            # Loss function: policy gradient loss with Advantage
            loss = -log_prob_t * A_t  # Using advantage

            # Backpropagation and policy update
            optimizer_policy.zero_grad()
            loss.backward()
            optimizer_policy.step()

        # Update Q network using the return
        for t in range(len(episode_data['rewards'])):
            state_t = torch.FloatTensor(episode_data['states'][t]).unsqueeze(0).to(device)
            R = sum(gamma**(k-t) * episode_data['rewards'][k] for k in range(t, len(episode_data['rewards'])))
            Q_t = q_net(state_t)  # Predicted Q value
            q_loss = F.mse_loss(Q_t, torch.tensor([R], device=device))  # MSE loss for Q-network
            optimizer_q.zero_grad()
            q_loss.backward()
            optimizer_q.step()

        if episode % 100 == 0:
            print(f'Episode {episode}, Reward: {scores[-1]:.1f}')

    return scores, steps_per_episode


if __name__ == "__main__":
    all_scores = []
    all_steps = []

    for run in range(NUM_RUNS):
        scores, steps = run_reinforce_with_q_baseline(seed=run)
        all_scores.append(scores)
        all_steps.append(steps)

    # Pad or truncate all_runs to the same length
    max_len = max(len(run) for run in all_scores)  # Find the maximum length across runs

    # Ensure all runs have the same length by padding with NaN or truncating
    all_scores = [run + [np.nan] * (max_len - len(run)) for run in all_scores]
    all_steps = [run + [0] * (max_len - len(run)) for run in all_steps]

    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)

    # Calculate cumulative steps
    cum_steps = np.cumsum(all_steps[0])

    df = pd.DataFrame({
        'gamma': [gamma] * max_len,
        'episode': np.arange(max_len),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'cum_steps': cum_steps
    })

    os.makedirs('./results', exist_ok=True)
    csv_path = './results/reinforce_with_q_baseline_results.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")
    print("\nSummary:")
    print(df['avg_reward'].agg(['mean', 'max']))

