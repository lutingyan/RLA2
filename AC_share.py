import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

torch.backends.cudnn.benchmark = True

# Hyperparameters
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99
hidden_dim = 128
max_steps = int(1e6)
NUM_RUNS = 5

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Shared Network (Base Network)
class SharedNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(SharedNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

# Actor Head
class Actor(nn.Module):
    def __init__(self, shared_network, action_dim):
        super(Actor, self).__init__()
        self.shared_network = shared_network
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.shared_network(x)
        return F.softmax(self.fc_out(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# Critic Head
class Critic(nn.Module):
    def __init__(self, shared_network):
        super(Critic, self).__init__()
        self.shared_network = shared_network
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared_network(x)
        return self.fc_out(x)

# n-step returns
def compute_returns(rewards, dones, values, gamma=0.99, n_steps=10):
    returns = np.zeros(len(rewards), dtype=np.float32)
    T = len(rewards)
    for t in range(T):
        R = 0
        step_count = 0
        for k in range(t, min(t + n_steps, T)):
            R += (gamma ** step_count) * rewards[k]
            step_count += 1
            if dones[k]: 
                break
        if (k < T - 1) and not dones[k]:
            R += (gamma ** step_count) * values[k + 1]
        returns[t] = R
    return torch.FloatTensor(returns)

def run_ac(seed, n_steps=10):
    shared_network = SharedNetwork(state_dim, hidden_dim)
    actor = Actor(shared_network, action_dim)
    critic = Critic(shared_network)
    
    optimizer = optim.Adam([
        {'params': shared_network.parameters()},
        {'params': actor.fc_out.parameters(), 'lr': lr_actor},
        {'params': critic.fc_out.parameters(), 'lr': lr_critic}
    ])
    
    episode_rewards = []
    eval_scores = []
    eval_steps = []
    total_steps = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed)
        episode_data = []
        done = False
        episode_reward = []
        
        while not done:
            action, log_prob = actor.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_tensor = torch.FloatTensor(state)
            value = critic(state_tensor).item()
            episode_data.append((state, reward, value, log_prob, done))
            episode_reward.append(reward)
            state = next_state
            total_steps += 1

            if total_steps >= 1250 and total_steps % 250 == 0:
                eval_reward = 0
                eval_state, _ = env.reset(seed=seed)
                done_eval = False
                while not done_eval:
                    state_tensor = torch.tensor(eval_state, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        probs = actor(state_tensor)
                    action = torch.argmax(probs, dim=-1).item()
                    eval_state, reward, terminated, truncated, _ = env.step(action)
                    eval_reward += reward
                    done_eval = terminated or truncated
                eval_scores.append(eval_reward)
                eval_steps.append(total_steps)
                print(f"[Eval @ Step {total_steps}] Reward: {eval_reward}")
                episode_rewards.append(sum(episode_reward))

        states, rewards, values, log_probs, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        returns = compute_returns(rewards, dones, values, gamma, n_steps)

        value_preds = critic(states).squeeze()
        value_loss = F.mse_loss(value_preds, returns)
        policy_loss = -(returns.detach() * torch.stack(log_probs)).mean()
        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return episode_rewards, eval_scores, eval_steps


if __name__ == "__main__":
    all_scores = []
    all_eval_scores = []
    all_eval_steps = []
    all_steps = []

    for run in range(NUM_RUNS):
        scores, eval_scores, eval_steps = run_ac(seed=run)
        all_scores.append(scores)
        all_eval_scores.append(eval_scores)
        all_eval_steps.append(eval_steps)
        all_steps.append(len(scores))

    max_len = max(len(run) for run in all_scores)
    all_scores = [run + [np.nan] * (max_len - len(run)) for run in all_scores]
    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)

    avg_reward = np.pad(avg_reward, (0, max_len - len(avg_reward)), constant_values=np.nan)
    std_reward = np.pad(std_reward, (0, max_len - len(std_reward)), constant_values=np.nan)

    max_eval_len = max(len(run) for run in all_eval_scores)
    all_eval_scores = [run + [np.nan] * (max_eval_len - len(run)) for run in all_eval_scores]
    all_eval_steps = [run + [np.nan] * (max_eval_len - len(run)) for run in all_eval_steps]

    avg_eval_scores = np.nanmean(all_eval_scores, axis=0)
    std_eval_scores = np.nanstd(all_eval_scores, axis=0)

    df_eval = pd.DataFrame({
        'steps': all_eval_steps[0], 
        'avg_reward': avg_eval_scores,
        'std_reward': std_eval_scores
    })
    os.makedirs('./results', exist_ok=True)
    df_eval.to_csv('./results/acshare_score.csv', index=False)
    
    df = pd.DataFrame({
        'steps': all_eval_steps[0], 
        'avg_reward': avg_reward,
        'std_reward': std_reward
    })

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/acshare_results.csv', index=False)

    print("\nResults saved to ./results/")
    print("\nSummary:")
    print(df[['avg_reward']].agg(['mean', 'max']))
