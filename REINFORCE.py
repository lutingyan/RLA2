import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

# ✅ GPU acceleration flags
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

fixed_lr = 1e-4
gamma = 0.99
hidden_dim = 128
max_steps = int(1e6)
NUM_RUNS = 5

# Precompute gamma powers
max_episode_len = 1000
gamma_powers = torch.tensor([gamma**t for t in range(max_episode_len)], dtype=torch.float32, device=device)

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
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), probs, action

def run_reinforce(seed=0):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=fixed_lr)
    episode_rewards = []
    eval_scores = []
    eval_steps = []
    total_steps = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed)
        episode_data = {'states': [], 'actions': [], 'probs': [], 'rewards': []}
        done = False

        while not done and total_steps < max_steps:
            action, probs, action_tensor = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data['states'].append(state)
            episode_data['actions'].append(action_tensor)
            episode_data['probs'].append(probs)
            episode_data['rewards'].append(reward)
            state = next_state
            total_steps += 1

            # ✅ Evaluation logic inserted here
            if total_steps > 1250 and total_steps % 250 == 0:
                eval_reward = 0
                eval_state, _ = env.reset(seed=seed)
                done_eval = False
                while not done_eval:
                    state_tensor = torch.tensor(eval_state, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                    with torch.no_grad():
                        probs = policy(state_tensor)
                    action = torch.argmax(probs, dim=-1).item()
                    eval_state, reward, terminated, truncated, _ = env.step(action)
                    eval_reward += reward
                    done_eval = terminated or truncated
                eval_scores.append(eval_reward)
                eval_steps.append(total_steps)
                print(f"[Eval @ Step {total_steps}] Reward: {eval_reward}")

        episode_rewards.append(sum(episode_data['rewards']))
        T = len(episode_data['rewards'])

        # Batch convert states
        states_tensor = torch.tensor(episode_data['states'], dtype=torch.float32, device=device)
        for t in range(T):
            rewards = episode_data['rewards'][t:]
            steps = len(rewards)
            gammas = gamma_powers[:steps]
            R = sum(gammas[:steps].cpu().numpy() * np.array(rewards))

            state_t = states_tensor[t].unsqueeze(0)
            probs_t = policy(state_t)
            dist_t = torch.distributions.Categorical(probs_t)
            log_prob_t = dist_t.log_prob(episode_data['actions'][t])
            loss = -log_prob_t * (gamma ** t) * R

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return episode_rewards, eval_scores, eval_steps


if __name__ == "__main__":
    all_scores = []
    all_eval_scores = []
    all_eval_steps = []
    all_steps = []

    for run in range(NUM_RUNS):
        scores, eval_scores, eval_steps = run_reinforce(seed=run)
        all_scores.append(scores)
        all_eval_scores.append(eval_scores)
        all_eval_steps.append(eval_steps)
        all_steps.append(len(scores))

    max_len = max(len(run) for run in all_scores)
    all_scores = [run + [np.nan] * (max_len - len(run)) for run in all_scores]
    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)

    # 确保长度一致，通过填充 NaN
    avg_reward = np.pad(avg_reward, (0, max_len - len(avg_reward)), constant_values=np.nan)
    std_reward = np.pad(std_reward, (0, max_len - len(std_reward)), constant_values=np.nan)
    cum_steps = np.cumsum(all_steps[0])

    # 确保 cum_steps 长度与 max_len 一致
    cum_steps = np.pad(cum_steps, (0, max_len - len(cum_steps)), constant_values=0)

    df = pd.DataFrame({
        'gamma': [gamma] * max_len,
        'steps': np.arange(max_len),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'cum_steps': cum_steps
    })

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/reinforce_results.csv', index=False)

    # ✅ Save eval results from first run
    if all_eval_scores and all_eval_steps:
        df_eval = pd.DataFrame({
            'eval_step': all_eval_steps[0],
            'eval_reward': all_eval_scores[0]
        })
        df_eval.to_csv('./results/reinforce_eval_scores.csv', index=False)

    print("\nResults saved to ./results/")
    print("\nSummary:")
    print(df[['avg_reward']].agg(['mean', 'max']))
