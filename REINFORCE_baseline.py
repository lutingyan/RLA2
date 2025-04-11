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

def run_reinforce_with_constant_baseline(seed=0):
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

        while not done:
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
            if total_steps >= 1250 and total_steps % 250 == 0:
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
        baseline = np.mean(episode_data['rewards'])
        # Batch convert states
        states_tensor = torch.tensor(episode_data['states'], dtype=torch.float32, device=device)

        for t in range(T):
            R = sum(gamma**(k-t) * episode_data['rewards'][k] for k in range(t, len(episode_data['rewards'])))

            state_t = states_tensor[t].unsqueeze(0)
            probs_t = policy(state_t)
            dist_t = torch.distributions.Categorical(probs_t)
            log_prob_t = dist_t.log_prob(episode_data['actions'][t])

            # Advantage with constant baseline
            A = R - baseline
            loss = -log_prob_t * A

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
        scores, eval_scores, eval_steps = run_reinforce_with_constant_baseline(seed=run)
        all_scores.append(scores)
        all_eval_scores.append(eval_scores)
        all_eval_steps.append(eval_steps)
        all_steps.append(len(scores))

    max_len = max(len(run) for run in all_scores)
    all_scores = [run + [np.nan] * (max_len - len(run)) for run in all_scores]
    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)

    # Ensure length consistency by padding NaN
    avg_reward = np.pad(avg_reward, (0, max_len - len(avg_reward)), constant_values=np.nan)
    std_reward = np.pad(std_reward, (0, max_len - len(std_reward)), constant_values=np.nan)


    max_eval_len = max(len(run) for run in all_eval_scores)
    all_eval_scores = [run + [np.nan] * (max_eval_len - len(run)) for run in all_eval_scores]
    all_eval_steps = [run + [np.nan] * (max_eval_len - len(run)) for run in all_eval_steps]

    # Calculate mean and std for each step
    avg_eval_scores = np.nanmean(all_eval_scores, axis=0)
    std_eval_scores = np.nanstd(all_eval_scores, axis=0)

    # Construct DataFrame for evaluation results
    df_eval = pd.DataFrame({
        'eval_step': all_eval_steps[0],  # Ensure that eval_steps corresponds to eval_scores
        'avg_eval_reward': avg_eval_scores,
        'std_eval_reward': std_eval_scores
    })
    os.makedirs('./results', exist_ok=True)
    df_eval.to_csv('./results/reinforce_baseline_score.csv', index=False)
    
    df = pd.DataFrame({
        'steps': all_eval_steps[0],  # Use eval_steps as the steps
        'avg_reward': avg_reward,
        'std_reward': std_reward
    })

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/reinforce_baseline_results.csv', index=False)

    print("\nResults saved to ./results/")
    print("\nSummary:")
    print(df[['avg_reward']].agg(['mean', 'max']))
