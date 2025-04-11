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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr_actor = 1e-4
lr_critic = 0.001
gamma = 0.99
hidden_dim = 128
max_steps = int(1e6)
NUM_RUNS = 5

class Actor(nn.Module):
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
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()  # Return entropy as well


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



def compute_advantages_and_returns(rewards, values, dones, gamma=0.99, n_steps=5):
    advantages = []
    returns = []
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
        returns.append(R)
        advantages.append(R - values[t])
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)

def run_reinforce_with_Net(seed, clip=False, entropy=False, entropy_weight=0.01):
    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, hidden_dim)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    episode_rewards = []
    eval_scores = []
    eval_steps = []
    total_steps = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    while total_steps < max_steps:
        state, _ = env.reset()
        done = False
        states, rewards, log_probs, entropy_list, dones = [], [], [], [], []

        while not done:
            action, log_prob, entropy = actor.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropy_list.append(entropy)
            dones.append(done)
            state = next_state
            total_steps += 1

            # ✅ Evaluation logic inserted here
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
                episode_rewards.append(sum(rewards))

        states_tensor = torch.FloatTensor(np.array(states))
        with torch.no_grad():
            values = critic(states_tensor).squeeze().numpy()

        # 计算优势
        advantages, returns = compute_advantages_and_returns(rewards, values, dones, gamma)

        # 计算损失
        log_probs = torch.stack(log_probs)
        entropy = torch.stack(entropy_list).mean()
        value_loss = F.mse_loss(returns, critic(states_tensor).squeeze())
        if entropy:
            policy_loss = -(advantages * log_probs).mean() - entropy_weight * entropy
        else:
            policy_loss = -(advantages * log_probs).mean()

        # 更新Actor
        optimizer_actor.zero_grad()
        policy_loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        optimizer_actor.step()

        # 更新Critic
        optimizer_critic.zero_grad()
        value_loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        optimizer_critic.step()

    return episode_rewards, eval_scores, eval_steps


if __name__ == "__main__":
    all_scores = []
    all_eval_scores = []
    all_eval_steps = []
    all_steps = []

    for run in range(NUM_RUNS):
        scores, eval_scores, eval_steps = run_reinforce_with_Net(seed=run, entropy=True)
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
        'steps': all_eval_steps[0],  # Ensure that eval_steps corresponds to eval_scores
        'avg_reward': avg_eval_scores,
        'std_reward': std_eval_scores
    })
    os.makedirs('./results', exist_ok=True)
    df_eval.to_csv('./results/reinforce_a2c_entropy_score.csv', index=False)
    
    df = pd.DataFrame({
        'steps': all_eval_steps[0],  # Use eval_steps as the steps
        'avg_reward': avg_reward,
        'std_reward': std_reward
    })

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/reinforce_a2c_entropy_results.csv', index=False)

    print("\nResults saved to ./results/")
    print("\nSummary:")
    print(df[['avg_reward']].agg(['mean', 'max']))