import time, random
import gymnasium as gym
import numpy as np
import tensorflow as tf
import utils
from collections import deque, namedtuple
import pandas as pd
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# ********* hyperparameter
BUFFER_SIZE = 100_000
GAMMA = 0.99
ALPHA = 1e-4
STEPS_FOR_UPDATE = 2
EPSILON = 0.4

TOTAL_STEPS = 1_000_0  # Total steps to run
NUM_RUNS = 2

env = gym.make("CartPole-v1")
MAX_TIMESTEPS = env.spec.max_episode_steps

state_size = env.observation_space.shape  # (4,)
num_actions = env.action_space.n  # 2 (left, right)

# Initialize lists to store results
all_eval_scores = []
all_eval_steps = []
all_returns_per_step = []

for run in range(NUM_RUNS):
    random.seed(run)
    np.random.seed(run)
    tf.random.set_seed(run)
    print(f"\nRunning {run+1}...\n")

    Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    memory_buffer = deque(maxlen=BUFFER_SIZE)

    # Create networks
    q_network = Sequential([
        Input(shape=state_size),
        Dense(units=128, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=num_actions, activation='linear'),
    ])
    
    target_q_network = Sequential([
        Input(shape=state_size),
        Dense(units=128, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=num_actions, activation='linear'),
    ])
    
    optimizer = Adam(learning_rate=ALPHA)
    target_q_network.set_weights(q_network.get_weights()) 

    @tf.function
    def nn_update(experiences, gamma, optimizer):
        with tf.GradientTape() as tape:
            loss = compute_loss(experiences, gamma)
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    def compute_loss(experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
        y_targets = rewards + (gamma * max_qsa * (1 - dones))         
        q_values = q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                        tf.cast(actions, tf.int32)], axis=1))
        loss = MSE(y_targets, q_values)
        return loss

    total_steps = 0  
    epsilon = EPSILON
    all_returns = []  # Store returns for each step per run
    start_time = time.time()

    while total_steps < TOTAL_STEPS:
        state, _ = env.reset(seed=run)
        episodic_return = 0
        episode_steps = 0  # Count steps in each episode
        done = False

        while not done:
            state_qn = tf.convert_to_tensor(state, dtype=tf.float32)[None, :]
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1
            episode_steps += 1
            episodic_return += reward

            memory_buffer.append(Experience(state, action, reward, next_state, done))
            if utils.check_update_conditions(episode_steps, STEPS_FOR_UPDATE, memory_buffer):
                experiences = utils.get_experiences(memory_buffer)
                nn_update(experiences, GAMMA, optimizer)
                utils.update_target_network(q_network, target_q_network, softupdate=True)

            state = next_state

            # Perform greedy evaluation every 250 steps after 1250 steps
            if total_steps >= 1250 and total_steps % 250 == 0:
                eval_reward = 0
                eval_state, _ = env.reset(seed=run)
                done_eval = False
                while not done_eval:
                    state_qn_eval = tf.convert_to_tensor(eval_state, dtype=tf.float32)[None, :]
                    greedy_action = tf.argmax(q_network(state_qn_eval), axis=-1).numpy()[0]  # Greedy action selection
                    eval_state, reward, terminated, truncated, _ = env.step(greedy_action)
                    eval_reward += reward
                    done_eval = terminated or truncated
                
                # Print and record episodic return, eval_reward, and steps
                print(f"[Eval @ Step {total_steps}] Reward: {eval_reward}")
                
                all_eval_scores.append(eval_reward)
                all_eval_steps.append(total_steps)

                all_returns_per_step.extend([episodic_return] * episode_steps)

        print(f"\rRun {run+1} | Total points: {episodic_return:.2f} | Steps: {total_steps}  ", end="")

        if total_steps >= TOTAL_STEPS:
            print(f"\n\nRun {run+1} completed!\n")
            break

# Ensure all lengths are consistent for saving to CSV
min_len = min(len(all_eval_steps), len(all_eval_scores), len(all_returns_per_step))

# Save the evaluation results CSV
df_eval = pd.DataFrame({
    'eval_step': all_eval_steps[:min_len],
    'avg_eval_reward': all_eval_scores[:min_len],
    'std_eval_reward': [np.std(all_eval_scores)] * min_len
})

# Save the step-wise results CSV
df_results = pd.DataFrame({
    'steps': all_eval_steps[:min_len],
    'avg_reward': all_returns_per_step[:min_len],
    'std_reward': [np.std(all_returns_per_step)] * min_len
})

# Save both DataFrames into CSV
os.makedirs('./results', exist_ok=True)
df_eval.to_csv('./results/reinforce_DQN_score.csv', index=False)
df_results.to_csv('./results/reinforce_DQN_results.csv', index=False)

print(f"\nResults saved to ./results/")
print(f"\nSummary:")
print(df_results[['avg_reward']].agg(['mean', 'max']))

total_time = time.time() - start_time
print(f"\nTotal Runtime: {total_time:.2f} s ({(total_time/60):.2f} min)")
