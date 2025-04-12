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

all_train_rewards = []  # 每个元素是一个列表，记录单次运行中每个step的回报
all_eval_scores = []    # 每个元素是一个列表，记录单次运行的评估结果（分数）
all_eval_steps = []     # 每个元素是一个列表，记录单次运行的评估时间点（步骤）

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
    run_train_rewards = []  # 仅在评估时记录训练回报
    run_eval_scores = []    # 记录评估分数
    run_eval_steps = []     # 记录评估步骤

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

            run_train_rewards.append(episodic_return)  # 注意这里需要根据具体需求调整
            
            # 评估逻辑
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
                
                run_eval_scores.append(eval_reward)
                run_eval_steps.append(total_steps)
                print(f"[Eval @ Step {total_steps}] Reward: {eval_reward}")
                
                # 同时记录当前训练回报
                run_train_rewards.append(episodic_return)

    all_train_rewards.append(run_train_rewards)
    all_eval_scores.append(run_eval_scores)
    all_eval_steps.append(run_eval_steps)

common_eval_steps = sorted(list(set(step for run_steps in all_eval_steps for step in run_steps)))

# 构建评估矩阵（确保维度正确）
eval_matrix = np.full((NUM_RUNS, len(common_eval_steps)), np.nan)
for run_idx in range(NUM_RUNS):
    step_score = dict(zip(all_eval_steps[run_idx], all_eval_scores[run_idx]))
    for col_idx, step in enumerate(common_eval_steps):
        eval_matrix[run_idx, col_idx] = step_score.get(step, np.nan)

train_matrix = np.full((NUM_RUNS, len(common_eval_steps)), np.nan)
for run_idx in range(NUM_RUNS):
    # 确保使用相同长度的列表
    step_reward = dict(zip(all_eval_steps[run_idx], all_train_rewards[run_idx]))
    for col_idx, step in enumerate(common_eval_steps):
        train_matrix[run_idx, col_idx] = step_reward.get(step, np.nan)
        
# 生成最终DataFrame
df_eval = pd.DataFrame({
    'step': common_eval_steps,
    'avg_eval_reward': np.nanmean(eval_matrix, axis=0),
    'std_eval_reward': np.nanstd(eval_matrix, axis=0)
})

df_train = pd.DataFrame({
    'step': common_eval_steps,
    'avg_train_reward': np.nanmean(train_matrix, axis=0),
    'std_train_reward': np.nanstd(train_matrix, axis=0)
})

# 保存结果
os.makedirs('./results', exist_ok=True)
df_eval.to_csv('./results/eval_greedy_scores.csv', index=False)
df_train.to_csv('./results/train_returns.csv', index=False)
print("数据已保存至 ./results/ 目录")