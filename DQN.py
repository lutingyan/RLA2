import time, random
import gymnasium as gym
import numpy as np
import tensorflow as tf
import utils
from collections import deque, namedtuple

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

TOTAL_STEPS = 1_000_000 
NUM_RUNS = 5

env = gym.make("CartPole-v1")
MAX_TIMESTEPS = env.spec.max_episode_steps
NUM_EPISODES = TOTAL_STEPS // MAX_TIMESTEPS

state_size = env.observation_space.shape  # (4,)
num_actions = env.action_space.n  # 2 (left, right)

returns_per_episode = np.zeros((NUM_RUNS, NUM_EPISODES))  
all_returns_per_step = []  

for run in range(NUM_RUNS):
    random.seed(run)
    np.random.seed(run)
    tf.random.set_seed(run)
    print(f"\nRunning {run+1}...\n")

    Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    memory_buffer = deque(maxlen=BUFFER_SIZE)

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
    per_step_returns = [] 
    start_time = time.time()
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=run)
        episodic_return = 0
        total_points = 0

        for t in range(MAX_TIMESTEPS):
            state_qn = tf.convert_to_tensor(state, dtype=tf.float32)[None, :]
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1
            total_points += reward

            memory_buffer.append(Experience(state, action, reward, next_state, done))
            if utils.check_update_conditions(t, STEPS_FOR_UPDATE, memory_buffer):
                experiences = utils.get_experiences(memory_buffer)
                nn_update(experiences, GAMMA, optimizer)
                utils.update_target_network(q_network, target_q_network, softupdate=True)
            state = next_state.copy()
            if done:
                break
        per_step_returns.append(total_points)        
        returns_per_episode[run, episode] = total_points  

        print(f"\rRun {run+1} | Episode {episode+1} | Total points: {total_points:.2f} | Steps: {total_steps}  ", end="")

        if (episode + 1) % 100 == 0:
            print(f"\rRun {run+1} | Episode {episode+1} | Total points: {total_points:.2f} | Steps: {total_steps}")

        epsilon = utils.get_new_eps(epsilon)  # epsilon 

        if total_steps >= TOTAL_STEPS:
            print(f"\n\nRun {run+1} completed!\n")
            q_network.save(f"ET{run+1}.h5")
            break

    all_returns_per_step.append(per_step_returns)  

all_returns_per_step = np.array(all_returns_per_step)
average_per_step = np.nanmean(all_returns_per_step, axis=0)
std_per_step = np.nanstd(all_returns_per_step, axis=0)
np.save("TN_ER_softupdate_decay.npy", all_returns_per_step)


# utils.plot_history(average_per_step, std_dev=std_per_step)

total_time = time.time() - start_time
print(f"\nTotal Runtime: {total_time:.2f} s ({(total_time/60):.2f} min)")