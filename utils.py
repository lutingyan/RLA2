import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

MINIBATCH_SIZE = 64   
TAU = 1e-3          
E_DECAY = 0.995      
E_MIN = 0.01    

def get_experiences_batchsize(memory_buffer, batch_size):
    experiences = random.sample(memory_buffer, k=batch_size)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)

def check_update_conditions_batchsize(t, num_steps_upd, memory_buffer, batch_size):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > batch_size:
        return True
    else:
        return False

def check_update_conditions(t, num_steps_upd, memory_buffer):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False
    
def check_update_conditions_update(t, num_steps_upd):
    if (t + 1) % num_steps_upd == 0:
        return True
    else:
        return False
    
def get_new_eps(epsilon):
    return max(E_MIN, E_DECAY*epsilon)


def get_action(q_values, epsilon, num_actions=2):
    q_values_np = q_values.numpy()[0]
    if random.random() > epsilon:
        return np.argmax(q_values_np)
    else:
        return random.choice(np.arange(num_actions))
    
    
def update_target_network(q_network, target_q_network, softupdate):
    if softupdate:
        for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
    else:
        for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
            target_weights.assign(q_net_weights)

def plot(reward_history, cumulative_steps, std_dev=None, rolling_window=50, label=None, color='red'):
    rh = np.array(reward_history)
    rolling_mean = pd.Series(rh).rolling(rolling_window, min_periods=1).mean()

    plt.plot(cumulative_steps, rolling_mean, linewidth=2, label=f"{label} (Smoothed)", color=color)

    if std_dev is not None:
        smoothed_std = pd.Series(std_dev).rolling(rolling_window, min_periods=1).mean()
        plt.fill_between(cumulative_steps, rolling_mean - smoothed_std, rolling_mean + smoothed_std, color=color, alpha=0.1, label=f"{label} (Std Dev)")

    plt.xlabel("Cumulative Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.title(f"{label} Reward History")
    plt.show()