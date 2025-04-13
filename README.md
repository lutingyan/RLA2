# Policy Gradient Algorithms - Implementation and Experiments

This repository contains implementations and experimental variations of key policy gradient reinforcement learning algorithms, including REINFORCE, Actor-Critic (AC), and Advantage Actor-Critic (A2C). All experiments are conducted in the `CartPole-v1` environment using PyTorch.

## File Overview

### A2C and Actor-Critic Variants

- **`A2C.py`**  
  Implementation of the Advantage Actor-Critic (A2C) algorithm, incorporating advantage estimation and entropy regularization, gradient clipping.

- **`AC.py`**  
  Standard implementation of the Actor-Critic (AC) algorithm.

- **`AC_lr.py`**  
  Experiments on Actor-Critic with different learning rates for the critic network.

- **`AC_montecarlo.py`**  
  Actor-Critic variant using Monte Carlo returns instead of bootstrapped n-step returns for value estimation.

- **`AC_nstep.py`**  
  Experiments with Actor-Critic using various n-step returns to study the impact on bias-variance trade-off and convergence.

- **`AC_share.py`**  
  Actor-Critic implementation with shared bottom network between actor and critic, used to compare with separate network architectures.

### REINFORCE Variants

- **`REINFORCE.py`**  
  Vanilla REINFORCE algorithm using Monte Carlo policy gradient.

- **`REINFORCE_baseNet.py`**  
  REINFORCE with a learnable value network as a baseline.

- **`REINFORCE_constantbaseline.py`**  
  REINFORCE variant with a constant baseline.

- **`REINFORCE_normalize.py`**  
  REINFORCE implementation with reward normalization applied.

## Environment

- **Gym Environment**: `CartPole-v1`
- **Framework**: PyTorch

## Usage

Each script can be run independently:
```bash
python A2C.py
python AC_lr.py
