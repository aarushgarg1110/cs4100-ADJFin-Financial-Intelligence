# ADJ Fin: Adaptive Financial Intelligence

A reinforcement learning–based personal finance simulator that learns to optimize saving, investment, and debt repayment strategies under real-world market and life conditions.

---

## Overview

**ADJ Fin** is a CS4100 project by **Jesvin Jerry**, **Dylan Vo**, and **Aarush Garg**.  
The system uses **reinforcement learning** (DQN/PPO) to train an agent that makes monthly financial decisions — balancing income allocation, debt management, and savings in a stochastic environment with random life events.

---

## Key Features
- Custom **Gym-style environment** simulating income, debt, investments, and expenses.
- **RL agent (DQN)** that learns optimal financial strategies over simulated lifetimes.
- **Baseline strategies** (equal allocation, debt avalanche, 60/40 portfolio) for comparison.
- **Evaluation tools** for net worth growth, bankruptcy rate, and crisis survival.

---

## Repository Structure
- TBD

## Agent Interface

Agents implement a minimal common interface in `agents/base_agent.py` to enable swapping RL and baseline strategies. See `agents/dqn_agent.py` (SB3 DQN wrapper; discrete actions) and `agents/baseline_agents.py` (equal-weight heuristic) for examples.

## Algorithm Choice (Week 1 notes)

- DQN: efficient for discrete action spaces; requires discretization for our Box(6) actions.
- PPO/A2C: better for continuous control; PPO generally more stable sample-wise.
- Recommendation: keep the DQN stub for comparisons with discretized actions; consider PPO/SAC if continuous actions are retained.
