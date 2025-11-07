#!/usr/bin/env python3
"""
Training script for RL agents with learning curve visualization
Usage: python train_agents.py [ppo|dqn|both] [num_episodes]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')

from environment.finance_env import FinanceEnv
from agents.ppo_agent import PPOAgent
from agents.continuous_dqn_agent import ContinuousDQNAgent

def plot_training_curves(ppo_rewards=None, dqn_rewards=None):
    """Plot training progress with cumulative running averages like Q-learning"""
    plt.figure(figsize=(12, 8))
    
    if ppo_rewards:
        # Calculate cumulative running average
        ppo_running_avg = []
        cumulative_sum = 0
        for i, reward in enumerate(ppo_rewards):
            cumulative_sum += reward
            ppo_running_avg.append(cumulative_sum / (i + 1))
        
        # Plot raw episodes (light) and running average (dark)
        plt.plot(ppo_rewards, linewidth=0.5, color='lightblue', alpha=0.5, label='PPO Episode Rewards')
        plt.plot(ppo_running_avg, linewidth=2, color='blue', label='PPO Running Average')
    
    if dqn_rewards:
        # Calculate cumulative running average
        dqn_running_avg = []
        cumulative_sum = 0
        for i, reward in enumerate(dqn_rewards):
            cumulative_sum += reward
            dqn_running_avg.append(cumulative_sum / (i + 1))
        
        # Plot raw episodes (light) and running average (dark)
        plt.plot(dqn_rewards, linewidth=0.5, color='lightcoral', alpha=0.5, label='DQN Episode Rewards')
        plt.plot(dqn_running_avg, linewidth=2, color='red', label='DQN Running Average')
    
    plt.title('RL Agent Training Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Training Episode', fontsize=14)
    plt.ylabel('Total Reward per Episode', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save to visualization directory
    os.makedirs('visualization', exist_ok=True)
    plt.savefig('visualization/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    agent_type = sys.argv[1] if len(sys.argv) > 1 else 'both'
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    print(f"Training for {num_episodes} episodes...")
    
    env = FinanceEnv()
    ppo_rewards = None
    dqn_rewards = None
    
    if agent_type in ['ppo', 'both']:
        print("=== Training PPO Agent ===")
        ppo_agent = PPOAgent()
        ppo_rewards = ppo_agent.train(env, num_episodes=num_episodes)
        print(f"PPO training complete. Final avg reward: {np.mean(ppo_rewards[-100:]):.2f}")
    
    if agent_type in ['dqn', 'both']:
        print("\n=== Training DQN Agent ===")
        dqn_agent = ContinuousDQNAgent()
        dqn_rewards = dqn_agent.train(env, num_episodes=num_episodes)
        print(f"DQN training complete. Final avg reward: {np.mean(dqn_rewards[-100:]):.2f}")
    
    # Plot training curves
    plot_training_curves(ppo_rewards, dqn_rewards)
    print("Training curves saved to visualization/training_curves.png")

if __name__ == "__main__":
    main()
