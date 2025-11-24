#!/usr/bin/env python3
"""
Training script for RL agents with learning curve visualization
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.finance_env import FinanceEnv
from agents.ppo_agent import PPOAgent
from agents.continuous_dqn_agent import ContinuousDQNAgent
from agents.sac_agent import SACAgent

def plot_training_curves(ppo_data=None, dqn_data=None, sac_data=None):
    """Plot training progress with rewards and losses"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Episode Rewards
    ax = axes[0]
    if ppo_data:
        rewards, _, _ = ppo_data
        ax.plot(rewards, linewidth=0.5, color='lightblue', alpha=0.5)
        running_avg = [sum(rewards[:i+1])/(i+1) for i in range(len(rewards))]
        ax.plot(running_avg, linewidth=2, color='blue', label='PPO')
    
    if dqn_data:
        rewards, _, _ = dqn_data
        ax.plot(rewards, linewidth=0.5, color='lightcoral', alpha=0.5)
        running_avg = [sum(rewards[:i+1])/(i+1) for i in range(len(rewards))]
        ax.plot(running_avg, linewidth=2, color='red', label='DQN')

    if sac_data:
        rewards, _, _ = sac_data
        ax.plot(rewards, linewidth=0.5, color='lightgreen', alpha=0.5)
        running_avg = [sum(rewards[:i+1])/(i+1) for i in range(len(rewards))]
        ax.plot(running_avg, linewidth=2, color='green', label='SAC')
    
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Actor/Policy Loss
    ax = axes[1]
    if ppo_data:
        _, policy_losses, _ = ppo_data
        ax.plot(policy_losses, linewidth=1, color='blue', label='PPO Policy Loss')
    
    if dqn_data:
        _, actor_losses, _ = dqn_data
        ax.plot(actor_losses, linewidth=1, color='red', label='DQN Actor Loss')

    if sac_data:
        _, actor_losses, _ = sac_data
        ax.plot(actor_losses, linewidth=1, color='green', label='SAC Actor Loss')
    
    ax.set_title('Actor/Policy Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Critic/Value Loss
    ax = axes[2]
    if ppo_data:
        _, _, value_losses = ppo_data
        ax.plot(value_losses, linewidth=1, color='blue', label='PPO Value Loss')
    
    if dqn_data:
        _, _, critic_losses = dqn_data
        ax.plot(critic_losses, linewidth=1, color='red', label='DQN Critic Loss')

    if sac_data:
        _, _, critic_losses = sac_data
        ax.plot(critic_losses, linewidth=1, color='green', label='SAC Critic Loss')
    
    ax.set_title('Critic/Value Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs('visualization', exist_ok=True)
    plt.savefig('visualization/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train RL agents for financial planning')
    parser.add_argument('--agents', nargs='+', choices=['ppo', 'dqn', 'sac', 'all'], 
                        default=['all'], help='Agents to train (default: all)')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='Number of training episodes (default: 1000)')
    args = parser.parse_args()
    
    agents_to_train = ['ppo', 'dqn', 'sac'] if 'all' in args.agents else args.agents
    print(f"Training {', '.join(agents_to_train).upper()} for {args.episodes} episodes...")
    
    env = FinanceEnv()
    ppo_data = dqn_data = sac_data = None
    
    if 'ppo' in agents_to_train:
        print("=== Training PPO Agent ===")
        ppo_agent = PPOAgent()
        ppo_data = ppo_agent.train(env, num_episodes=args.episodes)
        print(f"PPO training complete. Final avg reward: {np.mean(ppo_data[0][-100:]):.2f}")
    
    if 'dqn' in agents_to_train:
        print("\n=== Training DQN Agent ===")
        dqn_agent = ContinuousDQNAgent()
        dqn_data = dqn_agent.train(env, num_episodes=args.episodes)
        print(f"DQN training complete. Final avg reward: {np.mean(dqn_data[0][-100:]):.2f}")

    if 'sac' in agents_to_train:
        print("\n=== Training SAC Agent ===")
        sac_agent = SACAgent()
        sac_data = sac_agent.train(env, num_episodes=args.episodes)
        tail = sac_data[0][-100:] if len(sac_data[0]) >= 100 else sac_data[0]
        print(f"SAC training complete. Final avg reward: {np.mean(tail):.2f}")
    
    plot_training_curves(ppo_data, dqn_data, sac_data)
    print("Training curves saved to visualization/training_curves.png")

if __name__ == "__main__":
    main()
