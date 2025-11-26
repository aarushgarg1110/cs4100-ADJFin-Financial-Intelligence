#!/usr/bin/env python3
"""
Training script for discrete RL agents with loss visualization
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.finance_env import FinanceEnv
from agents import DiscreteDQNAgent, DiscretePPOAgent

def plot_dqn_training(rewards, losses):
    """Plot DQN training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rewards
    ax1.plot(rewards, linewidth=0.5, color='lightcoral', alpha=0.5)
    running_avg = np.convolve(rewards, np.ones(25)/25, mode='valid')
    ax1.plot(range(24, len(rewards)), running_avg, linewidth=2, color='red', label='Running Avg (25 eps)')
    ax1.set_title('Episode Rewards', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Q-Loss
    ax2.plot(losses, linewidth=1, color='red')
    ax2.set_title('Q-Loss', fontweight='bold')
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('visualization', exist_ok=True)
    plt.savefig('visualization/dqn_training.png', dpi=300)
    print("✓ Training curves saved to visualization/dqn_training.png")
    plt.show()

def plot_ppo_training(rewards, policy_losses, value_losses):
    """Plot PPO training curves"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Rewards
    ax1.plot(rewards, linewidth=0.5, color='lightblue', alpha=0.5)
    running_avg = np.convolve(rewards, np.ones(25)/25, mode='valid')
    ax1.plot(range(24, len(rewards)), running_avg, linewidth=2, color='blue', label='Running Avg (25 eps)')
    ax1.set_title('Episode Rewards', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Policy Loss
    ax2.plot(policy_losses, linewidth=1, color='blue')
    ax2.set_title('Policy Loss', fontweight='bold')
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Value Loss
    ax3.plot(value_losses, linewidth=1, color='darkblue')
    ax3.set_title('Value Loss', fontweight='bold')
    ax3.set_xlabel('Update Step')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('visualization', exist_ok=True)
    plt.savefig('visualization/ppo_training.png', dpi=300)
    print("✓ Training curves saved to visualization/ppo_training.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train discrete RL agents')
    parser.add_argument('--agent', type=str, choices=['dqn', 'ppo'], required=True, help='Agent to train')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save model (default: models/discrete_{agent}_model.pth)')
    
    # DQN hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (DQN only)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay (DQN only)')
    parser.add_argument('--target-update-freq', type=int, default=10, help='Target update frequency (DQN only)')
    
    # PPO hyperparameters
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--epochs', type=int, default=10, help='PPO epochs per update')
    
    args = parser.parse_args()
    
    # Set default save path if not provided
    if args.save_path is None:
        args.save_path = f'models/discrete_{args.agent}_model.pth'
    
    print(f"Training {args.agent.upper()} for {args.episodes} episodes...")
    print(f"Model will be saved to: {args.save_path}")
    
    env = FinanceEnv()
    
    if args.agent == 'dqn':
        print(f"Hyperparameters: lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size}, epsilon_decay={args.epsilon_decay}, target_update_freq={args.target_update_freq}")
        agent = DiscreteDQNAgent(
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq
        )
        rewards, losses = agent.train(env, num_episodes=args.episodes, seed=args.seed, 
                                      save_path=args.save_path)
        plot_dqn_training(rewards, losses)
        
    elif args.agent == 'ppo':
        print(f"Hyperparameters: lr={args.lr}, gamma={args.gamma}, clip_epsilon={args.clip_epsilon}, epochs={args.epochs}")
        agent = DiscretePPOAgent(
            lr=args.lr,
            gamma=args.gamma,
            clip_epsilon=args.clip_epsilon,
            epochs=args.epochs
        )
        rewards, policy_losses, value_losses = agent.train(env, num_episodes=args.episodes, seed=args.seed,
                                                           save_path=args.save_path)
        plot_ppo_training(rewards, policy_losses, value_losses)
    
    print(f"\n✓ Training complete! Final avg reward: {np.mean(rewards[-100:]):.2f}")

if __name__ == "__main__":
    main()
