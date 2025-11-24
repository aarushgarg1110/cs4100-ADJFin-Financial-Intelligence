#!/usr/bin/env python3
"""
Unified hyperparameter tuning for all RL agents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environment.finance_env import FinanceEnv
from agents.ppo_agent import PPOAgent
from agents.continuous_dqn_agent import ContinuousDQNAgent
from agents.sac_agent import SACAgent

# Hyperparameter configurations for each agent
CONFIGS = {
    'dqn': [
        {"name": "baseline", "lr": 1e-3, "batch_size": 128, "tau": 0.005, "noise_std": 0.1, "noise_decay": 0.999},
        {"name": "high_lr", "lr": 3e-3, "batch_size": 128, "tau": 0.005, "noise_std": 0.1, "noise_decay": 0.999},
        {"name": "low_lr", "lr": 3e-4, "batch_size": 128, "tau": 0.005, "noise_std": 0.1, "noise_decay": 0.999},
        {"name": "large_batch", "lr": 1e-3, "batch_size": 256, "tau": 0.005, "noise_std": 0.1, "noise_decay": 0.999},
        {"name": "fast_target", "lr": 1e-3, "batch_size": 128, "tau": 0.01, "noise_std": 0.1, "noise_decay": 0.999},
        {"name": "slow_target", "lr": 1e-3, "batch_size": 128, "tau": 0.001, "noise_std": 0.1, "noise_decay": 0.999},
        {"name": "high_explore", "lr": 1e-3, "batch_size": 128, "tau": 0.005, "noise_std": 0.2, "noise_decay": 0.995},
        {"name": "low_explore", "lr": 1e-3, "batch_size": 128, "tau": 0.005, "noise_std": 0.05, "noise_decay": 0.999},
    ],
    'ppo': [
        {"name": "baseline", "lr": 3e-4, "gamma": 0.99, "eps_clip": 0.2, "update_freq": 30},
        {"name": "high_lr", "lr": 1e-3, "gamma": 0.99, "eps_clip": 0.2, "update_freq": 30},
        {"name": "low_lr", "lr": 1e-4, "gamma": 0.99, "eps_clip": 0.2, "update_freq": 30},
        {"name": "high_gamma", "lr": 3e-4, "gamma": 0.995, "eps_clip": 0.2, "update_freq": 30},
        {"name": "low_gamma", "lr": 3e-4, "gamma": 0.95, "eps_clip": 0.2, "update_freq": 30},
        {"name": "tight_clip", "lr": 3e-4, "gamma": 0.99, "eps_clip": 0.1, "update_freq": 30},
        {"name": "loose_clip", "lr": 3e-4, "gamma": 0.99, "eps_clip": 0.3, "update_freq": 30},
        {"name": "freq_update", "lr": 3e-4, "gamma": 0.99, "eps_clip": 0.2, "update_freq": 60},
    ],
    'sac': [
        {"name": "baseline", "lr": 3e-4, "alpha": 0.2, "tau": 0.005, "batch_size": 128},
        {"name": "high_lr", "lr": 1e-3, "alpha": 0.2, "tau": 0.005, "batch_size": 128},
        {"name": "low_lr", "lr": 1e-4, "alpha": 0.2, "tau": 0.005, "batch_size": 128},
        {"name": "high_alpha", "lr": 3e-4, "alpha": 0.5, "tau": 0.005, "batch_size": 128},
        {"name": "low_alpha", "lr": 3e-4, "alpha": 0.1, "tau": 0.005, "batch_size": 128},
        {"name": "slow_target", "lr": 3e-4, "alpha": 0.2, "tau": 0.001, "batch_size": 128},
        {"name": "fast_target", "lr": 3e-4, "alpha": 0.2, "tau": 0.01, "batch_size": 128},
        {"name": "large_batch", "lr": 3e-4, "alpha": 0.2, "tau": 0.005, "batch_size": 256},
    ]
}

def print_progress(episode, num_episodes, rewards, loss1, loss2, loss1_name, loss2_name, print_freq):
    """Print training progress at specified frequency"""
    if (episode + 1) % print_freq == 0:
        avg_reward = np.mean(rewards[-print_freq:])
        avg_loss1 = np.mean(loss1[-print_freq:]) if len(loss1) >= print_freq else 0
        avg_loss2 = np.mean(loss2[-print_freq:]) if len(loss2) >= print_freq else 0
        print(f"  Ep {episode+1}/{num_episodes} | Reward: {avg_reward:.2f} | {loss1_name}: {avg_loss1:.4f} | {loss2_name}: {avg_loss2:.4f}")

def train_dqn(config, env, num_episodes, print_freq=10):
    """Train DQN with specific config"""
    agent = ContinuousDQNAgent(lr=config['lr'], tau=config['tau'], 
                               noise_std=config['noise_std'], name=f"DQN_{config['name']}")
    agent.batch_size = config['batch_size']
    
    episode_rewards, actor_losses, critic_losses = [], [], []
    
    for episode in tqdm(range(num_episodes), desc=f"  {config['name']}", leave=False):
        state, _ = env.reset(seed=42 + episode)
        episode_reward, done = 0, False
        step_count = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            step_count += 1
            
            # Update every 30 steps (matches training script)
            if step_count % 30 == 0 and len(agent.replay_buffer) > agent.batch_size:
                actor_loss, critic_loss = agent._update_networks()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            state, episode_reward = next_state, episode_reward + reward
        
        episode_rewards.append(episode_reward)
        agent.noise_std = max(0.05, agent.noise_std * config['noise_decay'])
        print_progress(episode, num_episodes, episode_rewards, actor_losses, critic_losses, "Actor Loss", "Critic Loss", print_freq)
    
    return episode_rewards, actor_losses, critic_losses

def train_ppo(config, env, num_episodes, print_freq=10):
    """Train PPO with specific config"""
    agent = PPOAgent(lr=config['lr'], gamma=config['gamma'], 
                     eps_clip=config['eps_clip'], name=f"PPO_{config['name']}")
    
    episode_rewards, policy_losses, value_losses = [], [], []
    
    for episode in tqdm(range(num_episodes), desc=f"  {config['name']}", leave=False):
        state, _ = env.reset(seed=42 + episode)
        episode_reward, done = 0, False
        states, actions, rewards = [], [], []
        step_count = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            step_count += 1
            
            if step_count % config['update_freq'] == 0 and len(states) > 0:
                policy_loss, value_loss = agent.learn_from_experience(states, actions, rewards, [], [])
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                states, actions, rewards = [], [], []
            
            state, episode_reward = next_state, episode_reward + reward
        
        if len(states) > 0:
            policy_loss, value_loss = agent.learn_from_experience(states, actions, rewards, [], [])
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
        
        episode_rewards.append(episode_reward)
        print_progress(episode, num_episodes, episode_rewards, policy_losses, value_losses, "Policy Loss", "Value Loss", print_freq)
    
    return episode_rewards, policy_losses, value_losses

def train_sac(config, env, num_episodes, print_freq=10):
    """Train SAC with specific config"""
    agent = SACAgent(lr=config['lr'], alpha=config['alpha'], 
                     tau=config['tau'], name=f"SAC_{config['name']}")
    agent.batch_size = config['batch_size']
    
    episode_rewards, actor_losses, critic_losses = [], [], []
    
    for episode in tqdm(range(num_episodes), desc=f"  {config['name']}", leave=False):
        state, _ = env.reset(seed=42 + episode)
        episode_reward, done = 0, False
        step_count = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            step_count += 1
            
            # Update every 30 steps (matches training script)
            if step_count % 30 == 0 and len(agent.replay_buffer) >= agent.batch_size:
                actor_loss, critic_loss = agent._update_networks()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            state, episode_reward = next_state, episode_reward + reward
        
        episode_rewards.append(episode_reward)
        print_progress(episode, num_episodes, episode_rewards, actor_losses, critic_losses, "Actor Loss", "Critic Loss", print_freq)
    
    return episode_rewards, actor_losses, critic_losses

def plot_results(agent_name, results, num_episodes):
    """Plot comparison of all configs for one agent"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Reward curves
    ax = axes[0, 0]
    for name, (rewards, _, _) in results.items():
        ax.plot(rewards, alpha=0.6, label=name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'{agent_name.upper()} Reward Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Actor/Policy loss
    ax = axes[0, 1]
    loss_name = 'Policy Loss' if agent_name == 'ppo' else 'Actor Loss'
    for name, (_, loss1, _) in results.items():
        if len(loss1) > 0:
            ax.plot(loss1, alpha=0.6, label=name)
    ax.set_xlabel('Update Step')
    ax.set_ylabel(loss_name)
    ax.set_title(f'{agent_name.upper()} {loss_name} Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Critic/Value loss
    ax = axes[1, 0]
    loss_name = 'Value Loss' if agent_name == 'ppo' else 'Critic Loss'
    for name, (_, _, loss2) in results.items():
        if len(loss2) > 0:
            ax.plot(loss2, alpha=0.6, label=name)
    ax.set_xlabel('Update Step')
    ax.set_ylabel(loss_name)
    ax.set_title(f'{agent_name.upper()} {loss_name} Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final performance
    ax = axes[1, 1]
    names = list(results.keys())
    final_scores = [np.mean(rewards[-10:]) for rewards, _, _ in results.values()]
    colors = ['green' if score == max(final_scores) else 'skyblue' for score in final_scores]
    ax.barh(names, final_scores, color=colors)
    ax.set_xlabel('Avg Reward (Last 10 Episodes)')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    os.makedirs('visualization', exist_ok=True)
    filename = f'visualization/{agent_name}_hyperparameter_tuning.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {filename}")
    plt.show()

def tune_agent(agent_name, env, num_episodes, print_freq):
    """Tune hyperparameters for one agent"""
    print(f"\n{'='*60}")
    print(f"{agent_name.upper()} Hyperparameter Tuning")
    print(f"Testing {len(CONFIGS[agent_name])} configurations over {num_episodes} episodes each")
    print(f"{'='*60}")
    
    train_fn = {'dqn': train_dqn, 'ppo': train_ppo, 'sac': train_sac}[agent_name]
    results = {}
    
    for config in CONFIGS[agent_name]:
        print(f"\nTesting: {config['name']}")
        config_str = ', '.join([f"{k}={v}" for k, v in config.items() if k != 'name'])
        print(f"  {config_str}")
        
        rewards, loss1, loss2 = train_fn(config, env, num_episodes, print_freq)
        results[config['name']] = (rewards, loss1, loss2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, (rewards, _, _) in results.items():
        avg_final = np.mean(rewards[-10:])
        print(f"{name:15s}: {avg_final:8.2f} (last 10 eps avg)")
    
    best = max(results.items(), key=lambda x: np.mean(x[1][0][-10:]))
    print(f"\nBest config: {best[0]} with {np.mean(best[1][0][-10:]):.2f}")
    
    plot_results(agent_name, results, num_episodes)

def main():
    parser = argparse.ArgumentParser(description='Tune hyperparameters for RL agents')
    parser.add_argument('--agents', nargs='+', choices=['ppo', 'dqn', 'sac', 'all'], 
                        default=['all'], help='Agents to tune (default: all)')
    parser.add_argument('--episodes', type=int, default=50, 
                        help='Number of episodes per config (default: 50)')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='Print progress every N episodes (default: 10)')
    args = parser.parse_args()
    
    agents_to_tune = ['ppo', 'dqn', 'sac'] if 'all' in args.agents else args.agents
    
    # Create environment ONCE for fair comparison
    print("Creating environment (one-time setup)...")
    env = FinanceEnv()
    
    for agent_name in agents_to_tune:
        tune_agent(agent_name, env, args.episodes, args.print_freq)

if __name__ == "__main__":
    main()
