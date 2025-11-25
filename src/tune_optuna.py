#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization with parallel support via Supabase
Run multiple instances in parallel: each terminal runs this script simultaneously
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import numpy as np
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment.finance_env import FinanceEnv
from agents.ppo_agent import PPOAgent
from agents.continuous_dqn_agent import ContinuousDQNAgent
from agents.sac_agent import SACAgent


def train_dqn_trial(config, env, num_episodes, trial_num, print_freq=50):
    """Train DQN with given config and return final performance"""
    agent = ContinuousDQNAgent(lr=config['lr'], tau=config['tau'], 
                               noise_std=config['noise_std'])
    agent.batch_size = config['batch_size']
    
    episode_rewards, actor_losses, critic_losses = [], [], []
    
    # Print config
    config_str = f"lr={config['lr']:.6f}, batch_size={config['batch_size']}, tau={config['tau']:.6f}, noise_std={config['noise_std']:.2f}, noise_decay={config['noise_decay']:.3f}"
    print(f"\nTrial {trial_num}:")
    print(f"  {config_str}")
    
    for episode in tqdm(range(num_episodes), desc=f"  Trial {trial_num}", leave=False):
        state, _ = env.reset(seed=42 + episode)
        episode_reward, done, step_count = 0, False, 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            step_count += 1
            if step_count % 30 == 0 and len(agent.replay_buffer) > agent.batch_size:
                actor_loss, critic_loss = agent._update_networks()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            state, episode_reward = next_state, episode_reward + reward
        
        episode_rewards.append(episode_reward)
        agent.noise_std = max(0.05, agent.noise_std * config['noise_decay'])
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_actor = np.mean(actor_losses[-print_freq:]) if len(actor_losses) >= print_freq else 0
            avg_critic = np.mean(critic_losses[-print_freq:]) if len(critic_losses) >= print_freq else 0
            print(f"  Ep {episode+1}/{num_episodes} | Reward: {avg_reward:.2f} | Actor Loss: {avg_actor:.4f} | Critic Loss: {avg_critic:.4f}")
    
    final_score = np.mean(episode_rewards[-10:])
    print(f"  Trial {trial_num} complete: {final_score:.2f}\n")
    return final_score
    print(f"  Trial {trial_num} complete: {final_score:.2f}")
    return final_score


def train_ppo_trial(config, env, num_episodes, trial_num, print_freq=50):
    """Train PPO with given config and return final performance"""
    agent = PPOAgent(lr=config['lr'], gamma=config['gamma'], eps_clip=config['eps_clip'])
    
    episode_rewards, policy_losses, value_losses = [], [], []
    
    # Print config
    config_str = f"lr={config['lr']:.6f}, gamma={config['gamma']:.3f}, eps_clip={config['eps_clip']:.2f}, update_freq={config['update_freq']}"
    print(f"\nTrial {trial_num}:")
    print(f"  {config_str}")
    
    for episode in tqdm(range(num_episodes), desc=f"  Trial {trial_num}", leave=False):
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
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_policy = np.mean(policy_losses[-print_freq:]) if len(policy_losses) >= print_freq else 0
            avg_value = np.mean(value_losses[-print_freq:]) if len(value_losses) >= print_freq else 0
            print(f"  Ep {episode+1}/{num_episodes} | Reward: {avg_reward:.2f} | Policy Loss: {avg_policy:.4f} | Value Loss: {avg_value:.4f}")
    
    final_score = np.mean(episode_rewards[-10:])
    print(f"  Trial {trial_num} complete: {final_score:.2f}\n")
    return final_score


def train_sac_trial(config, env, num_episodes, trial_num, print_freq=50):
    """Train SAC with given config and return final performance"""
    agent = SACAgent(lr=config['lr'], alpha=config['alpha'], tau=config['tau'])
    agent.batch_size = config['batch_size']
    
    episode_rewards, actor_losses, critic_losses = [], [], []
    
    # Print config
    config_str = f"lr={config['lr']:.6f}, alpha={config['alpha']:.2f}, tau={config['tau']:.6f}, batch_size={config['batch_size']}"
    print(f"\nTrial {trial_num}:")
    print(f"  {config_str}")
    
    for episode in tqdm(range(num_episodes), desc=f"  Trial {trial_num}", leave=False):
        state, _ = env.reset(seed=42 + episode)
        episode_reward, done, step_count = 0, False, 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            step_count += 1
            if step_count % 30 == 0 and len(agent.replay_buffer) >= agent.batch_size:
                actor_loss, critic_loss = agent._update_networks()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            state, episode_reward = next_state, episode_reward + reward
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_actor = np.mean(actor_losses[-print_freq:]) if len(actor_losses) >= print_freq else 0
            avg_critic = np.mean(critic_losses[-print_freq:]) if len(critic_losses) >= print_freq else 0
            print(f"  Ep {episode+1}/{num_episodes} | Reward: {avg_reward:.2f} | Actor Loss: {avg_actor:.4f} | Critic Loss: {avg_critic:.4f}")
    
    final_score = np.mean(episode_rewards[-10:])
    print(f"  Trial {trial_num} complete: {final_score:.2f}\n")
    return final_score


def objective_dqn(trial, env, num_episodes, print_freq):
    """Optuna objective for DQN"""
    config = {
        'lr': trial.suggest_categorical('lr', [1e-4, 3e-4, 1e-3, 3e-3]),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'tau': trial.suggest_categorical('tau', [0.001, 0.003, 0.005, 0.01]),
        'noise_std': trial.suggest_categorical('noise_std', [0.05, 0.1, 0.15, 0.2]),
        'noise_decay': trial.suggest_categorical('noise_decay', [0.99, 0.995, 0.998, 0.999])
    }
    
    return train_dqn_trial(config, env, num_episodes, trial.number, print_freq)


def objective_ppo(trial, env, num_episodes, print_freq):
    """Optuna objective for PPO"""
    config = {
        'lr': trial.suggest_categorical('lr', [1e-4, 3e-4, 1e-3]),
        'gamma': trial.suggest_categorical('gamma', [0.95, 0.98, 0.99, 0.995]),
        'eps_clip': trial.suggest_categorical('eps_clip', [0.1, 0.15, 0.2, 0.25, 0.3]),
        'update_freq': trial.suggest_categorical('update_freq', [20, 30, 40, 60])
    }
    
    return train_ppo_trial(config, env, num_episodes, trial.number, print_freq)


def objective_sac(trial, env, num_episodes, print_freq):
    """Optuna objective for SAC"""
    config = {
        'lr': trial.suggest_categorical('lr', [1e-4, 3e-4, 1e-3]),
        'alpha': trial.suggest_categorical('alpha', [0.1, 0.2, 0.3, 0.4, 0.5]),
        'tau': trial.suggest_categorical('tau', [0.001, 0.003, 0.005, 0.01]),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256])
    }
    
    return train_sac_trial(config, env, num_episodes, trial.number, print_freq)

def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization')
    parser.add_argument('--agent', type=str, choices=['dqn', 'ppo', 'sac'], required=True,
                        help='Agent to optimize')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Episodes per trial (default: 100)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials (default: 50)')
    parser.add_argument('--print-freq', type=int, default=25,
                        help='Print progress every N episodes (default: 25)')
    parser.add_argument('--db-url', type=str, 
                        default=os.environ.get('OPTUNA_DB_URL'),
                        help='Supabase PostgreSQL URL for parallel optimization (default: hardcoded Supabase)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: {agent}_hpo)')
    args = parser.parse_args()
    
    # Set study name
    study_name = args.study_name or f"{args.agent}_hpo"
    
    # Setup logging to file
    import sys
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{args.agent}_optuna.log"
    
    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    print(f"Logging to: {log_file}\n")
    
    # Create environment once
    print("Creating environment...")
    env = FinanceEnv()
    
    # Setup storage (Supabase or local SQLite)
    if args.db_url:
        storage = args.db_url
        print(f"Using Supabase storage: {study_name}")
        print("You can run this script in multiple terminals for parallel optimization!")
    else:
        storage = f"sqlite:///{args.agent}_optuna.db"
        print(f"Using local SQLite storage: {storage}")
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize'
    )
    
    # Select objective function
    objectives = {
        'dqn': objective_dqn,
        'ppo': objective_ppo,
        'sac': objective_sac
    }
    objective_fn = objectives[args.agent]
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna optimization for {args.agent.upper()}")
    print(f"Trials: {args.trials} | Episodes per trial: {args.episodes}")
    print(f"{'='*60}\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective_fn(trial, env, args.episodes, args.print_freq),
        n_trials=args.trials,
        show_progress_bar=True
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.2f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best params
    best_params_file = f'models/{args.agent}_best_params.json'
    os.makedirs('models', exist_ok=True)
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest parameters saved to {best_params_file}")


if __name__ == "__main__":
    main()
