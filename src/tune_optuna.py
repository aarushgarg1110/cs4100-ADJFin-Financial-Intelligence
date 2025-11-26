"""
Optuna-based hyperparameter optimization for discrete agents
Run multiple instances in parallel: each terminal runs this script simultaneously
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import numpy as np
import optuna
import torch
from tqdm import tqdm
from environment.finance_env import FinanceEnv
from agents import DiscreteDQNAgent, DiscretePPOAgent


def train_trial(agent_type, config, env, num_episodes, trial_num, print_freq=50):
    """Generic training function for any agent type"""
    print(f"\nTrial {trial_num}:")
    
    # Create agent based on type
    if agent_type == 'dqn':
        print(f"  lr={config['lr']:.6f}, batch_size={config['batch_size']}, epsilon_decay={config['epsilon_decay']:.4f}, target_update_freq={config['target_update_freq']}, gamma={config['gamma']:.2f}")
        agent = DiscreteDQNAgent(
            lr=config['lr'],
            batch_size=config['batch_size'],
            epsilon_decay=config['epsilon_decay'],
            target_update_freq=config['target_update_freq'],
            gamma=config['gamma']
        )
    
    elif agent_type == 'ppo':
        print(f"  lr={config['lr']:.6f}, gamma={config['gamma']:.3f}, clip_epsilon={config['clip_epsilon']:.2f}, epochs={config['epochs']}")
        agent = DiscretePPOAgent(
            lr=config['lr'],
            gamma=config['gamma'],
            clip_epsilon=config['clip_epsilon'],
            epochs=config['epochs']
        )
    
    # Train using agent's method
    results = agent.train(
        env,
        num_episodes=num_episodes,
        seed=42,
        print_freq=print_freq,
        save_path=f'models/optuna_trials/{agent_type}_trial_{trial_num}.pth'
    )
    
    # Extract episode rewards (first element for both DQN and PPO)
    episode_rewards = results[0]
    
    final_score = np.mean(episode_rewards[-10:])
    print(f"  Trial {trial_num} complete: {final_score:.2f}\n")
    return final_score


def objective_dqn(trial, env, num_episodes, print_freq):
    """Optuna objective for DQN"""
    config = {
        'lr': trial.suggest_categorical('lr', [3e-06, 1e-05, 3e-05]),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'epsilon_decay': trial.suggest_categorical('epsilon_decay', [0.990, 0.995, 0.998]),
        'target_update_freq': trial.suggest_categorical('target_update_freq', [2, 5, 10]),
        'gamma': trial.suggest_categorical('gamma', [0.95, 0.98, 0.99, 1.0])
    }
    return train_trial('dqn', config, env, num_episodes, trial.number, print_freq)


def objective_ppo(trial, env, num_episodes, print_freq):
    """Optuna objective for PPO"""
    config = {
        'lr': trial.suggest_categorical('lr', [1e-4, 3e-4, 1e-3, 3e-3]),
        'gamma': trial.suggest_categorical('gamma', [0.95, 0.98, 0.99, 1.0]),
        'clip_epsilon': trial.suggest_categorical('clip_epsilon', [0.1, 0.2, 0.3]),
        'epochs': trial.suggest_categorical('epochs', [5, 10, 15, 20])
    }
    return train_trial('ppo', config, env, num_episodes, trial.number, print_freq)


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization')
    parser.add_argument('--agent', type=str, choices=['dqn', 'ppo'], required=True,
                        help='Agent to optimize')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Episodes per trial (default: 100)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials (default: 50)')
    parser.add_argument('--print-freq', type=int, default=25,
                        help='Print progress every N episodes (default: 25)')
    parser.add_argument('--db-url', type=str, 
                        default=os.environ.get('OPTUNA_DB_URL'),
                        help='Supabase PostgreSQL URL for parallel optimization (default: sqllite)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: {agent}_hpo)')
    args = parser.parse_args()
    
    # Set study name
    study_name = args.study_name or f"{args.agent}_hpo"
    
    # Create environment once
    print("Creating environment...")
    os.makedirs('models', exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    env = FinanceEnv()
    
    # Setup storage (Supabase or local SQLite)
    if args.db_url:
        storage = args.db_url
        print(f"Using Supabase storage: {study_name}")
    else:
        storage = f"sqlite:///{args.agent}_optuna.db"
        print(f"Using local SQLite storage: {storage}")

    print("You can run this script in multiple terminals for parallel optimization!")
    
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

    # Save the best model in the hpo directory to models directory
    best_model_src = f'models/optuna_trials/{args.agent}_trial_{study.best_trial.number}.pth'
    best_model_dst = f'models/{args.agent}_best_model.pth'
    os.replace(best_model_src, best_model_dst)
    print(f"Best model saved to {best_model_dst}")


if __name__ == "__main__":
    main()
