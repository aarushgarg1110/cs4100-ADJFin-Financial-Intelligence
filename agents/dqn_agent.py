"""
DQN agent implementation using stable-baselines3
Wrapper that conforms to BaseFinancialAgent interface
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from typing import Optional

from .base_agent import BaseFinancialAgent

class DQNFinancialAgent(BaseFinancialAgent):
    """Deep Q-Network agent for financial decision making"""
    
    def __init__(
        self,
        name: str = "DQN_Agent",
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 5000,
        batch_size: int = 64,
        gamma: float = 0.99,
        exploration_fraction: float = 0.3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        target_update_interval: int = 1000,
        verbose: int = 1
    ):
        super().__init__(name)
        
        # Store hyperparameters
        self.hyperparams = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'gamma': gamma,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps,
            'target_update_interval': target_update_interval,
            'verbose': verbose
        }
        
        self.model: Optional[DQN] = None
        self.env: Optional[gym.Env] = None
        
    def initialize(self, env: gym.Env):
        """
        Initialize DQN model with environment
        Must be called before training or inference
        """
        self.env = env
        
        self.model = DQN(
            policy='MlpPolicy',
            env=env,
            **self.hyperparams
        )
        
        print(f"Initialized {self.name} with hyperparameters:")
        for key, value in self.hyperparams.items():
            print(f"  {key}: {value}")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Select action using trained DQN policy
        
        Args:
            observation: Current state observation
            deterministic: If True, always pick best action. If False, use epsilon-greedy
        """
        if self.model is None:
            raise ValueError("Agent not initialized. Call initialize(env) first.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None):
        """
        Train the DQN agent
        
        Args:
            total_timesteps: Number of environment steps to train for
            callback: Optional callback for logging/checkpointing
        """
        if self.model is None:
            raise ValueError("Agent not initialized. Call initialize(env) first.")
        
        print(f"\nTraining {self.name} for {total_timesteps:,} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        print(f"Training complete!")
    
    def learn_from_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """
        For DQN, learning happens automatically during .learn()
        This method is kept for interface compatibility
        """
        pass
    
    def save(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str, env: Optional[gym.Env] = None):
        """Load trained model"""
        if env is None and self.env is None:
            raise ValueError("Must provide environment to load model")
        
        load_env = env if env is not None else self.env
        self.model = DQN.load(path, env=load_env)
        self.env = load_env
        print(f"Model loaded from {path}")
    
    def get_current_epsilon(self) -> float:
        """Get current exploration rate"""
        if self.model is None:
            return 0.0
        return self.model.exploration_rate


class TrainingCallback(BaseCallback):
    """Custom callback to track training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
        
        return True
    
    def _on_training_end(self) -> None:
        print(f"\nTraining Summary:")
        print(f"  Episodes completed: {len(self.episode_rewards)}")
        if self.episode_rewards:
            print(f"  Mean reward: {np.mean(self.episode_rewards):.2f}")
            print(f"  Mean episode length: {np.mean(self.episode_lengths):.2f}")
