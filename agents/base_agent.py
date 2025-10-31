"""
Base class for all financial agents (RL and rule-based)
Ensures consistent interface across different strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class BaseFinancialAgent(ABC):
    """Abstract base class for financial decision agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.episode_rewards = []
        self.episode_net_worths = []
        
    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select action given current state observation
        
        Args:
            observation: numpy array from environment (normalized state)
            
        Returns:
            action: integer representing strategy choice (0-5)
        """
        pass
    
    @abstractmethod
    def learn_from_experience(
        self, 
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """
        Update agent based on experience tuple
        No-op for rule-based agents, actual learning for RL agents
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent state/weights"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent state/weights"""
        pass
    
    def record_episode(self, total_reward: float, final_net_worth: float):
        """Track episode statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_net_worths.append(final_net_worth)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_net_worth': np.mean(self.episode_net_worths),
            'std_net_worth': np.std(self.episode_net_worths),
            'episodes': len(self.episode_rewards)
        }


