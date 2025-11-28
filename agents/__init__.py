"""
Financial agents for reinforcement learning research
"""

from .base_agent import BaseFinancialAgent
from .baseline_strategies import (
    SixtyFortyAgent,
    DebtAvalancheAgent,
    EqualWeightAgent,
    AgeBasedAgent,
    MarkowitzAgent
)
# Continuous RL agents (original)
from .ppo_agent import PPOAgent
from .continuous_dqn_agent import ContinuousDQNAgent
from .sac_agent import SACAgent

# Discrete RL agents (new - for 60 discrete actions)
from .discrete_dqn_agent import DiscreteDQNAgent
from .discrete_ppo_agent import DiscretePPOAgent

__all__ = [
    'BaseFinancialAgent',
    'SixtyFortyAgent',
    'DebtAvalancheAgent',
    'EqualWeightAgent', 
    'AgeBasedAgent',
    'MarkowitzAgent',
    'PPOAgent',
    'ContinuousDQNAgent',
    'SACAgent',
    'DiscreteDQNAgent',
    'DiscretePPOAgent',
]
