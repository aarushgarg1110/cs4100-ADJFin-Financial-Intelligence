"""
Wrapper to convert discrete actions (0-5) to continuous action vectors
for compatibility between DQN agent and FinanceEnv

This wrapper is REQUIRED because:
- DQN outputs discrete actions (integers 0-5)
- FinanceEnv expects continuous actions (array of 6 floats)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteToBoxActionWrapper(gym.ActionWrapper):
    """
    Converts discrete action space to Box action space.
    Maps 6 preset financial strategies to continuous action vectors.
    
    Action Format: [stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment]
    All values are percentages (0-1) of available money/income
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define 6 preset strategies as action vectors
        # Format: [stock%, bond%, real_estate%, emergency_fund%, cc_payment%, student_payment%]
        self.strategy_actions = {
            0: np.array([0.1, 0.3, 0.0, 0.4, 0.1, 0.1]),    # Ultra Conservative
            1: np.array([0.2, 0.4, 0.1, 0.2, 0.05, 0.05]),  # Conservative
            2: np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05]),  # Moderate
            3: np.array([0.6, 0.2, 0.1, 0.05, 0.025, 0.025]),  # Aggressive Growth
            4: np.array([0.1, 0.2, 0.0, 0.05, 0.3, 0.35]),  # Debt Crusher
            5: np.array([0.15, 0.25, 0.05, 0.45, 0.05, 0.05]),  # Emergency Builder
        }
        
        # Change action space from continuous Box to discrete
        self.action_space = spaces.Discrete(6)
        
        # Keep original observation space unchanged
        self.observation_space = env.observation_space
    
    def action(self, discrete_action):
        """
        Convert discrete action (0-5) to continuous action vector.
        
        This is called automatically by gym when you call env.step(discrete_action)
        
        Args:
            discrete_action: Integer from 0 to 5
            
        Returns:
            continuous_action: NumPy array of 6 floats
        """
        if discrete_action not in self.strategy_actions:
            raise ValueError(f"Invalid action {discrete_action}. Must be 0-5.")
        
        return self.strategy_actions[discrete_action]
    
    def reverse_action(self, continuous_action):
        """
        Optional: convert continuous action back to closest discrete action.
        Useful for analysis/debugging.
        
        Args:
            continuous_action: NumPy array of 6 floats
            
        Returns:
            closest_discrete_action: Integer 0-5
        """
        min_dist = float('inf')
        closest_action = 0
        
        for action_id, strategy_vector in self.strategy_actions.items():
            # Euclidean distance
            dist = np.linalg.norm(continuous_action - strategy_vector)
            if dist < min_dist:
                min_dist = dist
                closest_action = action_id
        
        return closest_action
    
    def get_strategy_description(self, action):
        """
        Get human-readable description of a strategy.
        
        Args:
            action: Integer 0-5
            
        Returns:
            description: String describing the strategy
        """
        descriptions = {
            0: "Ultra Conservative: Heavy bonds (30%), large emergency fund (40%), minimal debt payments. For risk-averse or near-retirement.",
            1: "Conservative: Balanced bonds/stocks (40%/20%), good emergency fund (20%). Traditional safe approach.",
            2: "Moderate: Balanced growth (40% stocks, 30% bonds). Standard long-term strategy.",
            3: "Aggressive Growth: High stock allocation (60%), minimal safety net. For young investors with high risk tolerance.",
            4: "Debt Crusher: Prioritize debt elimination (30% CC, 35% student loans). Get out of debt fast.",
            5: "Emergency Builder: Build cash reserves aggressively (45% to emergency fund). For unstable income or job insecurity."
        }
        return descriptions.get(action, "Unknown strategy")
    
    def get_strategy_vector(self, action):
        """
        Get the actual continuous action vector for a given strategy.
        
        Args:
            action: Integer 0-5
            
        Returns:
            vector: NumPy array showing the allocation percentages
        """
        return self.strategy_actions.get(action, None)
    
    def print_all_strategies(self):
        """Print all available strategies with their allocations."""
        labels = ['Stocks', 'Bonds', 'Real Estate', 'Emergency Fund', 'CC Payment', 'Student Payment']
        
        print("\n" + "="*80)
        print("AVAILABLE STRATEGIES")
        print("="*80)
        
        for action_id in range(6):
            desc = self.get_strategy_description(action_id)
            vector = self.strategy_actions[action_id]
            
            print(f"\nAction {action_id}: {desc.split(':')[0]}")
            print(f"  Description: {desc.split(':', 1)[1].strip()}")
            print(f"  Allocations:")
            for label, value in zip(labels, vector):
                print(f"    {label:20s}: {value:5.1%}")


# Convenience function
def wrap_finance_env(env):
    """
    Wrap FinanceEnv to work with discrete actions.
    
    Usage:
        base_env = FinanceEnv()
        env = wrap_finance_env(base_env)
        
        # Now env accepts discrete actions (0-5)
        obs, _ = env.reset()
        action = 3  # Aggressive Growth
        obs, reward, done, truncated, info = env.step(action)
    
    Args:
        env: FinanceEnv instance
        
    Returns:
        wrapped_env: Environment with discrete action space
    """
    return DiscreteToBoxActionWrapper(env)


# Test/demonstration code
if __name__ == "__main__":
    print("Discrete Action Wrapper Demo")
    print("="*80)
    
    # This would normally be your FinanceEnv
    # For demo, we'll just show the mappings
    
    wrapper = DiscreteToBoxActionWrapper(None)
    wrapper.print_all_strategies()
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    print("""
from environment.finance_env import FinanceEnv
from agents.discrete_action_wrapper import wrap_finance_env

# Create and wrap environment
base_env = FinanceEnv()
env = wrap_finance_env(base_env)

# Now use discrete actions!
obs, _ = env.reset()

# Agent picks action 3 (Aggressive Growth)
action = 3
obs, reward, done, truncated, info = env.step(action)

# Behind the scenes, wrapper converts:
# 3 → [0.6, 0.2, 0.1, 0.05, 0.025, 0.025]
# and passes that to FinanceEnv
""")