"""
Rule-based baseline agents for comparison
These follow fixed strategies without learning
"""

import numpy as np
from .base_agent import BaseFinancialAgent

class RandomAgent(BaseFinancialAgent):
    """Selects random actions - worst-case baseline"""
    
    def __init__(self, n_actions: int = 6):
        super().__init__("Random_Agent")
        self.n_actions = n_actions
    
    def select_action(self, observation: np.ndarray) -> int:
        return np.random.randint(0, self.n_actions)
    
    def learn_from_experience(self, obs, action, reward, next_obs, done):
        pass  # No learning
    
    def save(self, path: str):
        pass  # Nothing to save
    
    def load(self, path: str):
        pass  # Nothing to load


class ConservativeAgent(BaseFinancialAgent):
    """Always chooses the most conservative strategy"""
    
    def __init__(self):
        super().__init__("Conservative_Agent")
    
    def select_action(self, observation: np.ndarray) -> int:
        return 0  # Always action 0 (ultra conservative)
    
    def learn_from_experience(self, obs, action, reward, next_obs, done):
        pass
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass


class AggressiveAgent(BaseFinancialAgent):
    """Always chooses aggressive growth strategy"""
    
    def __init__(self):
        super().__init__("Aggressive_Agent")
    
    def select_action(self, observation: np.ndarray) -> int:
        return 3  # Always action 3 (aggressive growth)
    
    def learn_from_experience(self, obs, action, reward, next_obs, done):
        pass
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass


class AdaptiveRuleAgent(BaseFinancialAgent):
    """
    Rule-based agent that adapts based on simple heuristics
    Good baseline to compare against learned policy
    """
    
    def __init__(self):
        super().__init__("Adaptive_Rule_Agent")
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Simple rules based on observation:
        - High debt → Debt Crusher (action 4)
        - Low cash → Emergency Builder (action 5)
        - Young + low debt → Aggressive (action 3)
        - Old → Conservative (action 1)
        - Default → Moderate (action 2)
        """
        # Assuming observation order matches state space
        # [cash, stocks, bonds, RE, cc_debt, student_loan, income, age, market, events]
        
        cash_norm = observation[0]
        cc_debt_norm = observation[4]
        student_loan_norm = observation[5]
        age_norm = observation[7]
        
        # High debt priority
        total_debt = cc_debt_norm + student_loan_norm
        if total_debt > 0.6:  # High debt
            return 4  # Debt Crusher
        
        # Low emergency fund
        if cash_norm < 0.2:
            return 5  # Emergency Builder
        
        # Young and low debt → aggressive
        if age_norm < 0.4 and total_debt < 0.3:
            return 3  # Aggressive Growth
        
        # Older → conservative
        if age_norm > 0.7:
            return 1  # Conservative
        
        # Default to moderate
        return 2  # Moderate
    
    def learn_from_experience(self, obs, action, reward, next_obs, done):
        pass
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
