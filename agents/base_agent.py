"""
Unified base class for all financial agents (RL and rule-based)
Ensures consistent interface for fair evaluation
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseFinancialAgent(ABC):
    """Abstract base class for financial decision agents"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get continuous action given current state
        
        Args:
            state: Environment state array (15 dimensions)
            
        Returns:
            action: Continuous action array [stock_alloc, bond_alloc, re_alloc, 
                   emergency_contrib, cc_payment, student_payment]
                   All values between 0-1
        """
        pass
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """Update agent from experience (no-op for rule-based agents)"""
        pass
    
    def save(self, path: str):
        """Save agent state/weights"""
        pass
    
    def load(self, path: str):
        """Load agent state/weights"""
        pass
    
    def _parse_state(self, state):
        """Helper to parse state array into meaningful variables"""
        return {
            'net_worth': state[0],
            'stocks': state[1], 
            'bonds': state[2],
            'real_estate': state[3],
            'credit_card_debt': state[4],
            'student_loan': state[5],
            'monthly_income': state[6],
            'age': state[7],
            'emergency_fund': state[8],
            'stock_return_1m': state[9],
            'market_regime': state[10],
            'recent_event': state[11],
            'months_unemployed': state[12]
        }
