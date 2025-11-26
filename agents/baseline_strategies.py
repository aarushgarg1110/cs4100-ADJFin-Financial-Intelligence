"""
Rule-based financial strategies for baseline comparison
All inherit from BaseFinancialAgent for consistent evaluation

Adapted for discrete action space
Action = money_idx * len(INVEST_ALLOC) + invest_idx
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.finance_env import MONEY_ALLOC, INVEST_ALLOC
from .base_agent import BaseFinancialAgent

class SixtyFortyAgent(BaseFinancialAgent):
    """Traditional 60% stocks, 40% bonds portfolio"""
    
    def __init__(self):
        super().__init__("60/40 Rule")
    
    def get_action(self, state):
        # Money: Balanced Classic [60% invest, 20% debt, 20% emergency]
        # Invest: Classic 60/40 [60% stocks, 40% bonds, 0% RE]
        money_idx = 2  # Balanced Classic
        invest_idx = 6  # Classic 60/40
        return money_idx * len(INVEST_ALLOC) + invest_idx

class DebtAvalancheAgent(BaseFinancialAgent):
    """Pay highest interest debt first, conservative investing"""
    
    def __init__(self):
        super().__init__("Debt Avalanche")
    
    def get_action(self, state):
        # Money: Maximum Debt Reduction [30% invest, 50% debt, 20% emergency]
        # Invest: Conservative Bonds [40% stocks, 50% bonds, 10% RE]
        money_idx = 6  # Maximum Debt Reduction
        invest_idx = 4  # Conservative Bonds
        return money_idx * len(INVEST_ALLOC) + invest_idx

class EqualWeightAgent(BaseFinancialAgent):
    """Equal allocation across all asset classes"""
    
    def __init__(self):
        super().__init__("Equal Weight")
    
    def get_action(self, state):
        # Money: Balanced Classic [60% invest, 20% debt, 20% emergency]
        # Invest: Equal Weight [33% stocks, 33% bonds, 33% RE]
        money_idx = 2  # Balanced Classic
        invest_idx = 7  # Equal Weight
        return money_idx * len(INVEST_ALLOC) + invest_idx

class AgeBasedAgent(BaseFinancialAgent):
    """Age-based allocation: younger = more stocks (100-age rule, rounded to nearest 10%)"""
    
    def __init__(self):
        super().__init__("Age-Based")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # 100-age rule with 4 brackets
        if s['age'] < 35:
            # Young (25-34): ~70% stocks
            money_idx = 1  # Aggressive Invest [70, 15, 15]
            invest_idx = 1  # Growth Stocks [70, 20, 10]
        elif s['age'] < 42:
            # Mid-young (35-41): ~60% stocks
            money_idx = 2  # Balanced Classic [60, 20, 20]
            invest_idx = 2  # Moderate Growth [60, 30, 10]
        elif s['age'] < 50:
            # Mid-old (42-49): ~50% stocks
            money_idx = 2  # Balanced Classic [60, 20, 20]
            invest_idx = 3  # Balanced Portfolio [50, 40, 10]
        else:
            # Older (50+): ~40% stocks
            money_idx = 4  # Moderate Safety-Focused [50, 20, 30]
            invest_idx = 4  # Conservative Bonds [40, 50, 10]
        
        return money_idx * len(INVEST_ALLOC) + invest_idx

class MarkowitzAgent(BaseFinancialAgent):
    """Mean-variance optimization based on market regime"""
    
    def __init__(self):
        super().__init__("Markowitz")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        if s['market_regime'] == 2:  # Bear market
            # Conservative: Moderate Safety + Bond Heavy [30% stocks, 60% bonds]
            money_idx = 4  # Moderate Safety-Focused
            invest_idx = 8  # Bond Heavy
        elif s['market_regime'] == 1:  # Bull market
            # Aggressive: Aggressive Invest + Growth Stocks [70% stocks, 20% bonds]
            money_idx = 1  # Aggressive Invest
            invest_idx = 1  # Growth Stocks
        else:  # Normal market
            # Balanced: Balanced Classic + Balanced Portfolio [50% stocks, 40% bonds]
            money_idx = 2  # Balanced Classic
            invest_idx = 3  # Balanced Portfolio
        
        return money_idx * len(INVEST_ALLOC) + invest_idx
