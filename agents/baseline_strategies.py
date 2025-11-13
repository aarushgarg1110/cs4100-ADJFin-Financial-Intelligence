"""
Rule-based financial strategies for baseline comparison
All inherit from BaseFinancialAgent for consistent evaluation

Expert Strategies + Naive Strategies
"""

import numpy as np
from .base_agent import BaseFinancialAgent

class SixtyFortyAgent(BaseFinancialAgent):
    """Traditional 60% stocks, 40% bonds portfolio"""
    
    def __init__(self):
        super().__init__("60/40 Rule")
    
    def get_action(self, state):
        return np.array([0.6, 0.4, 0.0, 0.02, 0.01, 0.01])

class DebtAvalancheAgent(BaseFinancialAgent):
    """Pay highest interest debt first, conservative investing"""
    
    def __init__(self):
        super().__init__("Debt Avalanche")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # Conservative allocation while paying debt
        stock_alloc = 0.2
        bond_alloc = 0.6
        re_alloc = 0.2
        
        # Emergency fund priority
        emergency_contrib = 0.05 if s['emergency_fund'] < 6000 else 0.02
        
        # Aggressive debt payments
        if s['credit_card_debt'] > 0:
            cc_payment = 0.15  # Focus on high-interest debt
            student_payment = 0.02
        else:
            cc_payment = 0.0
            student_payment = 0.10  # Then focus on student loans
        
        return np.array([stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment])

class EqualWeightAgent(BaseFinancialAgent):
    """Equal allocation across all asset classes"""
    
    def __init__(self):
        super().__init__("Equal Weight")
    
    def get_action(self, state):
        return np.array([1/3, 1/3, 1/3, 0.02, 0.01, 0.01])

class AgeBasedAgent(BaseFinancialAgent):
    """Age-based allocation: (100-age)% stocks, age% bonds"""
    
    def __init__(self):
        super().__init__("Age-Based")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # Classic "100 minus age" rule
        stock_alloc = max(0.2, min(0.9, (100 - s['age']) / 100))
        bond_alloc = 1 - stock_alloc
        re_alloc = 0.0
        
        # Standard contributions
        emergency_contrib = 0.03
        cc_payment = 0.02
        student_payment = 0.02
        
        return np.array([stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment])

class MarkowitzAgent(BaseFinancialAgent):
    """Mean-variance optimization (simplified)"""
    
    def __init__(self):
        super().__init__("Markowitz")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # Simplified mean-variance: adjust based on market regime
        if s['market_regime'] == 2:  # Bear market
            stock_alloc = 0.3
            bond_alloc = 0.6
            re_alloc = 0.1
        elif s['market_regime'] == 1:  # Bull market
            stock_alloc = 0.7
            bond_alloc = 0.2
            re_alloc = 0.1
        else:  # Normal market
            stock_alloc = 0.5
            bond_alloc = 0.4
            re_alloc = 0.1
        
        return np.array([stock_alloc, bond_alloc, re_alloc, 0.03, 0.02, 0.02])

"""
Simple/naive financial strategies for comparison
"""

class AllStocksAgent(BaseFinancialAgent):
    """Naive: Put everything in stocks (high risk)"""
    
    def __init__(self):
        super().__init__("All Stocks")
    
    def get_action(self, state):
        return np.array([1.0, 0.0, 0.0, 0.1, 0.0, 0.0])  # 100% stocks, minimal emergency

class CashHoarderAgent(BaseFinancialAgent):
    """Naive: Keep everything in emergency fund (no growth)"""
    
    def __init__(self):
        super().__init__("Cash Hoarder")
    
    def get_action(self, state):
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # All to emergency fund

class DebtIgnorerAgent(BaseFinancialAgent):
    """Bad: Ignore debt, invest everything"""
    
    def __init__(self):
        super().__init__("Debt Ignorer")
    
    def get_action(self, state):
        return np.array([0.6, 0.4, 0.0, 0.0, 0.0, 0.0])  # Invest, ignore debt
