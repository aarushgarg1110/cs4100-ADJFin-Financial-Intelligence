import numpy as np

class BaselineAgent:
    """Base class for all baseline financial strategies."""
    
    def __init__(self, name="BaselineAgent"):
        self.name = name
    
    def get_action(self, state):
        """
        Return action array: [stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment]
        All values should be between 0 and 1 (percentages of available money/income)
        """
        raise NotImplementedError
    
    def _parse_state(self, state):
        """Helper to parse state array into meaningful variables."""
        return {
            'cash': state[0],
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
            'interest_rate': state[11],
            'recent_event': state[12],
            'months_unemployed': state[13],
            'month': state[14]
        }


class SixtyFortyAgent(BaselineAgent):
    """Traditional 60% stocks, 40% bonds portfolio with minimum debt payments."""
    
    def __init__(self):
        super().__init__("60/40 Rule")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # 60% stocks, 40% bonds, 0% real estate
        stock_alloc = 0.6
        bond_alloc = 0.4
        re_alloc = 0.0
        
        # Small emergency fund contribution (2% of income)
        emergency_contrib = 0.02
        
        # Minimum debt payments only (1% extra each)
        cc_payment = 0.01
        student_payment = 0.01
        
        return np.array([stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment])


class DebtAvalancheAgent(BaselineAgent):
    """Pay highest interest debt first, then invest conservatively."""
    
    def __init__(self):
        super().__init__("Debt Avalanche")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # Conservative allocation while paying debt
        stock_alloc = 0.2
        bond_alloc = 0.6
        re_alloc = 0.2
        
        # Build emergency fund first
        emergency_contrib = 0.05 if s['emergency_fund'] < 6000 else 0.02
        
        # Aggressive debt payments (credit cards are higher interest)
        if s['credit_card_debt'] > 0:
            cc_payment = 0.15  # 15% of income to credit cards
            student_payment = 0.02  # Minimum to student loans
        else:
            cc_payment = 0.0
            student_payment = 0.10  # Focus on student loans after CC paid off
        
        return np.array([stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment])


class EqualWeightAgent(BaselineAgent):
    """Equal allocation across all asset classes."""
    
    def __init__(self):
        super().__init__("Equal Weight")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # Equal weight across all assets (33.33% each)
        stock_alloc = 1/3
        bond_alloc = 1/3
        re_alloc = 1/3
        
        # Moderate emergency fund
        emergency_contrib = 0.03
        
        # Balanced debt payments
        cc_payment = 0.05
        student_payment = 0.05
        
        return np.array([stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment])


class AgeBasedAgent(BaselineAgent):
    """'100 minus age' rule for stock allocation."""
    
    def __init__(self):
        super().__init__("Age-Based")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # "100 minus age" rule for stocks
        stock_percentage = max(0.2, min(0.8, (100 - s['age']) / 100))
        bond_percentage = 1 - stock_percentage
        
        stock_alloc = stock_percentage * 0.9  # 90% of allocation to stocks/bonds
        bond_alloc = bond_percentage * 0.9
        re_alloc = 0.1  # 10% to real estate
        
        # Age-appropriate emergency fund (older = more conservative)
        emergency_contrib = 0.02 + (s['age'] - 25) * 0.001
        
        # Moderate debt payments
        cc_payment = 0.06
        student_payment = 0.04
        
        return np.array([stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment])


class MarkowitzAgent(BaselineAgent):
    """Simplified Modern Portfolio Theory - optimized risk/return."""
    
    def __init__(self):
        super().__init__("Markowitz")
    
    def get_action(self, state):
        s = self._parse_state(state)
        
        # Simplified MPT: adjust allocation based on market regime
        if s['market_regime'] == 1:  # Bull market
            stock_alloc = 0.7
            bond_alloc = 0.2
            re_alloc = 0.1
        elif s['market_regime'] == 2:  # Bear market
            stock_alloc = 0.3
            bond_alloc = 0.6
            re_alloc = 0.1
        else:  # Normal market
            stock_alloc = 0.5
            bond_alloc = 0.4
            re_alloc = 0.1
        
        # Risk-adjusted emergency fund
        emergency_contrib = 0.04 if s['market_regime'] == 2 else 0.02
        
        # Optimize debt payments (higher interest first)
        cc_payment = 0.08
        student_payment = 0.03
        
        return np.array([stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment])


# Convenience function to get all agents
def get_all_baseline_agents():
    """Return list of all baseline agents for comparison."""
    return [
        SixtyFortyAgent(),
        DebtAvalancheAgent(), 
        EqualWeightAgent(),
        AgeBasedAgent(),
        MarkowitzAgent()
    ]


if __name__ == "__main__":
    # Test all baseline agents
    print("Testing Baseline Agents...")
    
    # Mock state for testing
    test_state = np.array([
        5000,    # cash
        10000,   # stocks
        5000,    # bonds  
        0,       # real_estate
        3000,    # credit_card_debt
        15000,   # student_loan
        4000,    # monthly_income
        28,      # age
        2000,    # emergency_fund
        0.01,    # stock_return_1m
        0,       # market_regime (normal)
        0.05,    # interest_rate
        0,       # recent_event
        0,       # months_unemployed
        12       # month
    ])
    
    agents = get_all_baseline_agents()
    
    for agent in agents:
        action = agent.get_action(test_state)
        print(f"\n{agent.name}:")
        print(f"  Stock: {action[0]:.1%}, Bond: {action[1]:.1%}, RE: {action[2]:.1%}")
        print(f"  Emergency: {action[3]:.1%}, CC Pay: {action[4]:.1%}, Student Pay: {action[5]:.1%}")
    
    print("\nSUCCESS: All baseline agents working!")