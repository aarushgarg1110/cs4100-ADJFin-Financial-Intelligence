import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from market_data import MarketDataManager

class FinanceEnv(gym.Env):
    """
    Enhanced personal finance environment for 30-year lifecycle simulation.
    
    State: 15 variables including assets, debts, income, market conditions, life events
    Action: 6 decisions for asset allocation and debt payments
    """
    
    def __init__(self):
        super().__init__()
        
        # Action space: [stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment]
        self.action_space = spaces.Box(low=0, high=1, shape=(6,))
        
        # State space: 15 variables
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(15,))
        
        # Initialize market data manager
        print("Loading market data...")
        self.market_data = MarketDataManager()
        self.market_data.download_data()
        
        # Classify regimes for all assets
        for asset in self.market_data.returns:
            self.market_data.classify_regimes(asset)
        
        print("Market data loaded successfully!")
        
        # Financial parameters (realistic 2024 values)
        self.params = {
            'credit_card_apr': 0.22,
            'student_loan_apr': 0.065,
            'savings_apy': 0.045,
            'job_loss_prob': 0.0033,       # 4% annual
            'medical_prob': 0.008,         # 10% annual
            'raise_prob': 0.004,           # 5% annual
        }
        
        # Simulate macro factors for entire simulation (30 years = 360 months)
        self.inflation, self.rates = self.market_data.simulate_macro_factors(months=360)

        # Initialize current regime
        self.current_regime = 0  # start in Normal (0 = normal, 1 = bull, 2 = bear)

        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize financial state
        self.cash = np.random.uniform(1000, 5000)
        self.stocks = np.random.uniform(0, 10000)
        self.bonds = np.random.uniform(0, 5000)
        self.real_estate = 0
        
        self.credit_card_debt = np.random.uniform(0, 8000)
        self.student_loan = np.random.uniform(5000, 25000)
        
        self.monthly_income = np.random.uniform(3500, 6000)
        self.age = 25
        self.emergency_fund = np.random.uniform(500, 3000)
        
        # Market and life event state
        self.stock_return_1m = 0
        self.interest_rate = 0.05
        self.recent_event = 0   # 0=none, 1=job_loss, 2=medical, 3=raise
        self.months_unemployed = 0
        self.month = 0
        
        return self._get_state(), {}
    
    def _get_state(self):
        return np.array([
            self.cash,
            self.stocks,
            self.bonds,
            self.real_estate,
            self.credit_card_debt,
            self.student_loan,
            self.monthly_income,
            self.age,
            self.emergency_fund,
            self.stock_return_1m,
            self.current_regime,
            self.inflation[self.month],
            self.rates[self.month],
            self.recent_event,
            self.months_unemployed
        ])
    
    def step(self, action):
        stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment = action
        
        # Normalize allocations
        total_alloc = stock_alloc + bond_alloc + re_alloc
        if total_alloc > 0:
            stock_alloc /= total_alloc
            bond_alloc /= total_alloc  
            re_alloc /= total_alloc
        
        # Generate market returns
        self._update_market()

        # Apply macroeconomic effects
        inflation = self.inflation[self.month]
        interest = self.rates[self.month]

        # Inflation increases monthly expenses
        base_expenses = 2000 * (1 + inflation)
        available_money = max(0, self.monthly_income - base_expenses)

        # Interest rate affects debt growth
        self.credit_card_debt *= (1 + interest / 12)
        self.student_loan *= (1 + interest / 12)
        
        # Apply life events
        self._apply_life_events()
        
        # Calculate available money for investing
        available_money = max(0, self.monthly_income - 2000)  # Keep $2k for expenses
        
        # Asset allocation
        if available_money > 0:
            self.stocks += stock_alloc * available_money
            self.bonds += bond_alloc * available_money
            self.real_estate += re_alloc * available_money
            self.cash += (1 - stock_alloc - bond_alloc - re_alloc) * available_money
        
        # Apply interest and payments
        self.cash *= (1 + self.params['savings_apy']/12)
        
        # Debt payments
        cc_min = max(25, self.credit_card_debt * 0.02)
        student_min = max(50, self.student_loan * 0.01)
        
        total_cc_payment = min(cc_min + cc_payment * self.monthly_income, self.credit_card_debt)
        total_student_payment = min(student_min + student_payment * self.monthly_income, self.student_loan)
        
        self.credit_card_debt = max(0, self.credit_card_debt - total_cc_payment)
        self.student_loan = max(0, self.student_loan - total_student_payment)
        
        # Apply interest on remaining debt
        self.credit_card_debt *= (1 + self.params['credit_card_apr']/12)
        self.student_loan *= (1 + self.params['student_loan_apr']/12)
        
        # Emergency fund contribution
        self.emergency_fund += emergency_contrib * self.monthly_income
        
        # Update time
        self.month += 1
        self.age = 25 + self.month / 12
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done (30 years = 360 months)
        done = self.month >= 360
        
        return self._get_state(), reward, done, False, {}
    
    def _update_market(self):
        """Advance the market one month using realistic regime transitions."""
        # Update regime stochastically using transition matrix
        self.current_regime = self.market_data.next_regime(self.current_regime)

        # Sample asset returns based on the current regime
        self.stock_return_1m = self.market_data.sample_return('stocks', self.current_regime)
        self.bond_return_1m = self.market_data.sample_return('bonds', self.current_regime)
        self.re_return_1m = self.market_data.sample_return('real_estate', self.current_regime)
    
        # Apply returns to assets
        self.stocks *= (1 + self.stock_return_1m)
        self.bonds *= (1 + self.bond_return_1m)
        self.real_estate *= (1 + self.re_return_1m)
    
    def _apply_life_events(self):
        self.recent_event = 0
        
        # Job loss
        if self.months_unemployed == 0 and np.random.rand() < self.params['job_loss_prob']:
            self.recent_event = 1
            self.months_unemployed = np.random.randint(2, 7)
            
        if self.months_unemployed > 0:
            self.monthly_income = 0
            self.months_unemployed -= 1
            if self.months_unemployed == 0:
                # New job with different salary
                self.monthly_income = np.random.uniform(3500, 6000) * np.random.uniform(0.9, 1.2)
        
        # Medical emergency
        if np.random.rand() < self.params['medical_prob']:
            self.recent_event = 2
            cost = np.random.uniform(1000, 10000)
            if self.emergency_fund >= cost:
                self.emergency_fund -= cost
            else:
                self.cash -= (cost - self.emergency_fund)
                self.emergency_fund = 0
        
        # Salary raise
        if np.random.rand() < self.params['raise_prob']:
            self.recent_event = 3
            self.monthly_income *= np.random.uniform(1.02, 1.07)
    
    def _calculate_reward(self):
        # Net worth
        net_worth = (self.cash + self.stocks + self.bonds + self.real_estate + 
                    self.emergency_fund - self.credit_card_debt - self.student_loan)
        
        # Base reward: net worth growth
        reward = net_worth / 100000  # Scale down
        
        # Penalties
        if net_worth < -10000:  # Bankruptcy threshold
            reward -= 100
            
        if self.credit_card_debt > 15000:  # Excessive debt
            reward -= (self.credit_card_debt - 15000) / 1000
            
        # Bonuses
        if self.emergency_fund >= 6000:  # Good emergency fund
            reward += 1
            
        if self.credit_card_debt == 0:  # Debt free
            reward += 2
            
        return reward
