import numpy as np
import gymnasium as gym
from .market_data import MarketDataManager

class FinanceEnv(gym.Env):
    """
    Enhanced personal finance environment for 30-year lifecycle simulation.
    
    State: 15 variables including assets, debts, income, market conditions, life events
    Action: 6 decisions for asset allocation and debt payments
    """
    
    # Class-level market data (shared across all instances)
    _market_data = None
    
    def __init__(self):
        super().__init__()
        
        # Action space: [stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(6,))
        
        # State space: 15 variables
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(15,))
        
        # Initialize market data manager (only once)
        if FinanceEnv._market_data is None:
            print("Loading market data (one-time setup)...")
            FinanceEnv._market_data = MarketDataManager()
            FinanceEnv._market_data.download_data()
        
        self.market_data = FinanceEnv._market_data
        
        # Classify regimes for all assets
        for asset in self.market_data.returns:
            self.market_data.classify_regimes(asset)
        
        print("Market data loaded successfully!")
        
        # ===== FINANCIAL CONSTANTS =====
        # Expense Model
        self.BASE_MONTHLY_EXPENSES = 2000  # Base living costs
        self.LIFESTYLE_SCALING_FACTOR = 0.15  # % of income for lifestyle expenses (reduced from 0.3)
        
        # Income Growth
        self.ANNUAL_RAISE_RATE = 1.035  # 3.5% annual raise
        self.CAREER_PLATEAU_AGE = 50  # Age when income growth slows
        self.LATE_CAREER_VARIATION = (0.98, 1.01)  # Income variation after plateau
        
        # Debt Payments
        self.CC_MIN_PAYMENT = 25  # Minimum credit card payment
        self.CC_MIN_PAYMENT_PCT = 0.02  # 2% of balance
        self.STUDENT_MIN_PAYMENT = 50  # Minimum student loan payment
        self.STUDENT_MIN_PAYMENT_PCT = 0.01  # 1% of balance
        
        # Reward Parameters
        self.NET_WORTH_SCALE = 100000  # Divisor for net worth reward
        self.BANKRUPTCY_THRESHOLD = -10000  # Net worth bankruptcy level
        self.BANKRUPTCY_PENALTY = 100  # Penalty for bankruptcy
        self.EXCESSIVE_DEBT_THRESHOLD = 15000  # Credit card debt threshold
        self.DEBT_PENALTY_SCALE = 100  # Divisor for debt penalty (reduced from 1000)
        self.EMERGENCY_FUND_MONTHS = 6  # Months of expenses for emergency fund
        self.EMERGENCY_FUND_BONUS = 5  # Reward for adequate emergency fund (increased from 2)
        self.DEBT_FREE_BONUS = 10  # Reward for being debt-free (increased from 4)
        
        # Simulation Parameters
        self.STARTING_AGE = 25
        self.SIMULATION_YEARS = 30
        self.SIMULATION_MONTHS = 360  # 30 years
        
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
        self.inflation, self.rates = self.market_data.simulate_macro_factors(months=self.SIMULATION_MONTHS)

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
        self.age = self.STARTING_AGE
        self.emergency_fund = np.random.uniform(500, 3000)
        
        # Market and life event state
        self.stock_return_1m = 0
        self.interest_rate = 0.05
        self.recent_event = 0   # 0=none, 1=job_loss, 2=medical, 3=raise
        self.months_unemployed = 0
        self.month = 0
        
        # Initialize market regime and macro data
        self.current_regime = 0
        self.inflation, self.rates = self.market_data.simulate_macro_factors()
        
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
        
        # Apply life events (affects income)
        self._apply_life_events()
        
        # Apply macroeconomic effects
        inflation = self.inflation[self.month]
        
        # Hybrid expense model: base living costs + lifestyle scaling
        # Lower earners spend higher %, higher earners save more %
        base_expenses = self.BASE_MONTHLY_EXPENSES * (1 + inflation) + self.LIFESTYLE_SCALING_FACTOR * self.monthly_income
        available_money = max(0, self.monthly_income - base_expenses)
        
        # Calculate debt payments first (from available money)
        cc_min = max(self.CC_MIN_PAYMENT, self.credit_card_debt * self.CC_MIN_PAYMENT_PCT)
        student_min = max(self.STUDENT_MIN_PAYMENT, self.student_loan * self.STUDENT_MIN_PAYMENT_PCT)
        
        total_cc_payment = min(cc_min + cc_payment * available_money, self.credit_card_debt, available_money)
        total_student_payment = min(student_min + student_payment * available_money, self.student_loan, available_money - total_cc_payment)
        
        # Reduce debt
        self.credit_card_debt = max(0, self.credit_card_debt - total_cc_payment)
        self.student_loan = max(0, self.student_loan - total_student_payment)
        
        # Money left after debt payments
        money_after_debt = available_money - total_cc_payment - total_student_payment
        
        # Emergency fund contribution (from remaining money)
        emergency_contribution = min(emergency_contrib * money_after_debt, money_after_debt)
        self.emergency_fund += emergency_contribution
        
        # Money left for investments
        money_for_investments = money_after_debt - emergency_contribution
        
        # Asset allocation
        if money_for_investments > 0:
            self.stocks += stock_alloc * money_for_investments
            self.bonds += bond_alloc * money_for_investments
            self.real_estate += re_alloc * money_for_investments
            self.cash += (1 - stock_alloc - bond_alloc - re_alloc) * money_for_investments
        
        # Apply interest to cash
        self.cash *= (1 + self.params['savings_apy']/12)
        
        # Apply interest on remaining debt (ONCE per month)
        self.credit_card_debt *= (1 + self.params['credit_card_apr']/12)
        self.student_loan *= (1 + self.params['student_loan_apr']/12)
        
        # Update time
        self.month += 1
        self.age = self.STARTING_AGE + self.month / 12
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done (30 years = 360 months)
        done = self.month >= self.SIMULATION_MONTHS - 1
        
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
        
        # Annual salary growth (applied once per year = every 12 months)
        if self.month > 0 and self.month % 12 == 0 and self.months_unemployed == 0:
            if self.age < self.CAREER_PLATEAU_AGE:
                # 3.5% annual raise until age 50
                self.monthly_income *= self.ANNUAL_RAISE_RATE
            elif self.age < 55:
                # Plateau/slight decline after 50
                self.monthly_income *= np.random.uniform(*self.LATE_CAREER_VARIATION)
        
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
        
        # Occasional promotion/bonus (5% annual chance for 10-15% bump)
        if np.random.rand() < self.params['raise_prob']:
            self.recent_event = 3
            self.monthly_income *= np.random.uniform(1.02, 1.07)
    
    def _calculate_reward(self):
        # Net worth
        net_worth = (self.cash + self.stocks + self.bonds + self.real_estate + 
                    self.emergency_fund - self.credit_card_debt - self.student_loan)
        
        # Base reward: net worth growth
        reward = net_worth / self.NET_WORTH_SCALE
        
        # Penalties
        if net_worth < self.BANKRUPTCY_THRESHOLD:
            reward -= self.BANKRUPTCY_PENALTY
            
        # Continuous debt penalty (penalize by monthly interest cost)
        if self.credit_card_debt > 0:
            monthly_cc_interest = self.credit_card_debt * (self.params['credit_card_apr'] / 12)
            reward -= monthly_cc_interest / self.DEBT_PENALTY_SCALE
            
        if self.student_loan > 0:
            monthly_student_interest = self.student_loan * (self.params['student_loan_apr'] / 12)
            reward -= monthly_student_interest / self.DEBT_PENALTY_SCALE
            
        # Extra penalty for excessive CC debt
        if self.credit_card_debt > self.EXCESSIVE_DEBT_THRESHOLD:
            reward -= (self.credit_card_debt - self.EXCESSIVE_DEBT_THRESHOLD) / self.DEBT_PENALTY_SCALE
            
        # Calculate emergency fund threshold based on BASE expenses only
        # (grows slowly with inflation, not income - more learnable for RL)
        inflation = self.inflation[self.month]
        base_monthly_expenses = self.BASE_MONTHLY_EXPENSES * (1 + inflation)
        emergency_threshold = base_monthly_expenses * self.EMERGENCY_FUND_MONTHS
        
        # Bonuses (scaled to be meaningful)
        if self.emergency_fund >= emergency_threshold:
            reward += self.EMERGENCY_FUND_BONUS
            
        if self.credit_card_debt == 0:
            reward += self.DEBT_FREE_BONUS
            
        return reward
