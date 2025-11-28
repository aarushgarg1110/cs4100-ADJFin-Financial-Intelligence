import numpy as np
import gymnasium as gym
from collections import deque
from .market_data import MarketDataManager

# ===== DISCRETE ACTION SPACE DEFINITIONS =====
# 60 total actions: 10 money allocation strategies × 9 investment allocations

# Part 1: Money Allocation ([% invest, % debt, % emergency], description)
MONEY_ALLOC = [
    ([80, 10, 10], "Very Aggressive Invest"),
    ([70, 15, 15], "Aggressive Invest"), # Baseline Strats: AgeBased-Young, Markowitz-Bull
    ([60, 20, 20], "Balanced Classic"), # Baseline Strats: 60/40, EqualWeight, AgeBased-Mid, Markowitz-Normal
    ([50, 30, 20], "Moderate Debt-Focused"),
    ([50, 20, 30], "Moderate Safety-Focused"), # Baseline Strats: AgeBased-Old, Markowitz-Bear
    ([40, 40, 20], "Debt-Heavy"),
    ([30, 50, 20], "Maximum Debt Reduction"), # Baseline Strats: DebtAvalanche
    ([30, 20, 50], "Maximum Safety"),
    ([20, 35, 45], "Conservative Both"),
    ([40, 10, 50], "Income Protection"),
]

# Part 2: Investment Allocation ([% stocks, % bonds, % real estate], description)
INVEST_ALLOC = [
    ([80, 15, 5], "Very Aggressive Stocks"),
    ([70, 20, 10], "Growth Stocks"), # Baseline Strats: AgeBased-Young, Markowitz-Bull
    ([60, 30, 10], "Moderate Growth"), # Baseline Strats: AgeBased-Mid
    ([50, 40, 10], "Balanced Portfolio"), # Baseline Strats: AgeBased-Mid, Markowitz-Normal
    ([40, 50, 10], "Conservative Bonds"), # Baseline Strats: Debt Avalanche, AgeBased-Old
    ([30, 30, 40], "Real Estate Focus"),
    ([60, 40, 0], "Classic 60/40"),  # Baseline Strats: 60/40
    ([33, 33, 33], "Equal Weight"),  # Baseline Strats: EqualWeight
    ([30, 60, 10], "Bond Heavy"), # Baseline Strats: Markowitz-Bear
]

NUM_ACTIONS = len(MONEY_ALLOC) * len(INVEST_ALLOC)

# Generate action descriptions for all 60 combinations
ACTION_DESCRIPTIONS = {}
for money_idx in range(len(MONEY_ALLOC)):
    for invest_idx in range(len(INVEST_ALLOC)):
        action_idx = money_idx * len(INVEST_ALLOC) + invest_idx
        money_desc = MONEY_ALLOC[money_idx][1]
        invest_desc = INVEST_ALLOC[invest_idx][1]
        ACTION_DESCRIPTIONS[action_idx] = f"{money_desc} + {invest_desc}"

class FinanceEnv(gym.Env):
    """
    Enhanced personal finance environment for 30-year lifecycle simulation.
    
    State: 13 variables including net worth, assets, debts, income, market conditions, life events
    Action: 90 discrete strategies (10 money allocations x 9 investment allocations)
    """
    
    # Class-level market data (shared across all instances)
    _market_data = None
    
    def __init__(self, seed=42, sharpe_ratio=0.5):
        super().__init__()
        self.seed = seed
        self.sharpe_ratio = sharpe_ratio
        print(f"Initializing FinanceEnv with sharpe_ratio weight: {self.sharpe_ratio}")
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        
        # State space: 13 variables
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(13,))
        
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
        
        # Reward Parameters
        self.TARGET_NET_WORTH = 2_000_000  # $2M retirement goal
        self.BANKRUPTCY_THRESHOLD = -10_000  # Net worth bankruptcy level
        self.BANKRUPTCY_PENALTY = -5  # Penalty for bankruptcy
        self.EMERGENCY_TARGET = 12_000  # 6 months of base expenses
        self.EMERGENCY_BONUS = 0.2  # Bonus for adequate emergency fund
        self.DEBT_FREE_BONUS = 0.3  # Bonus for being debt-free
        self.EXCEED_BONUS_MAX = 0.3  # Max bonus for exceeding target
        
        # Simulation Parameters
        self.STARTING_AGE = 25
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

        self.reset(seed=seed)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize financial state
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
        
        # Sharpe ratio tracking
        self.portfolio_returns = deque(maxlen=12)  # Last 12 months
        self.prev_net_worth = None
        
        return self._get_state(), {}
    
    def _get_state(self):
        # Calculate net worth (primary metric for reward)
        net_worth = (self.stocks + self.bonds + self.real_estate + 
                     self.emergency_fund - self.credit_card_debt - self.student_loan)
        
        return np.array([
            net_worth,              # 0: Primary goal metric
            self.stocks,            # 1: Asset breakdown
            self.bonds,             # 2
            self.real_estate,       # 3
            self.credit_card_debt,  # 4: Debt levels
            self.student_loan,      # 5
            self.monthly_income,    # 6: Resources
            self.age,               # 7: Time/lifecycle signal
            self.emergency_fund,    # 8: Safety buffer
            self.stock_return_1m,   # 9: Market signal
            self.current_regime,    # 10: Market condition (0=normal, 1=bull, 2=bear)
            self.recent_event,      # 11: Life event (0=none, 1=job_loss, 2=medical, 3=raise)
            self.months_unemployed  # 12: Unemployment duration
        ])
    
    def step(self, action_idx):
        """
        Execute one timestep with discrete action.
        
        Args:
            action_idx: Integer 0-59 representing chosen strategy
        
        Returns:
            state, reward, done, truncated, info
        """
        # Decode discrete action into money and investment allocations
        money_idx = action_idx // len(INVEST_ALLOC)
        invest_idx = action_idx % len(INVEST_ALLOC)
        
        # Get allocation percentages
        invest_pct, debt_pct, emergency_pct = MONEY_ALLOC[money_idx][0]
        stock_pct, bond_pct, re_pct = INVEST_ALLOC[invest_idx][0]
        
        # Convert to decimals
        invest_pct /= 100
        debt_pct /= 100
        emergency_pct /= 100
        stock_pct /= 100
        bond_pct /= 100
        re_pct /= 100
        
        # Generate market returns
        self._update_market()
        
        # Apply life events (affects income)
        self._apply_life_events()
        
        # Calculate expenses (with inflation)
        inflation = self.inflation[self.month]
        base_expenses = self.BASE_MONTHLY_EXPENSES * (1 + inflation) + self.LIFESTYLE_SCALING_FACTOR * self.monthly_income
        available_money = max(0, self.monthly_income - base_expenses)
        
        # Apply money allocation
        debt_amount = available_money * debt_pct
        emergency_amount = available_money * emergency_pct
        invest_amount = available_money * invest_pct
        
        # Pay debts (prioritize high-interest CC over student loan)
        if self.credit_card_debt > 0 and self.student_loan > 0:
            # Split debt payment: 70% to CC (22% APR), 30% to student (6.5% APR)
            cc_payment = min(debt_amount * 0.7, self.credit_card_debt)
            student_payment = min(debt_amount * 0.3, self.student_loan)
        elif self.credit_card_debt > 0:
            cc_payment = min(debt_amount, self.credit_card_debt)
            student_payment = 0
        else:
            cc_payment = 0
            student_payment = min(debt_amount, self.student_loan)
        
        self.credit_card_debt -= cc_payment
        self.student_loan -= student_payment
        
        # Add to emergency fund
        self.emergency_fund += emergency_amount
        
        # Invest according to investment allocation
        self.stocks += invest_amount * stock_pct
        self.bonds += invest_amount * bond_pct
        self.real_estate += invest_amount * re_pct
        
        # Apply interest on remaining debt
        self.credit_card_debt *= (1 + self.params['credit_card_apr'] / 12)
        self.student_loan *= (1 + self.params['student_loan_apr'] / 12)
        
        # Update time
        self.month += 1
        self.age = self.STARTING_AGE + self.month / 12
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self.month >= self.SIMULATION_MONTHS
        
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
                # If emergency fund insufficient, remaining cost goes to credit card debt
                remaining_cost = cost - self.emergency_fund
                self.emergency_fund = 0
                self.credit_card_debt += remaining_cost
        
        # Occasional promotion/bonus (5% annual chance for 10-15% bump)
        if np.random.rand() < self.params['raise_prob']:
            self.recent_event = 3
            self.monthly_income *= np.random.uniform(1.02, 1.07)
    
    def _calculate_reward(self):
        """
        Hybrid reward: 50% wealth accumulation + 50% Sharpe ratio (risk-adjusted return)
        
        Components:
        - Wealth reward: net_worth / $1M (encourages getting rich)
        - Sharpe ratio: (return - risk_free) / volatility (encourages smart risk-taking)
        - Penalties: bankruptcy, debt interest
        - Bonuses: emergency fund, debt-free
        """
        # Calculate net worth
        net_worth = (self.stocks + self.bonds + self.real_estate + 
                     self.emergency_fund - self.credit_card_debt - self.student_loan)
        
        # === WEALTH REWARD ===
        wealth_reward = net_worth / 1_000_000  # $1M = 1.0, $2M = 2.0
        
        # === SHARPE RATIO REWARD ===
        if self.prev_net_worth is None or self.prev_net_worth <= 0:
            # First step or bankruptcy: no Sharpe calculation
            self.prev_net_worth = max(net_worth, 1)
            portfolio_return = 0
            sharpe_reward = 0
        else:
            # Calculate portfolio return
            portfolio_return = (net_worth - self.prev_net_worth) / self.prev_net_worth
            self.portfolio_returns.append(portfolio_return)
            
            # Calculate Sharpe ratio
            if len(self.portfolio_returns) > 1:
                volatility = np.std(self.portfolio_returns)
                risk_free_rate = 0.02 / 12  # 2% annual = 0.00167 monthly
                sharpe_reward = (portfolio_return - risk_free_rate) / (volatility + 1e-6)
            else:
                # Not enough data yet
                sharpe_reward = 0
            
            # Update for next step
            self.prev_net_worth = net_worth
        
        # === PENALTIES ===
        bankruptcy_penalty = self.BANKRUPTCY_PENALTY if net_worth < self.BANKRUPTCY_THRESHOLD else 0
        
        debt_interest_penalty = 0
        if self.credit_card_debt > 0:
            debt_interest_penalty -= (self.credit_card_debt * self.params['credit_card_apr'] / 12) / 1000
        if self.student_loan > 0:
            debt_interest_penalty -= (self.student_loan * self.params['student_loan_apr'] / 12) / 1000
        
        # === BONUSES ===
        emergency_bonus = self.EMERGENCY_BONUS if self.emergency_fund >= self.EMERGENCY_TARGET else 0
        debt_free_bonus = self.DEBT_FREE_BONUS if (self.credit_card_debt + self.student_loan) == 0 else 0
        
        # === HYBRID REWARD ===
        total_reward = (
            (1 - self.sharpe_ratio) * wealth_reward +
            self.sharpe_ratio * sharpe_reward +
            bankruptcy_penalty +
            debt_interest_penalty +
            emergency_bonus +
            debt_free_bonus
        )
        
        return np.clip(total_reward, -5, 10)
