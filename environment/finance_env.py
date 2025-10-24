# environment/finance_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FinanceEnv(gym.Env):
    """
    Custom Gym environment for personal finance simulation.
    State: [income, savings, debt, investments, emergency_fund, month]
    Action: % allocation to [invest, save, pay_debt]
    """

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(3,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,))
        self.reset()

    def reset(self, seed=None, options=None):
        self.income = 4000
        self.savings = 2000
        self.debt = 10000
        self.investments = 0
        self.emergency = 500
        self.month = 0
        state = np.array([self.income, self.savings, self.debt, self.investments, self.emergency, self.month])
        return state, {}

    def step(self, action):
        invest, save, debt_pay = action
        # basic transition logic placeholder
        self.investments += invest * self.income
        self.savings += save * self.income
        self.debt -= debt_pay * self.income
        self.debt = max(0, self.debt)
        self.month += 1

        # reward = net worth change
        net_worth = self.savings + self.investments - self.debt
        reward = net_worth / 10000
        done = self.month >= 12
        state = np.array([self.income, self.savings, self.debt, self.investments, self.emergency, self.month])
        return state, reward, done, False, {}
