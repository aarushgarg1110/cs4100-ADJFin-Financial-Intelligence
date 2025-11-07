"""
Tabular Q-Learning agent for personal finance
"""

import numpy as np
import pickle
from typing import Optional, Dict
from .base_agent import BaseFinancialAgent


class QLearningAgent(BaseFinancialAgent):
    """
    Tabular Q-Learning with state discretization.
    """
    
    def __init__(
        self,
        n_actions: int = 6,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        n_bins: int = 5,
        name: str = "Q_Learning_Agent"
    ):
        super().__init__(name)
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins
        
        # Q-table: dict mapping state -> np.array of Q-values
        self.Q_table: Dict[tuple, np.ndarray] = {}
        
        # Track updates per (state, action) for adaptive learning rate
        self.num_updates: Dict[tuple, np.ndarray] = {}
        
        # State bounds for discretization
        self.state_bounds = None
        
        self.training_steps = 0
        
        print(f"Initialized {name}")
        print(f"  Actions: {n_actions}")
        print(f"  Bins per dimension: {n_bins}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma: {gamma}")
        print(f"  Epsilon: {epsilon} → {epsilon_min} (decay: {epsilon_decay})")
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Convert continuous state to discrete bins.
        Similar to hash() function from class but for finance states.
        
        State structure (15 dimensions):
        [cash, stocks, bonds, real_estate, cc_debt, student_loan, 
         monthly_income, age, emergency_fund, stock_return_1m, 
         market_regime, interest_rate, recent_event, months_unemployed, month]
        """
        
        # Initialize bounds on first call
        if self.state_bounds is None:
            self.state_bounds = np.array([
                [0, 100000],      # cash
                [0, 200000],      # stocks
                [0, 100000],      # bonds
                [0, 300000],      # real_estate
                [0, 30000],       # cc_debt
                [0, 50000],       # student_loan
                [0, 15000],       # monthly_income
                [25, 55],         # age
                [0, 20000],       # emergency_fund
                [-0.2, 0.2],      # stock_return_1m
                [0, 2],           # market_regime (discrete)
                [0, 0.15],        # interest_rate
                [0, 3],           # recent_event (discrete)
                [0, 12],          # months_unemployed
                [0, 360]          # month
            ])
        
        # Clip to bounds
        clipped = np.clip(state, self.state_bounds[:, 0], self.state_bounds[:, 1])
        
        # Discretize each dimension
        discrete_state = []
        for i, value in enumerate(clipped):
            low, high = self.state_bounds[i]
            
            # Already discrete (market_regime, recent_event)
            if i in [10, 12]:
                discrete_state.append(int(value))
            else:
                # Bin continuous values
                if high > low:
                    bin_idx = int((value - low) / (high - low) * (self.n_bins - 1))
                    bin_idx = max(0, min(self.n_bins - 1, bin_idx))
                    discrete_state.append(bin_idx)
                else:
                    discrete_state.append(0)
        
        return tuple(discrete_state)
    
    def _get_q_values(self, state: tuple) -> np.ndarray:
        """Get Q-values for a state, initialize if not seen before"""
        if state not in self.Q_table:
            self.Q_table[state] = np.zeros(self.n_actions)
            self.num_updates[state] = np.zeros(self.n_actions)
        return self.Q_table[state]
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        Similar to class implementation.
        
        Args:
            observation: Continuous state observation
            deterministic: If True, always exploit (no exploration)
        """
        state = self._discretize_state(observation)
        q_values = self._get_q_values(state)
        
        # Exploitation (greedy)
        if deterministic or np.random.random() > self.epsilon:
            return int(np.argmax(q_values))
        
        # Exploration (random)
        return np.random.randint(0, self.n_actions)
    
    def learn_from_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """
        Q-learning update rule with adaptive learning rate.
        Based on class implementation: η = 1/(1 + N(s,a))
        
        Q(s,a) ← (1-η)Q(s,a) + η[r + γ max Q(s',a')]
        """
        # Discretize states
        state = self._discretize_state(obs)
        next_state = self._discretize_state(next_obs)
        
        # Get Q-values
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        
        # Adaptive learning rate: η = 1/(1 + N(s,a))
        self.num_updates[state][action] += 1
        eta = 1.0 / (1 + self.num_updates[state][action])
        
        # Compute target
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q_values)
        
        # Q-learning update
        self.Q_table[state][action] = (1 - eta) * q_values[action] + eta * target
        
        self.training_steps += 1
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = True):
        """
        Train Q-learning agent.
        Similar structure to class implementation.
        
        Args:
            env: Wrapped finance environment
            n_episodes: Number of training episodes
            verbose: Print progress
        """
        from tqdm import tqdm
        
        print(f"\n{'='*60}")
        print(f"Training {self.name}")
        print(f"{'='*60}")
        
        episode_rewards = []
        
        for episode in tqdm(range(n_episodes), desc="Training"):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                # Select action
                action = self.select_action(obs, deterministic=False)
                
                # Take step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Learn from experience
                self.learn_from_experience(obs, action, reward, next_obs, done)
                
                obs = next_obs
                episode_reward += reward
                steps += 1
            
            # Record episode
            episode_rewards.append(episode_reward)
            self.episode_rewards.append(episode_reward)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Logging
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{n_episodes} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Q-table: {len(self.Q_table):,} states")
        
        print(f"\n✓ Training complete!")
        print(f"  Episodes: {n_episodes}")
        print(f"  Final Q-table size: {len(self.Q_table):,} states")
        print(f"  Final epsilon: {self.epsilon:.3f}")
        print(f"  Mean reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
        
        return episode_rewards
    
    def save(self, path: str):
        """Save Q-table to pickle file (like class implementation)"""
        data = {
            'Q_table': self.Q_table,
            'num_updates': self.num_updates,
            'state_bounds': self.state_bounds,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'hyperparams': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'n_bins': self.n_bins,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Q-table saved to {path}")
    
    def load(self, path: str):
        """Load Q-table from pickle file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.Q_table = data['Q_table']
        self.num_updates = data['num_updates']
        self.state_bounds = data['state_bounds']
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']
        self.episode_rewards = data.get('episode_rewards', [])
        
        print(f"Q-table loaded from {path}")
        print(f"  States: {len(self.Q_table):,}")
        print(f"  Epsilon: {self.epsilon:.3f}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get Q-learning statistics"""
        stats = super().get_statistics()
        stats['q_table_size'] = len(self.Q_table)
        stats['epsilon'] = self.epsilon
        stats['training_steps'] = self.training_steps
        
        # State space coverage
        theoretical_max = self.n_bins ** 13 * 3 * 4  # 13 continuous, 2 discrete
        stats['state_coverage'] = len(self.Q_table) / theoretical_max if theoretical_max > 0 else 0
        
        return stats