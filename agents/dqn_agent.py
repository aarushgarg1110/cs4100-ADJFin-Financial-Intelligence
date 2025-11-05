"""
Deep Q-Network (DQN) implementation using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional, Tuple
from .base_agent import BaseFinancialAgent


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    Architecture: Input → Hidden → Hidden → Output
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """Forward pass: state → Q-values for all actions"""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    Stores transitions: (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseFinancialAgent):
    """
    Deep Q-Network agent from scratch.
    Implements: experience replay, target network, epsilon-greedy exploration.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 6,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        hidden_dim: int = 128,
        name: str = "DQN_Agent"
    ):
        super().__init__(name)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks (policy and target)
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_steps = 0
        self.losses = []
        
        print(f"Initialized {name}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma: {gamma}")
        print(f"  Epsilon: {epsilon_start} → {epsilon_end} (decay: {epsilon_decay})")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Target update freq: {target_update_freq}")
        print(f"  Device: {self.device}")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            observation: State observation
            deterministic: If True, always exploit (greedy)
        """
        # Exploitation
        if deterministic or np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return int(q_values.argmax().item())
        
        # Exploration
        return np.random.randint(0, self.action_dim)
    
    def learn_from_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """
        Store experience and train if buffer is large enough.
        """
        # Add to replay buffer
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        
        # Train if we have enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """
        Perform one training step:
        1. Sample batch from replay buffer
        2. Compute TD target using target network
        3. Update policy network
        4. Update target network (periodically)
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values: r + γ max Q'(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (MSE)
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (stabilize training)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track loss
        self.losses.append(loss.item())
        self.training_steps += 1
        
        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = True):
        """
        Train DQN agent.
        
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
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Logging
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                mean_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode+1}/{n_episodes} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"Loss: {mean_loss:.4f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Buffer: {len(self.replay_buffer):,}")
        
        print(f"\n✓ Training complete!")
        print(f"  Episodes: {n_episodes}")
        print(f"  Training steps: {self.training_steps:,}")
        print(f"  Final epsilon: {self.epsilon:.3f}")
        print(f"  Mean reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  Mean loss (last 100): {np.mean(self.losses[-100:]):.4f}")
        
        return episode_rewards
    
    def save(self, path: str):
        """Save model weights and hyperparameters"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards,
            'hyperparams': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }, path)
        
        print(f"DQN model saved to {path}")
    
    def load(self, path: str):
        """Load model weights and hyperparameters"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.losses = checkpoint.get('losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        
        print(f"DQN model loaded from {path}")
        print(f"  Training steps: {self.training_steps:,}")
        print(f"  Epsilon: {self.epsilon:.3f}")
    
    def get_statistics(self):
        """Get DQN statistics"""
        stats = super().get_statistics()
        stats['training_steps'] = self.training_steps
        stats['epsilon'] = self.epsilon
        stats['buffer_size'] = len(self.replay_buffer)
        stats['mean_loss'] = np.mean(self.losses[-100:]) if self.losses else 0
        return stats