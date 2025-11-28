"""
Discrete DQN Agent for Financial Planning
Uses standard Q-learning with discrete action space (60 actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .base_agent import BaseFinancialAgent


class QNetwork(nn.Module):
    """Q-Network that outputs Q-values for each discrete action"""
    
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DiscreteDQNAgent(BaseFinancialAgent):
    """
    Discrete DQN Agent using epsilon-greedy exploration.
    
    Args:
        state_dim: Dimension of state space (default: 13)
        num_actions: Number of discrete actions (default: 90)
        lr: Learning rate (default: 3e-05)
        gamma: Discount factor (default: 0.98)
        epsilon_start: Initial exploration rate (default: 1.0)
        epsilon_end: Final exploration rate (default: 0.05)
        epsilon_decay: Epsilon decay rate per episode (default: 0.9927)
        batch_size: Batch size for training (default: 128)
        target_update_freq: Frequency to update target network in gradient steps (default: 500)
        update_freq: Steps between gradient updates (default: 10)
        min_buffer_size: Minimum samples before learning starts (default: 7200)
    """
    
    def __init__(self, state_dim=13, num_actions=90, lr=3e-05, gamma=0.98,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9927,
                 batch_size=128, target_update_freq=500, update_freq=10, 
                 min_buffer_size=7200, name="Discrete_DQN"):
        super().__init__(name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.q_network = QNetwork(state_dim, num_actions).to(self.device)
        self.target_network = QNetwork(state_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq
        self.min_buffer_size = min_buffer_size
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training state
        self.training = True
        self.update_counter = 0
    
    def get_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
        
        Returns:
            action: Integer action index (0-89)
        """
        # Epsilon-greedy exploration
        if self.training and random.random() < self.epsilon:
            return random.randint(0, self.q_network.network[-1].out_features - 1)
        
        # Exploitation: choose action with highest Q-value
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update Q-network using experience replay.
        Performs one gradient step if enough samples in buffer.
        
        Returns:
            loss: Float loss value, or None if not enough samples
        """
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss (Huber loss is more robust to outliers than MSE)
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
    
    def train(self, env, num_episodes=1000, seed=None, print_freq=25, save_path='models/discrete_dqn_model.pth'):
        """Train Discrete DQN agent"""
        from tqdm import tqdm
        from collections import deque, Counter
        
        episode_rewards = []
        q_losses = []
        
        # Diversity tracking
        recent_actions = deque(maxlen=360)  # Track last episode worth of actions
        action_counts = Counter()
        
        for episode in tqdm(range(num_episodes), desc="Training Discrete DQN"):
            episode_seed = seed + episode if seed is not None else None
            state, _ = env.reset(seed=episode_seed)
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Track action for diversity metrics
                recent_actions.append(action)
                action_counts[action] += 1
                
                self.store_experience(state, action, reward, next_state, done or truncated)
                
                step_count += 1
                
                # Update every self.update_freq steps
                if step_count % self.update_freq == 0:
                    loss = self.update()
                    if loss is not None:
                        q_losses.append(loss)
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon
            self.decay_epsilon()
            
            if (episode + 1) % print_freq == 0 or (episode + 1) == num_episodes:
                avg_reward = np.mean(episode_rewards[-print_freq:]) if len(episode_rewards) >= print_freq else np.mean(episode_rewards)
                avg_loss = np.mean(q_losses[-print_freq:]) if len(q_losses) >= print_freq else 0
                
                # Calculate diversity metrics
                unique_actions = len(set(recent_actions))
                if len(action_counts) > 1:
                    probs = np.array(list(action_counts.values())) / sum(action_counts.values())
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                else:
                    entropy = 0.0
                
                tqdm.write(f"Ep {episode+1}/{num_episodes} | Reward: {avg_reward:.2f} | Q Loss: {avg_loss:.4f} | "
                          f"Eps: {self.epsilon:.3f} | Actions: {unique_actions}/90 | H: {entropy:.2f}")
        
        # Save model if path provided
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.save(save_path)
        
        return episode_rewards, q_losses
