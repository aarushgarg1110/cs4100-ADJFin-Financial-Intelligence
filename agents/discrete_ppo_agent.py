"""
Discrete PPO Agent for Financial Planning
Uses Categorical distribution for discrete action space (60 actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from .base_agent import BaseFinancialAgent


class PolicyNetwork(nn.Module):
    """Policy network that outputs logits for Categorical distribution"""
    
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


class ValueNetwork(nn.Module):
    """Value network that estimates state value"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class DiscretePPOAgent(BaseFinancialAgent):
    """
    Discrete PPO Agent using Categorical policy.
    
    Args:
        state_dim: Dimension of state space (default: 13)
        num_actions: Number of discrete actions (default: 90)
        lr: Learning rate (default: 3e-05)
        gamma: Discount factor (default: 0.98)
        clip_epsilon: PPO clipping parameter (default: 0.2)
        epochs: Number of optimization epochs per update (default: 10)
        batch_size: Batch size for training (default: 128)
        entropy_coef: Entropy bonus coefficient for exploration (default: 0.01)
    """
    
    def __init__(self, state_dim=13, num_actions=90, lr=3e-05, gamma=0.98,
                 clip_epsilon=0.2, epochs=10, batch_size=128, entropy_coef=0.01, name="Discrete_PPO"):
        super().__init__(name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim, num_actions).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        # Episode buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        # Training state
        self.training = True
    
    def get_action(self, state):
        """
        Select action from Categorical policy.
        
        Args:
            state: Current state (numpy array)
        
        Returns:
            action: Integer action index (0-89)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            dist = Categorical(logits=logits)
            
            if self.training:
                action = dist.sample()
            else:
                action = logits.argmax()
            
            log_prob = dist.log_prob(action)
        
        # Store for training
        if self.training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def update(self):
        """
        Update policy and value networks using PPO.
        Called at end of episode.
        """
        if len(self.states) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Calculate returns (discounted rewards)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(self.epochs):
            # Get current policy distribution
            logits = self.policy_net(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            
            # Get current value estimates
            values = self.value_net(states).squeeze()
            
            # Calculate advantages
            advantages = returns - values.detach()
            
            # Policy loss (PPO clipped objective)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Entropy bonus to encourage exploration
            entropy = dist.entropy().mean()
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.policy_optimizer.step()
            
            # Update value
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.value_optimizer.step()
        
        # Clear episode buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
    
    def train(self, env, num_episodes=1000, seed=None, print_freq=25, save_path='models/discrete_ppo_model.pth'):
        """Train Discrete PPO agent"""
        from tqdm import tqdm
        
        episode_rewards = []
        policy_losses = []
        value_losses = []
        
        for episode in tqdm(range(num_episodes), desc="Training Discrete PPO"):
            episode_seed = seed + episode if seed is not None else None
            state, _ = env.reset(seed=episode_seed)
            episode_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                self.store_reward(reward)
                
                state = next_state
                episode_reward += reward
            
            # Update at end of episode
            if len(self.states) > 0:
                # Calculate losses before update
                states = torch.FloatTensor(np.array(self.states)).to(self.device)
                actions = torch.LongTensor(self.actions).to(self.device)
                
                # Calculate returns
                returns = []
                G = 0
                for r in reversed(self.rewards):
                    G = r + self.gamma * G
                    returns.insert(0, G)
                returns = torch.FloatTensor(returns).to(self.device)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                # Get values for loss calculation
                values = self.value_net(states).squeeze()
                policy_loss = -torch.min(torch.zeros(1), returns - values.detach()).mean()
                value_loss = F.mse_loss(values, returns)
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                
                # Perform update
                self.update()
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % print_freq == 0 or (episode + 1) == num_episodes:
                avg_reward = np.mean(episode_rewards[-print_freq:]) if len(episode_rewards) >= print_freq else np.mean(episode_rewards)
                avg_policy = np.mean(policy_losses[-print_freq:]) if len(policy_losses) >= print_freq else 0
                avg_value = np.mean(value_losses[-print_freq:]) if len(value_losses) >= print_freq else 0
                tqdm.write(f"Ep {episode+1}/{num_episodes} | Reward: {avg_reward:.2f} | Policy Loss: {avg_policy:.4f} | Value Loss: {avg_value:.4f}")
        
        # Save model if path provided
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.save(save_path)
        
        return episode_rewards, policy_losses, value_losses
