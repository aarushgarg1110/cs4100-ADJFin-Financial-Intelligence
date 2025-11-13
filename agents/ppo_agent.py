"""
Proximal Policy Optimization (PPO) agent for continuous financial control
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta
from tqdm import tqdm
from .base_agent import BaseFinancialAgent

class PolicyNetwork(nn.Module):
    """Actor network outputting Beta distribution parameters"""
    
    def __init__(self, state_dim=15, action_dim=6, hidden_dim=128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Beta distribution parameters (alpha, beta) for each action
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()  # Ensure positive
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim), 
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, state):
        shared_out = self.shared(state)
        alpha = self.alpha_head(shared_out) + 1  # Add 1 for numerical stability
        beta = self.beta_head(shared_out) + 1
        return alpha, beta

class ValueNetwork(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim=15, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOAgent(BaseFinancialAgent):
    """PPO agent for continuous financial control"""
    
    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, name="PPO_Agent"):
        super().__init__(name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        # Training mode
        self.training = True
        
    def get_action(self, state):
        """Get action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            alpha, beta = self.policy_net(state_tensor)
            
            if self.training:
                # Sample from Beta distribution during training
                dist = Beta(alpha, beta)
                action = dist.sample()
            else:
                # Use mean during evaluation
                action = alpha / (alpha + beta)
        
        return action.cpu().numpy().flatten()
    
    def learn_from_experience(self, states, actions, rewards, next_states, dones):
        """Update policy using PPO algorithm"""
        # Convert to tensors (fix the warning)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards)
        values = self.value_net(states).squeeze()
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update (simplified single-step version)
        self._update_policy(states, actions, advantages, returns)
    
    def _compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns).to(self.device)
    
    def _update_policy(self, states, actions, advantages, returns):
        """Update policy and value networks"""
        # Get current policy outputs
        alpha, beta = self.policy_net(states)
        dist = Beta(alpha, beta)
        
        # Policy loss (simplified)
        log_probs = dist.log_prob(actions).sum(dim=1)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        values = self.value_net(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
    
    def train(self, env, num_episodes=1000):
        """Train PPO agent"""
        episode_rewards = []
        
        for episode in tqdm(range(num_episodes), desc="Training PPO"):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            states, actions, rewards = [], [], []
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                episode_reward += reward
            
            # Update after each episode
            if len(states) > 0:
                self.learn_from_experience(states, actions, rewards, [], [])
            
            episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                tqdm.write(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        # Save model
        import os
        os.makedirs('models', exist_ok=True)
        self.save('models/ppo_model.pth')
        
        return episode_rewards
