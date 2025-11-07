"""
Continuous DQN agent using deterministic policy gradient (DDPG-style)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
from .base_agent import BaseFinancialAgent

class ActorNetwork(nn.Module):
    """Deterministic policy network (actor)"""
    
    def __init__(self, state_dim=15, action_dim=6, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        return self.network(state)

class CriticNetwork(nn.Module):
    """Q-value network (critic)"""
    
    def __init__(self, state_dim=15, action_dim=6, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ContinuousDQNAgent(BaseFinancialAgent):
    """Continuous DQN using actor-critic architecture"""
    
    def __init__(self, lr=1e-3, gamma=0.99, tau=0.005, noise_std=0.1, name="Continuous_DQN"):
        super().__init__(name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork().to(self.device)
        self.critic = CriticNetwork().to(self.device)
        self.target_actor = ActorNetwork().to(self.device)
        self.target_critic = CriticNetwork().to(self.device)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.batch_size = 64
        
        # Training mode
        self.training = True
        
    def get_action(self, state):
        """Get action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
            
            if self.training:
                # Add exploration noise during training
                noise = torch.normal(0, self.noise_std, action.shape).to(self.device)
                action = torch.clamp(action + noise, 0, 1)
        
        return action.cpu().numpy().flatten()
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """Store experience and update networks"""
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Update networks if enough experiences
        if len(self.replay_buffer) > self.batch_size:
            self._update_networks()
    
    def _update_networks(self):
        """Update actor and critic networks"""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q.squeeze()
        
        current_q = self.critic(states, actions).squeeze()
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
    
    def _soft_update(self, target, source):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    def train(self, env, num_episodes=1000):
        """Train Continuous DQN agent"""
        episode_rewards = []
        
        for episode in tqdm(range(num_episodes), desc="Training DQN"):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                if len(self.replay_buffer) > self.batch_size:
                    self._update_networks()
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # Decay exploration noise
            self.noise_std = max(0.01, self.noise_std * 0.995)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                tqdm.write(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Noise: {self.noise_std:.3f}")
        
        # Save model
        import os
        os.makedirs('models', exist_ok=True)
        self.save('models/dqn_model.pth')
        
        return episode_rewards
