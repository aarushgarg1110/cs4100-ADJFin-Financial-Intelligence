"""
Soft Actor-Critic agent tailored for the personal finance environment
"""

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .base_agent import BaseFinancialAgent


class GaussianPolicy(nn.Module):
    """Stochastic policy with tanh-squashed Gaussian actions"""

    def __init__(self, state_dim=15, action_dim=6, hidden_dim=256):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        """Reparameterized action sample + log-prob for SAC update"""
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        noise = normal.rsample()
        squashed = torch.tanh(noise)

        # Map from [-1, 1] to [0, 1] action bounds
        action = (squashed + 1) / 2

        # Log-prob with tanh correction
        log_prob = normal.log_prob(noise) - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def deterministic(self, state):
        """Mean action for evaluation"""
        mean, _ = self(state)
        squashed = torch.tanh(mean)
        return (squashed + 1) / 2


class QNetwork(nn.Module):
    """State-action value network"""

    def __init__(self, state_dim=15, action_dim=6, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ReplayBuffer:
    """Simple FIFO replay buffer"""

    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SACAgent(BaseFinancialAgent):
    """Soft Actor-Critic implementation"""

    def __init__(
        self,
        lr=3e-4,
        gamma=0.995,
        tau=0.005,
        alpha=0.2,
        target_entropy=None,
        batch_size=64,
        name="SAC_Agent",
    ):
        super().__init__(name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = 6

        # Networks
        self.actor = GaussianPolicy().to(self.device)
        self.q1 = QNetwork().to(self.device)
        self.q2 = QNetwork().to(self.device)
        self.target_q1 = QNetwork().to(self.device)
        self.target_q2 = QNetwork().to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy or -float(self.action_dim)
        self.batch_size = batch_size

        # Experience replay
        self.replay_buffer = ReplayBuffer()

        # Training flag
        self.training = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state):
        """Sample action for environment interaction"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            if self.training:
                action, _ = self.actor.sample(state_tensor)
            else:
                action = self.actor.deterministic(state_tensor)
        self.actor.train()
        return action.cpu().numpy().flatten()

    def _update_networks(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # Critic losses
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = nn.MSELoss()(current_q1, target_value)
        q2_loss = nn.MSELoss()(current_q2, target_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update targets
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        
        return actor_loss.item(), (q1_loss.item() + q2_loss.item()) / 2

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "target_q1_state_dict": self.target_q1.state_dict(),
                "target_q2_state_dict": self.target_q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
        self.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])
        if "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))

    def train(self, env, num_episodes=1000, seed=None):
        """Self-contained training loop (mirrors other agents)"""
        episode_rewards = []
        actor_losses = []
        critic_losses = []

        progress = tqdm(range(num_episodes), desc="Training SAC")
        for episode in progress:
            episode_seed = seed + episode if seed is not None else None
            state, _ = env.reset(seed=episode_seed)
            done = False
            total_reward = 0.0
            step_count = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, float(done))

                step_count += 1
                
                # Update every 30 steps
                if step_count % 30 == 0 and len(self.replay_buffer) >= self.batch_size:
                    actor_loss, critic_loss = self._update_networks()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                tqdm.write(f"[SAC] Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}")

        # Persist models
        import os
        os.makedirs("models", exist_ok=True)
        self.save("models/sac_model.pth")

        return episode_rewards, actor_losses, critic_losses
