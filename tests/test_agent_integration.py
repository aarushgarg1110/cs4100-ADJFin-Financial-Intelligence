"""
Integration test for agent classes
Tests that agents work with a dummy environment
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.dqn_agent import DQNFinancialAgent
from agents.baseline_agents import RandomAgent, AdaptiveRuleAgent

class DummyFinanceEnv(gym.Env):
    """Minimal environment for testing agent interface"""
    
    def __init__(self):
        super().__init__()
        
        # 10-dimensional observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )
        
        # 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        self.state = None
        self.steps = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.random.rand(10).astype(np.float32)
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        
        # Dummy dynamics
        self.state = np.random.rand(10).astype(np.float32)
        reward = np.random.randn()
        done = self.steps >= 100
        truncated = False
        
        return self.state, reward, done, truncated, {}


def test_dqn_agent():
    """Test DQN agent can initialize, train briefly, and infer"""
    print("\n" + "="*50)
    print("Testing DQN Agent")
    print("="*50)
    
    env = DummyFinanceEnv()
    agent = DQNFinancialAgent(
        name="Test_DQN",
        verbose=0
    )
    
    # Test initialization
    agent.initialize(env)
    print("✓ Agent initialized")
    
    # Test training (very short)
    agent.train(total_timesteps=1000)
    print("✓ Agent trained")
    
    # Test inference
    obs, _ = env.reset()
    action = agent.select_action(obs)
    assert 0 <= action < 6, f"Invalid action: {action}"
    print(f"✓ Agent selected action: {action}")
    
    # Test save/load
    agent.save("test_agent")
    agent.load("test_agent", env=env)
    print("✓ Save/load works")
    
    # Cleanup
    import os
    os.remove("test_agent.zip")
    
    print("✓ DQN Agent: ALL TESTS PASSED\n")


def test_baseline_agents():
    """Test baseline agents work"""
    print("="*50)
    print("Testing Baseline Agents")
    print("="*50)
    
    env = DummyFinanceEnv()
    
    agents = [
        RandomAgent(n_actions=6),
        AdaptiveRuleAgent()
    ]
    
    for agent in agents:
        obs, _ = env.reset()
        action = agent.select_action(obs)
        assert 0 <= action < 6, f"{agent.name}: Invalid action {action}"
        print(f"✓ {agent.name} works (action={action})")
    
    print("✓ Baseline Agents: ALL TESTS PASSED\n")


def test_full_episode():
    """Test agent can complete full episode"""
    print("="*50)
    print("Testing Full Episode")
    print("="*50)
    
    env = DummyFinanceEnv()
    agent = RandomAgent(n_actions=6)
    
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = agent.select_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        if done or truncated:
            break
    
    print(f"✓ Episode completed in {step+1} steps")
    print(f"✓ Total reward: {total_reward:.2f}")
    print("✓ Full Episode: TEST PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("AGENT INTEGRATION TESTS")
    print("="*50)
    
    test_baseline_agents()
    test_full_episode()
    test_dqn_agent()
    
    print("="*50)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("="*50)

