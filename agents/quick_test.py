"""
Quick test script to verify agent-environment integration
From-scratch implementations (no stable-baselines3)
"""

import numpy as np
import sys
import os


# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from environment.finance_env import FinanceEnv
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.discrete_action_wrapper import wrap_finance_env

print("All imports successful!")

def test_environment():
    """Test basic environment functionality"""
    print("="*60)
    print("TEST 1: Environment Setup")
    print("="*60)
    
    try:
        env = FinanceEnv()
        print("✓ FinanceEnv created successfully")
        
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step completed: reward={reward:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper():
    """Test discrete action wrapper"""
    print("\n" + "="*60)
    print("TEST 2: Action Wrapper")
    print("="*60)
    
    try:
        base_env = FinanceEnv()
        wrapped_env = wrap_finance_env(base_env)
        print("✓ Wrapper applied successfully")
        
        print(f"  Original action space: {base_env.action_space}")
        print(f"  Wrapped action space: {wrapped_env.action_space}")
        
        # Test discrete actions
        obs, info = wrapped_env.reset()
        for action in range(6):
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            strategy_desc = wrapped_env.get_strategy_description(action)
            print(f"✓ Action {action}: {strategy_desc[:40]}...")
        
        return True
    except Exception as e:
        print(f"✗ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_q_learning_agent():
    """Test Q-Learning agent"""
    print("\n" + "="*60)
    print("TEST 3: Q-Learning Agent")
    print("="*60)
    
    try:
        base_env = FinanceEnv()
        env = wrap_finance_env(base_env)
        
        # Create agent
        agent = QLearningAgent(n_actions=6, n_bins=3, epsilon=1.0)
        print("✓ Q-Learning agent created")
        
        # Test action selection
        obs, _ = env.reset()
        action = agent.select_action(obs, deterministic=False)
        print(f"✓ Action selected: {action}")
        
        # Test learning
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.learn_from_experience(obs, action, reward, next_obs, terminated)
        print(f"✓ Learning step successful")
        print(f"  Q-table size: {len(agent.Q_table)} states")
        
        return True
    except Exception as e:
        print(f"✗ Q-Learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dqn_agent():
    """Test DQN agent"""
    print("\n" + "="*60)
    print("TEST 4: DQN Agent")
    print("="*60)
    
    try:
        base_env = FinanceEnv()
        env = wrap_finance_env(base_env)
        
        # Create agent
        agent = DQNAgent(
            state_dim=15,
            action_dim=6,
            learning_rate=1e-4,
            buffer_size=1000,
            batch_size=32
        )
        print("✓ DQN agent created")
        
        # Test action selection
        obs, _ = env.reset()
        action = agent.select_action(obs, deterministic=False)
        print(f"✓ Action selected: {action}")
        
        # Test learning (add multiple experiences first)
        for _ in range(5):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.learn_from_experience(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        
        print(f"✓ Learning steps successful")
        print(f"  Buffer size: {len(agent.replay_buffer)}")
        print(f"  Training steps: {agent.training_steps}")
        
        return True
    except Exception as e:
        print(f"✗ DQN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_episode():
    """Run a complete short episode with Q-Learning"""
    print("\n" + "="*60)
    print("TEST 5: Complete Episode Run")
    print("="*60)
    
    try:
        base_env = FinanceEnv()
        env = wrap_finance_env(base_env)
        agent = QLearningAgent(n_actions=6, n_bins=3)
        
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        print("Running episode (max 100 steps)...")
        while not done and steps < 100:
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Learn
            agent.learn_from_experience(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_reward += reward
            steps += 1
        
        print(f"✓ Episode completed:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final cash: ${obs[0]:,.0f}")
        final_nw = obs[0] + obs[1] + obs[2] + obs[3] - obs[4] - obs[5]
        print(f"  Final net worth: ${final_nw:,.0f}")
        print(f"  Q-table size: {len(agent.Q_table)} states")
        
        return True
    except Exception as e:
        print(f"✗ Episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("RUNNING INTEGRATION TESTS")
    print("From-Scratch Implementation (No stable-baselines3)")
    print("🧪 "*20 + "\n")
    
    tests = [
        ("Environment Setup", test_environment),
        ("Action Wrapper", test_wrapper),
        ("Q-Learning Agent", test_q_learning_agent),
        ("DQN Agent", test_dqn_agent),
        ("Complete Episode", test_short_episode)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed! Ready for training.")
    else:
        print("\n  Some tests failed. Fix issues before training.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)