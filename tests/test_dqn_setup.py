"""
Test DQN installation and basic functionality
Run this to verify your environment is ready
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

def test_installation():
    """Test 1: Verify stable-baselines3 works"""
    print("Testing SB3 installation...")
    try:
        env = gym.make('CartPole-v1')
        model = DQN('MlpPolicy', env, verbose=0)
        print("✓ SB3 installed correctly")
        return True
    except Exception as e:
        print(f"✗ Installation error: {e}")
        return False

def test_training():
    """Test 2: Verify DQN can train"""
    print("\nTesting DQN training (this takes ~30 seconds)...")
    try:
        env = gym.make('CartPole-v1')
        model = DQN(
            'MlpPolicy', 
            env,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=100,
            batch_size=32,
            verbose=0
        )
        model.learn(total_timesteps=5000)
        
        # Test the trained agent
        obs, _ = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        
        print("✓ DQN training works")
        return True
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

def test_save_load():
    """Test 3: Verify model persistence"""
    print("\nTesting model save/load...")
    try:
        env = gym.make('CartPole-v1')
        model = DQN('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=1000)
        
        # Save
        model.save("test_model")
        
        # Load
        loaded_model = DQN.load("test_model", env=env)
        
        # Test loaded model works
        obs, _ = env.reset()
        action, _ = loaded_model.predict(obs)
        
        print("✓ Save/load works")
        
        # Cleanup
        import os
        os.remove("test_model.zip")
        return True
    except Exception as e:
        print(f"✗ Save/load error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("DQN Setup Verification")
    print("=" * 50)
    
    results = []
    results.append(test_installation())
    results.append(test_training())
    results.append(test_save_load())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ ALL TESTS PASSED - Ready to build!")
    else:
        print("✗ Some tests failed - check errors above")
    print("=" * 50)

