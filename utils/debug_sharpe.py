import sys
sys.path.append('.')
import numpy as np
from environment.finance_env import FinanceEnv
from collections import Counter

print('='*60)
print('DEBUGGING SHARPE RATIO REWARD')
print('='*60)

# Test 1: Check Sharpe calculation
print('\n=== TEST 1: Sharpe Calculation ===')
env = FinanceEnv()
state, _ = env.reset(seed=42)
sharpe_values = []

for step in range(20):
    action = np.random.randint(0, 90)
    state, reward, done, truncated, _ = env.step(action)

    net_worth = state[0]
    wealth_reward = net_worth / 1_000_000

    if len(env.portfolio_returns) > 1:
        volatility = np.std(env.portfolio_returns)
        portfolio_return = list(env.portfolio_returns)[-1]
        sharpe = (portfolio_return - 0.00167) / (volatility + 1e-6)
    else:
        sharpe = 0

    sharpe_values.append(sharpe)

    if step < 5:
        print(f'Step {step+1}: Reward={reward:.3f}, Wealth={wealth_reward:.3f}, Sharpe={sharpe:.3f}')

print(f'\nAverage Sharpe: {np.mean(sharpe_values):.3f}')
print(f'Sharpe range: [{min(sharpe_values):.3f}, {max(sharpe_values):.3f}]')

# Test 2: Compare actions
print('\n=== TEST 2: Action 5 vs Others (50 steps each) ===')

def test_action(action_idx, num_steps=50):
    env_test = FinanceEnv()
    state, _ = env_test.reset(seed=100)
    total_reward = 0
    sharpe_vals = []

    for _ in range(num_steps):
        state, reward, done, truncated, _ = env_test.step(action_idx)
        total_reward += reward

        if len(env_test.portfolio_returns) > 1:
            vol = np.std(env_test.portfolio_returns)
            ret = list(env_test.portfolio_returns)[-1]
            sharpe = (ret - 0.00167) / (vol + 1e-6)
            sharpe_vals.append(sharpe)

        if done:
            break

    avg_sharpe = np.mean(sharpe_vals) if sharpe_vals else 0
    return total_reward, avg_sharpe

reward_5, sharpe_5 = test_action(5, 50)
print(f'Action 5 (70%% bonds): Reward={reward_5:.1f}, Sharpe={sharpe_5:.3f}')

reward_20, sharpe_20 = test_action(20, 50)
print(f'Action 20 (60/40):     Reward={reward_20:.1f}, Sharpe={sharpe_20:.3f}')

reward_35, sharpe_35 = test_action(35, 50)
print(f'Action 35 (balanced):  Reward={reward_35:.1f}, Sharpe={sharpe_35:.3f}')

print(f'\nDifference (Action 20 - Action 5): {reward_20 - reward_5:.1f}')
print(f'Action 5 still best? {reward_5 > reward_20 and reward_5 > reward_35}')

print('\n' + '='*60)
print('DIAGNOSIS')
print('='*60)

if reward_5 > reward_20:
    print('PROBLEM: Action 5 still gives higher reward')
    print('Solution: Increase Sharpe weight from 30%% to 50%% or 60%%')
else:
    print('GOOD: Action 20 gives higher reward than action 5')


# ============================================================
# DEBUGGING SHARPE RATIO REWARD
# ============================================================

# === TEST 1: Sharpe Calculation ===
# Loading market data (one-time setup)...
# Downloading market data...
# SUCCESS stocks: 310 months of data
# SUCCESS bonds: 223 months of data
# SUCCESS real_estate: 254 months of data
# Market data loaded successfully!
# Step 1: Reward=-5.000, Wealth=-0.012, Sharpe=0.000
# Step 2: Reward=-5.000, Wealth=-0.011, Sharpe=0.000
# Step 3: Reward=-5.000, Wealth=-0.010, Sharpe=0.000
# Step 4: Reward=-2.438, Wealth=-0.009, Sharpe=-7.467
# Step 5: Reward=-0.196, Wealth=-0.008, Sharpe=-7.467

# Average Sharpe: -1.235
# Sharpe range: [-7.467, 0.001]

# === TEST 2: Action 5 vs Others (50 steps each) ===
# Market data loaded successfully!
# Action 5 (70%% bonds): Reward=-54.0, Sharpe=-1.795
# Market data loaded successfully!
# Action 20 (60/40):     Reward=-48.5, Sharpe=-1.758
# Market data loaded successfully!
# Action 35 (balanced):  Reward=-46.7, Sharpe=-1.670

# Difference (Action 20 - Action 5): 5.5
# Action 5 still best? False

# ============================================================
# DIAGNOSIS
# ============================================================
# GOOD: Action 20 gives higher reward than action 5