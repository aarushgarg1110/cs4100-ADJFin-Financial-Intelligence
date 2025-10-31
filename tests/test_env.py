#!/usr/bin/env python3

import sys
sys.path.append('../environment')

from finance_env import FinanceEnv
import numpy as np

def test_environment():
    print("Testing Enhanced Finance Environment...")
    
    # Create environment
    env = FinanceEnv()
    
    # Test reset
    state, info = env.reset(seed=42)
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    # Test a few steps
    for i in range(5):
        # Random action: [stock_alloc, bond_alloc, re_alloc, emergency_contrib, cc_payment, student_payment]
        action = np.random.rand(6) * 0.5  # Keep actions reasonable
        
        state, reward, done, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"Action: {action}")
        print(f"Reward: {reward:.2f}")
        print(f"Net worth: ${(state[0] + state[1] + state[2] + state[3] + state[8] - state[4] - state[5]):.0f}")
        print(f"Month: {state[14]}")
        
        if done:
            break
    
    print("\n✅ Environment test completed successfully!")

if __name__ == "__main__":
    test_environment()


# Testing Enhanced Finance Environment...
# Initial state shape: (15,)
# Initial state: [2.49816048e+03 9.50714306e+03 3.65996971e+03 0.00000000e+00
#  4.78926787e+03 8.12037281e+03 3.88998630e+03 2.50000000e+01
#  6.45209030e+02 0.00000000e+00 0.00000000e+00 5.00000000e-02
#  0.00000000e+00 0.00000000e+00 0.00000000e+00]

# Step 1:
# Action: [0.43308807 0.30055751 0.35403629 0.01029225 0.48495493 0.41622132]
# Reward: 0.09
# Net worth: $8833
# Month: 1.0

# Step 2:
# Action: [0.06974693 0.14607232 0.18318092 0.22803499 0.39258798 0.09983689]
# Reward: 0.13
# Net worth: $13066
# Month: 2.0

# Step 3:
# Action: [0.04883606 0.34211651 0.22007625 0.06101912 0.24758846 0.01719426]
# Reward: 0.17
# Net worth: $16610
# Month: 3.0

# Step 4:
# Action: [0.46093712 0.04424625 0.09799143 0.02261364 0.16266517 0.19433864]
# Reward: 2.20
# Net worth: $20141
# Month: 4.0

# Step 5:
# Action: [0.40109849 0.03727532 0.49344347 0.38612238 0.09935784 0.00276106]
# Reward: 2.24
# Net worth: $24233
# Month: 5.0
