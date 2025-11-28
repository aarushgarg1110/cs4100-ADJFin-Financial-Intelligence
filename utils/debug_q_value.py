import sys
sys.path.append('.')
import torch
import numpy as np
from environment.finance_env import FinanceEnv
from agents import DiscreteDQNAgent

# Load trained model
agent = DiscreteDQNAgent(gamma=1.0)
try:
    agent.load('models/dqn_best_model.pth')
    print('Loaded trained model')
except:
    print('No trained model found, using random')

# Get a state
env = FinanceEnv()
state, _ = env.reset(seed=42)

# Get Q-values for all actions
state_tensor = torch.FloatTensor(state).unsqueeze(0)
with torch.no_grad():
    q_values = agent.q_network(state_tensor).squeeze().numpy()

print('\n=== Q-VALUES FOR ALL 90 ACTIONS ===')
print(f'Action 5 Q-value: {q_values[5]:.2f}')
print(f'Action 20 Q-value: {q_values[20]:.2f}')
print(f'Action 35 Q-value: {q_values[35]:.2f}')

print(f'\nTop 5 actions by Q-value:')
top_actions = np.argsort(q_values)[-5:][::-1]
for i, action in enumerate(top_actions):
    print(f'  {i+1}. Action {action}: Q={q_values[action]:.2f}')

print(f'\nQ-value statistics:')
print(f'  Mean: {q_values.mean():.2f}')
print(f'  Std: {q_values.std():.2f}')
print(f'  Min: {q_values.min():.2f}')
print(f'  Max: {q_values.max():.2f}')

# Check if all Q-values are similar (collapsed)
if q_values.std() < 10:
    print('\nPROBLEM: Q-values have collapsed (std < 10)')
    print('All actions look the same to the agent!')
else:
    print('\nQ-values are differentiated')

# Loaded trained model
# Loading market data (one-time setup)...
# Downloading market data...
# SUCCESS stocks: 310 months of data
# SUCCESS bonds: 223 months of data
# SUCCESS real_estate: 254 months of data
# Market data loaded successfully!

# === Q-VALUES FOR ALL 90 ACTIONS ===
# Action 5 Q-value: 173265.11
# Action 20 Q-value: 124336.60
# Action 35 Q-value: 121665.27

# Top 5 actions by Q-value:
#   1. Action 5: Q=173265.11
#   2. Action 86: Q=132421.67
#   3. Action 19: Q=130464.30
#   4. Action 70: Q=129880.91
#   5. Action 32: Q=129112.59

# Q-value statistics:
#   Mean: 117297.72
#   Std: 10422.78
#   Min: 95223.24
#   Max: 173265.11

# Q-values are differentiated