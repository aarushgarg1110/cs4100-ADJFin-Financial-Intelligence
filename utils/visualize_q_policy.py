"""
Visualize learned Q-value policy across different states.
Shows which action the agent prefers in different situations.

Usage: python visualize_q_policy.py [model_path]
"""
import sys
sys.path.append('.')

import torch
import numpy as np
import pandas as pd
from environment.finance_env import FinanceEnv
from agents import DiscreteDQNAgent

def analyze_q_policy(model_path='models/dqn_best_model.pth', gamma=1.0):
    """Analyze Q-value policy across different states"""
    
    # Load model
    agent = DiscreteDQNAgent(gamma=gamma)
    try:
        agent.load(model_path)
        print(f'Loaded model from {model_path}')
    except:
        print(f'Could not load model from {model_path}')
        return
    
    env = FinanceEnv()
    
    print('\n' + '='*70)
    print('Q-VALUE POLICY ANALYSIS')
    print('='*70)
    
    # Define test scenarios
    scenarios = [
        ('Young, Bull Market, High Debt', 42, 1),
        ('Young, Bear Market, High Debt', 43, 2),
        ('Young, Normal Market, Low Debt', 44, 0),
        ('Middle Age, Bull Market, Med Debt', 45, 1),
        ('Middle Age, Bear Market, Med Debt', 46, 2),
        ('Middle Age, Normal Market, No Debt', 47, 0),
        ('Old, Bull Market, No Debt', 48, 1),
        ('Old, Bear Market, No Debt', 49, 2),
        ('Old, Normal Market, High Wealth', 50, 0),
    ]
    
    results = []
    action_5_count = 0
    
    for name, seed, target_regime in scenarios:
        state, _ = env.reset(seed=seed)
        
        # Force regime for testing
        env.current_regime = target_regime
        state[10] = target_regime
        
        # Get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.q_network(state_tensor).squeeze().numpy()
        
        # Find best action
        best_action = q_values.argmax()
        best_q = q_values[best_action]
        
        # Get top 3
        top_3 = np.argsort(q_values)[-3:][::-1]
        
        print(f'\n{name}:')
        print(f'  State: age={state[7]:.0f}, net_worth=${state[0]:.0f}, regime={state[10]:.0f}')
        print(f'  Best action: {best_action} (Q={best_q:.0f})')
        print(f'  Top 3: {top_3[0]} (Q={q_values[top_3[0]]:.0f}), '
              f'{top_3[1]} (Q={q_values[top_3[1]]:.0f}), '
              f'{top_3[2]} (Q={q_values[top_3[2]]:.0f})')
        
        if best_action == 5:
            action_5_count += 1
            print('  -> Action 5 (bond-heavy)')
        
        results.append({
            'scenario': name,
            'best_action': best_action,
            'best_q': best_q,
            'q_mean': q_values.mean(),
            'q_std': q_values.std(),
            'q_max': q_values.max(),
        })
    
    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Action 5 chosen in {action_5_count}/{len(scenarios)} scenarios ({action_5_count/len(scenarios)*100:.0f}%)')
    
    df = pd.DataFrame(results)
    print(f'\nQ-value statistics:')
    print(f'  Average Q-max: {df["q_max"].mean():.0f}')
    print(f'  Average Q-mean: {df["q_mean"].mean():.0f}')
    print(f'  Average Q-std: {df["q_std"].mean():.0f}')
    
    if df['q_max'].mean() > 10000:
        print(f'\nDIAGNOSIS: Q-values exploded (avg max = {df["q_max"].mean():.0f})')
        print('Cause: gamma=1.0 with long episodes (360 steps)')
        print('Fix: Use gamma=0.99 and retrain')
    
    if action_5_count == len(scenarios):
        print(f'\nDIAGNOSIS: Degenerate policy (action 5 always chosen)')
        print('Cause: Q-value explosion + early convergence')
        print('Fix: gamma=0.99 + train longer (500-1000 episodes)')
    elif action_5_count > len(scenarios) * 0.7:
        print(f'\nDIAGNOSIS: Poor diversity (action 5 dominates)')
        print('Fix: Increase exploration or Sharpe weight')
    else:
        print(f'\nGOOD: Policy shows diversity across states')
    
    return df


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/dqn_best_model.pth'
    analyze_q_policy(model_path)
