#!/usr/bin/env python3
"""
Evaluation script for discrete action space financial agents
Supports discrete RL agents and baseline strategies
"""

import numpy as np
from tqdm import tqdm
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.finance_env import FinanceEnv, MONEY_ALLOC, INVEST_ALLOC, ACTION_DESCRIPTIONS
from agents import (
    DiscreteDQNAgent,
    DiscretePPOAgent,
    SixtyFortyAgent,
    DebtAvalancheAgent,
    EqualWeightAgent,
    AgeBasedAgent,
    MarkowitzAgent,
)
from plots import (
    plot_net_worth_trajectories,
    plot_final_net_worth_comparison,
    plot_action_heatmap,
    plot_allocation_evolution,
    plot_investment_allocation_evolution,
    plot_debt_timeline,
    plot_portfolio_snapshots,
    plot_metrics_comparison,
)

def calculate_net_worth(state):
    """Calculate net worth from state (now at index 0 in new state structure)"""
    return state[0]  # Net worth is first element in 13D state

def decode_discrete_action(action_idx):
    """
    Decode discrete action index into money and investment allocations.
    
    Args:
        action_idx: Integer 0-89 (90 total actions)
    
    Returns:
        dict with 'money' and 'investment' allocation percentages
    """
    money_idx = action_idx // len(INVEST_ALLOC)
    invest_idx = action_idx % len(INVEST_ALLOC)
    
    invest_pct, debt_pct, emergency_pct = MONEY_ALLOC[money_idx][0]
    stock_pct, bond_pct, re_pct = INVEST_ALLOC[invest_idx][0]
    
    return {
        'money': {'invest': invest_pct, 'debt': debt_pct, 'emergency': emergency_pct},
        'investment': {'stocks': stock_pct, 'bonds': bond_pct, 'real_estate': re_pct},
        'description': ACTION_DESCRIPTIONS[action_idx]
    }

def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate discrete action agent"""
    
    print(f"Evaluating {agent.name} over {num_episodes} episodes...")
    
    rewards = []
    final_net_worths = []
    bankruptcy_count = 0
    debt_free_count = 0
    
    # Track net worth trajectory
    monthly_net_worths = []
    
    # Track portfolio snapshots at years 0, 10, 20, 30
    snapshot_months = [0, 120, 240, 359]
    portfolio_snapshots = {month: [] for month in snapshot_months}
    
    # Track actions and allocations
    action_history = {age: [] for age in range(25, 56)}
    allocation_history = {age: {'invest': [], 'debt': [], 'emergency': []} for age in range(25, 56)}
    investment_history = {age: {'stocks': [], 'bonds': [], 'real_estate': []} for age in range(25, 56)}
    debt_history = {age: {'cc_debt': [], 'student_loan': []} for age in range(25, 56)}
    
    # Track all actions
    all_actions = []
    
    # Set agent to evaluation mode if it's an RL agent
    if hasattr(agent, 'training'):
        agent.training = False
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset(seed=env.seed + episode)
        total_reward = 0
        episode_net_worths = []
        
        # Capture initial state (Year 0)
        portfolio_snapshots[0].append({
            'stocks': state[1],
            'bonds': state[2],
            'real_estate': state[3],
            'cc_debt': state[4],
            'student_loan': state[5],
            'emergency_fund': state[8]
        })
        
        for step in range(359):
            action = agent.get_action(state)
            all_actions.append(action)
            
            current_age = int(state[7])
            
            # Track action and decode allocations
            action_history[current_age].append(action)
            decoded = decode_discrete_action(action)
            allocation_history[current_age]['invest'].append(decoded['money']['invest'])
            allocation_history[current_age]['debt'].append(decoded['money']['debt'])
            allocation_history[current_age]['emergency'].append(decoded['money']['emergency'])
            investment_history[current_age]['stocks'].append(decoded['investment']['stocks'])
            investment_history[current_age]['bonds'].append(decoded['investment']['bonds'])
            investment_history[current_age]['real_estate'].append(decoded['investment']['real_estate'])
            
            # Track debt levels
            debt_history[current_age]['cc_debt'].append(state[4])
            debt_history[current_age]['student_loan'].append(state[5])
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Track net worth at each month
            episode_net_worths.append(calculate_net_worth(state))
            
            # Capture snapshots at specific months (fix state indices)
            if step + 1 in snapshot_months:
                portfolio_snapshots[step + 1].append({
                    'stocks': state[1],
                    'bonds': state[2],
                    'real_estate': state[3],
                    'cc_debt': state[4],
                    'student_loan': state[5],
                    'emergency_fund': state[8]
                })
            
            if terminated or truncated:
                break
        
        # Store this episode's trajectory
        monthly_net_worths.append(episode_net_worths)
        
        # Final metrics
        final_net_worth = calculate_net_worth(state)
        final_net_worths.append(final_net_worth)
        rewards.append(total_reward)
        
        # Check bankruptcy and debt-free status
        if final_net_worth < 0:
            bankruptcy_count += 1
        if state[4] + state[5] < 100:  # Credit card + student debt < $100
            debt_free_count += 1
    
    # Calculate average trajectory across all episodes
    avg_monthly_trajectory = np.mean(monthly_net_worths, axis=0)
    std_monthly_trajectory = np.std(monthly_net_worths, axis=0)
    
    # Calculate average portfolio snapshots
    avg_snapshots = {}
    for month in snapshot_months:
        avg_snapshots[month] = {
            'stocks': np.mean([s['stocks'] for s in portfolio_snapshots[month]]),
            'bonds': np.mean([s['bonds'] for s in portfolio_snapshots[month]]),
            'real_estate': np.mean([s['real_estate'] for s in portfolio_snapshots[month]]),
            'cc_debt': np.mean([s['cc_debt'] for s in portfolio_snapshots[month]]),
            'student_loan': np.mean([s['student_loan'] for s in portfolio_snapshots[month]]),
            'emergency_fund': np.mean([s['emergency_fund'] for s in portfolio_snapshots[month]])
        }
    
    # Calculate action diversity
    unique_actions = len(set(all_actions))
    action_diversity = unique_actions / len(INVEST_ALLOC) / len(MONEY_ALLOC)  # Percentage of action space used
    
    return {
        'agent_name': agent.name,
        'avg_trajectory': avg_monthly_trajectory,
        'std_trajectory': std_monthly_trajectory,
        'final_net_worths': final_net_worths,
        'avg_reward': np.mean(rewards),
        'avg_net_worth': np.mean(final_net_worths),
        'bankruptcy_rate': bankruptcy_count / num_episodes,
        'debt_free_rate': debt_free_count / num_episodes,
        'min_net_worth': np.min(final_net_worths),
        'max_net_worth': np.max(final_net_worths),
        'portfolio_snapshots': avg_snapshots,
        'action_diversity': action_diversity,
        'action_history': action_history,
        'allocation_history': allocation_history,
        'investment_history': investment_history,
        'debt_history': debt_history,
    }

def generate_plots(all_results):
    """Generate interactive Plotly visualizations"""
    os.makedirs('visualization', exist_ok=True)
    
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # 1. Net worth trajectories (all agents)
    print("Creating net worth trajectories plot...")
    plot_net_worth_trajectories(all_results)
    
    # 2. Final net worth comparison (all agents)
    print("Creating final net worth comparison...")
    plot_final_net_worth_comparison(all_results)
    
    # 3. Portfolio snapshots (all agents)
    print("Creating portfolio snapshots...")
    plot_portfolio_snapshots(all_results)
    
    # 4. Debt timeline (all agents combined)
    print("Creating combined debt timeline...")
    plot_debt_timeline(all_results)
    
    # 5. Allocation evolution (all agents combined)
    print("Creating combined allocation evolution...")
    plot_allocation_evolution(all_results)
    
    # 6. Investment allocation (all agents combined)
    print("Creating combined investment allocation...")
    plot_investment_allocation_evolution(all_results)
    
    # Agent-specific plots (only for RL agents with diverse actions)
    rl_agents = ['Discrete_DQN', 'Discrete_PPO']
    rl_results = [r for r in all_results if r['agent_name'] in rl_agents]
    
    if rl_results:
        print("\n=== RL AGENT-SPECIFIC VISUALIZATIONS ===")
        for result in rl_results:
            agent_name = result['agent_name']
            
            # Action heatmap (only meaningful for RL agents)
            print(f"Creating action heatmap for {agent_name}...")
            plot_action_heatmap(result['action_history'], agent_name)
    
    print("\n✓ All visualizations saved to visualization/ directory")

def main():
    parser = argparse.ArgumentParser(description='Evaluate discrete financial agents')
    parser.add_argument('--agents', nargs='+', choices=['dqn', 'ppo', '60/40', 'debt', 'equal', 'age', 'markowitz', 'all'],
                        default=['all'], help='Agents to evaluate')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    args = parser.parse_args()
    
    print("=== DISCRETE AGENT EVALUATION ===\n")
    
    env = FinanceEnv()
    agents = []
    
    # Map agent names
    agent_map = {
        'dqn': ('models/dqn_best_model.pth', DiscreteDQNAgent),
        'ppo': ('models/ppo_best_model.pth', DiscretePPOAgent),
        '60/40': (None, SixtyFortyAgent),
        'debt': (None, DebtAvalancheAgent),
        'equal': (None, EqualWeightAgent),
        'age': (None, AgeBasedAgent),
        'markowitz': (None, MarkowitzAgent),
    }
    
    # Select agents
    if 'all' in args.agents:
        selected = agent_map.keys()
    else:
        selected = args.agents
    
    # Load agents
    for name in selected:
        model_path, agent_class = agent_map[name]
        
        if model_path:  # RL agent
            if os.path.exists(model_path):
                agent = agent_class()
                agent.load(model_path)
                agents.append(agent)
                print(f"Loaded {agent.name}")
            else:
                print(f"Model not found: {model_path}")
        else:  # Baseline agent
            agents.append(agent_class())
            print(f"Loaded {agent_class().name}")
    
    if not agents:
        print("No agents to evaluate")
        return
    
    # Evaluate agents
    all_results = []
    for agent in agents:
        results = evaluate_agent(agent, env, args.episodes)
        all_results.append(results)
        
        print(f"\n=== {results['agent_name']} RESULTS ===")
        print(f"Average Net Worth: ${results['avg_net_worth']:,.0f}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Bankruptcy Rate: {results['bankruptcy_rate']:.1%}")
        print(f"Debt-Free Rate: {results['debt_free_rate']:.1%}")
        print(f"Net Worth Range: ${results['min_net_worth']:,.0f} to ${results['max_net_worth']:,.0f}")
        print(f"Action Diversity: {results['action_diversity']:.1%} of action space used")
    
    # Generate visualizations
    generate_plots(all_results)
    
    # Final rankings
    print(f"\n=== FINAL RANKINGS ===")
    sorted_results = sorted(all_results, key=lambda x: x['avg_net_worth'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['agent_name']}: ${result['avg_net_worth']:,.0f}")
    
    print(f"\nEvaluation complete!")

if __name__ == "__main__":
    main()
