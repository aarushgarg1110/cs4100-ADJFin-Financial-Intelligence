#!/usr/bin/env python3
"""
Unified evaluation script for all financial agents (RL and rule-based)
All agents use the same interface: get_action(state) -> continuous_action
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.finance_env import FinanceEnv
from agents import (
    SixtyFortyAgent, DebtAvalancheAgent, EqualWeightAgent, 
    AgeBasedAgent, MarkowitzAgent, PPOAgent, ContinuousDQNAgent,
    AllStocksAgent, CashHoarderAgent, DebtIgnorerAgent
)

def calculate_net_worth(state):
    """Calculate net worth from state"""
    return state[0] + state[1] + state[2] + state[3] + state[8] - state[4] - state[5]

def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate any agent using unified interface"""
    
    print(f"Evaluating {agent.name} over {num_episodes} episodes...")
    
    rewards = []
    final_net_worths = []
    bankruptcy_count = 0
    debt_free_count = 0
    
    # Track net worth trajectory (average across episodes)
    monthly_net_worths = []
    
    # Set agent to evaluation mode if it's an RL agent
    if hasattr(agent, 'training'):
        agent.training = False
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset(seed=42 + episode)
        total_reward = 0
        episode_net_worths = []
        
        for step in range(359):  # 30 years * 12 months - 1
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Track net worth at each month
            episode_net_worths.append(calculate_net_worth(state))
            
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
    
    return {
        'agent_name': agent.name,
        'avg_trajectory': avg_monthly_trajectory,  # Average net worth growth over 30 years
        'std_trajectory': std_monthly_trajectory,  # Standard deviation for error bars
        'final_net_worths': final_net_worths,  # Episode-by-episode final values
        'avg_reward': np.mean(rewards),
        'avg_net_worth': np.mean(final_net_worths),
        'bankruptcy_rate': bankruptcy_count / num_episodes,
        'debt_free_rate': debt_free_count / num_episodes,
        'min_net_worth': np.min(final_net_worths),
        'max_net_worth': np.max(final_net_worths),
    }

def plot_results(all_results):
    """Create comparison plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    agent_names = [r['agent_name'] for r in all_results]
    
    # 1. Net Worth Growth Trajectories Over 30 Years
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    months = range(len(all_results[0]['avg_trajectory']))
    years = [m/12 for m in months]
    
    for i, result in enumerate(all_results):
        color = colors[i % len(colors)]
        trajectory = result['avg_trajectory']
        
        # Plot average trajectory
        ax1.plot(years, trajectory, color=color, linewidth=2, label=result['agent_name'])
    
    ax1.set_title('Net Worth Growth Over 30 Years')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Net Worth ($)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Final Net Worth Comparison (Bar Chart with Error Bars)
    avg_net_worths = [r['avg_net_worth'] for r in all_results]
    std_net_worths = [np.std(r['final_net_worths']) for r in all_results]
    
    bars = ax2.bar(agent_names, avg_net_worths, color=colors[:len(agent_names)], 
                   yerr=std_net_worths, capsize=5)
    ax2.set_title('Average Final Net Worth')
    ax2.set_ylabel('Net Worth ($)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_net_worths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_net_worths)*0.01,
                f'${value:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Average Reward Comparison
    avg_rewards = [r['avg_reward'] for r in all_results]
    ax3.bar(agent_names, avg_rewards, color=colors[:len(agent_names)])
    ax3.set_title('Average Episode Reward')
    ax3.set_ylabel('Reward')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Risk Metrics
    bankruptcy_rates = [r['bankruptcy_rate'] * 100 for r in all_results]
    debt_free_rates = [r['debt_free_rate'] * 100 for r in all_results]
    
    x = np.arange(len(agent_names))
    width = 0.35
    
    ax4.bar(x - width/2, bankruptcy_rates, width, label='Bankruptcy Rate', color='red', alpha=0.7)
    ax4.bar(x + width/2, debt_free_rates, width, label='Debt-Free Rate', color='green', alpha=0.7)
    
    ax4.set_title('Risk Metrics')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(agent_names, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save to visualization directory
    import os
    os.makedirs('../visualization', exist_ok=True)
    plt.savefig('../visualization/agent_comparison_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to visualization/agent_comparison_results.png")
    plt.show()

def main():
    print("=== UNIFIED AGENT EVALUATION ===")
    print("All agents use continuous actions for fair comparison\n")
    
    # Create environment (single instance for all agents)
    env = FinanceEnv()
    
    # Initialize all agents
    agents = [
        # Simple/naive strategies
        AllStocksAgent(),
        CashHoarderAgent(), 
        DebtIgnorerAgent(),
        
        # Expert rule-based strategies
        SixtyFortyAgent(),
        DebtAvalancheAgent(),
        EqualWeightAgent(),
        AgeBasedAgent(),
        MarkowitzAgent(),
    ]
    
    # Add trained RL agents if models exist
    if os.path.exists('models/ppo_model.pth'):
        ppo_agent = PPOAgent()
        ppo_agent.load('models/ppo_model.pth')
        agents.append(ppo_agent)
    
    if os.path.exists('models/dqn_model.pth'):
        dqn_agent = ContinuousDQNAgent()
        dqn_agent.load('models/dqn_model.pth')
        agents.append(dqn_agent)

    if os.path.exists('models/sac_model.pth'):
        sac_agent = SACAgent()
        sac_agent.load('models/sac_model.pth')
        agents.append(sac_agent)
    
    # Evaluate each agent
    all_results = []
    for agent in agents:
        num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
        results = evaluate_agent(agent, env, num_episodes)
        all_results.append(results)
        
        print(f"\n=== {results['agent_name']} RESULTS ===")
        print(f"Average Net Worth: ${results['avg_net_worth']:,.0f}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Bankruptcy Rate: {results['bankruptcy_rate']:.1%}")
        print(f"Debt-Free Rate: {results['debt_free_rate']:.1%}")
        print(f"Net Worth Range: ${results['min_net_worth']:,.0f} to ${results['max_net_worth']:,.0f}")
    
    # Create visualizations
    plot_results(all_results)
    
    # Final rankings
    print(f"\n=== FINAL RANKINGS ===")
    sorted_results = sorted(all_results, key=lambda x: x['avg_net_worth'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['agent_name']}: ${result['avg_net_worth']:,.0f}")
    
    print(f"\nSUCCESS: Unified evaluation complete!")

if __name__ == "__main__":
    main()
