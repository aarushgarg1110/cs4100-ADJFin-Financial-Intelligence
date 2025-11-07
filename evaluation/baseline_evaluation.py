"""
Baseline Strategy Evaluation
Follows Q-learning evaluation pattern with 1000+ episodes per strategy
"""

import os
import sys
sys.path.append('environment')
sys.path.append('agents')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from finance_env import FinanceEnv
from baseline_human_strats import SixtyFortyAgent, DebtAvalancheAgent, EqualWeightAgent, AgeBasedAgent, MarkowitzAgent

def calculate_net_worth(state):
    """Calculate net worth from state array"""
    # assets - debts
    return state[0] + state[1] + state[2] + state[3] + state[8] - state[4] - state[5]

def evaluate_baseline_agent(agent, env, num_episodes=1000):
    """Evaluate baseline agent following Q-learning evaluation pattern"""
    
    print(f"Evaluating {agent.name} over {num_episodes} episodes...")
    
    # Metrics to track (like Q-learning evaluation)
    rewards = []
    final_net_worths = []
    episode_lengths = []
    bankruptcy_count = 0
    debt_free_count = 0
    net_worth_trajectories = []  # Track net worth over time for line plot
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset(seed=42 + episode)
        total_reward = 0
        episode_length = 0
        episode_net_worths = []  # Track net worth each month for this episode
        
        # Run 30-year simulation (360 months)
        for step in range(360):
            action = agent.get_action(state)  # Fixed strategy (like Q-table lookup)
            state, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            # Track net worth each month for first few episodes (for line plot)
            if episode < 5:  # Only track first 5 episodes to avoid memory issues
                current_net_worth = calculate_net_worth(state)
                episode_net_worths.append(current_net_worth)
            
            if done:  # Early termination (bankruptcy)
                break
        
        # Store trajectory for line plot
        if episode < 5:
            net_worth_trajectories.append(episode_net_worths)
        
        # Calculate final metrics
        final_net_worth = calculate_net_worth(state)
        final_debt = state[4] + state[5]  # CC + student debt
        
        # Track metrics
        rewards.append(total_reward)
        final_net_worths.append(final_net_worth)
        episode_lengths.append(episode_length)
        
        # Count special outcomes
        if final_net_worth < -5000:  # Bankruptcy threshold
            bankruptcy_count += 1
        if final_debt < 1000:  # Essentially debt-free
            debt_free_count += 1
    
    # Calculate aggregate statistics (like Q-learning)
    results = {
        'agent_name': agent.name,
        'avg_reward': np.mean(rewards),
        'avg_net_worth': np.mean(final_net_worths),
        'std_net_worth': np.std(final_net_worths),
        'avg_episode_length': np.mean(episode_lengths),
        'bankruptcy_rate': bankruptcy_count / num_episodes,
        'debt_free_rate': debt_free_count / num_episodes,
        'min_net_worth': np.min(final_net_worths),
        'max_net_worth': np.max(final_net_worths),
        'all_net_worths': final_net_worths,
        'all_rewards': rewards,
        'net_worth_trajectories': net_worth_trajectories
    }
    
    return results

def plot_results(all_results):
    """Create visualization plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    agent_names = [r['agent_name'] for r in all_results]
    
    # 1. Net Worth Trajectories Over Time (Like Q-learning reward curves)
    months = range(360)  # 30 years * 12 months
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, result in enumerate(all_results):
        trajectories = result['net_worth_trajectories']
        if trajectories:  # If we have trajectory data
            # Average the trajectories across episodes
            avg_trajectory = np.mean(trajectories, axis=0)
            ax1.plot(months[:len(avg_trajectory)], avg_trajectory, 
                    label=result['agent_name'], color=colors[i], linewidth=2)
    
    ax1.set_title('Net Worth Growth Over 30 Years')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Net Worth ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average Net Worth Comparison
    avg_net_worths = [r['avg_net_worth'] for r in all_results]
    std_net_worths = [r['std_net_worth'] for r in all_results]
    
    bars1 = ax2.bar(agent_names, avg_net_worths, yerr=std_net_worths, capsize=5)
    ax2.set_title('Average Final Net Worth (30 Years)')
    ax2.set_ylabel('Net Worth ($)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars1, avg_net_worths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'${val:,.0f}', ha='center', va='bottom')
    
    # 3. Risk Metrics
    bankruptcy_rates = [r['bankruptcy_rate'] * 100 for r in all_results]
    debt_free_rates = [r['debt_free_rate'] * 100 for r in all_results]
    
    x = np.arange(len(agent_names))
    width = 0.35
    
    bars2 = ax3.bar(x - width/2, bankruptcy_rates, width, label='Bankruptcy Rate', color='red', alpha=0.7)
    bars3 = ax3.bar(x + width/2, debt_free_rates, width, label='Debt-Free Rate', color='green', alpha=0.7)
    
    ax3.set_title('Risk Metrics')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(agent_names, rotation=45)
    ax3.legend()
    
    # 4. Performance Summary Table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for r in all_results:
        table_data.append([
            r['agent_name'],
            f"${r['avg_net_worth']:,.0f}",
            f"{r['bankruptcy_rate']:.1%}",
            f"{r['debt_free_rate']:.1%}",
            f"{r['avg_reward']:.1f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Strategy', 'Avg Net Worth', 'Bankruptcy Rate', 'Debt-Free Rate', 'Avg Reward'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('baseline_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== BASELINE STRATEGY EVALUATION ===")
    print("Following Q-learning evaluation pattern...\n")
    
    # Create environment
    env = FinanceEnv()
    
    # Create baseline agents
    agents = [
        SixtyFortyAgent(),
        DebtAvalancheAgent(),
        EqualWeightAgent(),
        AgeBasedAgent(),
        MarkowitzAgent()
    ]
    
    # Evaluate each agent (like Q-learning evaluation loop)
    all_results = []
    
    for agent in agents:
        results = evaluate_baseline_agent(agent, env, num_episodes=1000)  # 1000 episodes
        all_results.append(results)
        
        # Print metrics (like Q-learning output)
        print(f"\n=== {agent.name} RESULTS ===")
        print(f"Average Net Worth: ${results['avg_net_worth']:,.0f} ± ${results['std_net_worth']:,.0f}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Bankruptcy Rate: {results['bankruptcy_rate']:.1%}")
        print(f"Debt-Free Rate: {results['debt_free_rate']:.1%}")
        print(f"Net Worth Range: ${results['min_net_worth']:,.0f} to ${results['max_net_worth']:,.0f}")
    
    # Rank strategies by performance
    print(f"\n=== FINAL RANKINGS ===")
    sorted_results = sorted(all_results, key=lambda x: x['avg_net_worth'], reverse=True)
    
    for i, result in enumerate(sorted_results):
        print(f"{i+1}. {result['agent_name']}: ${result['avg_net_worth']:,.0f}")
    
    # Create visualization
    plot_results(all_results)
    
    print("\nSUCCESS: Baseline evaluation complete!")
    print("Results saved to: baseline_evaluation_results.png")
    
    return all_results

if __name__ == "__main__":
    results = main()
