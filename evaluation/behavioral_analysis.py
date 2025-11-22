"""
Behavioral Analysis Script for ADJFin RL Project
Analyzes agent decision-making patterns across market regimes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import sys
import os
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from environment.finance_env import FinanceEnv
from agents import (
    SixtyFortyAgent, DebtAvalancheAgent, EqualWeightAgent,
    AgeBasedAgent, MarkowitzAgent, PPOAgent, ContinuousDQNAgent
)
from agents.sac_agent import SACAgent


def calculate_net_worth(state):
    """Calculate net worth from state"""
    return state[0] + state[1] + state[2] + state[3] + state[8] - state[4] - state[5]


class BehavioralAnalyzer:
    def __init__(self, output_dir: str = "behavioral_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.action_labels = [
            'Stock Allocation',
            'Bond Allocation', 
            'Real Estate Allocation',
            'Emergency Fund Contribution',
            'Credit Card Payment',
            'Student Loan Payment'
        ]
        
        self.behavioral_data = {}
    
    def _get_agent(self, agent_name: str):
        """Load or create agent"""
        agent_map = {
            '60_40': SixtyFortyAgent(),
            'debt_avalanche': DebtAvalancheAgent(),
            'equal_weight': EqualWeightAgent(),
            'age_based': AgeBasedAgent(),
            'markowitz': MarkowitzAgent()
        }
        
        # RL agents
        if agent_name == 'ppo':
            agent = PPOAgent()
            if os.path.exists('../models/ppo_model.pth'):
                agent.load('../models/ppo_model.pth')
                agent.training = False
            return agent
        elif agent_name == 'dqn':
            agent = ContinuousDQNAgent()
            if os.path.exists('../models/dqn_model.pth'):
                agent.load('../models/dqn_model.pth')
                agent.training = False
            return agent
        elif agent_name == 'sac':
            agent = SACAgent()
            if os.path.exists('../models/sac_model.pth'):
                agent.load('../models/sac_model.pth')
                agent.training = False
            return agent
        
        return agent_map.get(agent_name)
    
    def run_agent_analysis(self, agent_name: str, n_episodes: int = 5):
        """Run episodes and collect behavioral data"""
        print(f"\nAnalyzing {agent_name}...")
        
        agent = self._get_agent(agent_name)
        
        agent_behavior = {
            'actions_by_regime': {0: [], 1: [], 2: []},  # Normal, Bull, Bear
            'allocations_by_age': {},
            'action_sequences': [],
            'market_timing_score': 0,
            'debt_priority_score': 0,
            'emergency_fund_priority': 0
        }
        
        for episode in range(n_episodes):
            env = FinanceEnv()
            state, _ = env.reset(seed=42 + episode)
            
            done = False
            episode_data = []
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Parse state
                regime = int(state[10])  # Market regime
                age = int(state[7])
                
                # Record action by regime
                agent_behavior['actions_by_regime'][regime].append(action)
                
                # Record allocation by age decade
                age_bucket = (age // 10) * 10
                if age_bucket not in agent_behavior['allocations_by_age']:
                    agent_behavior['allocations_by_age'][age_bucket] = []
                
                agent_behavior['allocations_by_age'][age_bucket].append({
                    'emergency': state[8],
                    'stocks': state[1],
                    'bonds': state[2],
                    'cc_debt': state[4],
                    'student_debt': state[5],
                    'action': action
                })
                
                episode_data.append({
                    'action': action.copy(),
                    'regime': regime,
                    'age': age,
                    'state': state.copy()
                })
                
                state = next_state
                done = terminated or truncated
            
            agent_behavior['action_sequences'].append(episode_data)
        
        # Calculate behavioral scores
        agent_behavior['market_timing_score'] = self._calculate_market_timing_score(
            agent_behavior['actions_by_regime']
        )
        agent_behavior['debt_priority_score'] = self._calculate_debt_priority_score(
            agent_behavior['action_sequences']
        )
        agent_behavior['emergency_fund_priority'] = self._calculate_emergency_priority(
            agent_behavior['allocations_by_age']
        )
        
        self.behavioral_data[agent_name] = agent_behavior
        return agent_behavior
    
    def _calculate_market_timing_score(self, actions_by_regime: Dict) -> float:
        """
        Score how well agent times the market
        Higher score = more risk reduction in bear markets
        """
        if not actions_by_regime[2]:  # No bear market data
            return 0.0
        
        # Calculate average stock allocation by regime
        def avg_stock_allocation(actions):
            if not actions:
                return 0.0
            stock_allocs = [a[0] for a in actions]  # First action is stock allocation
            return np.mean(stock_allocs)
        
        bull_stocks = avg_stock_allocation(actions_by_regime[1])
        bear_stocks = avg_stock_allocation(actions_by_regime[2])
        
        # Good market timing = reduce stocks in bear markets
        return max(0, bull_stocks - bear_stocks)
    
    def _calculate_debt_priority_score(self, action_sequences: List) -> float:
        """
        Score how well agent prioritizes high-interest debt
        """
        cc_focus = []
        student_focus = []
        
        for sequence in action_sequences:
            for step in sequence:
                state = step['state']
                action = step['action']
                
                # When has credit card debt, how much does agent pay?
                if state[4] > 100:  # Has credit card debt
                    cc_focus.append(action[4])  # CC payment action
                
                # When has student debt, how much does agent pay?
                if state[5] > 100:  # Has student debt
                    student_focus.append(action[5])  # Student payment action
        
        # Higher score = prioritizes credit card debt (higher interest)
        cc_priority = np.mean(cc_focus) if cc_focus else 0
        student_priority = np.mean(student_focus) if student_focus else 0
        
        return cc_priority / (student_priority + 0.01)  # Avoid division by zero
    
    def _calculate_emergency_priority(self, allocations_by_age: Dict) -> float:
        """
        Score how well agent builds emergency fund early
        """
        if 20 not in allocations_by_age:
            return 0.0
        
        early_allocations = allocations_by_age[20]
        avg_emergency = np.mean([a['emergency'] for a in early_allocations])
        
        # Normalize by typical emergency fund target (6 months expenses = ~12k)
        return min(1.0, avg_emergency / 12000)
    
    def plot_action_heatmap(self):
        """Create heatmap showing action preferences by agent and regime"""
        agents_to_plot = ['dqn', 'ppo', 'sac', '60_40', 'age_based', 'markowitz']
        agents_in_data = [a for a in agents_to_plot if a in self.behavioral_data]
        
        if not agents_in_data:
            print("No agents with behavioral data to plot")
            return
        
        # Prepare data - average actions by regime
        data_matrix = []
        agent_labels = []
        
        for agent_name in agents_in_data:
            agent_labels.append(agent_name)
            actions_by_regime = self.behavioral_data[agent_name]['actions_by_regime']
            
            row = []
            for regime in [0, 1, 2]:  # Normal, Bull, Bear
                actions = actions_by_regime[regime]
                if not actions:
                    row.extend([0] * 6)
                    continue
                
                # Average each action dimension
                avg_actions = np.mean(actions, axis=0)
                row.extend(avg_actions)
            
            data_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        
        heatmap_data = np.array(data_matrix)
        
        # Column labels
        col_labels = []
        for regime in ['Normal', 'Bull', 'Bear']:
            for action in self.action_labels:
                col_labels.append(f"{regime[:4]}_{action.split()[0]}")
        
        sns.heatmap(heatmap_data, 
                   xticklabels=col_labels,
                   yticklabels=agent_labels,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Action Value (0-1)'},
                   ax=ax)
        
        plt.title('Agent Action Patterns by Market Regime', fontsize=16, fontweight='bold')
        plt.xlabel('Action by Market Regime', fontsize=12)
        plt.ylabel('Agent', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(self.output_dir / "action_heatmap_by_regime.png", dpi=300)
        print("Action heatmap saved!")
        plt.close()
    
    def plot_allocation_evolution(self):
        """Plot how allocations change with age"""
        agents_to_plot = ['dqn', 'ppo', 'sac', '60_40', 'age_based']
        agents_in_data = [a for a in agents_to_plot if a in self.behavioral_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Evolution by Age', fontsize=16, fontweight='bold')
        
        components = ['emergency', 'stocks', 'bonds', 'cc_debt']
        titles = ['Emergency Fund', 'Stock Holdings', 'Bond Holdings', 'Credit Card Debt']
        
        for idx, (component, title) in enumerate(zip(components, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for agent_name in agents_in_data:
                allocations = self.behavioral_data[agent_name]['allocations_by_age']
                
                ages = sorted(allocations.keys())
                means = []
                stds = []
                
                for age in ages:
                    values = [a[component] for a in allocations[age]]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                
                ax.plot(ages, means, marker='o', label=agent_name, linewidth=2)
                ax.fill_between(ages, 
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.2)
            
            ax.set_xlabel('Age', fontsize=12)
            ax.set_ylabel(f'{title} ($)', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k' if x >= 1000 else f'${x:.0f}'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "allocation_evolution.png", dpi=300)
        print("Allocation evolution plot saved!")
        plt.close()
    
    def plot_market_timing_comparison(self):
        """Compare market timing scores across agents"""
        if not self.behavioral_data:
            print("No behavioral data to plot")
            return
        
        agents = []
        scores = []
        
        for agent_name, data in self.behavioral_data.items():
            agents.append(agent_name)
            scores.append(data['market_timing_score'])
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        agents = [agents[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#2E86AB' if a in ['dqn', 'ppo', 'sac'] else '#A23B72' for a in agents]
        
        ax.barh(agents, scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Market Timing Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Agent', fontsize=12, fontweight='bold')
        ax.set_title('Market Timing Ability (Higher = Better Risk Management)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "market_timing_scores.png", dpi=300)
        print("Market timing comparison saved!")
        plt.close()
    
    def plot_action_distribution(self):
        """Plot distribution of actions for key agents"""
        agents_to_plot = ['dqn', 'ppo', '60_40', 'markowitz']
        agents_in_data = [a for a in agents_to_plot if a in self.behavioral_data]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Action Distribution Comparison', fontsize=16, fontweight='bold')
        
        for action_idx, action_name in enumerate(self.action_labels):
            ax = axes[action_idx // 3, action_idx % 3]
            
            for agent_name in agents_in_data:
                # Collect all actions across all regimes
                all_actions = []
                for regime in [0, 1, 2]:
                    actions = self.behavioral_data[agent_name]['actions_by_regime'][regime]
                    if actions:
                        all_actions.extend([a[action_idx] for a in actions])
                
                if all_actions:
                    ax.hist(all_actions, bins=20, alpha=0.5, label=agent_name, density=True)
            
            ax.set_xlabel(action_name, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(action_name, fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "action_distributions.png", dpi=300)
        print("Action distribution plot saved!")
        plt.close()
    
    def generate_behavioral_report(self):
        """Generate comprehensive behavioral analysis report"""
        report = []
        report.append("="*80)
        report.append("BEHAVIORAL ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        for agent_name, data in self.behavioral_data.items():
            report.append(f"\n{'='*60}")
            report.append(f"Agent: {agent_name.upper()}")
            report.append(f"{'='*60}")
            
            # Market timing
            report.append(f"\nMarket Timing Score: {data['market_timing_score']:.3f}")
            report.append("  (Higher = better risk reduction in bear markets)")
            
            # Debt priority
            report.append(f"\nDebt Priority Score: {data['debt_priority_score']:.3f}")
            report.append("  (Higher = prioritizes high-interest credit card debt)")
            
            # Emergency fund
            report.append(f"\nEmergency Fund Priority: {data['emergency_fund_priority']:.3f}")
            report.append("  (Target: 1.0 = $12k emergency fund in 20s)")
            
            # Average actions by regime
            report.append("\nAverage Actions by Market Regime:")
            for regime, regime_name in [(0, 'NORMAL'), (1, 'BULL'), (2, 'BEAR')]:
                actions = data['actions_by_regime'][regime]
                if not actions:
                    continue
                
                report.append(f"\n  {regime_name} Market:")
                avg_action = np.mean(actions, axis=0)
                for i, (action_val, action_name) in enumerate(zip(avg_action, self.action_labels)):
                    report.append(f"    {action_name}: {action_val:.3f}")
        
        report.append("\n" + "="*80)
        
        # Save report
        with open(self.output_dir / "behavioral_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        print("\nBehavioral report saved!")
        return '\n'.join(report)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Behavioral analysis for ADJFin')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='Episodes per agent for analysis')
    parser.add_argument('--agents', nargs='+', default=None,
                       help='Specific agents to analyze')
    
    args = parser.parse_args()
    
    analyzer = BehavioralAnalyzer()
    
    # Default agents to analyze
    default_agents = ['dqn', 'ppo', 'sac', '60_40', 'debt_avalanche', 
                     'age_based', 'markowitz']
    agents_to_analyze = args.agents if args.agents else default_agents
    
    print("="*60)
    print("RUNNING BEHAVIORAL ANALYSIS")
    print("="*60)
    
    for agent_name in agents_to_analyze:
        try:
            analyzer.run_agent_analysis(agent_name, n_episodes=args.episodes)
        except Exception as e:
            print(f"Error analyzing {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    analyzer.plot_action_heatmap()
    analyzer.plot_allocation_evolution()
    analyzer.plot_market_timing_comparison()
    analyzer.plot_action_distribution()
    
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    report = analyzer.generate_behavioral_report()
    print(report)
    
    print("\n" + "="*60)
    print(f"Analysis complete! Results saved to: {analyzer.output_dir}")
    print("="*60)