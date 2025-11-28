"""
Statistical Analysis Script
Runs multiple seeds per agent and performs statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
from typing import Dict, List
import sys
import os
sys.path.append('..')

from environment.finance_env import FinanceEnv
from agents import (
    SixtyFortyAgent, DebtAvalancheAgent, EqualWeightAgent,
    AgeBasedAgent, MarkowitzAgent
)

# Import discrete RL agents (you'll need to create these)
# For now, we'll just use baselines


def calculate_net_worth(state):
    """Calculate net worth from state vector (now at index 0)"""
    return state[0]  # Net worth is now directly in state


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Bonferroni correction for multiple comparisons"""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]


class StatisticalAnalyzer:
    def __init__(self, n_seeds: int = 5, output_dir: str = "statistical_results"):
        self.n_seeds = n_seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        
    def _get_agent(self, agent_name: str):
        """Load or create agent"""
        agent_map = {
            '60_40': SixtyFortyAgent(),
            'debt_avalanche': DebtAvalancheAgent(),
            'equal_weight': EqualWeightAgent(),
            'age_based': AgeBasedAgent(),
            'markowitz': MarkowitzAgent()
        }
        
        if agent_name == 'ppo':
            from agents.discrete_ppo_agent import DiscretePPOAgent
            agent = DiscretePPOAgent()
            if os.path.exists('../models/ppo_best_model.pth'):
                agent.load('../models/ppo_best_model.pth')
                agent.training = False
                print(f"Loaded discrete PPO model")
            else:
                print(f"No discrete PPO model found")
            return agent
    
        elif agent_name == 'dqn':
            from agents.discrete_dqn_agent import DiscreteDQNAgent
            agent = DiscreteDQNAgent()
            if os.path.exists('../models/dqn_best_model.pth'):
                agent.load('../models/dqn_best_model.pth')
                agent.training = False
                print(f"Loaded discrete DQN model")
            else:
                print(f"No discrete DQN model found")
            return agent
        
        return agent_map.get(agent_name)
    
    def run_single_evaluation(self, agent_name: str, seed: int) -> Dict:
        """Run one evaluation for an agent with given seed"""
        env = FinanceEnv(seed=seed)
        agent = self._get_agent(agent_name)
        
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        
        trajectory = {
            'net_worths': [],
            'rewards': [],
            'ages': [],
            'actions': []
        }
        
        step = 0
        while not done and step < 360:  # Safety limit
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Track trajectory
            trajectory['net_worths'].append(state[0])  # Net worth at index 0
            trajectory['rewards'].append(reward)
            trajectory['ages'].append(state[7])
            trajectory['actions'].append(action)
            
            total_reward += reward
            state = next_state
            done = terminated or truncated
            step += 1
        
        # Final metrics
        final_net_worth = state[0]  # Net worth is at index 0
        
        return {
            'final_net_worth': final_net_worth,
            'total_reward': total_reward,
            'trajectory': trajectory,
            'bankruptcy': final_net_worth < 0,
            'debt_free': state[4] + state[5] < 100  # cc_debt + student_loan
        }
    
    def run_all_experiments(self):
        """Run all agents with multiple seeds"""
        # Only baseline agents for now (no trained RL agents yet)
        agents = [
            '60_40', 'debt_avalanche', 'equal_weight',
            'age_based', 'markowitz', 'ppo', 'dqn'
        ]
        
        print(f"Running {len(agents)} agents with {self.n_seeds} seeds each...")
        print(f"Total evaluations: {len(agents) * self.n_seeds}")
        
        for agent_name in agents:
            print(f"\n{'='*60}")
            print(f"Evaluating agent: {agent_name.upper()}")
            print(f"{'='*60}")
            
            agent_results = []
            
            for seed in range(self.n_seeds):
                print(f"  Seed {seed+1}/{self.n_seeds}...", end=' ')
                
                try:
                    result = self.run_single_evaluation(agent_name, seed=10000 + seed)
                    agent_results.append(result)
                    print(f"Final NW: ${result['final_net_worth']:,.0f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.results[agent_name] = agent_results
            
            # Save intermediate results
            self.save_results()
        
        print("\n" + "="*60)
        print("All experiments complete!")
        print("="*60)
    
    def save_results(self):
        """Save results to JSON"""
        output_file = self.output_dir / "raw_results.json"
        
        # Convert to serializable format
        serializable_results = {}
        for agent, runs in self.results.items():
            serializable_results[agent] = []
            for run in runs:
                serializable_run = {
                    'final_net_worth': float(run['final_net_worth']),
                    'total_reward': float(run['total_reward']),
                    'bankruptcy': bool(run['bankruptcy']),
                    'debt_free': bool(run['debt_free'])
                }
                serializable_results[agent].append(serializable_run)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def load_results(self):
        """Load results from JSON"""
        input_file = self.output_dir / "raw_results.json"
        with open(input_file, 'r') as f:
            self.results = json.load(f)
        print(f"Results loaded from {input_file}")
    
    def compute_statistics(self) -> pd.DataFrame:
        """Compute summary statistics for all agents"""
        stats_data = []
        
        for agent_name, runs in self.results.items():
            if not runs:  # Skip empty results
                continue
                
            net_worths = [r['final_net_worth'] for r in runs]
            
            if not net_worths:  # Skip if no valid net worths
                continue
            
            stats_data.append({
                'Agent': agent_name,
                'Mean': np.mean(net_worths),
                'Std': np.std(net_worths, ddof=1),
                'Median': np.median(net_worths),
                'Min': np.min(net_worths),
                'Max': np.max(net_worths),
                'CI_95_Lower': np.percentile(net_worths, 2.5),
                'CI_95_Upper': np.percentile(net_worths, 97.5),
                'Bankruptcy_Rate': np.mean([r['bankruptcy'] for r in runs]),
                'Debt_Free_Rate': np.mean([r['debt_free'] for r in runs])
            })
        
        df = pd.DataFrame(stats_data)
        df = df.sort_values('Mean', ascending=False)
        
        # Save to CSV
        df.to_csv(self.output_dir / "summary_statistics.csv", index=False)
        print("\nSummary statistics saved!")
        
        return df
    
    def pairwise_comparisons(self, baseline_agent: str = '60_40') -> pd.DataFrame:
        """Perform pairwise statistical tests against baseline"""
        comparison_data = []
        
        if baseline_agent not in self.results or not self.results[baseline_agent]:
            print(f"Warning: Baseline agent '{baseline_agent}' has no results")
            return pd.DataFrame()
        
        baseline_results = [r['final_net_worth'] for r in self.results[baseline_agent]]
        p_values = []
        
        for agent_name, runs in self.results.items():
            if agent_name == baseline_agent or not runs:
                continue
            
            agent_results = [r['final_net_worth'] for r in runs]
            
            if not agent_results:
                continue
            
            # T-test
            t_stat, p_value = stats.ttest_ind(agent_results, baseline_results)
            p_values.append(p_value)
            
            # Cohen's d
            effect_size = cohens_d(np.array(agent_results), np.array(baseline_results))
            
            # Interpretation
            if abs(effect_size) < 0.2:
                effect_interp = "negligible"
            elif abs(effect_size) < 0.5:
                effect_interp = "small"
            elif abs(effect_size) < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            
            comparison_data.append({
                'Agent': agent_name,
                'Mean_Diff': np.mean(agent_results) - np.mean(baseline_results),
                'Pct_Improvement': ((np.mean(agent_results) / np.mean(baseline_results)) - 1) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': effect_size,
                'effect_size': effect_interp,
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Mean_Diff', ascending=False)
        
        # Apply Bonferroni correction
        if p_values:
            df['significant_bonferroni'] = bonferroni_correction(p_values)
            df['significant_uncorrected'] = df['p_value'] < 0.05
        
        # Save to CSV
        df.to_csv(self.output_dir / f"pairwise_vs_{baseline_agent}.csv", index=False)
        print(f"\nPairwise comparisons vs {baseline_agent} saved!")
        
        return df
    
    def plot_results_with_error_bars(self):
        """Create publication-quality plots with error bars"""
        # Prepare data
        agents = []
        means = []
        stds = []
        
        for agent_name, runs in self.results.items():
            if not runs:
                continue
            net_worths = [r['final_net_worth'] for r in runs]
            if not net_worths:
                continue
            agents.append(agent_name)
            means.append(np.mean(net_worths))
            stds.append(np.std(net_worths, ddof=1))
        
        if not agents:
            print("No data to plot")
            return
        
        # Sort by mean
        sorted_indices = np.argsort(means)[::-1]
        agents = [agents[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color coding
        colors = ['#A23B72'] * len(agents)  # All purple for baselines
        
        # Bar plot with error bars
        bars = ax.bar(range(len(agents)), means, yerr=stds, 
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Agent', fontsize=14, fontweight='bold')
        ax.set_ylabel('Final Net Worth ($)', fontsize=14, fontweight='bold')
        ax.set_title('Baseline Agent Performance (Mean ± Std, n=5)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_with_error_bars.png", dpi=300)
        print("\nPlot with error bars saved!")
        plt.close()
    
    def plot_confidence_intervals(self):
        """Plot with 95% confidence intervals"""
        # Prepare data
        agents = []
        means = []
        ci_lower = []
        ci_upper = []
        
        for agent_name, runs in self.results.items():
            if not runs:
                continue
            net_worths = [r['final_net_worth'] for r in runs]
            if not net_worths:
                continue
            agents.append(agent_name)
            means.append(np.mean(net_worths))
            ci_lower.append(np.percentile(net_worths, 2.5))
            ci_upper.append(np.percentile(net_worths, 97.5))
        
        if not agents:
            print("No data to plot")
            return
        
        # Sort by mean
        sorted_indices = np.argsort(means)[::-1]
        agents = [agents[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        ci_lower = [ci_lower[i] for i in sorted_indices]
        ci_upper = [ci_upper[i] for i in sorted_indices]
        
        # Calculate error bars
        yerr_lower = [means[i] - ci_lower[i] for i in range(len(means))]
        yerr_upper = [ci_upper[i] - means[i] for i in range(len(means))]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Point plot with error bars
        ax.errorbar(range(len(agents)), means, 
                   yerr=[yerr_lower, yerr_upper],
                   fmt='o', markersize=10, capsize=7, capthick=2,
                   color='#A23B72', ecolor='#666666', linewidth=2)
        
        ax.set_xlabel('Agent', fontsize=14, fontweight='bold')
        ax.set_ylabel('Final Net Worth ($)', fontsize=14, fontweight='bold')
        ax.set_title('Agent Performance with 95% Confidence Intervals (n=5)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confidence_intervals.png", dpi=300)
        print("Confidence interval plot saved!")
        plt.close()
    
    def create_significance_table(self):
        """Create LaTeX-formatted significance table"""
        # Get summary stats
        stats_df = self.compute_statistics()
        
        if stats_df.empty:
            print("No statistics to create table")
            return stats_df
        
        # Get pairwise comparisons
        comparison_df = self.pairwise_comparisons()
        
        if comparison_df.empty:
            print("No comparisons to merge")
            return stats_df
        
        # Merge
        merged = stats_df.merge(
            comparison_df[['Agent', 'p_value', 'cohens_d', 'significant_bonferroni']], 
            on='Agent', 
            how='left'
        )
        
        # Format for LaTeX
        merged['Mean_Formatted'] = merged['Mean'].apply(lambda x: f"${x/1e6:.2f}M")
        merged['CI_Formatted'] = merged.apply(
            lambda row: f"[${row['CI_95_Lower']/1e6:.2f}M, ${row['CI_95_Upper']/1e6:.2f}M]", 
            axis=1
        )
        merged['Significance'] = merged['significant_bonferroni'].apply(
            lambda x: '***' if pd.notna(x) and x else ''
        )
        
        # Save to file
        with open(self.output_dir / "significance_table.txt", 'w') as f:
            f.write("Agent Performance with Statistical Significance\n")
            f.write("="*80 + "\n\n")
            f.write(merged[['Agent', 'Mean_Formatted', 'CI_Formatted', 'cohens_d', 'Significance']].to_string(index=False))
            f.write("\n\n*** = Significant after Bonferroni correction (p < 0.05/n)\n")
        
        print("\nSignificance table saved!")
        
        return merged


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run statistical analysis for ADJFin (discrete)')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds per agent')
    parser.add_argument('--run', action='store_true', help='Run experiments (otherwise load existing)')
    parser.add_argument('--baseline', type=str, default='60_40', help='Baseline agent for comparisons')
    
    args = parser.parse_args()
    
    analyzer = StatisticalAnalyzer(n_seeds=args.seeds)
    
    if args.run:
        analyzer.run_all_experiments()
    else:
        try:
            analyzer.load_results()
        except FileNotFoundError:
            print("No existing results found. Running experiments...")
            analyzer.run_all_experiments()
    
    # Compute all statistics and create visualizations
    print("\n" + "="*60)
    print("COMPUTING STATISTICS")
    print("="*60)
    
    stats_df = analyzer.compute_statistics()
    if not stats_df.empty:
        print("\n" + stats_df.to_string(index=False))
    
    comparison_df = analyzer.pairwise_comparisons(baseline_agent=args.baseline)
    if not comparison_df.empty:
        print("\n" + comparison_df.to_string(index=False))
    
    analyzer.plot_results_with_error_bars()
    analyzer.plot_confidence_intervals()
    analyzer.create_significance_table()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*60)