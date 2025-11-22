"""
Ablation Study Script for ADJFin RL Project
Tests impact of market volatility on RL performance
Includes stable market environment and financial metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from environment.finance_env import FinanceEnv
from agents import (
    SixtyFortyAgent, AgeBasedAgent, MarkowitzAgent,
    PPOAgent, ContinuousDQNAgent, DebtAvalancheAgent, EqualWeightAgent
)
from agents.sac_agent import SACAgent


def calculate_net_worth(state):
    """Calculate net worth from state"""
    return state[0] + state[1] + state[2] + state[3] + state[8] - state[4] - state[5]


class StableMarketEnvironment(FinanceEnv):
    """Modified environment with no market volatility"""
    
    def _update_market(self):
        """Override with constant average returns (no volatility)"""
        # Use constant average returns instead of regime-based sampling
        # Historical averages: stocks ~10% annual, bonds ~4% annual, RE ~7% annual
        self.stock_return_1m = 0.008  # ~10% annual
        self.bond_return_1m = 0.003   # ~4% annual
        self.re_return_1m = 0.006     # ~7% annual
        
        # Apply returns
        self.stocks *= (1 + self.stock_return_1m)
        self.bonds *= (1 + self.bond_return_1m)
        self.real_estate *= (1 + self.re_return_1m)
        
        # Keep regime at normal (0) always
        self.current_regime = 0


class FinancialMetrics:
    """Calculate financial performance metrics"""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (risk-adjusted returns)"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 12)  # Monthly risk-free rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(12)
    
    @staticmethod
    def max_drawdown(net_worths: np.ndarray) -> float:
        """Calculate maximum drawdown (worst peak-to-trough decline)"""
        if len(net_worths) == 0:
            return 0.0
        
        peak = net_worths[0]
        max_dd = 0.0
        
        for value in net_worths:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    @staticmethod
    def volatility(returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(12)
    
    @staticmethod
    def calculate_all_metrics(net_worths: List[float]) -> Dict:
        """Calculate all financial metrics from net worth trajectory"""
        net_worths = np.array(net_worths)
        
        # Calculate returns
        returns = np.diff(net_worths) / (net_worths[:-1] + 1e-6)  # Avoid division by zero
        
        return {
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns),
            'max_drawdown': FinancialMetrics.max_drawdown(net_worths),
            'volatility': FinancialMetrics.volatility(returns),
            'final_net_worth': net_worths[-1] if len(net_worths) > 0 else 0,
            'total_return': (net_worths[-1] - net_worths[0]) / (net_worths[0] + 1e-6) if len(net_worths) > 0 else 0
        }


class AblationStudy:
    """Conducts ablation experiments"""
    
    def __init__(self, output_dir: str = "ablation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            'normal_market': {},
            'stable_market': {}
        }
    
    def _get_agent(self, agent_name: str):
        """Load or create agent"""
        from agents.baseline_strategies import AllStocksAgent, CashHoarderAgent, DebtIgnorerAgent
        
        agent_map = {
            '60_40': SixtyFortyAgent(),
            'age_based': AgeBasedAgent(),
            'markowitz': MarkowitzAgent(),
            'debt_avalanche': DebtAvalancheAgent(),
            'equal_weight': EqualWeightAgent(),
            'all_stocks': AllStocksAgent(),
            'cash_hoarder': CashHoarderAgent(),
            'debt_ignorer': DebtIgnorerAgent()
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
    
    def run_ablation_experiment(self, n_seeds: int = 5):
        """Run agents in both normal and stable market conditions"""
        
        agents = [
            'dqn', 'ppo', 'sac',  # RL agents
            '60_40', 'age_based', 'markowitz',  # Traditional
            'debt_avalanche', 'equal_weight',  # Expert
            'all_stocks', 'cash_hoarder', 'debt_ignorer'  # Naive
        ]
        
        print("="*60)
        print("ABLATION STUDY: Market Volatility Impact")
        print("="*60)
        
        for agent_name in agents:
            print(f"\nTesting {agent_name.upper()}...")
            
            # Normal market (with volatility)
            print("  Running with market volatility...")
            normal_results = self._run_agent_seeds(agent_name, n_seeds, stable=False)
            self.results['normal_market'][agent_name] = normal_results
            
            # Stable market (no volatility)
            print("  Running without market volatility...")
            stable_results = self._run_agent_seeds(agent_name, n_seeds, stable=True)
            self.results['stable_market'][agent_name] = stable_results
            
            # Calculate impact
            normal_mean = np.mean([r['final_net_worth'] for r in normal_results])
            stable_mean = np.mean([r['final_net_worth'] for r in stable_results])
            impact = ((normal_mean - stable_mean) / stable_mean) * 100
            
            print(f"    Normal market: ${normal_mean:,.0f}")
            print(f"    Stable market: ${stable_mean:,.0f}")
            print(f"    Impact: {impact:+.1f}%")
    
    def _run_agent_seeds(self, agent_name: str, n_seeds: int, stable: bool) -> List[Dict]:
        """Run agent for multiple seeds"""
        results = []
        
        for seed in range(n_seeds):
            # Create environment
            if stable:
                env = StableMarketEnvironment()
            else:
                env = FinanceEnv()
            
            # Load agent
            agent = self._get_agent(agent_name)
            
            # Run episode
            state, _ = env.reset(seed=42 + seed)
            done = False
            total_reward = 0
            net_worths = []
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                net_worths.append(calculate_net_worth(state))
                
                total_reward += reward
                state = next_state
                done = terminated or truncated
            
            # Calculate metrics
            metrics = FinancialMetrics.calculate_all_metrics(net_worths)
            metrics['total_reward'] = total_reward
            results.append(metrics)
        
        return results
    
    def plot_ablation_results(self):
        """Visualize ablation study results"""
        
        # Prepare data
        agents = []
        normal_means = []
        stable_means = []
        improvements = []
        
        for agent_name in self.results['normal_market'].keys():
            normal_results = self.results['normal_market'][agent_name]
            stable_results = self.results['stable_market'][agent_name]
            
            normal_mean = np.mean([r['final_net_worth'] for r in normal_results])
            stable_mean = np.mean([r['final_net_worth'] for r in stable_results])
            
            agents.append(agent_name)
            normal_means.append(normal_mean)
            stable_means.append(stable_mean)
            improvements.append(((normal_mean - stable_mean) / stable_mean) * 100)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Absolute performance comparison
        x = np.arange(len(agents))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, normal_means, width, 
                       label='With Volatility', color='#2E86AB', alpha=0.7)
        bars2 = ax1.bar(x + width/2, stable_means, width,
                       label='Without Volatility', color='#A23B72', alpha=0.7)
        
        ax1.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Final Net Worth ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance: Normal vs Stable Markets', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=45, ha='right', fontsize=10)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Plot 2: Relative improvement
        colors = ['#2E86AB' if imp > 0 else '#C73E1D' for imp in improvements]
        bars = ax2.barh(agents, improvements, color=colors, alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Performance Change (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Agent', fontsize=12, fontweight='bold')
        ax2.set_title('Impact of Market Volatility', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels with better positioning to avoid overlap
        x_range = max(improvements) - min(improvements) if improvements else 1
        offset = max(0.02 * x_range, 0.5)  # Dynamic offset based on data range
        
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            # Position label outside the bar
            x_pos = val + offset if val >= 0 else val - offset
            ax2.text(x_pos, i, f'{val:+.1f}%', 
                    va='center', ha='left' if val >= 0 else 'right',
                    fontweight='bold', fontsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(self.output_dir / "ablation_comparison.png", dpi=300)
        print("\nAblation comparison plot saved!")
        plt.close()
    
    def plot_financial_metrics(self):
        """Plot financial performance metrics"""
        
        metrics_to_plot = ['sharpe_ratio', 'max_drawdown', 'volatility']
        metric_names = ['Sharpe Ratio', 'Max Drawdown', 'Volatility']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx]
            
            agents = []
            normal_vals = []
            stable_vals = []
            
            for agent_name in self.results['normal_market'].keys():
                normal_results = self.results['normal_market'][agent_name]
                stable_results = self.results['stable_market'][agent_name]
                
                agents.append(agent_name)
                normal_vals.append(np.mean([r[metric] for r in normal_results]))
                stable_vals.append(np.mean([r[metric] for r in stable_results]))
            
            x = np.arange(len(agents))
            width = 0.35
            
            ax.bar(x - width/2, normal_vals, width, 
                  label='With Volatility', color='#2E86AB', alpha=0.7)
            ax.bar(x + width/2, stable_vals, width,
                  label='Without Volatility', color='#A23B72', alpha=0.7)
            
            ax.set_xlabel('Agent', fontsize=11, fontweight='bold')
            ax.set_ylabel(name, fontsize=11, fontweight='bold')
            ax.set_title(name, fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(agents, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Format specific metrics
            if 'drawdown' in metric or 'volatility' in metric:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "financial_metrics.png", dpi=300)
        print("Financial metrics plot saved!")
        plt.close()
    
    def generate_ablation_report(self):
        """Generate comprehensive ablation study report"""
        
        report = []
        report.append("="*80)
        report.append("ABLATION STUDY REPORT: Market Volatility Impact")
        report.append("="*80)
        report.append("")
        
        for agent_name in self.results['normal_market'].keys():
            report.append(f"\n{'='*60}")
            report.append(f"Agent: {agent_name.upper()}")
            report.append(f"{'='*60}")
            
            normal_results = self.results['normal_market'][agent_name]
            stable_results = self.results['stable_market'][agent_name]
            
            # Calculate statistics
            normal_nw = [r['final_net_worth'] for r in normal_results]
            stable_nw = [r['final_net_worth'] for r in stable_results]
            
            report.append(f"\nFinal Net Worth:")
            report.append(f"  With Volatility:    ${np.mean(normal_nw):,.0f} ± ${np.std(normal_nw):,.0f}")
            report.append(f"  Without Volatility: ${np.mean(stable_nw):,.0f} ± ${np.std(stable_nw):,.0f}")
            
            improvement = ((np.mean(normal_nw) - np.mean(stable_nw)) / np.mean(stable_nw)) * 100
            report.append(f"  Impact: {improvement:+.1f}%")
            
            # Financial metrics
            report.append(f"\nFinancial Metrics (With Volatility):")
            report.append(f"  Sharpe Ratio:   {np.mean([r['sharpe_ratio'] for r in normal_results]):.3f}")
            report.append(f"  Max Drawdown:   {np.mean([r['max_drawdown'] for r in normal_results]):.1%}")
            report.append(f"  Volatility:     {np.mean([r['volatility'] for r in normal_results]):.1%}")
            
            report.append(f"\nFinancial Metrics (Without Volatility):")
            report.append(f"  Sharpe Ratio:   {np.mean([r['sharpe_ratio'] for r in stable_results]):.3f}")
            report.append(f"  Max Drawdown:   {np.mean([r['max_drawdown'] for r in stable_results]):.1%}")
            report.append(f"  Volatility:     {np.mean([r['volatility'] for r in stable_results]):.1%}")
        
        report.append("\n" + "="*80)
        report.append("KEY FINDINGS")
        report.append("="*80)
        
        # Calculate RL advantage
        rl_agents = ['dqn', 'ppo', 'sac']
        baseline_agents = ['60_40', 'age_based', 'markowitz', 'debt_avalanche', 'equal_weight']
        
        rl_agents_present = [a for a in rl_agents if a in self.results['normal_market']]
        baseline_agents_present = [a for a in baseline_agents if a in self.results['normal_market']]
        
        if rl_agents_present and baseline_agents_present: 
            rl_normal_avg = np.mean([np.mean([r['final_net_worth'] 
                                              for r in self.results['normal_market'][a]]) 
                                    for a in rl_agents_present])
            rl_stable_avg = np.mean([np.mean([r['final_net_worth'] 
                                              for r in self.results['stable_market'][a]]) 
                                    for a in rl_agents_present if a in self.results['stable_market']])
            
            baseline_normal_avg = np.mean([np.mean([r['final_net_worth'] 
                                                     for r in self.results['normal_market'][a]]) 
                                           for a in baseline_agents_present])
            baseline_stable_avg = np.mean([np.mean([r['final_net_worth'] 
                                                     for r in self.results['stable_market'][a]]) 
                                           for a in baseline_agents_present if a in self.results['stable_market']])
            
            rl_advantage_normal = ((rl_normal_avg - baseline_normal_avg) / baseline_normal_avg) * 100
            rl_advantage_stable = ((rl_stable_avg - baseline_stable_avg) / baseline_stable_avg) * 100
            
            report.append(f"\nRL Advantage (vs Expert Baselines):")
            report.append(f"  With Volatility:    {rl_advantage_normal:+.1f}%")
            report.append(f"  Without Volatility: {rl_advantage_stable:+.1f}%")
            report.append(f"  Advantage Loss:     {rl_advantage_normal - rl_advantage_stable:.1f} percentage points")
            
            report.append("\nCONCLUSION:")
            if rl_advantage_normal - rl_advantage_stable > 5:
                report.append("  RL agents' advantage comes PRIMARILY from market timing ability")
            elif rl_advantage_stable > 10:
                report.append("  RL agents maintain strong advantage even without volatility")
                report.append("  suggesting superior long-term strategic planning")
            else:
                report.append("  RL agents show modest advantage over traditional strategies")
        else:
            report.append("\nRL Advantage: Cannot calculate (missing RL or baseline agents)")
        
        report.append("\n" + "="*80)
        
        # Save report
        with open(self.output_dir / "ablation_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        print("\nAblation report saved!")
        return '\n'.join(report)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation study for ADJFin')
    parser.add_argument('--seeds', type=int, default=5, 
                       help='Number of seeds per condition')
    
    args = parser.parse_args()
    
    study = AblationStudy()
    
    print("="*60)
    print("ABLATION STUDY: MARKET VOLATILITY IMPACT")
    print("="*60)
    
    study.run_ablation_experiment(n_seeds=args.seeds)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    study.plot_ablation_results()
    study.plot_financial_metrics()
    
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    report = study.generate_ablation_report()
    print(report)
    
    print("\n" + "="*60)
    print(f"Study complete! Results saved to: {study.output_dir}")
    print("="*60)