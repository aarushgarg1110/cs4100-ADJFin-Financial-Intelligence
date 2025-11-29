"""
Ablation Study Script for ADJFin
Tests impact of market volatility on RL performance
Tests DQN models with different Sharpe ratios + human baselines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys
import os
sys.path.append('..')

from environment.finance_env import FinanceEnv
from agents import (
    SixtyFortyAgent, AgeBasedAgent, MarkowitzAgent,
    DebtAvalancheAgent, EqualWeightAgent
)


def calculate_net_worth(state):
    """Calculate net worth from state (now at index 0)"""
    return state[0]


class StableMarketEnvironment(FinanceEnv):
    """Modified environment with no market volatility"""
    
    def _update_market(self):
        """Override with constant average returns (no volatility)"""
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
        
        excess_returns = returns - (risk_free_rate / 12)
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(12)
    
    @staticmethod
    def max_drawdown(net_worths: np.ndarray) -> float:
        """Calculate maximum drawdown"""
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
        
        if len(net_worths) == 0:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'final_net_worth': 0.0,
                'total_return': 0.0
            }
        
        # Calculate returns
        returns = np.diff(net_worths) / (net_worths[:-1] + 1e-6)
        
        return {
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns),
            'max_drawdown': FinancialMetrics.max_drawdown(net_worths),
            'volatility': FinancialMetrics.volatility(returns),
            'final_net_worth': net_worths[-1],
            'total_return': (net_worths[-1] - net_worths[0]) / (net_worths[0] + 1e-6)
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
        agent_map = {
            '60_40': SixtyFortyAgent(),
            'age_based': AgeBasedAgent(),
            'markowitz': MarkowitzAgent(),
            'debt_avalanche': DebtAvalancheAgent(),
            'equal_weight': EqualWeightAgent()
        }
        
        # Check if it's a DQN model with sharpe ratio
        if agent_name.startswith('dqn_sharpe'):
            from agents.discrete_dqn_agent import DiscreteDQNAgent
            
            # Extract sharpe percentage (e.g., 'dqn_sharpe60' -> 60)
            sharpe_pct = agent_name.replace('dqn_sharpe', '')
            model_path = f'../models/dqn_sharpe{sharpe_pct}.pth'
            
            if os.path.exists(model_path):
                agent = DiscreteDQNAgent()
                agent.load(model_path)
                agent.training = False
                print(f"Loaded DQN model: {model_path}")
                return agent
            else:
                print(f"Warning: Model not found: {model_path}")
                return None

        return agent_map.get(agent_name)
    
    def run_ablation_experiment(self, n_seeds: int = 5):
        """Run agents in both normal and stable market conditions"""
        
        # Focus on key agents: select DQN models + human baselines
        agents = [
            'dqn_sharpe30', 'dqn_sharpe60', 'dqn_sharpe80',
            '60_40', 'age_based', 'markowitz'
        ]
        
        print("="*60)
        print("ABLATION STUDY: Market Volatility Impact (DQN Sharpe Models)")
        print("="*60)
        
        for agent_name in agents:
            print(f"\nTesting {agent_name.upper()}...")
            
            # Normal market (with volatility)
            print("  Running with market volatility...")
            normal_results = self._run_agent_seeds(agent_name, n_seeds, stable=False)
            if normal_results:
                self.results['normal_market'][agent_name] = normal_results
            
            # Stable market (no volatility)
            print("  Running without market volatility...")
            stable_results = self._run_agent_seeds(agent_name, n_seeds, stable=True)
            if stable_results:
                self.results['stable_market'][agent_name] = stable_results
            
            # Calculate impact
            if normal_results and stable_results:
                normal_mean = np.mean([r['final_net_worth'] for r in normal_results])
                stable_mean = np.mean([r['final_net_worth'] for r in stable_results])
                impact = ((normal_mean - stable_mean) / stable_mean) * 100
                
                print(f"    Normal market: ${normal_mean:,.0f}")
                print(f"    Stable market: ${stable_mean:,.0f}")
                print(f"    Impact: {impact:+.1f}%")
    
    def _run_agent_seeds(self, agent_name: str, n_seeds: int, stable: bool) -> List[Dict]:
        """Run agent for multiple seeds"""
        results = []
        
        agent = self._get_agent(agent_name)
        if agent is None:
            print(f"    Skipping {agent_name} - agent not found")
            return None
        
        for seed in range(n_seeds):
            # Create environment
            if stable:
                env = StableMarketEnvironment(seed=10000 + seed)
            else:
                env = FinanceEnv(seed=10000 + seed)
            
            # Run episode
            state, _ = env.reset(seed=10000 + seed)
            done = False
            total_reward = 0
            net_worths = []
            step = 0
            
            while not done and step < 360:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                net_worths.append(state[0])  # Net worth at index 0
                
                total_reward += reward
                state = next_state
                done = terminated or truncated
                step += 1
            
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
        colors = []
        
        for agent_name in self.results['normal_market'].keys():
            normal_results = self.results['normal_market'][agent_name]
            stable_results = self.results['stable_market'][agent_name]
            
            normal_mean = np.mean([r['final_net_worth'] for r in normal_results])
            stable_mean = np.mean([r['final_net_worth'] for r in stable_results])
            
            agents.append(agent_name)
            normal_means.append(normal_mean)
            stable_means.append(stable_mean)
            improvements.append(((normal_mean - stable_mean) / stable_mean) * 100)
            
            # Color coding
            if agent_name.startswith('dqn_sharpe'):
                colors.append('#2E86AB')
            else:
                colors.append('#A23B72')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Absolute performance
        x = np.arange(len(agents))
        width = 0.35
        
        ax1.bar(x - width/2, normal_means, width, 
               label='With Volatility', color='#2E86AB', alpha=0.7)
        ax1.bar(x + width/2, stable_means, width,
               label='Without Volatility', color='#A23B72', alpha=0.7)
        
        ax1.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Final Net Worth ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance: Normal vs Stable Markets', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Plot 2: Relative improvement
        bar_colors = ['#2E86AB' if imp > 0 else '#C73E1D' for imp in improvements]
        ax2.barh(agents, improvements, color=bar_colors, alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Performance Change (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Agent', fontsize=12, fontweight='bold')
        ax2.set_title('Impact of Market Volatility', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (val) in enumerate(improvements):
            ax2.text(val + (1 if val > 0 else -1), i, f'{val:+.1f}%', 
                    va='center', ha='left' if val > 0 else 'right',
                    fontweight='bold')
        
        plt.tight_layout()
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
        report.append("ABLATION STUDY: Market Volatility Impact (DQN Sharpe Models)")
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
        
        # Save report
        with open(self.output_dir / "ablation_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        print("\nAblation report saved!")
        return '\n'.join(report)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation study for ADJFin (DQN Sharpe models)')
    parser.add_argument('--seeds', type=int, default=5, 
                       help='Number of seeds per condition')
    
    args = parser.parse_args()
    
    study = AblationStudy()
    
    print("="*60)
    print("ABLATION STUDY: MARKET VOLATILITY (DQN Sharpe Models)")
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