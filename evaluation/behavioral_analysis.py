"""
Behavioral Analysis Script for ADJFin 
Analyzes agent decision-making patterns across market regimes
Tracks which of the 90 discrete strategies agents prefer
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

from environment.finance_env import FinanceEnv, MONEY_ALLOC, INVEST_ALLOC, ACTION_DESCRIPTIONS
from agents import (
    SixtyFortyAgent, DebtAvalancheAgent, EqualWeightAgent,
    AgeBasedAgent, MarkowitzAgent
)


def calculate_net_worth(state):
    """Calculate net worth from state (at index 0)"""
    return state[0]


def decode_action(action_idx):
    """Decode discrete action into money and investment allocations"""
    money_idx = action_idx // len(INVEST_ALLOC)
    invest_idx = action_idx % len(INVEST_ALLOC)
    
    money_alloc = MONEY_ALLOC[money_idx][0]  # [invest%, debt%, emergency%]
    invest_alloc = INVEST_ALLOC[invest_idx][0]  # [stock%, bond%, re%]
    
    return {
        'action_idx': action_idx,
        'money_idx': money_idx,
        'invest_idx': invest_idx,
        'invest_pct': money_alloc[0],
        'debt_pct': money_alloc[1],
        'emergency_pct': money_alloc[2],
        'stock_pct': invest_alloc[0],
        'bond_pct': invest_alloc[1],
        're_pct': invest_alloc[2],
        'description': ACTION_DESCRIPTIONS[action_idx]
    }


class BehavioralAnalyzer:
    def __init__(self, output_dir: str = "behavioral_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
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
    
    def run_agent_analysis(self, agent_name: str, n_episodes: int = 5):
        """Run episodes and collect behavioral data"""
        print(f"\nAnalyzing {agent_name}...")
        
        agent = self._get_agent(agent_name)
        if agent is None:
            print(f"Skipping {agent_name} - agent not found")
            return None
        
        agent_behavior = {
            'actions_by_regime': {0: [], 1: [], 2: []},  # Normal, Bull, Bear
            'action_counts': {},  # Count of each action taken
            'decoded_actions_by_regime': {0: [], 1: [], 2: []},
            'allocations_by_age': {},
            'market_timing_score': 0,
            'debt_priority_score': 0,
            'emergency_fund_priority': 0
        }
        
        for episode in range(n_episodes):
            env = FinanceEnv(seed=10000 + episode)
            state, _ = env.reset(seed=10000 + episode)
            
            done = False
            step = 0
            
            while not done and step < 360:
                action_idx = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action_idx)
                
                # Decode action
                decoded = decode_action(action_idx)
                
                # Parse state
                regime = int(state[10])  # Market regime
                age = int(state[7])
                
                # Record raw action by regime
                agent_behavior['actions_by_regime'][regime].append(action_idx)
                
                # Record decoded action by regime
                agent_behavior['decoded_actions_by_regime'][regime].append(decoded)
                
                # Count action usage
                if action_idx not in agent_behavior['action_counts']:
                    agent_behavior['action_counts'][action_idx] = 0
                agent_behavior['action_counts'][action_idx] += 1
                
                # Record by age
                age_bucket = (age // 10) * 10
                if age_bucket not in agent_behavior['allocations_by_age']:
                    agent_behavior['allocations_by_age'][age_bucket] = []
                
                agent_behavior['allocations_by_age'][age_bucket].append({
                    'emergency': state[8],
                    'stocks': state[1],
                    'bonds': state[2],
                    'cc_debt': state[4],
                    'student_debt': state[5],
                    'action_idx': action_idx,
                    'decoded': decoded
                })
                
                state = next_state
                done = terminated or truncated
                step += 1
        
        # Calculate behavioral scores
        agent_behavior['market_timing_score'] = self._calculate_market_timing_score(
            agent_behavior['decoded_actions_by_regime']
        )
        agent_behavior['debt_priority_score'] = self._calculate_debt_priority_score(
            agent_behavior['decoded_actions_by_regime']
        )
        
        self.behavioral_data[agent_name] = agent_behavior
        return agent_behavior
    
    def _calculate_market_timing_score(self, decoded_actions_by_regime: Dict) -> float:
        """Score how well agent times the market (higher = reduces stocks in bear markets)"""
        if not decoded_actions_by_regime[2]:  # No bear market data
            return 0.0
        
        # Average stock % in each regime
        def avg_stock_pct(actions):
            if not actions:
                return 0.0
            return np.mean([a['stock_pct'] for a in actions])
        
        bull_stocks = avg_stock_pct(decoded_actions_by_regime[1])
        bear_stocks = avg_stock_pct(decoded_actions_by_regime[2])
        
        # Good market timing = reduce stocks in bear markets
        return max(0, bull_stocks - bear_stocks)
    
    def _calculate_debt_priority_score(self, decoded_actions_by_regime: Dict) -> float:
        """Score how well agent prioritizes debt repayment"""
        all_decoded = []
        for regime in [0, 1, 2]:
            all_decoded.extend(decoded_actions_by_regime[regime])
        
        if not all_decoded:
            return 0.0
        
        avg_debt_pct = np.mean([a['debt_pct'] for a in all_decoded])
        return avg_debt_pct  # Higher = more focus on debt
    
    def plot_action_frequency_heatmap(self):
        """Plot heatmap of action usage (10 money × 9 investment grid) using Plotly"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        agents_to_plot = list(self.behavioral_data.keys())
        
        if not agents_to_plot:
            print("No agents with behavioral data")
            return
        
        # Calculate appropriate vertical spacing
        n_agents = len(agents_to_plot)
        if n_agents > 1:
            vertical_spacing = min(0.05, 1.0 / (n_agents - 1) * 0.8)
        else:
            vertical_spacing = 0.1
        
        # Create subplots
        fig = make_subplots(
            rows=n_agents, cols=1,
            subplot_titles=[f'{agent.upper()} Action Usage' for agent in agents_to_plot],
            vertical_spacing=vertical_spacing
        )
        
        for idx, agent_name in enumerate(agents_to_plot):
            # Create 10×9 grid (money allocation × investment allocation)
            grid = np.zeros((len(MONEY_ALLOC), len(INVEST_ALLOC)))
            
            action_counts = self.behavioral_data[agent_name]['action_counts']
            total_actions = sum(action_counts.values())
            
            for action_idx, count in action_counts.items():
                money_idx = action_idx // len(INVEST_ALLOC)
                invest_idx = action_idx % len(INVEST_ALLOC)
                grid[money_idx, invest_idx] = (count / total_actions) * 100
            
            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=grid,
                    x=[desc for _, desc in INVEST_ALLOC],
                    y=[desc for _, desc in MONEY_ALLOC],
                    colorscale='YlOrRd',
                    text=np.round(grid, 1),
                    texttemplate='%{text:.1f}',
                    textfont={"size": 10},
                    colorbar=dict(title='Usage %', x=1.02),
                    hovertemplate='Money: %{y}<br>Investment: %{x}<br>Usage: %{z:.1f}%<extra></extra>'
                ),
                row=idx+1, col=1
            )
        
        fig.update_layout(
            height=400 * n_agents,
            width=1400,
            title_text="Action Usage Heatmaps (10 Money Strategies × 9 Investment Portfolios)",
            showlegend=False,
            template='plotly_white'
        )
        
        # Update all x and y axes
        for i in range(n_agents):
            fig.update_xaxes(title_text="Investment Allocation", row=i+1, col=1)
            fig.update_yaxes(title_text="Money Allocation", row=i+1, col=1)
        
        output_file = self.output_dir / "action_usage_heatmap.html"
        fig.write_html(str(output_file))
        print(f"Action usage heatmap saved: {output_file}")
    
    def plot_regime_preferences(self):
        """Plot how agents change strategies by market regime using Plotly"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        agents_to_plot = list(self.behavioral_data.keys())
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Normal Market', 'Bull Market', 'Bear Market']
        )
        
        regime_names = ['Normal', 'Bull', 'Bear']
        
        for regime_idx, regime in enumerate([0, 1, 2]):
            for agent_name in agents_to_plot:
                decoded_actions = self.behavioral_data[agent_name]['decoded_actions_by_regime'][regime]
                
                if not decoded_actions:
                    continue
                
                # Average allocations in this regime
                avg_stock = np.mean([a['stock_pct'] for a in decoded_actions])
                avg_bond = np.mean([a['bond_pct'] for a in decoded_actions])
                avg_re = np.mean([a['re_pct'] for a in decoded_actions])
                
                fig.add_trace(
                    go.Scatter(
                        x=['Stocks', 'Bonds', 'Real Estate'],
                        y=[avg_stock, avg_bond, avg_re],
                        mode='lines+markers',
                        name=agent_name,
                        showlegend=(regime_idx == 0),  # Only show legend once
                        legendgroup=agent_name,
                        line=dict(width=3),
                        marker=dict(size=10),
                        hovertemplate=f'<b>{agent_name}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>'
                    ),
                    row=1, col=regime_idx+1
                )
            
            fig.update_yaxes(title_text='Allocation %', range=[0, 100], row=1, col=regime_idx+1)
            fig.update_xaxes(row=1, col=regime_idx+1)
        
        fig.update_layout(
            height=600,
            width=1800,
            title_text="Investment Allocation by Market Regime",
            template='plotly_white',
            hovermode='closest'
        )
        
        output_file = self.output_dir / "regime_preferences.html"
        fig.write_html(str(output_file))
        print(f"Regime preferences plot saved: {output_file}")
    
    def plot_top_actions(self):
        """Plot most frequently used actions for each agent using Plotly"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        agents_in_data = list(self.behavioral_data.keys())
        n_agents = len(agents_in_data)
        
        # Calculate appropriate vertical spacing
        if n_agents > 1:
            vertical_spacing = min(0.05, 1.0 / (n_agents - 1) * 0.8)
        else:
            vertical_spacing = 0.1
        
        fig = make_subplots(
            rows=n_agents, cols=1,
            subplot_titles=[f'{agent.upper()} Top 10 Actions' for agent in agents_in_data],
            vertical_spacing=vertical_spacing
        )
        
        for idx, agent_name in enumerate(agents_in_data):
            action_counts = self.behavioral_data[agent_name]['action_counts']
            
            # Get top 10 actions
            top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            action_labels = [f"Action {a}: {ACTION_DESCRIPTIONS[a][:40]}" for a, _ in top_actions]
            counts = [c for _, c in top_actions]
            total = sum(action_counts.values())
            percentages = [(c/total)*100 for c in counts]
            
            fig.add_trace(
                go.Bar(
                    y=action_labels,
                    x=percentages,
                    orientation='h',
                    marker=dict(color='#2E86AB', line=dict(color='black', width=1)),
                    text=[f'{p:.1f}%' for p in percentages],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Usage: %{x:.1f}%<extra></extra>',
                    showlegend=False
                ),
                row=idx+1, col=1
            )
            
            fig.update_xaxes(title_text='Usage %', row=idx+1, col=1, gridcolor='lightgray')
            fig.update_yaxes(row=idx+1, col=1)
        
        fig.update_layout(
            height=350 * n_agents,
            width=1400,
            title_text="Top 10 Most Used Actions by Agent",
            template='plotly_white'
        )
        
        output_file = self.output_dir / "top_actions.html"
        fig.write_html(str(output_file))
        print(f"Top actions plot saved: {output_file}")
    
    def generate_behavioral_report(self):
        """Generate comprehensive behavioral analysis report"""
        report = []
        report.append("="*80)
        report.append("BEHAVIORAL ANALYSIS REPORT (Discrete Action Space)")
        report.append("="*80)
        report.append("")
        
        for agent_name, data in self.behavioral_data.items():
            report.append(f"\n{'='*60}")
            report.append(f"Agent: {agent_name.upper()}")
            report.append(f"{'='*60}")
            
            # Market timing
            report.append(f"\nMarket Timing Score: {data['market_timing_score']:.1f}%")
            report.append("  (Stock % reduction from bull to bear markets)")
            
            # Debt priority
            report.append(f"\nDebt Priority Score: {data['debt_priority_score']:.1f}%")
            report.append("  (Average % of income allocated to debt)")
            
            # Top 5 most used actions
            report.append("\nTop 5 Most Used Strategies:")
            action_counts = data['action_counts']
            total_actions = sum(action_counts.values())
            top_5 = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for rank, (action_idx, count) in enumerate(top_5, 1):
                pct = (count / total_actions) * 100
                report.append(f"  {rank}. Action {action_idx} ({pct:.1f}%): {ACTION_DESCRIPTIONS[action_idx]}")
            
            # Average allocations by regime
            report.append("\nAverage Allocations by Market Regime:")
            for regime, regime_name in [(0, 'NORMAL'), (1, 'BULL'), (2, 'BEAR')]:
                decoded = data['decoded_actions_by_regime'][regime]
                if not decoded:
                    continue
                
                avg_invest = np.mean([a['invest_pct'] for a in decoded])
                avg_debt = np.mean([a['debt_pct'] for a in decoded])
                avg_emergency = np.mean([a['emergency_pct'] for a in decoded])
                avg_stock = np.mean([a['stock_pct'] for a in decoded])
                avg_bond = np.mean([a['bond_pct'] for a in decoded])
                avg_re = np.mean([a['re_pct'] for a in decoded])
                
                report.append(f"\n  {regime_name} Market:")
                report.append(f"    Money: {avg_invest:.0f}% invest, {avg_debt:.0f}% debt, {avg_emergency:.0f}% emergency")
                report.append(f"    Invest: {avg_stock:.0f}% stocks, {avg_bond:.0f}% bonds, {avg_re:.0f}% RE")
        
        report.append("\n" + "="*80)
        
        # Save report
        with open(self.output_dir / "behavioral_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        print("\nBehavioral report saved!")
        return '\n'.join(report)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Behavioral analysis for ADJFin (discrete)')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='Episodes per agent for analysis')
    parser.add_argument('--agents', nargs='+', default=None,
                       help='Specific agents to analyze')
    
    args = parser.parse_args()
    
    analyzer = BehavioralAnalyzer()
    
    # DQN models with different sharpe ratios + human baselines
    default_agents = [
        'dqn_sharpe10', 'dqn_sharpe20', 'dqn_sharpe30', 'dqn_sharpe40',
        'dqn_sharpe50', 'dqn_sharpe60', 'dqn_sharpe70', 'dqn_sharpe80',
        '60_40', 'age_based', 'markowitz', 'equal_weight'
    ]
    agents_to_analyze = args.agents if args.agents else default_agents
    
    print("="*60)
    print("RUNNING BEHAVIORAL ANALYSIS (DQN Sharpe Ratios + Baselines)")
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
    
    analyzer.plot_action_frequency_heatmap()
    analyzer.plot_regime_preferences()
    analyzer.plot_top_actions()
    
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    report = analyzer.generate_behavioral_report()
    print(report)
    
    print("\n" + "="*60)
    print(f"Analysis complete! Results saved to: {analyzer.output_dir}")
    print("="*60)