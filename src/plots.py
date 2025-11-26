"""
Interactive Plotly visualizations for agent evaluation
Each function creates a standalone interactive HTML plot
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os


def plot_net_worth_trajectories(all_results, output_dir='visualization'):
    """
    Plot net worth growth over 30 years for all agents.
    Interactive line plot with hover details.
    """
    fig = go.Figure()
    
    for result in all_results:
        trajectory = result['avg_trajectory']
        std_trajectory = result.get('std_trajectory', np.zeros_like(trajectory))
        
        months = np.arange(len(trajectory))
        years = months / 12
        
        # Add main trajectory line
        fig.add_trace(go.Scatter(
            x=years,
            y=trajectory,
            mode='lines',
            name=result['agent_name'],
            line=dict(width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Year: %{x:.1f}<br>' +
                         'Net Worth: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add confidence band (±1 std)
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([trajectory + std_trajectory, 
                             (trajectory - std_trajectory)[::-1]]),
            fill='toself',
            fillcolor=fig.data[-1].line.color,
            opacity=0.2,
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add $2M target line
    fig.add_hline(y=2_000_000, line_dash="dash", line_color="red",
                  annotation_text="$2M Target", annotation_position="right")
    
    fig.update_layout(
        title='Net Worth Growth Over 30 Years',
        xaxis_title='Years',
        yaxis_title='Net Worth ($)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f'{output_dir}/net_worth_trajectories.html')
    fig.show()
    print(f"✓ Saved: {output_dir}/net_worth_trajectories.html")
    return fig


def plot_final_net_worth_comparison(all_results, output_dir='visualization'):
    """
    Bar chart comparing final net worth across agents.
    Shows mean with error bars (std).
    """
    agent_names = [r['agent_name'] for r in all_results]
    avg_net_worths = [r['avg_net_worth'] for r in all_results]
    std_net_worths = [np.std(r['final_net_worths']) for r in all_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=agent_names,
        y=avg_net_worths,
        error_y=dict(type='data', array=std_net_worths),
        text=[f'${v:,.0f}' for v in avg_net_worths],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                     'Avg Net Worth: $%{y:,.0f}<br>' +
                     'Std Dev: $%{error_y.array:,.0f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Final Net Worth Comparison (30 Years)',
        xaxis_title='Agent',
        yaxis_title='Average Net Worth ($)',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f'{output_dir}/final_net_worth_comparison.html')
    fig.show()
    print(f"✓ Saved: {output_dir}/final_net_worth_comparison.html")
    return fig


def plot_action_heatmap(action_history, agent_name, output_dir='visualization'):
    """
    Heatmap showing which actions (0-59) the agent picks at each age (25-55).
    Only for discrete agents.
    
    Args:
        action_history: dict {age: [action_indices]}
        agent_name: str
    """
    ages = sorted(action_history.keys())
    
    # Create matrix: rows = actions (0-59), cols = ages
    matrix = np.zeros((60, len(ages)))
    
    for age_idx, age in enumerate(ages):
        actions = action_history[age]
        if len(actions) > 0:
            # Count frequency of each action at this age
            unique, counts = np.unique(actions, return_counts=True)
            for action, count in zip(unique, counts):
                matrix[action, age_idx] = count / len(actions) * 100  # Percentage
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=ages,
        y=list(range(60)),
        colorscale='YlOrRd',
        hovertemplate='Age: %{x}<br>' +
                     'Action: %{y}<br>' +
                     'Frequency: %{z:.1f}%<br>' +
                     '<extra></extra>',
        colorbar=dict(title='Frequency (%)')
    ))
    
    fig.update_layout(
        title=f'Action Selection Heatmap - {agent_name}',
        xaxis_title='Age',
        yaxis_title='Action Index (0-59)',
        template='plotly_white',
        height=800,
        width=1200
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/action_heatmap_{agent_name.replace(" ", "_").replace("/", "_")}.html'
    fig.write_html(filename)
    fig.show()
    print(f"✓ Saved: {filename}")
    return fig


def plot_allocation_evolution(all_results, output_dir='visualization'):
    """
    Combined line plot showing money allocation (% to investing) over time for all agents.
    
    Args:
        all_results: list of result dicts with 'agent_name' and 'allocation_history'
    """
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, result in enumerate(all_results):
        agent_name = result['agent_name']
        allocation_history = result['allocation_history']
        
        ages = sorted(allocation_history.keys())
        invest_pcts = [np.mean(allocation_history[age]['invest']) if allocation_history[age]['invest'] else 0 for age in ages]
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=ages, y=invest_pcts,
            mode='lines',
            name=agent_name,
            line=dict(width=2, color=color),
            hovertemplate=f'{agent_name}<br>Age: %{{x}}<br>Invest: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Investment Allocation % Over Time (All Agents)',
        xaxis_title='Age',
        yaxis_title='% of Income to Investing',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/allocation_evolution_combined.html'
    fig.write_html(filename)
    fig.show()
    print(f"✓ Saved: {filename}")
    return fig


def plot_investment_allocation_evolution(all_results, output_dir='visualization'):
    """
    Combined line plot showing investment allocation (% stocks) over time for all agents.
    
    Args:
        all_results: list of result dicts with 'agent_name' and 'investment_history'
    """
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, result in enumerate(all_results):
        agent_name = result['agent_name']
        investment_history = result['investment_history']
        
        ages = sorted(investment_history.keys())
        stock_pcts = [np.mean(investment_history[age]['stocks']) if investment_history[age]['stocks'] else 0 for age in ages]
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=ages, y=stock_pcts,
            mode='lines',
            name=agent_name,
            line=dict(width=2, color=color),
            hovertemplate=f'{agent_name}<br>Age: %{{x}}<br>Stocks: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Stock Allocation % Over Time (All Agents)',
        xaxis_title='Age',
        yaxis_title='% of Investments in Stocks',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/investment_allocation_combined.html'
    fig.write_html(filename)
    fig.show()
    print(f"✓ Saved: {filename}")
    return fig



def plot_debt_timeline(all_results, output_dir='visualization'):
    """
    Combined line plot showing debt elimination over time for all agents.
    
    Args:
        all_results: list of result dicts with 'agent_name' and 'debt_history'
    """
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, result in enumerate(all_results):
        agent_name = result['agent_name']
        debt_history = result['debt_history']
        
        ages = sorted(debt_history.keys())
        cc_debt = [np.mean(debt_history[age]['cc_debt']) if debt_history[age]['cc_debt'] else 0 for age in ages]
        student_debt = [np.mean(debt_history[age]['student_loan']) if debt_history[age]['student_loan'] else 0 for age in ages]
        total_debt = [cc + sl for cc, sl in zip(cc_debt, student_debt)]
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=ages, y=total_debt,
            mode='lines',
            name=agent_name,
            line=dict(width=2, color=color),
            hovertemplate=f'{agent_name}<br>Age: %{{x}}<br>Total Debt: $%{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Debt Elimination Timeline (All Agents)',
        xaxis_title='Age',
        yaxis_title='Total Debt ($)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/debt_timeline_combined.html'
    fig.write_html(filename)
    fig.show()
    print(f"✓ Saved: {filename}")
    return fig



def plot_portfolio_snapshots(all_results, output_dir='visualization'):
    """
    Grouped bar chart showing portfolio composition at key milestones (Years 0, 10, 20, 30).
    
    Args:
        all_results: list of result dicts with 'portfolio_snapshots'
    """
    milestones = ['Year 0', 'Year 10', 'Year 20', 'Year 30']
    snapshot_months = [0, 120, 240, 359]
    
    fig = go.Figure()
    
    # For each agent
    for result in all_results:
        agent_name = result['agent_name']
        snapshots = result['portfolio_snapshots']
        
        # Calculate net worth at each milestone
        net_worths = []
        for month in snapshot_months:
            snap = snapshots[month]
            nw = (snap['stocks'] + snap['bonds'] + snap['real_estate'] + 
                  snap['emergency_fund'] - snap['cc_debt'] - snap['student_loan'])
            net_worths.append(nw)
        
        fig.add_trace(go.Bar(
            name=agent_name,
            x=milestones,
            y=net_worths,
            text=[f'${v:,.0f}' for v in net_worths],
            textposition='outside',
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         '%{x}<br>' +
                         'Net Worth: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Net Worth at Key Milestones',
        xaxis_title='Milestone',
        yaxis_title='Net Worth ($)',
        barmode='group',
        template='plotly_white',
        height=600
    )
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f'{output_dir}/portfolio_snapshots.html')
    fig.show()
    print(f"✓ Saved: {output_dir}/portfolio_snapshots.html")
    return fig


def plot_metrics_comparison(all_results, output_dir='visualization'):
    """
    Radar chart comparing multiple metrics across agents.
    Metrics: Final Net Worth, Debt-Free Rate, Emergency Fund, Consistency
    """
    agent_names = [r['agent_name'] for r in all_results]
    
    # Normalize metrics to 0-100 scale
    metrics = {
        'Net Worth': [r['avg_net_worth'] / 20_000 for r in all_results],  # $2M = 100
        'Debt-Free Rate': [r['debt_free_rate'] * 100 for r in all_results],
        'Consistency': [100 - (np.std(r['final_net_worths']) / r['avg_net_worth'] * 100) 
                       for r in all_results],
    }
    
    fig = go.Figure()
    
    for i, agent_name in enumerate(agent_names):
        values = [metrics[m][i] for m in metrics.keys()]
        values.append(values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(metrics.keys()) + [list(metrics.keys())[0]],
            fill='toself',
            name=agent_name
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title='Multi-Metric Agent Comparison',
        template='plotly_white',
        height=600
    )
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f'{output_dir}/metrics_comparison.html')
    fig.show()
    print(f"✓ Saved: {output_dir}/metrics_comparison.html")
    return fig
