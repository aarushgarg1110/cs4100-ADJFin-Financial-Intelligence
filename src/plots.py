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
    matrix = np.zeros((90, len(ages)))
    
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
        y=list(range(90)),
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
    Grouped bar chart showing portfolio at key milestones.
    Each agent has a unique color, with stacked segments using opacity to show composition.
    
    Args:
        all_results: list of result dicts with 'portfolio_snapshots'
    """
    # Milestones: Age 25, 35, 45, 50, 55 (Years 0, 10, 20, 25, 30)
    milestones = ['Age 25\n(Year 0)', 'Age 35\n(Year 10)', 'Age 45\n(Year 20)', 'Age 50\n(Year 25)', 'Age 55\n(Year 30)']
    snapshot_months = [0, 120, 240, 300, 359]
    
    # Agent colors (distinct colors for each agent)
    agent_colors = {
        '60/40': '#1f77b4',           # Blue
        'Age-Based': '#ff7f0e',       # Orange
        'Markowitz': '#2ca02c',       # Green
        'Equal Weight': '#d62728',    # Red
        'Debt Avalanche': '#9467bd',  # Purple
        'DQN': '#8c564b',             # Brown
        'PPO': '#e377c2',             # Pink
    }
    
    # Asset categories with opacity levels
    categories = [
        ('stocks', 1.0, 'Stocks'),
        ('bonds', 0.7, 'Bonds'),
        ('real_estate', 0.5, 'Real Estate'),
        ('emergency_fund', 0.3, 'Emergency Fund')
    ]
    
    fig = go.Figure()
    
    # For each agent
    for result in all_results:
        agent_name = result['agent_name']
        snapshots = result['portfolio_snapshots']
        base_color = agent_colors.get(agent_name, '#17becf')  # Default cyan if not found
        
        # For each asset category (stacked)
        for asset_key, opacity, asset_label in categories:
            values = []
            hover_texts = []
            
            for month in snapshot_months:
                snap = snapshots[month]
                value = snap[asset_key]
                values.append(value)
                
                # Calculate percentage
                total = snap['stocks'] + snap['bonds'] + snap['real_estate'] + snap['emergency_fund']
                pct = (value / total * 100) if total > 0 else 0
                
                hover_texts.append(
                    f"<b>{agent_name}</b><br>" +
                    f"{asset_label}: ${value:,.0f} ({pct:.1f}%)<br>" +
                    f"<extra></extra>"
                )
            
            # Convert hex color to rgba with opacity
            rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
            
            # Only show legend for first category of each agent
            show_legend = (asset_key == 'stocks')
            legend_name = agent_name if show_legend else None
            
            fig.add_trace(go.Bar(
                name=legend_name,
                x=milestones,
                y=values,
                marker_color=rgba_color,
                legendgroup=agent_name,
                showlegend=show_legend,
                hovertemplate='%{hovertext}',
                hovertext=hover_texts,
                offsetgroup=agent_name  # Group bars by agent
            ))
    
    fig.update_layout(
        title='Portfolio Value at Key Life Milestones',
        xaxis_title='Milestone',
        yaxis_title='Portfolio Value ($)',
        barmode='stack',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            title="Agent",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
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



def plot_q_policy_heatmap(q_policy_results, output_dir='visualization'):
    """
    Heatmap showing which action each model chooses in each scenario.
    
    Args:
        q_policy_results: List of dicts with keys:
            - 'model_name': str
            - 'scenarios': List of 9 dicts with keys:
                - 'name': str (e.g., "Young, Bull Market, High Debt")
                - 'best_action': int
                - 'q_value': float
                - 'action_desc': str
    """
    from environment.finance_env import MONEY_ALLOC, INVEST_ALLOC
    
    # Define action categories based on money allocation
    def get_action_category(action_num):
        money_idx = action_num // 9
        if money_idx in [0, 1]:  # Very Aggressive, Aggressive
            return 'Aggressive'
        elif money_idx in [2, 3, 4]:  # Balanced, Moderate
            return 'Balanced'
        elif money_idx in [5, 6]:  # Debt-Heavy, Max Debt
            return 'Debt-Focused'
        elif money_idx in [7, 8]:  # Max Safety, Conservative
            return 'Conservative'
        else:  # Income Protection
            return 'Safety-Focused'
    
    # Category colors
    category_colors = {
        'Aggressive': '#d62728',      # Red
        'Balanced': '#ff7f0e',         # Orange
        'Debt-Focused': '#2ca02c',    # Green
        'Conservative': '#1f77b4',    # Blue
        'Safety-Focused': '#9467bd'   # Purple
    }
    
    # Build data matrix
    model_names = [r['model_name'] for r in q_policy_results]
    scenario_names = [s['name'] for s in q_policy_results[0]['scenarios']]
    
    # Create matrix of actions
    action_matrix = []
    hover_text = []
    colors = []
    
    for result in q_policy_results:
        row_actions = []
        row_hover = []
        row_colors = []
        
        for scenario in result['scenarios']:
            action = scenario['best_action']
            q_val = scenario['q_value']
            desc = scenario['action_desc']
            category = get_action_category(action)
            
            row_actions.append(action)
            row_hover.append(
                f"<b>{result['model_name']}</b><br>"
                f"{scenario['name']}<br>"
                f"Action {action}: {desc}<br>"
                f"Q-value: {q_val:.0f}<br>"
                f"Category: {category}"
            )
            row_colors.append(category_colors[category])
        
        action_matrix.append(row_actions)
        hover_text.append(row_hover)
        colors.append(row_colors)
    
    # Create heatmap
    fig = go.Figure()
    
    # Add colored rectangles for each cell
    for i, model in enumerate(model_names):
        for j, scenario in enumerate(scenario_names):
            fig.add_trace(go.Scatter(
                x=[j],
                y=[i],
                mode='markers+text',
                marker=dict(
                    size=80,
                    color=colors[i][j],
                    symbol='square',
                    line=dict(color='white', width=2)
                ),
                text=str(action_matrix[i][j]),
                textfont=dict(size=14, color='white', family='Arial Black'),
                hovertext=hover_text[i][j],
                hoverinfo='text',
                showlegend=False
            ))
    
    # Add legend for categories
    for category, color in category_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol='square'),
            name=category,
            showlegend=True
        ))
    
    fig.update_layout(
        title='Q-Policy Action Selection Across Scenarios',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(scenario_names))),
            ticktext=[s.replace(', ', '<br>') for s in scenario_names],
            tickangle=0,
            side='bottom'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(model_names))),
            ticktext=model_names,
            autorange='reversed'
        ),
        template='plotly_white',
        height=400 + len(model_names) * 80,
        width=1400,
        hovermode='closest',
        legend=dict(
            title='Action Category',
            orientation='v',
            x=1.02,
            y=1
        )
    )
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f'{output_dir}/q_policy_heatmap.html')
    print(f"✓ Saved: {output_dir}/q_policy_heatmap.html")
    return fig



def plot_strategy_profile(q_policy_results, output_dir='visualization'):
    """
    Stacked bar chart showing % of scenarios using each strategy type per model.
    
    Args:
        q_policy_results: List of dicts with keys:
            - 'model_name': str
            - 'scenarios': List of 9 dicts with 'best_action': int
    """
    # Define action categories
    def get_action_category(action_num):
        money_idx = action_num // 9
        if money_idx in [0, 1]:  # Very Aggressive, Aggressive
            return 'Aggressive'
        elif money_idx in [2, 3, 4]:  # Balanced, Moderate
            return 'Balanced'
        elif money_idx in [5, 6]:  # Debt-Heavy, Max Debt
            return 'Debt-Focused'
        elif money_idx in [7, 8]:  # Max Safety, Conservative
            return 'Conservative'
        else:  # Income Protection
            return 'Safety-Focused'
    
    # Count categories for each model
    model_names = []
    category_counts = {
        'Aggressive': [],
        'Balanced': [],
        'Debt-Focused': [],
        'Conservative': [],
        'Safety-Focused': []
    }
    
    for result in q_policy_results:
        model_names.append(result['model_name'])
        
        # Count categories
        counts = {'Aggressive': 0, 'Balanced': 0, 'Debt-Focused': 0, 
                 'Conservative': 0, 'Safety-Focused': 0}
        
        for scenario in result['scenarios']:
            category = get_action_category(scenario['best_action'])
            counts[category] += 1
        
        # Convert to percentages
        total = len(result['scenarios'])
        for cat in category_counts.keys():
            category_counts[cat].append(counts[cat] / total * 100)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    colors = {
        'Aggressive': '#d62728',
        'Balanced': '#ff7f0e',
        'Debt-Focused': '#2ca02c',
        'Conservative': '#1f77b4',
        'Safety-Focused': '#9467bd'
    }
    
    for category, color in colors.items():
        fig.add_trace(go.Bar(
            name=category,
            x=model_names,
            y=category_counts[category],
            marker_color=color,
            hovertemplate='<b>%{x}</b><br>' +
                         f'{category}: %{{y:.1f}}%<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Strategy Profile: Action Category Distribution by Model',
        xaxis_title='Model',
        yaxis_title='% of Scenarios',
        barmode='stack',
        template='plotly_white',
        height=500,
        width=800,
        legend=dict(
            title='Action Category',
            orientation='v',
            x=1.02,
            y=1
        ),
        yaxis=dict(range=[0, 100])
    )
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f'{output_dir}/strategy_profile.html')
    print(f"✓ Saved: {output_dir}/strategy_profile.html")
    return fig
