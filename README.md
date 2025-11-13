# RL vs Human Financial Strategies
## CS 4100 Final Project - Reinforcement Learning for Personal Finance

### **Research Question**
Can reinforcement learning agents learn superior long-term financial strategies compared to traditional rule-based approaches used by human financial advisors?

### **Approach**
We created a comprehensive 30-year personal finance simulation environment where RL agents compete against sophisticated human financial strategies. The environment models realistic market volatility, life events (job loss, medical emergencies), and complex financial decisions including asset allocation, debt management, and emergency fund planning.

---

## **Repository Structure**

```
├── agents/                     # All financial agents
│   ├── __init__.py            # Agent imports and exports
│   ├── base_agent.py          # Unified interface for all agents
│   ├── baseline_strategies.py  # 5 human financial strategies
│   ├── ppo_agent.py          # PPO reinforcement learning agent
│   └── continuous_dqn_agent.py # DQN reinforcement learning agent
├── environment/               # Financial simulation environment
│   ├── finance_env.py        # 30-year lifecycle financial environment
│   └── market_data.py        # Historical market data manager
├── src/                      # Main execution scripts
│   ├── train_agents.py       # Training script for RL agents
│   └── evaluate_agents.py    # Unified evaluation script
├── models/                   # Trained RL model weights
│   ├── ppo_model.pth        # Trained PPO agent
│   └── dqn_model.pth        # Trained DQN agent
├── visualization/            # Generated plots and results
│   ├── training_curves.png  # RL learning progress
│   └── agent_comparison_results.png # Performance comparison
├── PROJECT_ROADMAP.md       # Detailed project timeline
└── README.md               # This file
```

---

## **Environment Design (Detailed)**

### **State Space (15 Dimensions)**
The environment provides a comprehensive 15-dimensional continuous state vector representing the complete financial situation:

**Financial Assets (4 dimensions):**
- `cash`: Liquid cash available for immediate use
- `stocks`: Value of stock market investments (S&P 500 equivalent)
- `bonds`: Value of bond investments (total bond market)
- `real_estate`: Value of real estate investments (REITs)

**Debts & Obligations (2 dimensions):**
- `credit_card_debt`: High-interest revolving debt (15-25% APR)
- `student_loan`: Education debt with moderate interest (4-8% APR)

**Income & Demographics (3 dimensions):**
- `monthly_income`: Current monthly salary (varies with job changes/raises)
- `age`: Current age (starts at 25, increases monthly)
- `emergency_fund`: Dedicated emergency savings (separate from cash)

**Market Conditions (3 dimensions):**
- `stock_return_1m`: Last month's stock market return
- `current_regime`: Market regime (0=bear, 1=neutral, 2=bull)
- `inflation[month]`: Current monthly inflation rate

**Economic Factors (2 dimensions):**
- `rates[month]`: Current interest rates affecting debt growth
- `recent_event`: Life event indicator (0=none, 1=job loss, 2=medical, 3=raise)

**Life Circumstances (1 dimension):**
- `months_unemployed`: Unemployment duration (affects income)

### **Action Space (6 Dimensions)**
All actions are continuous values in [0,1] representing allocation percentages:

**Investment Allocation (3 dimensions):**
- `stock_alloc`: Percentage of available income to invest in stocks
- `bond_alloc`: Percentage of available income to invest in bonds  
- `re_alloc`: Percentage of available income to invest in real estate

**Financial Management (3 dimensions):**
- `emergency_contrib`: Percentage of income to emergency fund
- `cc_payment`: Percentage of income toward credit card debt
- `student_payment`: Percentage of income toward student loans

**Constraints & Realism:**
- Total allocation percentages are normalized if they exceed 100%
- Minimum $2,000/month reserved for living expenses
- Debt payments have minimum requirements to avoid default
- Actions are applied to available income after expenses

### **Reward Function (Multi-Objective)**
The reward function balances growth with financial responsibility:

**Base Reward:**
- `reward = net_worth / 100,000` (scaled net worth)

**Penalties (Risk Management):**
- `-100` for bankruptcy (net worth < -$10,000)
- `-(debt - 15,000) / 1,000` for excessive credit card debt

**Bonuses (Financial Health):**
- `+1` for maintaining emergency fund ≥ $6,000
- `+2` for achieving debt-free status (no credit card debt)

### **Environmental Realism**

**Market Dynamics:**
- **Historical Data**: Real S&P 500, bond, and REIT returns (2000-2024)
- **Regime Switching**: Bull/bear/neutral market cycles with transition probabilities
- **Volatility**: Monthly return sampling from historical distributions
- **Correlation**: Asset classes move with realistic correlations

**Economic Cycles:**
- **Inflation**: Dynamic cost-of-living adjustments affecting expenses
- **Interest Rates**: Variable rates affecting debt growth and investment returns
- **GDP Growth**: Economic expansion/contraction cycles

**Life Events (Stochastic):**
- **Job Loss**: 2-6 month unemployment periods with income loss
- **Medical Emergencies**: $1,000-$10,000 unexpected expenses
- **Salary Changes**: Raises, promotions, or job changes affecting income
- **Career Progression**: Income growth over 30-year career

**Time Horizon:**
- **Duration**: 360 months (30 years) from age 25 to 55
- **Career Phase**: Early career through pre-retirement
- **Compound Growth**: Long-term investment returns with realistic volatility

---

## **Financial Agents**

### **Human Strategies (Baselines)**
- **`agents/baseline_strategies.py`**: 5 sophisticated rule-based strategies
  - **60/40 Rule**: Classic 60% stocks, 40% bonds allocation
  - **Debt Avalanche**: Prioritizes highest interest debt first
  - **Equal Weight**: Balanced allocation across all categories
  - **Age-Based**: Dynamic allocation based on age (100 - age)% stocks
  - **Markowitz**: Modern Portfolio Theory optimization

### **RL Agents**
- **`agents/ppo_agent.py`**: Proximal Policy Optimization
  - Uses Beta distribution for bounded continuous actions
  - Policy gradient method with clipped objective
  - Learns adaptive allocation strategies

- **`agents/continuous_dqn_agent.py`**: Continuous Deep Q-Network
  - Actor-critic architecture for continuous control
  - Experience replay and target networks
  - Deterministic policy with exploration noise

### **Training & Evaluation**
- **`src/train_agents.py`**: RL agent training with learning curve visualization
  - Trains PPO and/or DQN agents for specified episodes
  - Generates training progress plots (raw rewards + running average)
  - Saves trained models for evaluation

- **`src/evaluate_agents.py`**: Comprehensive agent comparison
  - Evaluates all agents (baselines + trained RL) on same environment
  - Tracks net worth growth trajectories over 30 years
  - Calculates financial performance and risk metrics
  - Generates 4-panel comparison visualization

---

## **Quick Start**

### **1. Setup Environment**
```bash
# Install dependencies
pip install torch numpy matplotlib scipy yfinance gymnasium tqdm

# Or with uv
uv sync
```

### **2. Train RL Agents**
```bash
# Train both PPO and DQN for 1000 episodes
python src/train_agents.py both 1000

# Train specific agent
python src/train_agents.py ppo 500
```

### **3. Evaluate All Agents**
```bash
# Compare all agents (baselines + trained RL)
python src/evaluate_agents.py
```

### **4. View Results**
- Training curves: `visualization/training_curves.png`
- Performance comparison: `visualization/agent_comparison_results.png`

---

## **Research Methodology**

### **Agent Comparison**
- **Unified Interface**: All agents use same `get_action(state) -> action` interface for fair comparison
- **Continuous Actions**: All agents make continuous allocation decisions (no discrete strategy selection)
- **Realistic Constraints**: Action space bounded to [0,1] representing allocation percentages

### **Evaluation Metrics**
- **Primary**: Average final net worth after 30 years
- **Risk**: Bankruptcy rate, debt-free rate, volatility
- **Consistency**: Performance across multiple random seeds and market conditions

---

## **Next Steps (Research Extensions)**

### **Statistical Analysis**
- [ ] Multiple seed evaluation (5+ runs per agent)
- [ ] Statistical significance testing (t-tests, Cohen's d)
- [ ] Confidence intervals and error bars

### **Behavioral Analysis**
- [ ] Action pattern analysis by market regime
- [ ] Market timing capabilities comparison
- [ ] Strategy adaptation over lifecycle

### **Advanced Experiments**
- [ ] Ablation study (stable vs volatile markets)
- [ ] Additional RL algorithms (SAC, TD3)
- [ ] Transaction costs and realistic constraints
- [ ] Multi-objective optimization (risk vs return)

---

## **Technical Details**

### **RL Implementation**
- **PPO**: Policy gradient with Beta distribution for bounded actions
- **DQN**: Actor-critic with deterministic policy and experience replay
- **Training**: 1000 episodes with progress tracking and model persistence
- **Evaluation**: Deterministic policies for consistent performance measurement

### **Baseline Sophistication**
- **Modern Portfolio Theory**: Markowitz mean-variance optimization
- **Age-Based Allocation**: Dynamic risk adjustment over lifecycle  
- **Debt Optimization**: Mathematically optimal debt payoff strategies
- **Professional Standards**: Strategies used by actual financial advisors

### **Environment Realism**
- **Market Data**: Historical S&P 500, bond, and REIT returns
- **Economic Cycles**: Bull/bear market regime switching
- **Life Events**: Job loss, medical emergencies, salary changes
- **Inflation**: Dynamic cost of living adjustments

---

## **Contributors**
- **Jesvin Jerry**
- **Dylan Vo**
- **Aarush Garg**

## **Acknowledgments**
This project builds upon modern reinforcement learning techniques applied to the important real-world problem of personal financial planning. We thank the CS 4100 teaching team for guidance on this research direction.
