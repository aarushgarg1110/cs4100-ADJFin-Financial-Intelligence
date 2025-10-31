# Agent Architecture Documentation

## Overview
Our agent system uses an abstract base class to ensure all agents (RL and rule-based) have consistent interfaces.

## Class Hierarchy
```
BaseFinancialAgent (abstract)
├── DQNFinancialAgent (RL learner)
└── Baseline Agents (rule-based)
    ├── RandomAgent
    ├── ConservativeAgent
    ├── AggressiveAgent
    └── AdaptiveRuleAgent
```

## Key Design Decisions

### 1. Discrete Action Space (6 actions)
- Action 0: Ultra Conservative
- Action 1: Conservative
- Action 2: Moderate
- Action 3: Aggressive Growth
- Action 4: Debt Crusher
- Action 5: Emergency Builder

### 2. State Representation (10 features, normalized 0-1)
- Liquid assets: cash, stocks, bonds, real estate
- Liabilities: credit card debt, student loans
- Income and demographics: monthly income, age
- Market context: current trend, recent life events

### 3. Interface Methods
All agents must implement:
- `select_action(observation) -> int`
- `learn_from_experience(...)` (no-op for baselines)
- `save(path)` and `load(path)`

## Usage Example
```python
# DQN Agent
agent = DQNFinancialAgent(learning_rate=1e-4)
agent.initialize(env)
agent.train(total_timesteps=100000)

obs, _ = env.reset()
action = agent.select_action(obs)

# Baseline Agent
baseline = AdaptiveRuleAgent()
action = baseline.select_action(obs)
```

## Action Space Conversion (Week 2 Note)

The current `FinanceEnv` uses continuous `Box(6)` actions, but our agents expect discrete actions (0-5). In Week 2, we'll need to either:

1. **Option A**: Create an action wrapper that maps discrete actions to continuous strategy vectors
2. **Option B**: Modify `FinanceEnv` to accept discrete actions directly

Each discrete action (0-5) should map to a predefined allocation strategy vector:
- Action 0 → [0.1, 0.3, 0.0, 0.4, 0.1, 0.1] (Ultra Conservative)
- Action 1 → [0.2, 0.4, 0.1, 0.2, 0.05, 0.05] (Conservative)
- etc.

## Next Steps
- Week 2: Integrate with real FinanceEnv from Jesvin (handle action space conversion)
- Week 3: Train on full 30-year simulations
- Week 4: Comparative evaluation

