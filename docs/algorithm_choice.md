# RL Algorithm Selection for Personal Finance Agent

## Decision: Deep Q-Network (DQN)

### Why DQN?
- **Course appropriate**: Value-based RL covered in CS 4200
- **Discrete action space**: Natural fit for preset financial strategies
- **Proven in finance**: Used in BBVA trading, portfolio optimization
- **Stable implementation**: Mature stable-baselines3 support

### Action Space Design
Using 6 preset financial strategies (discrete actions 0-5):
- Ultra Conservative, Conservative, Moderate, Aggressive Growth, 
  Debt Crusher, Emergency Builder

### State Space (10 features)
- Normalized cash, stocks, bonds, real estate values
- Debt levels (credit card, student loans)
- Income, age, market conditions, recent events

### Training Plan
- 100K-150K timesteps expected
- Epsilon decay from 1.0 → 0.05 over 30% of training
- Experience replay buffer: 100K transitions
- Target network updates every 1000 steps

### Expected Performance
- Training time: 2-4 hours on laptop
- Convergence: 50K-100K steps
- Final performance: 20-40% better than random baseline

