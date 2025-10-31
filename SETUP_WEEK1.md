# Week 1 Setup Instructions

## Installation

Install dependencies using one of these methods:

### Option 1: Using uv (recommended if available)
```bash
uv sync
```

### Option 2: Using pip
```bash
pip install stable-baselines3[extra] gymnasium numpy torch
```

## Verify Installation

Run the setup test:
```bash
python tests/test_dqn_setup.py
```

You should see:
```
✓ ALL TESTS PASSED - Ready to build!
```

## Run Integration Tests

After installation, test the agent integration:
```bash
python tests/test_agent_integration.py
```

## Week 1 Deliverables Checklist

- [x] `docs/algorithm_choice.md` - DQN justification
- [x] `tests/test_dqn_setup.py` - Installation verification
- [x] `agents/base_agent.py` - Abstract base class
- [x] `agents/dqn_agent.py` - DQN wrapper implementation
- [x] `agents/baseline_agents.py` - 4 baseline agents
- [x] `tests/test_agent_integration.py` - Integration tests
- [x] `docs/agent_architecture.md` - Architecture documentation

## Next Steps

1. Install dependencies (see above)
2. Run verification tests
3. Coordinate with Jesvin on exact observation space dimensions
4. Coordinate with Aarush on baseline strategy priorities

