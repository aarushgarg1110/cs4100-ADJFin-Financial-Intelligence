from environment.finance_env import FinanceEnv
from agents.baseline_human_strats import get_all_baseline_agents
import numpy as np

def test_baseline_agents_in_environment():
    print("Testing Baseline Agents in Finance Environment...")
    
    # Create environment
    env = FinanceEnv()
    agents = get_all_baseline_agents()
    
    results = {}
    
    for agent in agents:
        print(f"\nTesting {agent.name}...")
        
        # Run short simulation
        state, _ = env.reset(seed=42)
        total_reward = 0
        
        for step in range(12):  # 1 year simulation
            action = agent.get_action(state)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if step == 0:  # Show first action
                print(f"  First action: Stock={action[0]:.1%}, Bond={action[1]:.1%}, RE={action[2]:.1%}")
        
        # Calculate final net worth
        s = state
        net_worth = s[0] + s[1] + s[2] + s[3] + s[8] - s[4] - s[5]  # assets - debts
        
        results[agent.name] = {
            'total_reward': total_reward,
            'final_net_worth': net_worth,
            'final_cc_debt': s[4],
            'final_student_debt': s[5]
        }
        
        print(f"  Final net worth: ${net_worth:.0f}")
        print(f"  Total reward: {total_reward:.2f}")
    
    # Compare results
    print(f"\n=== 1-Year Performance Comparison ===")
    sorted_agents = sorted(results.items(), key=lambda x: x[1]['final_net_worth'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_agents):
        print(f"{i+1}. {name}: ${result['final_net_worth']:.0f} net worth")
    
    print("\nSUCCESS: All baseline agents completed simulation!")
    return results

if __name__ == "__main__":
    results = test_baseline_agents_in_environment()
