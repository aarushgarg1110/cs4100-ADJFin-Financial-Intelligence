"""
Main training script for Q-Learning and DQN agents
From-scratch implementations (no stable-baselines3)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.finance_env import FinanceEnv
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.discrete_action_wrapper import wrap_finance_env


class FinanceTrainer:
    """Manages training and evaluation of financial agents"""
    
    def __init__(self, save_dir="models", log_dir="logs"):
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create wrapped environment
        print("Initializing environment...")
        base_env = FinanceEnv()
        self.env = wrap_finance_env(base_env)
        print(f"Environment ready!")
        print(f"  Action space: {self.env.action_space}")
        print(f"  Observation space: {self.env.observation_space}")
    
    def evaluate_agent(self, agent, n_episodes: int = 20, render: bool = False):
        """Run evaluation episodes and collect statistics"""
        
        print(f"\n{'='*60}")
        print(f"EVALUATING: {agent.name}")
        print(f"{'='*60}")
        
        episode_rewards = []
        episode_net_worths = []
        episode_lengths = []
        bankruptcies = 0
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                if render and episode == 0:
                    self._render_state(obs, action, reward, steps)
            
            # Calculate final net worth
            final_net_worth = self._calculate_net_worth(obs)
            
            episode_rewards.append(episode_reward)
            episode_net_worths.append(final_net_worth)
            episode_lengths.append(steps)
            
            if final_net_worth < -10000:
                bankruptcies += 1
            
            if episode < 3 or episode % 5 == 0:  # Print some episodes
                print(f"Episode {episode+1}/{n_episodes}: "
                      f"Reward={episode_reward:.1f}, "
                      f"Net Worth=${final_net_worth:,.0f}, "
                      f"Steps={steps}")
        
        # Print summary statistics
        results = {
            'agent_name': agent.name,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_net_worth': float(np.mean(episode_net_worths)),
            'std_net_worth': float(np.std(episode_net_worths)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'bankruptcy_rate': bankruptcies / n_episodes,
            'episodes': n_episodes
        }
        
        print(f"\n{agent.name} RESULTS:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Net Worth: ${results['mean_net_worth']:,.0f} ± ${results['std_net_worth']:,.0f}")
        print(f"  Bankruptcy Rate: {results['bankruptcy_rate']:.1%}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f} steps")
        
        return results
    
    def compare_agents(self, agents_dict, n_episodes=20):
        """Compare multiple agents and visualize results"""
        
        print("\n" + "="*60)
        print("AGENT COMPARISON")
        print("="*60)
        
        all_results = {}
        
        for name, agent in agents_dict.items():
            results = self.evaluate_agent(agent, n_episodes=n_episodes)
            all_results[name] = results
        
        # Create comparison plot
        self._plot_comparison(all_results)
        
        return all_results
    
    def _calculate_net_worth(self, state):
        """Calculate net worth from state vector"""
        # state = [cash, stocks, bonds, real_estate, cc_debt, student_loan, ...]
        assets = state[0] + state[1] + state[2] + state[3] + state[8]  # includes emergency fund
        liabilities = state[4] + state[5]
        return assets - liabilities
    
    def _render_state(self, state, action, reward, step):
        """Print readable state information"""
        strategy_names = [
            "Ultra Conservative", "Conservative", "Moderate",
            "Aggressive Growth", "Debt Crusher", "Emergency Builder"
        ]
        
        print(f"\nStep {step}:")
        print(f"  Cash: ${state[0]:,.0f}")
        print(f"  Stocks: ${state[1]:,.0f}")
        print(f"  Bonds: ${state[2]:,.0f}")
        print(f"  Real Estate: ${state[3]:,.0f}")
        print(f"  CC Debt: ${state[4]:,.0f}")
        print(f"  Student Loans: ${state[5]:,.0f}")
        print(f"  Monthly Income: ${state[6]:,.0f}")
        print(f"  Age: {state[7]:.1f}")
        print(f"  Emergency Fund: ${state[8]:,.0f}")
        print(f"  Action: {action} ({strategy_names[action]})")
        print(f"  Reward: {reward:.2f}")
    
    def _plot_comparison(self, results):
        """Create comparison visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        agents = list(results.keys())
        
        # Plot 1: Mean Net Worth
        net_worths = [results[agent]['mean_net_worth'] for agent in agents]
        net_worth_stds = [results[agent]['std_net_worth'] for agent in agents]
        axes[0].bar(agents, net_worths, yerr=net_worth_stds, capsize=5, color=['#2E86AB', '#A23B72'])
        axes[0].set_ylabel('Net Worth ($)', fontsize=12)
        axes[0].set_title('Final Net Worth Comparison', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=15)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Mean Reward
        rewards = [results[agent]['mean_reward'] for agent in agents]
        reward_stds = [results[agent]['std_reward'] for agent in agents]
        axes[1].bar(agents, rewards, yerr=reward_stds, capsize=5, color=['#2E86AB', '#A23B72'])
        axes[1].set_ylabel('Total Reward', fontsize=12)
        axes[1].set_title('Mean Episode Reward', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Bankruptcy Rate
        bankruptcy_rates = [results[agent]['bankruptcy_rate'] * 100 for agent in agents]
        axes[2].bar(agents, bankruptcy_rates, color=['#2E86AB', '#A23B72'])
        axes[2].set_ylabel('Bankruptcy Rate (%)', fontsize=12)
        axes[2].set_title('Bankruptcy Rate', fontsize=14, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=15)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_training_curves(self, q_rewards, dqn_rewards):
        """Plot training curves for both agents"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Q-Learning curve
        if q_rewards:
            episodes = range(len(q_rewards))
            # Running average
            window = min(100, len(q_rewards) // 10)
            running_avg = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
            
            axes[0].plot(episodes, q_rewards, alpha=0.3, color='lightblue', label='Raw')
            axes[0].plot(range(window-1, len(episodes)), running_avg, 
                        color='blue', linewidth=2, label=f'Avg (window={window})')
            axes[0].set_xlabel('Episode', fontsize=12)
            axes[0].set_ylabel('Reward', fontsize=12)
            axes[0].set_title('Q-Learning Training Progress', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        
        # DQN curve
        if dqn_rewards:
            episodes = range(len(dqn_rewards))
            window = min(100, len(dqn_rewards) // 10)
            running_avg = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
            
            axes[1].plot(episodes, dqn_rewards, alpha=0.3, color='lightcoral', label='Raw')
            axes[1].plot(range(window-1, len(episodes)), running_avg,
                        color='red', linewidth=2, label=f'Avg (window={window})')
            axes[1].set_xlabel('Episode', fontsize=12)
            axes[1].set_ylabel('Reward', fontsize=12)
            axes[1].set_title('DQN Training Progress', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = self.log_dir / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to: {plot_path}")
        
        plt.show()


def main():
    """Main training and evaluation pipeline"""
    
    # Initialize trainer
    trainer = FinanceTrainer()
    
    # Train Q-Learning agent
    print("\n" + "="*60)
    print("PHASE 1: Q-LEARNING TRAINING")
    print("="*60)
    
    q_agent = QLearningAgent(
        n_actions=6,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,  # Slower decay for better exploration
        n_bins=5
    )
    
    q_rewards = q_agent.train(trainer.env, n_episodes=500, verbose=True)
    q_agent.save(str(trainer.save_dir / "q_learning_finance.pkl"))
    
    # Train DQN agent
    print("\n" + "="*60)
    print("PHASE 2: DQN TRAINING")
    print("="*60)
    
    dqn_agent = DQNAgent(
        state_dim=15,
        action_dim=6,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=500,
        hidden_dim=128
    )
    
    dqn_rewards = dqn_agent.train(trainer.env, n_episodes=500, verbose=True)
    dqn_agent.save(str(trainer.save_dir / "dqn_finance.pth"))
    
    # Plot training curves
    print("\n" + "="*60)
    print("PLOTTING TRAINING CURVES")
    print("="*60)
    trainer.plot_training_curves(q_rewards, dqn_rewards)
    
    # Compare both agents
    print("\n" + "="*60)
    print("PHASE 3: AGENT COMPARISON")
    print("="*60)
    
    agents_dict = {
        'Q-Learning': q_agent,
        'DQN': dqn_agent
    }
    
    results = trainer.compare_agents(agents_dict, n_episodes=20)
    
    # Save results
    results_path = trainer.log_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for agent_name, stats in results.items():
        print(f"\n{agent_name}:")
        print(f"  Net Worth: ${stats['mean_net_worth']:,.0f}")
        print(f"  Reward: {stats['mean_reward']:.2f}")
        print(f"  Bankruptcy: {stats['bankruptcy_rate']:.1%}")
    
    # Calculate improvement
    if 'Q-Learning' in results and 'DQN' in results:
        q_nw = results['Q-Learning']['mean_net_worth']
        dqn_nw = results['DQN']['mean_net_worth']
        improvement = ((dqn_nw - q_nw) / q_nw) * 100 if q_nw != 0 else 0
        
        print(f"\n DQN Improvement: {improvement:+.1f}%")
    
    print("\n Training and evaluation complete!")
    print(" Check the 'models/' and 'logs/' directories for saved files.\n")


if __name__ == "__main__":
    main()