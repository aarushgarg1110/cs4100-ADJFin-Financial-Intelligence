#!/usr/bin/env python3

import sys
sys.path.append('../environment')
sys.path.append('../data')

from finance_env import FinanceEnv
import numpy as np

def test_market_integration():
    print("Testing Market Data Integration...")
    
    # Create environment (this will load market data)
    env = FinanceEnv()
    
    # Test that market data is loaded
    print(f"✅ Market data loaded for assets: {list(env.market_data.returns.keys())}")
    
    # Test regime statistics
    for asset in env.market_data.returns:
        stats = env.market_data.get_regime_stats(asset)
        print(f"\n{asset.upper()} regime stats:")
        for regime, stat in stats.items():
            regime_name = ['Normal', 'Bull', 'Bear'][regime]
            annual_return = (1 + stat['mean'])**12 - 1
            print(f"  {regime_name}: {annual_return:.1%} annual return")
    
    # Test environment with real market data
    state, info = env.reset(seed=42)
    
    print(f"\n=== Testing Environment Steps ===")
    returns_collected = []
    
    for i in range(10):
        action = np.random.rand(6) * 0.3  # Conservative actions
        state, reward, done, truncated, info = env.step(action)
        
        stock_return = state[9]  # stock_return_1m is at index 9
        returns_collected.append(stock_return)
        
        print(f"Month {i+1}: Stock return = {stock_return:.3f} ({stock_return*12:.1%} annualized)")
    
    # Verify returns are realistic (not all the same)
    return_std = np.std(returns_collected)
    print(f"\nReturn volatility: {return_std:.4f} (should be > 0)")
    
    if return_std > 0.001:
        print("✅ Market integration successful - returns are varying realistically!")
    else:
        print("❌ Market integration issue - returns too uniform")
    
    return env

if __name__ == "__main__":
    env = test_market_integration()
