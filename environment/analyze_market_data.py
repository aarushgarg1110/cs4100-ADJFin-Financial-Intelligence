"""
Analyze and save market data to CSV files for inspection.

This script downloads historical market data (2000-2025) from Yahoo Finance
and generates CSV files with monthly returns and market regime classifications.

USAGE:
    python environment/analyze_market_data.py

OUTPUT FILES (saved to project root):
    - market_stocks.csv: S&P 500 (SPY) monthly returns and regimes
    - market_bonds.csv: Total Bond Market (BND) monthly returns and regimes
    - market_real_estate.csv: REITs (VNQ) monthly returns and regimes
    - macro_factors.csv: Simulated inflation and interest rates (360 months)

ANALYSIS INCLUDES:
    - Annual returns and volatility for each asset class
    - Market regime breakdown (bull/bear/normal)
    - Correlation matrix between assets
    - Key insights about diversification benefits (or lack thereof)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from environment.market_data import MarketDataManager


def main():
    # Download and process market data
    print("Downloading market data...")
    manager = MarketDataManager()
    manager.download_data(start_date='2000-01-01')
    
    # Classify regimes
    for asset in manager.returns:
        manager.classify_regimes(asset)
    
    # Save each asset separately
    print("\nSaving CSV files...")
    for asset in ['stocks', 'bonds', 'real_estate']:
        df = pd.DataFrame({
            'return': manager.returns[asset],
            'regime': manager.regimes[asset]
        })
        df.to_csv(f'market_{asset}.csv', index=False)
        print(f'  Saved market_{asset}.csv ({len(df)} months)')
    
    # Save macro factors
    inflation, rates = manager.simulate_macro_factors(months=360)
    macro_df = pd.DataFrame({
        'inflation': inflation,
        'interest_rates': rates
    })
    macro_df.to_csv('macro_factors.csv', index=False)
    print(f'  Saved macro_factors.csv (360 months)')
    
    # Print summary
    print("\n" + "="*60)
    print("MARKET DATA SUMMARY")
    print("="*60)
    manager.get_summary()
    
    # Detailed analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    stocks = pd.read_csv('market_stocks.csv')
    bonds = pd.read_csv('market_bonds.csv')
    re = pd.read_csv('market_real_estate.csv')
    
    print('\n=== STOCKS (SPY) ===')
    print(f'Months: {len(stocks)}')
    print(f'Mean monthly return: {stocks["return"].mean():.4f} ({stocks["return"].mean()*12:.2%} annual)')
    print(f'Std dev: {stocks["return"].std():.4f}')
    print(f'Min: {stocks["return"].min():.4f} | Max: {stocks["return"].max():.4f}')
    print(f'Worst month: {stocks["return"].min()*100:.1f}%')
    print(f'Best month: {stocks["return"].max()*100:.1f}%')
    print(f'Negative months: {(stocks["return"] < 0).sum()} ({(stocks["return"] < 0).mean()*100:.1f}%)')
    
    print('\n=== BONDS (BND) ===')
    print(f'Months: {len(bonds)}')
    print(f'Mean monthly return: {bonds["return"].mean():.4f} ({bonds["return"].mean()*12:.2%} annual)')
    print(f'Std dev: {bonds["return"].std():.4f}')
    print(f'Min: {bonds["return"].min():.4f} | Max: {bonds["return"].max():.4f}')
    print(f'Negative months: {(bonds["return"] < 0).sum()} ({(bonds["return"] < 0).mean()*100:.1f}%)')
    
    print('\n=== REAL ESTATE (VNQ) ===')
    print(f'Months: {len(re)}')
    print(f'Mean monthly return: {re["return"].mean():.4f} ({re["return"].mean()*12:.2%} annual)')
    print(f'Std dev: {re["return"].std():.4f}')
    print(f'Min: {re["return"].min():.4f} | Max: {re["return"].max():.4f}')
    print(f'Negative months: {(re["return"] < 0).sum()} ({(re["return"] < 0).mean()*100:.1f}%)')
    
    print('\n=== MACRO FACTORS ===')
    print(f'Inflation mean: {macro_df["inflation"].mean():.4f} ({macro_df["inflation"].mean()*12:.2%} annual)')
    print(f'Interest rate mean: {macro_df["interest_rates"].mean():.4f} ({macro_df["interest_rates"].mean()*12:.2%} annual)')
    
    print('\n=== CORRELATION ANALYSIS ===')
    min_len = min(len(stocks), len(bonds), len(re))
    corr_data = pd.DataFrame({
        'stocks': stocks['return'][:min_len].values,
        'bonds': bonds['return'][:min_len].values,
        'real_estate': re['return'][:min_len].values
    })
    print(corr_data.corr())
    
    print('\n=== KEY INSIGHTS ===')
    print(f'- Stocks have {(stocks["return"] < 0).mean()*100:.0f}% negative months')
    print(f'- Bonds are stable: {(bonds["return"] < 0).mean()*100:.0f}% negative months')
    print(f'- Real estate is volatile: {re["return"].std():.4f} std dev')
    print(f'- Stock-bond correlation: {corr_data["stocks"].corr(corr_data["bonds"]):.3f} (near zero!)')
    print(f'- This means: diversification provides NO risk reduction')
    
    print("\n" + "="*60)
    print("Files saved")
    print("="*60)


if __name__ == "__main__":
    main()
