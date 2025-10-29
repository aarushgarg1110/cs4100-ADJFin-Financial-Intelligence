import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketDataManager:
    """
    Pulls and processes historical market data for realistic simulation.
    """
    
    def __init__(self):
        self.data = {}
        self.returns = {}
        self.regimes = {}
        
    def download_data(self, start_date="2000-01-01"):
        """Download historical data for major asset classes."""
        
        tickers = {
            'stocks': 'SPY',      # S&P 500
            'bonds': 'BND',       # Total Bond Market  
            'real_estate': 'VNQ'  # REITs
        }
        
        print("Downloading market data...")
        
        for asset, ticker in tickers.items():
            try:
                # Download data
                data = yf.download(ticker, start=start_date, progress=False)
                
                # Calculate monthly returns
                monthly_data = data['Adj Close'].resample('M').last()
                monthly_returns = monthly_data.pct_change().dropna()
                
                self.data[asset] = data
                self.returns[asset] = monthly_returns.values
                
                print(f"✅ {asset}: {len(monthly_returns)} months of data")
                
            except Exception as e:
                print(f"❌ Error downloading {asset}: {e}")
                # Fallback to synthetic data
                self.returns[asset] = self._generate_fallback_returns(asset)
    
    def _generate_fallback_returns(self, asset):
        """Generate synthetic returns if download fails."""
        np.random.seed(42)
        
        params = {
            'stocks': (0.0087, 0.04),      # ~10.5% annual, 4% monthly vol
            'bonds': (0.0041, 0.02),       # ~5% annual, 2% monthly vol  
            'real_estate': (0.006, 0.03)   # ~7% annual, 3% monthly vol
        }
        
        mean, std = params[asset]
        return np.random.normal(mean, std, 300)  # 25 years of monthly data
    
    def classify_regimes(self, asset='stocks', window=12):
        """Classify market periods into bull/bear/normal regimes."""
        
        returns = self.returns[asset]
        rolling_returns = pd.Series(returns).rolling(window).mean()
        
        # Simple regime classification
        regimes = []
        for ret in rolling_returns:
            if pd.isna(ret):
                regimes.append(0)  # Normal
            elif ret > 0.015:      # >18% annual
                regimes.append(1)  # Bull
            elif ret < -0.005:     # <-6% annual  
                regimes.append(2)  # Bear
            else:
                regimes.append(0)  # Normal
                
        self.regimes[asset] = regimes
        return regimes
    
    def get_regime_stats(self, asset='stocks'):
        """Get statistics for each market regime."""
        
        returns = self.returns[asset]
        regimes = self.regimes.get(asset, [0] * len(returns))
        
        stats = {}
        for regime in [0, 1, 2]:  # Normal, Bull, Bear
            regime_returns = [r for r, reg in zip(returns, regimes) if reg == regime]
            
            if regime_returns:
                stats[regime] = {
                    'mean': np.mean(regime_returns),
                    'std': np.std(regime_returns),
                    'count': len(regime_returns)
                }
            else:
                # Fallback stats
                fallback = {
                    0: {'mean': 0.0087, 'std': 0.04},   # Normal
                    1: {'mean': 0.015, 'std': 0.05},    # Bull  
                    2: {'mean': -0.01, 'std': 0.06}     # Bear
                }
                stats[regime] = fallback[regime]
                stats[regime]['count'] = 0
        
        return stats
    
    def sample_return(self, asset='stocks', regime=0):
        """Sample a return for given asset and market regime."""
        
        stats = self.get_regime_stats(asset)
        regime_stat = stats[regime]
        
        return np.random.normal(regime_stat['mean'], regime_stat['std'])
    
    def get_summary(self):
        """Print summary of downloaded data."""
        
        print("\n=== Market Data Summary ===")
        
        for asset in self.returns:
            returns = self.returns[asset]
            annual_return = (1 + np.mean(returns))**12 - 1
            annual_vol = np.std(returns) * np.sqrt(12)
            
            print(f"{asset.upper()}:")
            print(f"  Annual Return: {annual_return:.1%}")
            print(f"  Annual Volatility: {annual_vol:.1%}")
            print(f"  Months of Data: {len(returns)}")
            
            # Regime breakdown
            if asset in self.regimes:
                regimes = self.regimes[asset]
                regime_counts = {0: 0, 1: 0, 2: 0}
                for r in regimes:
                    regime_counts[r] += 1
                    
                total = len(regimes)
                print(f"  Normal: {regime_counts[0]/total:.1%}")
                print(f"  Bull: {regime_counts[1]/total:.1%}")  
                print(f"  Bear: {regime_counts[2]/total:.1%}")
            print()

# Convenience function
def load_market_data():
    """Load and return market data manager."""
    
    manager = MarketDataManager()
    manager.download_data()
    
    # Classify regimes for all assets
    for asset in manager.returns:
        manager.classify_regimes(asset)
    
    manager.get_summary()
    return manager

if __name__ == "__main__":
    # Test the market data module
    manager = load_market_data()
