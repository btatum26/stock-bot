from src.database import Database
from src.analysis import LevelsAnalyzer
import pandas as pd

def test_analysis():
    db = Database()
    analyzer = LevelsAnalyzer(window=5, threshold_pct=0.015) # 1.5% zone
    
    ticker = "AAPL"
    interval = "1d"
    
    print(f"Analyzing {ticker} ({interval})...")
    df = db.get_data(ticker, interval)
    
    if df.empty:
        print("No data found in DB. Please sync AAPL first.")
        return

    levels = analyzer.get_strong_levels(df, min_strength=5)
    
    print(f"\nIdentified {len(levels)} Strong Interest Zones:")
    print("-" * 55)
    print(f"{'Avg Price':<10} | {'Range (Low - High)':<22} | {'Strength':<10}")
    print("-" * 55)
    
    for l in sorted(levels, key=lambda x: x['price'], reverse=True):
        price_range = f"${l['min_price']:.2f} - ${l['max_price']:.2f}"
        print(f"${l['price']:<9.2f} | {price_range:<22} | {l['strength']}")

if __name__ == "__main__":
    test_analysis()
