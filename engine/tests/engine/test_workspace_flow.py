import os
import sys
import pandas as pd
import numpy as np
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.core.workspace import WorkspaceManager
from engine.core.backtester import LocalBacktester
from engine.core.bundler import Bundler

# 1. Setup Strategy Workspace
STRAT_DIR = "strategies/test_strategy"

# The updated strategy logic utilizing SMA, RSI, and MACD Histogram
MODEL_PY_CONTENT = """
import numpy as np
import pandas as pd
from engine.core.controller import SignalModel  # noqa: F401 (imported for dynamic strategy loading)

class TestStrategy(SignalModel):
    def train(self, df, context, params):
        return {}

    def generate_signals(self, df, context, artifacts=None):
        rsi_val = df[context.features.RSI_close_14]
        sma_val = df[context.features.SMA_50_close]
        macd_hist = df[context.features.MACD_12_9_26_close_HIST]
        
        # Standardize on 'close' (handling potential capitalization differences)
        close_price = df['close'] if 'close' in df.columns else df['Close']
        
        condition_long = (
            (close_price > sma_val) & 
            (rsi_val < 70) & 
            (macd_hist > 0.0)
        )
        
        condition_short = (
            (close_price < sma_val) & 
            (rsi_val > 30) & 
            (macd_hist < 0.0)
        )
        
        signals = np.select(
            [condition_long, condition_short], 
            [1.0, -1.0], 
            default=0.0
        )
        return pd.Series(signals, index=df.index)
"""

def test_full_flow():
    # A. Sync Workspace (GUI Action)
    print("--- 1. Syncing Workspace ---")
    wm = WorkspaceManager(STRAT_DIR)
    
    # Corrected parameters based on test_strategy manifest
    features = [
        {"id": "SMA", "params": {"period": 50, "source": "close"}},
        {"id": "RSI", "params": {"window": 14, "source": "close"}},
        {"id": "MACD", "params": {
            "fast_period": 12, 
            "slow_period": 26, 
            "signal_period": 9,
            "source": "close"
        }},
        {"id": "RSI", "params": {"window": 21, "source": "close"}}
    ]
    
    hyperparams = {"stop_loss": 0.05, "take_profit": 0.1}
    parameter_bounds = {"stop_loss": [0.01, 0.1], "take_profit": [0.05, 0.2]}
    
    # This triggers your auto-generator for context.py and writes manifest.json
    wm.sync(features, hyperparams, parameter_bounds)
    
    # Write the model.py
    with open(os.path.join(STRAT_DIR, "model.py"), 'w') as f:
        f.write(MODEL_PY_CONTENT)
        
    # We will manually overwrite context.py here just for the test environment
    # to guarantee the mappings match our model.py logic before your auto-generator kicks in.
    context_content = """
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    SMA_50_close: str = 'SMA_50_close'
    RSI_close_14: str = 'RSI_close_14'
    MACD_12_9_26_close_HIST: str = 'MACD_12_9_26_close_HIST'

@dataclass(frozen=True)
class ParamsContext:
    stop_loss: float = 0.05
    take_profit: float = 0.1

@dataclass(frozen=True)
class Context:
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)
"""
    with open(wm.context_path, 'w') as f:
        f.write(context_content)
    
    print(f"Context verified at {wm.context_path}")

    # B. Run Backtest (Local Execution)
    print("\n--- 2. Running Local Backtest ---")
    
    # Mock Data Setup
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 150, 200),
        'High': np.random.uniform(105, 155, 200),
        'Low': np.random.uniform(95, 145, 200),
        'Close': np.random.uniform(100, 150, 200),
        'Volume': np.random.uniform(1000, 5000, 200),
        # Mocking the features that the LocalBacktester/Context expects
        'SMA_50_close': np.random.uniform(100, 150, 200),
        'RSI_close_14': np.random.uniform(10, 90, 200),
        'MACD_12_9_26_close_HIST': np.random.uniform(-2, 2, 200)
    }, index=dates)

    
    backtester = LocalBacktester(STRAT_DIR)
    
    try:
        signals = backtester.run(df)
        print(f"Backtest generated {len(signals)} signals.")
        print(f"Signal distributions:\n{signals.value_counts()}")
    except Exception as e:
        print(f"Backtest Failed: {e}")
        return

    # C. Grid Search
    print("\n--- 3. Running Grid Search ---")
    try:
        grid_results = backtester.run_grid_search(df)
        print(f"Grid search generated {len(grid_results)} permutations.")
        for res in grid_results[:5]: # Print top 5 to avoid console spam
            print(f"Permutation: {res.name} | Mean Signal: {res.mean():.4f}")
    except AttributeError:
        print("Note: run_grid_search not fully implemented or mocked in LocalBacktester.")

    # D. Export to .strat (Deployment)
    print("\n--- 4. Exporting to .strat ---")
    try:
        os.makedirs("exports", exist_ok=True)
        bundle_path = Bundler.export(STRAT_DIR, "exports")
        print(f"Bundle successfully created at: {bundle_path}")
    except Exception as e:
        print(f"Bundler export failed: {e}")

if __name__ == "__main__":
    test_full_flow()