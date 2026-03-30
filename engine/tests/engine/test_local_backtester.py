import pytest
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import patch
from engine.core.backtester import LocalBacktester
from engine.core.controller import SignalModel

# Define a mock SignalModel subclass for testing
MOCK_MODEL_CONTENT = """
from engine.core.controller import SignalModel
import pandas as pd

class MockStrategy(SignalModel):
    def train(self, df, context, params):
        return {"mean": df['Close'].mean()}
    
    def generate_signals(self, df, context, params, artifacts):
        signals = pd.Series(0, index=df.index)
        signals.iloc[-1] = 1 if df['Close'].iloc[-1] > artifacts['mean'] else -1
        return signals
"""

MOCK_CONTEXT_CONTENT = """
class Context:
    def __init__(self):
        self.state = {}
"""

MOCK_MANIFEST = {
    "name": "Mock Strategy",
    "features": [
        {"id": "moving_avg", "params": {"window": 5}}
    ],
    "hyperparameters": {
        "threshold": 0.5
    },
    "parameter_bounds": {
        "threshold": [0.1, 0.5, 0.9]
    }
}

@pytest.fixture
def temp_strategy_dir(tmp_path):
    strat_dir = tmp_path / "mock_strat"
    strat_dir.mkdir()
    
    with open(strat_dir / "model.py", "w") as f:
        f.write(MOCK_MODEL_CONTENT)
    
    with open(strat_dir / "context.py", "w") as f:
        f.write(MOCK_CONTEXT_CONTENT)
        
    with open(strat_dir / "manifest.json", "w") as f:
        json.dump(MOCK_MANIFEST, f)
        
    return str(strat_dir)

def test_backtester_load_strategy(temp_strategy_dir):
    backtester = LocalBacktester(temp_strategy_dir)
    model, context_instance = backtester._load_user_model_and_context()
    
    assert issubclass(model, SignalModel)
    assert model.__name__ == "MockStrategy"
    # context_instance should be an object of the Context class
    assert context_instance is not None
    assert context_instance.__name__ == "Context"

def test_backtester_run(temp_strategy_dir):
    backtester = LocalBacktester(temp_strategy_dir)
    
    # Create mock data
    dates = pd.date_range('2023-01-01', periods=10)
    df = pd.DataFrame({
        'Open': range(10),
        'High': range(10),
        'Low': range(10),
        'Close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Volume': [100] * 10
    }, index=dates)
    
    # Mock compute_all_features to avoid actually running feature logic
    with patch('engine.core.backtester.compute_all_features') as mock_compute:
        mock_compute.return_value = (df, 0)
        
        signals = backtester.run(df)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == 10
        assert signals.iloc[-1] == 1 # 10 > mean(1..10)=5.5

def test_backtester_grid_search(temp_strategy_dir):
    backtester = LocalBacktester(temp_strategy_dir)
    
    dates = pd.date_range('2023-01-01', periods=10)
    df = pd.DataFrame({
        'Open': range(10),
        'High': range(10),
        'Low': range(10),
        'Close': range(10),
        'Volume': [100] * 10
    }, index=dates)
    
    with patch('engine.core.backtester.compute_all_features') as mock_compute:
        mock_compute.return_value = (df, 0)
        
        results = backtester.run_grid_search(df)
        
        # threshold has 3 values in parameter_bounds
        assert len(results) == 3
        assert all(isinstance(r, pd.Series) for r in results)
