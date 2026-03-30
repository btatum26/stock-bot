import pytest
import os
import pandas as pd
from unittest.mock import MagicMock
from engine.core.optimization.local_cache import LocalCache, SHM_PATH, load_data_from_shm
from engine.core.optimization.optimizer_core import OptimizerCore

def test_local_cache_operations():
    cache = LocalCache()
    df = pd.DataFrame({"close": [100, 101, 102]})
    dataset_ref = "test_data"
    
    ref = cache.load_to_ram(dataset_ref, df)
    assert ref == dataset_ref
    
    # Verify the file was written
    assert os.path.exists(SHM_PATH)
    
    loaded_df = load_data_from_shm()
    pd.testing.assert_frame_equal(df, loaded_df)
    
    get_ref = cache.get_ref(dataset_ref)
    assert get_ref == ref
    
    cache.clear_cache(dataset_ref)
    assert not os.path.exists(SHM_PATH)

def test_optimizer_core_instantiation():
    manifest = {
        "hyperparameters": {"rsi_lower": [20, 30], "rsi_upper": [70, 80]},
        "features": [{"id": "RSI_14", "type": "momentum", "params": {"period": 14}}]
    }
    optimizer = OptimizerCore(
        strategy_path="strategies/momentum_surge",
        dataset_ref="AAPL_1h",
        manifest=manifest,
        ticker="AAPL",
        interval="1h"
    )
    assert optimizer.ticker == "AAPL"
    assert optimizer.interval == "1h"
    assert optimizer.manifest == manifest

def test_optimizer_circuit_breaker_routing():
    # Small permutation (<=5000)
    manifest_small = {
        "parameters": {"p1": {"min": 1, "max": 50, "step": 1}, "p2": {"min": 1, "max": 50, "step": 1}}, # 2500 permutations
    }
    
    # Large permutation (>5000)
    manifest_large = {
        "parameters": {"p1": {"min": 1, "max": 100, "step": 1}, "p2": {"min": 1, "max": 100, "step": 1}}, # 10000 permutations
    }
    
    optimizer_small = OptimizerCore("path", "ref", manifest_small)
    optimizer_large = OptimizerCore("path", "ref", manifest_large)
    
    optimizer_small._run_grid_search = MagicMock(return_value={"p1": 0, "p2": 0})
    optimizer_small._run_optuna_search = MagicMock()
    
    optimizer_large._run_grid_search = MagicMock()
    optimizer_large._run_optuna_search = MagicMock(return_value={"p1": 0, "p2": 0})
    
    optimizer_small._phase_a_discovery()
    optimizer_small._run_grid_search.assert_called_once()
    optimizer_small._run_optuna_search.assert_not_called()
    
    optimizer_large._phase_a_discovery()
    optimizer_large._run_optuna_search.assert_called_once()
    optimizer_large._run_grid_search.assert_not_called()
