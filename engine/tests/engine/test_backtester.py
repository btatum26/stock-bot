import pytest
import pandas as pd
import json
import os
import sys
from engine.core.backtester import LocalBacktester
from engine.core.exceptions import StrategyError
from unittest.mock import patch, MagicMock

def test_init_missing_manifest(tmp_path):
    empty_path = tmp_path / "empty"
    empty_path.mkdir()
    with pytest.raises(StrategyError, match="Missing or invalid manifest"):
        LocalBacktester(str(empty_path))

def test_init_invalid_manifest_json(tmp_path):
    strat_dir = tmp_path / "bad_manifest"
    strat_dir.mkdir()
    (strat_dir / "manifest.json").write_text("{bad json")
    
    with pytest.raises(StrategyError, match="Missing or invalid manifest"):
        LocalBacktester(str(strat_dir))

def test_dynamic_import_isolation(dummy_strategy):
    # Setup backtester
    backtester = LocalBacktester(dummy_strategy)
    
    # Capture original sys.path
    original_path = list(sys.path)
    
    # Load model and context
    model, context = backtester._load_user_model_and_context()
    
    # Verify successful load
    assert model is not None
    assert context is not None
    
    # Assert that sys.path is restored
    assert sys.path == original_path

@patch("engine.core.backtester.logger.warning")
def test_nan_audit_trigger(mock_warning, dummy_strategy):
    backtester = LocalBacktester(dummy_strategy)
    df_with_nans = pd.DataFrame({"feature_1": [1.0, float('nan'), 3.0]})
    
    backtester._audit_nans(df_with_nans, ["feature_1"])
    
    mock_warning.assert_called_once_with("Feature 'feature_1' has 1 unexpected NaN values after l_max purge.")

@patch("engine.core.backtester.compute_all_features")
def test_single_run_output_shape(mock_compute, dummy_strategy):
    mock_compute.return_value = (pd.DataFrame(index=range(4)), 0)
    
    backtester = LocalBacktester(dummy_strategy)
    df = pd.DataFrame(index=range(4))
    
    output = backtester.run(df)
    
    assert isinstance(output, pd.Series)
    assert len(output) == len(df)
    assert output.index.equals(df.index)

@patch("engine.core.backtester.compute_all_features")
def test_single_run_deterministic_values(mock_compute, dummy_strategy):
    # Dummy strategy should output [1.0, -1.0, 1.0, -1.0] for length 4
    df = pd.DataFrame(index=range(4))
    mock_compute.return_value = (df, 0)
    
    backtester = LocalBacktester(dummy_strategy)
    output = backtester.run(df)
    
    expected = pd.Series([1.0, -1.0, 1.0, -1.0], index=df.index, name="conviction_signal")
    pd.testing.assert_series_equal(output, expected)

@patch("engine.core.backtester.compute_all_features")
def test_grid_search_permutation_count(mock_compute, dummy_strategy):
    df = pd.DataFrame(index=range(4))
    mock_compute.return_value = (df, 0)
    
    backtester = LocalBacktester(dummy_strategy)
    
    bounds = {"window": [10, 20], "threshold": [0.1, 0.2]}
    results = backtester.run_grid_search(df, param_bounds=bounds)
    
    assert len(results) == 4
    assert all(isinstance(res, pd.Series) for res in results)

@patch("engine.core.backtester.compute_all_features")
def test_grid_search_naming_convention(mock_compute, dummy_strategy):
    df = pd.DataFrame(index=range(4))
    mock_compute.return_value = (df, 0)
    
    backtester = LocalBacktester(dummy_strategy)
    bounds = {"window": [10], "threshold": [0.1]}
    results = backtester.run_grid_search(df, param_bounds=bounds)
    
    assert len(results) == 1
    # Check naming convention: <strategy_dir> (<params>)
    strat_dir_name = os.path.basename(dummy_strategy)
    assert results[0].name == f"{strat_dir_name} (window=10, threshold=0.1)"
