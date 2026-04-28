import os
import sys
import shutil
import pytest
import pandas as pd
import fakeredis
from unittest.mock import patch

# Ensure repo and engine roots are importable in local and Docker test runs.
_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ENGINE_ROOT)
sys.path.insert(0, _ENGINE_ROOT)
sys.path.insert(0, _REPO_ROOT)

# RQ's scheduler uses multiprocessing.get_context('fork') at import time,
# which is unavailable on Windows. Patch it before importing rq.
if sys.platform == "win32":
    import multiprocessing
    _real_get_context = multiprocessing.get_context
    multiprocessing.get_context = lambda m: _real_get_context("spawn") if m == "fork" else _real_get_context(m)

from rq import Queue

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Custom hook to track test failures. 
    
    This is used by the managed_artifact_dir fixture to determine whether 
    to preserve or delete generated artifact files after a test completes.
    """
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed:
        item.test_failed = True


@pytest.fixture
def mock_redis_client():
    """
    Provides a FakeRedis client for testing.
    
    Patches the global redis_client in daemon/main.py. The tasks.py worker now
    gets its connection dynamically from the RQ job, so it no longer needs 
    to be patched globally here.
    """
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    with patch("daemon.main.redis_client", fake_client):
        yield fake_client


@pytest.fixture
def mock_rq_queue(mock_redis_client):
    """
    Provides a synchronous fake RQ queue.
    
    Using is_async=False forces the queue to execute jobs immediately in the 
    same thread, which is highly useful for end-to-end integration tests.
    """
    fake_queue = Queue('default', connection=mock_redis_client, is_async=False)
    with patch("daemon.main.task_queue", fake_queue):
        yield fake_queue


@pytest.fixture
def mock_data_broker():
    """
    Mocks the DataBroker to return a static Pandas DataFrame.
    
    Ensures that backend execution tests do not make live network calls 
    to external data providers like yfinance.
    """
    df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [95.0, 96.0, 97.0],
        "Close": [102.0, 103.0, 104.0],
        "Volume": [1000, 1100, 1200]
    }, index=pd.date_range("2023-01-01", periods=3))
    
    with patch("engine.core.data_broker.data_broker.DataBroker.get_data", return_value=df):
        yield df


@pytest.fixture
def managed_artifact_dir(request):
    """
    Manages the lifecycle of heavy artifact files generated during testing.
    
    Creates an isolated directory for the current test. If the test passes, 
    the directory is wiped to save space. If the test fails, the directory 
    is preserved for manual debugging.
    """
    test_name = request.node.name
    artifact_dir = os.path.abspath(f"./tests/failed_artifacts/{test_name}")
    os.makedirs(artifact_dir, exist_ok=True)
    
    with patch("daemon.tasks.ARTIFACT_DIR", artifact_dir):
        yield artifact_dir
        
    if not getattr(request.node, "test_failed", False):
        shutil.rmtree(artifact_dir, ignore_errors=True)
    else:
        print(f"\nArtifacts preserved at {artifact_dir}")


@pytest.fixture
def dummy_strategy(tmp_path):
    """
    Generates a temporary, fully-formed strategy directory.
    
    Automatically patches the STRATEGIES_DIR in daemon/main.py to point to this
    temporary path, allowing the 'Fail Fast' validation to succeed during testing.
    """
    strat_dir = tmp_path / "dummy_strat"
    strat_dir.mkdir()
    
    # Generate the manifest payload
    manifest_content = """{
        "name": "Dummy Strategy",
        "description": "A dummy strategy for testing",
        "is_ml": false,
        "features": [{"id": "feature_1"}],
        "hyperparameters": {
            "window": 14,
            "threshold": 0.15
        },
        "parameter_bounds": {
            "window": [10, 20],
            "threshold": [0.1, 0.2]
        }
    }"""
    (strat_dir / "manifest.json").write_text(manifest_content)
    
    # Generate the strict-typed context
    context_content = """class Context:
    def __init__(self):
        self.mapping = {}
"""
    (strat_dir / "context.py").write_text(context_content)
    
    # Generate the user-defined signal model
    model_content = """import pandas as pd
from engine.core.controller import SignalModel

class DummyModel(SignalModel):
    def train(self, df, context, params):
        return {"trained": True}
        
    def generate_signals(self, df, context, params, artifacts):
        # Deterministic alternating signals
        signals = [1.0, -1.0] * (len(df) // 2) + [1.0] * (len(df) % 2)
        return pd.Series(signals, index=df.index, name="signal")
"""
    (strat_dir / "model.py").write_text(model_content)
    
    # Automatically patch the API's strategy directory to point to our temp pytest directory
    base_dir = os.path.dirname(str(strat_dir))
    with patch("daemon.main.STRATEGIES_DIR", base_dir):
        yield str(strat_dir)
