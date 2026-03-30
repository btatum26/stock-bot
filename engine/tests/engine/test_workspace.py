import os
import json
import pytest
from engine.core.features.base import Feature, FeatureResult, register_feature
# Adjust this import to match where WorkspaceManager actually lives
from engine.core.workspace import WorkspaceManager
from engine.core.exceptions import ValidationError

# --- Mock Features for Testing ---

@register_feature("mock_single")
class MockSingleFeature(Feature):
    @property
    def name(self): return "Mock Single"
    @property
    def description(self): return "Tests single output base naming."
    @property
    def category(self): return "Test"
    @property
    def outputs(self): return [None] # Single output

    def compute(self, df, params, cache=None):
        return FeatureResult(data={})

@register_feature("mock_multi")
class MockMultiFeature(Feature):
    @property
    def name(self): return "Mock Multi"
    @property
    def description(self): return "Tests multi-output suffix generation."
    @property
    def category(self): return "Test"
    @property
    def outputs(self): return ["macd", "signal", "hist"] # Multi output

    def compute(self, df, params, cache=None):
        return FeatureResult(data={})

# --- Setup Fixtures ---

@pytest.fixture
def workspace_env(tmp_path):
    """
    Creates temporary strategy and template directories.
    Writes dummy Jinja templates so the WorkspaceManager doesn't crash on load.
    """
    strat_dir = tmp_path / "test_strategy"
    strat_dir.mkdir()
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    # Write dummy Jinja templates for testing the render engine
    context_template = template_dir / "context.py.j2"
    context_template.write_text("FEATURES: {{ features | length }}, PARAMS: {{ params | length }}")
    
    model_template = template_dir / "model.py.j2"
    model_template.write_text("# Dummy Model Template")
    
    return str(strat_dir), str(template_dir)

@pytest.fixture
def wm(workspace_env):
    """Provides an initialized WorkspaceManager instance."""
    strat_dir, template_dir = workspace_env
    return WorkspaceManager(strat_dir, template_dir)

# --- Test Cases ---

def test_infer_type(wm):
    """Verifies that Python/JSON types are correctly mapped to type hint strings."""
    assert wm._infer_type(True) == "bool"
    assert wm._infer_type(42) == "int"
    assert wm._infer_type(3.14) == "float"
    assert wm._infer_type("test") == "str"
    assert wm._infer_type([1, 2, 3]) == "list"
    assert wm._infer_type({"a": 1}) == "dict"
    assert wm._infer_type(object()) == "Any"

def test_build_features_payload_single_output(wm):
    """Verifies flat naming for features with a single primary output."""
    config = [
        {
            "id": "mock_single", 
            # Note: "color" should be filtered out by the logic
            "params": {"window": 14, "color": "red"} 
        }
    ]
    payload = wm._build_features_payload(config)
    
    assert len(payload) == 1
    
    feat = payload[0]
    # base naming should extract 'window' and ignore 'color'
    assert feat["attr_name"] == "MOCK_SINGLE_14" 
    # check if generate_column_name generated the underlying string correctly
    assert feat["col_name"] == "mock_single_14"
    assert "Mock Single" in feat["docstring"]

def test_build_features_payload_multi_output(wm):
    """Verifies suffix appending for features like MACD or Bollinger Bands."""
    config = [
        {
            "id": "mock_multi", 
            "params": {"fast": 12, "slow": 26}
        }
    ]
    payload = wm._build_features_payload(config)
    
    assert len(payload) == 3
    
    attr_names = [p["attr_name"] for p in payload]
    
    # Expected deterministic flat names
    assert "MOCK_MULTI_12_26_MACD" in attr_names
    assert "MOCK_MULTI_12_26_SIGNAL" in attr_names
    assert "MOCK_MULTI_12_26_HIST" in attr_names

def test_sync_blocks_python_keywords(wm):
    """Verifies the defensive layer that stops Jinja from writing fatal syntax errors."""
    bad_hparams = {"class": 0.05, "valid_param": 10}
    
    with pytest.raises(ValidationError, match="reserved Python keyword"):
        wm.sync(features=[], hparams=bad_hparams, bounds={})

def test_sync_writes_manifest(wm):
    """Verifies manifest.json is created and formatted correctly."""
    features = [{"id": "mock_single", "params": {"window": 10}}]
    hparams = {"stop_loss": 0.05}
    bounds = {"stop_loss": [0.01, 0.1]}
    
    wm.sync(features, hparams, bounds)
    
    assert os.path.exists(wm.manifest_path)
    
    with open(wm.manifest_path, 'r') as f:
        manifest = json.load(f)
        
    assert manifest["hyperparameters"]["stop_loss"] == 0.05
    assert manifest["features"][0]["id"] == "mock_single"

def test_sync_renders_jinja_templates(wm):
    """Verifies the templating engine successfully reads payload and generates files."""
    features = [
        {"id": "mock_single", "params": {"window": 10}},
        {"id": "mock_multi", "params": {"fast": 12, "slow": 26}}
    ]
    hparams = {"stop_loss": 0.05, "take_profit": 0.10}
    bounds = {}
    
    wm.sync(features, hparams, bounds)
    
    # 1 mock_single + 3 from mock_multi = 4 total feature outputs
    # 2 hparams
    
    assert os.path.exists(wm.context_path)
    with open(wm.context_path, 'r') as f:
        context_content = f.read()
    
    # Verify the dummy template successfully unpacked the dictionary lists
    assert context_content == "FEATURES: 4, PARAMS: 2"
    
    assert os.path.exists(wm.model_path)
    with open(wm.model_path, 'r') as f:
        model_content = f.read()
        
    assert model_content == "# Dummy Model Template"

def test_sync_does_not_overwrite_model_py(wm, workspace_env):
    """Safety check: ensures the system never overwrites user trading logic."""
    strat_dir, _ = workspace_env
    model_path = os.path.join(strat_dir, "model.py")
    
    # Simulate a user having already written their code
    with open(model_path, 'w') as f:
        f.write("# USER LOGIC")
        
    wm.sync(features=[], hparams={}, bounds={})
    
    # The sync function should have skipped writing model.py
    with open(model_path, 'r') as f:
        assert f.read() == "# USER LOGIC"