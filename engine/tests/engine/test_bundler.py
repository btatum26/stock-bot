import os
import zipfile
import pytest

# Adjust these imports based on where you save the bundler script
from engine.core.bundler import Bundler, StrategyValidator, StrategyValidationError

# --- Mock Source Code Payloads ---

VALID_MODEL_CODE = """
import pandas as pd
import numpy as np
from datetime import datetime
from context import Context

class SignalModel:
    def generate_signals(self, df: pd.DataFrame, ctx: Context, artifacts: dict = None) -> pd.Series:
        return pd.Series(1.0, index=df.index)
"""

MISSING_CLASS_CODE = """
import pandas as pd
from context import Context

# Forgot to name it SignalModel
class MyStrategy:
    def generate_signals(self, df, ctx, artifacts):
        pass
"""

MISSING_METHOD_CODE = """
import pandas as pd
from context import Context

class SignalModel:
    # Typo in the method name
    def gen_signals(self, df, ctx, artifacts):
        pass
"""

MISSING_CONTEXT_CODE = """
import pandas as pd
import numpy as np

class SignalModel:
    def generate_signals(self, df, ctx, artifacts):
        pass
"""

MALICIOUS_IMPORT_CODE = """
import pandas as pd
import os
from context import Context

class SignalModel:
    def generate_signals(self, df, ctx, artifacts):
        os.system("echo 'boom'")
"""

MALICIOUS_FROM_IMPORT_CODE = """
import pandas as pd
from subprocess import Popen
from context import Context

class SignalModel:
    def generate_signals(self, df, ctx, artifacts):
        pass
"""

SYNTAX_ERROR_CODE = """
import pandas as pd
from context import Context

class SignalModel
    def generate_signals(self, df, ctx, artifacts):
        pass
"""

SUBMODULE_ALLOWED_CODE = """
import pandas.core.common as pd_common
from context import Context

class SignalModel:
    def generate_signals(self, df, ctx, artifacts):
        pass
"""

SUBMODULE_DENIED_CODE = """
import urllib.request
from context import Context

class SignalModel:
    def generate_signals(self, df, ctx, artifacts):
        pass
"""

# --- Test Cases: AST Strategy Validator ---

def test_validator_passes_valid_code():
    """Verifies that a structurally sound and safe model passes without raising exceptions."""
    validator = StrategyValidator()
    validator.validate(VALID_MODEL_CODE)
    
    assert validator.has_context_import is True
    assert validator.has_signal_model is True
    assert validator.has_generate_signals is True

def test_validator_catches_missing_class():
    """Verifies the validator enforces the SignalModel class name."""
    validator = StrategyValidator()
    with pytest.raises(StrategyValidationError, match="Missing required class: `class SignalModel:`"):
        validator.validate(MISSING_CLASS_CODE)

def test_validator_catches_missing_method():
    """Verifies the validator enforces the generate_signals method."""
    validator = StrategyValidator()
    with pytest.raises(StrategyValidationError, match="Missing required method: `def generate_signals"):
        validator.validate(MISSING_METHOD_CODE)

def test_validator_catches_missing_context():
    """Verifies the validator enforces the Context import."""
    validator = StrategyValidator()
    with pytest.raises(StrategyValidationError, match="Missing required import: `from context import Context`"):
        validator.validate(MISSING_CONTEXT_CODE)

def test_validator_catches_malicious_import():
    """Verifies the blocklist catches standard unauthorized imports."""
    validator = StrategyValidator()
    with pytest.raises(StrategyValidationError, match="Importing 'os' is not permitted"):
        validator.validate(MALICIOUS_IMPORT_CODE)

def test_validator_catches_malicious_from_import():
    """Verifies the blocklist catches unauthorized 'from x import y' statements."""
    validator = StrategyValidator()
    with pytest.raises(StrategyValidationError, match="Importing 'subprocess' is not permitted"):
        validator.validate(MALICIOUS_FROM_IMPORT_CODE)

def test_validator_allows_valid_submodules():
    """Verifies that allowed root modules (like pandas.core) pass the check."""
    validator = StrategyValidator()
    # Should not raise
    validator.validate(SUBMODULE_ALLOWED_CODE)

def test_validator_catches_invalid_submodules():
    """Verifies that unauthorized root modules bypasses (like urllib.request) are caught."""
    validator = StrategyValidator()
    with pytest.raises(StrategyValidationError, match="Importing 'urllib' is not permitted"):
        validator.validate(SUBMODULE_DENIED_CODE)

def test_validator_catches_syntax_errors():
    """Verifies the AST parser gracefully handles and reports invalid Python syntax."""
    validator = StrategyValidator()
    with pytest.raises(StrategyValidationError, match="Syntax Error in model.py on line"):
        validator.validate(SYNTAX_ERROR_CODE)


# --- Test Cases: The Bundler ---

@pytest.fixture
def strategy_env(tmp_path):
    """Sets up a mock strategy directory with required files."""
    strat_dir = tmp_path / "my_strategy"
    strat_dir.mkdir()
    
    # Create the required files
    (strat_dir / "manifest.json").write_text('{"mock": "data"}')
    (strat_dir / "context.py").write_text("# Mock Context")
    (strat_dir / "model.py").write_text(VALID_MODEL_CODE)
    
    out_dir = tmp_path / "exports"
    
    return str(strat_dir), str(out_dir)

def test_bundler_missing_files(tmp_path):
    """Verifies the bundler fails fast if core files are missing."""
    empty_strat = tmp_path / "empty_strat"
    empty_strat.mkdir()
    out_dir = tmp_path / "exports"
    
    with pytest.raises(FileNotFoundError, match="Required file 'manifest.json' not found"):
        Bundler.export(str(empty_strat), str(out_dir))

def test_bundler_aborts_on_invalid_code(strategy_env):
    """Verifies the bundler refuses to zip if the AST validation fails."""
    strat_dir, out_dir = strategy_env
    
    # Overwrite valid model with malicious one
    with open(os.path.join(strat_dir, "model.py"), "w") as f:
        f.write(MALICIOUS_IMPORT_CODE)
        
    with pytest.raises(StrategyValidationError):
        Bundler.export(strat_dir, out_dir)
        
    # Ensure no zip file was created
    assert not os.path.exists(out_dir) or len(os.listdir(out_dir)) == 0

def test_bundler_creates_valid_archive(strategy_env):
    """Verifies the bundler successfully creates a valid .strat zip archive."""
    strat_dir, out_dir = strategy_env
    
    bundle_path = Bundler.export(strat_dir, out_dir)
    
    # Verify the file exists and has the right extension
    assert os.path.exists(bundle_path)
    assert bundle_path.endswith("my_strategy.strat")
    
    # Open the zip and verify its contents
    with zipfile.ZipFile(bundle_path, 'r') as archive:
        contents = archive.namelist()
        assert len(contents) == 3
        assert "manifest.json" in contents
        assert "context.py" in contents
        assert "model.py" in contents