import ast
import os
import zipfile
from typing import Set

class StrategyValidationError(Exception):
    """Raised when a strategy fails AST static analysis or structural checks."""
    pass

class StrategyValidator(ast.NodeVisitor):
    """
    Parses and validates the abstract syntax tree of a user's model.py.
    Enforces a strict allowlist of imports and verifies the existence of 
    required classes and methods.
    """
    
    # Broad allowlist for project-supported strategy code plus local helpers.
    ALLOWED_MODULES: Set[str] = {
        "pandas",
        "numpy",
        "typing",
        "math",
        "datetime",
        "context",
        "engine",
        "xgboost",
        "sklearn",
        "regime",
        "confirmations",
    }

    BLOCKED_MODULES: Set[str] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pathlib",
        "urllib",
        "requests",
    }

    def __init__(self):
        self.has_context_import = False
        self.has_signal_model = False
        self.has_generate_signals = False

    def _check_module_allowed(self, module_name: str, lineno: int):
        """Validates a module against the allowlist, checking the root package."""
        if not module_name:
            return
            
        # Extract the root package (e.g., 'pandas' from 'pandas.core')
        root_module = module_name.split('.')[0]
        
        if root_module in self.BLOCKED_MODULES or root_module not in self.ALLOWED_MODULES:
            raise StrategyValidationError(
                f"Security Violation on line {lineno}: "
                f"Importing '{root_module}' is not permitted. Allowed modules: {', '.join(self.ALLOWED_MODULES)}"
            )

    def visit_Import(self, node: ast.Import):
        """Intercepts `import x` statements."""
        for alias in node.names:
            self._check_module_allowed(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Intercepts `from x import y` statements."""
        if node.module:
            self._check_module_allowed(node.module, node.lineno)
            
            # Check if they are properly importing the Context
            if node.module == 'context':
                for alias in node.names:
                    if alias.name == 'Context':
                        self.has_context_import = True
                        
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Intercepts class definitions to find SignalModel and its methods."""
        base_names = {
            getattr(base, "id", None) or getattr(base, "attr", None)
            for base in node.bases
        }
        if node.name == "SignalModel" or "SignalModel" in base_names:
            self.has_signal_model = True
            
            # Scan the body of the class for the required method
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "generate_signals":
                    self.has_generate_signals = True
                    
        self.generic_visit(node)

    def validate(self, source_code: str):
        """
        Executes the full AST validation pipeline.
        
        Raises:
            StrategyValidationError: If security or structural checks fail.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise StrategyValidationError(f"Syntax Error in model.py on line {e.lineno}: {e.msg}")

        # Walk the syntax tree (this triggers the visit_* methods above)
        self.visit(tree)

        # Post-Walk Structural Verification
        if not self.has_context_import:
            raise StrategyValidationError("Missing required import: `from context import Context`")
        
        if not self.has_signal_model:
            raise StrategyValidationError("Missing required class: `class SignalModel:` or SignalModel subclass.")
            
        if not self.has_generate_signals:
            raise StrategyValidationError("Missing required method: `def generate_signals(self, df, ctx, artifacts):` inside SignalModel")


class Bundler:
    """The Exporter: Validates and zips the strategy directory into a .strat bundle."""
    
    @staticmethod
    def export(strategy_dir: str, output_dir: str) -> str:
        """
        Validates the strategy source code and packages it into a deployment artifact.

        Args:
            strategy_dir (str): Path to the raw strategy files.
            output_dir (str): Destination directory for the .strat archive.

        Returns:
            str: The absolute path to the generated .strat bundle.

        Raises:
            FileNotFoundError: If core files are missing.
            StrategyValidationError: If model.py contains malicious imports or lacks required structure.
        """
        required_files = ["manifest.json", "context.py", "model.py"]
        for file in required_files:
            if not os.path.exists(os.path.join(strategy_dir, file)):
                raise FileNotFoundError(f"Required file '{file}' not found in {strategy_dir}")
                
        # Perform AST Static Analysis on model.py
        model_path = os.path.join(strategy_dir, "model.py")
        with open(model_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
        validator = StrategyValidator()
        validator.validate(source_code)

        # Runtime contract check: the strategy must import and expose a concrete
        # engine.core.controller.SignalModel subclass, not just a matching name.
        from .backtester import LocalBacktester
        LocalBacktester(strategy_dir)._load_user_model_and_context()
        
        # Create the Bundle
        strat_name = os.path.basename(os.path.normpath(strategy_dir))
        os.makedirs(output_dir, exist_ok=True)
        bundle_path = os.path.join(output_dir, f"{strat_name}.strat")
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(os.path.join(strategy_dir, "manifest.json"), arcname="manifest.json")
            for root, dirs, files in os.walk(strategy_dir):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    filepath = os.path.join(root, filename)
                    arcname = os.path.relpath(filepath, strategy_dir)
                    bundle.write(filepath, arcname=arcname)
                    
        return bundle_path
