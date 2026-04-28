import os
import re
import json
import keyword
from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader

from .features.base import FEATURE_REGISTRY
from .exceptions import ValidationError
from .features.features import load_features

load_features()  # Ensure all features are registered before any workspace operations
class WorkspaceManager:
    """Manages the synchronization of strategy configuration with local workspace files.

    This class handles validating user configurations, updating the strategy's
    manifest.json, and generating strictly-typed context.py and model.py files
    using Jinja2 templates.
    """
    
    def __init__(self, strategy_dir: str, template_dir: str = None):
        """Initializes the WorkspaceManager.

        Args:
            strategy_dir (str): The directory path of the active strategy.
            template_dir (str, optional): The directory path containing Jinja2 templates. 
                Defaults to "engine/core/templates".
        """
        self.strategy_dir = strategy_dir
        self.manifest_path = os.path.join(strategy_dir, "manifest.json")
        self.context_path = os.path.join(strategy_dir, "context.py")
        self.model_path = os.path.join(strategy_dir, "model.py")

        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), "templates")

        # Initialize Jinja2 Environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _infer_type(self, value: Any) -> str:
        """Maps JSON/Python runtime types to type hint strings.

        Args:
            value (Any): The hyperparameter value to infer the type for.

        Returns:
            str: A string representing the Python type hint (e.g., 'int', 'float', 'str').
        """
        if isinstance(value, bool): return "bool"
        if isinstance(value, int): return "int"
        if isinstance(value, float): return "float"
        if isinstance(value, str): return "str"
        if isinstance(value, list): return "list"
        if isinstance(value, dict): return "dict"
        return "Any"

    def _build_features_payload(self, features_config: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generates the flat naming payload for features, handling multi-outputs.

        Filters out UI/Engine-specific parameters, standardizes the base attribute name,
        and appends output suffixes for multi-output features to ensure a flat, 
        collision-free namespace.

        Args:
            features_config (List[Dict[str, Any]]): The list of feature configurations 
                from the strategy manifest.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing 'attr_name', 'col_name', 
                and 'docstring' keys for rendering in the Jinja2 template.
        """
        payload = []
        for config in features_config:
            fid = config.get("id")
            params = config.get("params", {})
            
            if fid not in FEATURE_REGISTRY:
                continue
                
            feature_instance = FEATURE_REGISTRY[fid]()
            outputs = feature_instance.outputs
            
            # Filter out UI/Engine parameters
            core_params = {k: v for k, v in params.items() if k not in ["color", "normalize", "overbought", "oversold"] and not k.startswith("color_")} # This may not need to be here anymore, but it serves as a safeguard against non-core params affecting naming
            
            # Strip redundant 'type' to match base.py exactly <-- Added this
            if "type" in core_params and str(core_params["type"]).upper() == fid.upper():
                del core_params["type"]
                
            # Generate the Base Attribute Name
            if len(core_params) == 1 and ("period" in core_params or "window" in core_params):
                val = core_params.get("period") or core_params.get("window")
                base_attr = f"{fid.upper()}_{val}"
            elif not core_params:
                base_attr = fid.upper()
            else:
                param_str = "_".join([str(v) for k, v in sorted(core_params.items())])
                # Sanitize: replace any character that isn't alphanumeric or _ with _
                safe_str  = re.sub(r'[^A-Za-z0-9_]', '_', param_str)
                base_attr = f"{fid.upper()}_{safe_str}"
                
            # Iterate over outputs to handle MACD, Bollinger Bands, etc.
            for output in outputs:
                col_name = feature_instance.generate_column_name(fid, params, output)
                
                if output:
                    suffix = output.upper().replace(" ", "_").replace("-", "_")
                    attr_name = f"{base_attr}_{suffix}"
                else:
                    attr_name = base_attr
                    
                payload.append({
                    "attr_name": attr_name,
                    "col_name": col_name,
                    "docstring": f"Feature: {feature_instance.name}\n    Outputs: {output or 'Primary'}\n    Params: {params}"
                })
        return payload

    def sync(self, features: List[Dict[str, Any]], hparams: Dict[str, Any], bounds: Dict[str, Any]):
        """Updates manifest and generates synchronized context/model files using Jinja2.

        Validates hyperparameters against Python keywords, writes the updated 
        configuration to manifest.json, constructs the template payload, and renders 
        both context.py and model.py safely.

        Args:
            features (List[Dict[str, Any]]): List of feature configurations.
            hparams (Dict[str, Any]): Dictionary of strategy hyperparameters.
            bounds (Dict[str, Any]): Dictionary of parameter bounds for optimization.

        Raises:
            ValueError: If a hyperparameter key is a reserved Python keyword.
        """
        
        # Validate Hyperparameters (Block Python Keywords)
        for key in hparams.keys():
            if keyword.iskeyword(key):
                raise ValidationError(f"Hyperparameter '{key}' is a reserved Python keyword and cannot be used.")
        
        # Update manifest.json
        manifest = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)

        manifest["features"] = features
        manifest["hyperparameters"] = hparams
        manifest["parameter_bounds"] = bounds

        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)

        # Build Jinja Context Payload
        params_payload = [
            {"name": k, "type": self._infer_type(v), "value": repr(v)}
            for k, v in hparams.items()
        ]
        
        template_data = {
            "features": self._build_features_payload(features),
            "params": params_payload
        }

        # Render and Write context.py (Always Overwrite)
        context_template = self.jinja_env.get_template("context.py.j2")
        rendered_context = context_template.render(**template_data)
        
        with open(self.context_path, 'w') as f:
            f.write(rendered_context)

        # Render and Write model.py (Only if it doesn't exist)
        if not os.path.exists(self.model_path):
            model_template = self.jinja_env.get_template("model.py.j2")
            rendered_model = model_template.render()
            with open(self.model_path, 'w') as f:
                f.write(rendered_model)
