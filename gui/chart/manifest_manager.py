"""
ManifestManager — reads and writes manifest.json and gui_prefs.json for a strategy.

No Qt dependencies; all widget-value extraction is done by FeatureManager before
calling sync(), keeping this class focused on filesystem I/O only.
"""

import json
import logging
import os

from engine import ModelEngine


class ManifestManager:
    """Handles manifest.json and gui_prefs.json I/O for chart-tab strategies."""

    def __init__(self, engine: ModelEngine):
        self._engine = engine

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------

    def _strategy_dir(self, name: str) -> str:
        return os.path.join(self._engine.workspace_dir, name)

    def color_prefs_path(self, strategy_name: str) -> str:
        return os.path.join(self._strategy_dir(strategy_name), "gui_prefs.json")

    # -----------------------------------------------------------------------
    # Color preferences
    # -----------------------------------------------------------------------

    def load_color_prefs(self, strategy_name: str) -> dict:
        try:
            with open(self.color_prefs_path(strategy_name)) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logging.warning(f"Could not read color prefs for {strategy_name}: {e}")
            return {}

    def save_color_prefs(self, strategy_name: str, prefs: dict) -> None:
        if not strategy_name or strategy_name == "(none)":
            return
        try:
            with open(self.color_prefs_path(strategy_name), "w") as f:
                json.dump(prefs, f, indent=4)
        except Exception as e:
            logging.warning(f"Failed to save color prefs for {strategy_name}: {e}")

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------

    def load_manifest(self, strategy_name: str) -> dict:
        path = os.path.join(self._strategy_dir(strategy_name), "manifest.json")
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logging.error(f"Could not read manifest for {strategy_name}: {e}")
            return {}

    def sync(self, strategy_name: str, features_list: list) -> str:
        """Update manifest.json with *features_list* and regenerate context.py.

        Args:
            strategy_name: The strategy directory name.
            features_list: Pre-built list from FeatureManager.build_features_list().

        Returns:
            A short status string suitable for display in the UI.
        """
        if not strategy_name or strategy_name == "(none)":
            return ""

        strategy_dir = self._strategy_dir(strategy_name)
        manifest_path = os.path.join(strategy_dir, "manifest.json")

        manifest = self.load_manifest(strategy_name)
        manifest["features"] = features_list

        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
        except Exception as e:
            return f"Manifest write failed: {e}"

        hparams = manifest.get("hyperparameters", {})
        try:
            self._engine.write_context_py(strategy_name, features_list, hparams)
        except Exception as e:
            logging.error(f"context.py write failed for {strategy_name}: {e}")
            return f"context.py update failed: {e}"

        return "Synced."
