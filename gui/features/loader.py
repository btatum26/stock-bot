"""Feature loader for the GUI.

Loads all features from the engine's global registry and wraps them
with the GUI adapter so they produce visual outputs (LineOutput, etc.)
from engine-computed data.
"""

from .engine_adapter import load_engine_features


def load_features():
    """Load all available features for GUI use.

    Returns:
        Dict[str, AdaptedFeature]: Features keyed by display name.
    """
    return load_engine_features()
