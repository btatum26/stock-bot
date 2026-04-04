"""Adapter that bridges engine features to the GUI rendering pipeline.

The engine computes raw data (pd.Series, levels, zones, heatmaps) and declares
structural output_schema. This adapter wraps each engine feature so the GUI sees
the same interface it always has (target_pane, y_range, compute → FeatureResult
with visuals), but computation is delegated entirely to the engine.

Colors, line widths, and other visual config come from a simple theme table —
never from the engine.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from engine.core.features.base import (
    Feature as EngineFeature,
    FeatureResult as EngineResult,
    OutputType,
    Pane,
    FEATURE_REGISTRY,
)
from engine.core.features.features import load_features as engine_load_features, FeatureCache

from .base import (
    Feature as GUIFeature,
    FeatureResult as GUIResult,
    LineOutput,
    LevelOutput,
    MarkerOutput,
    HeatmapOutput,
)

# ---------------------------------------------------------------------------
# Theme — default rendering config per category/output type
# ---------------------------------------------------------------------------
# Palette of colors cycled for multi-line features
_PALETTE = ["#aaff00", "#00bfff", "#ff6b6b", "#ffd700", "#da70d6", "#00fa9a"]

_CATEGORY_COLORS = {
    "Oscillators (Momentum)": "#aaff00",
    "Volatility":             "#00bfff",
    "Trend":                  "#ffd700",
    "Volume":                 "#da70d6",
    "Price Levels":           "#ff6b6b",
}

_LEVEL_COLORS = {
    "Overbought": "#ff4444",
    "Oversold":   "#44ff44",
}

_DEFAULT_LINE_WIDTH = 1.5


def _color_for_level(label: str) -> str:
    """Pick a color for a level based on its label."""
    label_lower = label.lower()
    if "overbought" in label_lower or "high" in label_lower:
        return "#ff4444"
    if "oversold" in label_lower or "low" in label_lower:
        return "#44ff44"
    return "#888888"


def _color_for_feature(engine_feat: EngineFeature, idx: int = 0) -> str:
    """Pick a default line color for a feature output."""
    cat_color = _CATEGORY_COLORS.get(engine_feat.category)
    if cat_color and idx == 0:
        return cat_color
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Adapted Feature — wraps one engine feature for the GUI
# ---------------------------------------------------------------------------
class AdaptedFeature(GUIFeature):
    """Wraps an engine Feature so the GUI can use it transparently.

    The GUI calls `compute(df, params)` and gets back a FeatureResult with
    visuals (LineOutput, LevelOutput, etc.) built automatically from the
    engine feature's output_schema and computed data.
    """

    def __init__(self, feature_id: str, engine_feat: EngineFeature):
        self._id = feature_id
        self._engine = engine_feat

    # --- Properties the GUI reads ---

    @property
    def name(self) -> str:
        return self._engine.name

    @property
    def description(self) -> str:
        return self._engine.description

    @property
    def category(self) -> str:
        return self._engine.category

    @property
    def parameters(self) -> Dict[str, Any]:
        base = dict(self._engine.parameters)
        # Inject per-output color params for LINE outputs.
        # Single-line features get a plain "color"; multi-line features get
        # "color_<output_name>" for each LINE so the user can control each
        # line independently.
        line_outputs = [s for s in self._engine.output_schema
                        if s.output_type == OutputType.LINE]
        if len(line_outputs) == 1:
            if "color" not in base:
                base["color"] = _color_for_feature(self._engine)
        else:
            for i, schema in enumerate(line_outputs):
                key = f"color_{schema.name}" if schema.name else "color"
                if key not in base:
                    base[key] = _color_for_feature(self._engine, i)
        return base

    @property
    def output_names(self) -> List[str]:
        """Short schema names for all renderable outputs (LINE, HISTOGRAM, MARKER)."""
        return [
            s.name for s in self._engine.output_schema
            if s.output_type in (OutputType.LINE, OutputType.HISTOGRAM, OutputType.MARKER)
            and s.name
        ]

    @property
    def target_pane(self) -> str:
        """If ANY output lives on a NEW pane, the feature gets its own subplot."""
        for schema in self._engine.output_schema:
            # Only count series types (LINE, HISTOGRAM, MARKER) — not LEVEL/BAND
            if schema.output_type in (OutputType.LINE, OutputType.HISTOGRAM, OutputType.MARKER):
                if schema.pane == Pane.NEW:
                    return "new"
        return "main"

    @property
    def y_range(self) -> Optional[List[float]]:
        for schema in self._engine.output_schema:
            if schema.y_range is not None:
                return list(schema.y_range)
        return None

    @property
    def source_param_keys(self) -> List[str]:
        """Parameter keys that are dynamic column selectors (e.g. high_col, low_col)."""
        return getattr(self._engine, 'source_param_keys', [])

    @property
    def y_padding(self) -> float:
        return 0.05 if self.y_range else 0.0

    # --- Computation ---

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> GUIResult:
        """Run the engine feature and translate the result into GUI visuals."""
        # Strip GUI-only params before passing to engine
        engine_params = {k: v for k, v in params.items()
                         if k not in ("color",) and not k.startswith("color_")}

        # Strip source-selector params from column-name generation so that
        # selecting "RSI" vs "Price" doesn't change the output column names.
        _source_keys = set(getattr(self._engine, 'source_param_keys', []))
        name_params = {k: v for k, v in engine_params.items() if k not in _source_keys}

        # Compute via engine with a shared cache for dependency optimization
        cache = FeatureCache()
        engine_result: EngineResult = self._engine.compute(df, engine_params, cache)

        visuals = []
        # Determine whether we're in single-color or per-output-color mode
        line_outputs = [s for s in self._engine.output_schema
                        if s.output_type == OutputType.LINE]
        multi_color = len(line_outputs) > 1

        # --- Build visuals from output_schema ---
        line_idx = 0
        for schema in self._engine.output_schema:

            if schema.output_type == OutputType.LINE:
                col_name = self._engine.generate_column_name(self._id, name_params, schema.name)
                series = (engine_result.data or {}).get(col_name)
                if series is not None:
                    if multi_color:
                        color_key = f"color_{schema.name}" if schema.name else "color"
                        color = params.get(color_key, _color_for_feature(self._engine, line_idx))
                    else:
                        color = params.get("color", _color_for_feature(self._engine))
                    visuals.append(LineOutput(
                        name=col_name,
                        data=series.where(pd.notnull(series), None).tolist(),
                        color=color,
                        width=_DEFAULT_LINE_WIDTH,
                        schema_name=schema.name,
                    ))
                    line_idx += 1

            elif schema.output_type == OutputType.HISTOGRAM:
                col_name = self._engine.generate_column_name(self._id, name_params, schema.name)
                series = (engine_result.data or {}).get(col_name)
                if series is not None:
                    # Render histogram as a line for now — the GUI can upgrade
                    # to a bar renderer later without changing the engine
                    visuals.append(LineOutput(
                        name=col_name,
                        data=series.where(pd.notnull(series), None).tolist(),
                        color="#888888",
                        width=1.0,
                        schema_name=schema.name,
                    ))

            elif schema.output_type == OutputType.MARKER:
                col_name = self._engine.generate_column_name(self._id, name_params, schema.name)
                series = (engine_result.data or {}).get(col_name)
                if series is not None:
                    non_null = series[series.astype(bool)]
                    if not non_null.empty:
                        # Map index positions to integer x coords
                        idx_positions = [df.index.get_loc(i) for i in non_null.index
                                         if i in df.index]
                        values = non_null.tolist()[:len(idx_positions)]
                        marker_color = params.get("color", _color_for_feature(self._engine))
                        visuals.append(MarkerOutput(
                            name=col_name,
                            indices=idx_positions,
                            values=values,
                            color=marker_color,
                            shape="t" if any(v > 0 for v in values) else "d",
                            schema_name=schema.name,
                        ))

            elif schema.output_type == OutputType.LEVEL:
                if engine_result.levels:
                    for level in engine_result.levels:
                        price = level.get("value", 0)
                        label = level.get("label", "")
                        visuals.append(LevelOutput(
                            name=label,
                            price=price,
                            min_price=level.get("min_price", price),
                            max_price=level.get("max_price", price),
                            strength=level.get("strength", 1.0),
                            color=_color_for_level(label),
                        ))

            elif schema.output_type == OutputType.BAND:
                # Bands are a visual instruction — the two boundary lines are
                # already rendered as LINE outputs above. The GUI could add a
                # fill-between later using band_pair to find the two lines.
                pass

            elif schema.output_type == OutputType.HEATMAP:
                if engine_result.heatmaps:
                    for hm_name, hm_data in engine_result.heatmaps.items():
                        visuals.append(HeatmapOutput(
                            name=hm_name,
                            price_grid=hm_data.get("price_grid", []),
                            density=hm_data.get("intensity", []),
                            color_map="viridis",
                        ))

            elif schema.output_type == OutputType.ZONE:
                # Zones need a ZoneOverlay in the GUI — skip for now, data is
                # still available in engine_result.zones for future rendering
                pass

        # Raw data for scoring/ML
        raw_data = engine_result.data if engine_result.data else {}

        return GUIResult(visuals=visuals, data=raw_data)


# ---------------------------------------------------------------------------
# Loader — replaces the old gui/features/loader.py discovery
# ---------------------------------------------------------------------------
def load_engine_features() -> Dict[str, AdaptedFeature]:
    """Load all engine features and wrap them for GUI consumption.

    Returns:
        Dict keyed by feature display name → AdaptedFeature instance.
    """
    engine_load_features()

    features = {}
    for feature_id, feature_cls in FEATURE_REGISTRY.items():
        engine_instance = feature_cls()
        adapted = AdaptedFeature(feature_id, engine_instance)
        features[adapted.name] = adapted

    return features
