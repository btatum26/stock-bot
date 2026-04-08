"""
FeatureManager — owns the lifecycle of every active chart feature.

Responsibilities:
  - add / remove / update features
  - own active_features and sub_plots dicts
  - build the features_list and color_prefs for manifest serialization
  - coordinate overlay and sub-plot management

Emits sync_needed when the manifest should be re-written (param change or
feature removal). ChartWindow connects to this signal and decides whether to
act (respecting its own _suppress_sync flag).
"""

import logging

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QCheckBox

from ..features.base import LineOutput, LevelOutput, MarkerOutput, FeatureResult
from ..gui_components.plots import UnifiedPlot, LineOverlay, LevelOverlay
from ..colors import BG_MAIN, CHART_CROSSHAIR


class FeatureManager(QObject):
    """Manages active chart features: add, remove, update, and source-selector sync."""

    sync_needed = pyqtSignal()   # emit when manifest + context.py should be re-written

    def __init__(self, available_features, main_plot, main_plot_widget,
                 plot_splitter, feature_panel, v_lines, parent=None):
        super().__init__(parent)

        self.active_features: dict = {}   # {name: {instance, inputs, visibility,
        #                                           overlays, plot, v_line,
        #                                           widget, container, raw_data}}
        self.sub_plots: dict = {}         # {feat_name: UnifiedPlot}

        self._available     = available_features
        self._main_plot     = main_plot
        self._main_plot_widget = main_plot_widget
        self._plot_splitter = plot_splitter
        self._feature_panel = feature_panel
        self._v_lines       = v_lines    # shared list owned by ChartWindow
        self._df            = None       # kept in sync by ChartWindow.load_chart

    # -----------------------------------------------------------------------
    # DataFrame sync
    # -----------------------------------------------------------------------

    def set_df(self, df) -> None:
        self._df = df

    # -----------------------------------------------------------------------
    # Public API — feature lifecycle
    # -----------------------------------------------------------------------

    def add(self, feat_name: str, initial_values=None) -> None:
        """Add a feature to the chart. Renders immediately if df is available."""
        if feat_name in self.active_features:
            return

        feature     = self._available[feat_name]
        output_names = getattr(feature, "output_names", None)
        source_keys  = getattr(feature, "source_param_keys", [])
        col_options  = (
            {key: self._get_column_options() for key in source_keys}
            if source_keys else None
        )

        input_widgets, visibility_widgets, group_widget = self._feature_panel.create_feature_widget(
            feat_name, feature.parameters,
            lambda fn=feat_name: self._on_param_changed(fn),
            self.remove,
            initial_values=initial_values,
            output_names=output_names if output_names and len(output_names) > 1 else None,
            column_options=col_options,
        )

        plot_target      = self._main_plot
        v_line           = None
        container_widget = self._main_plot_widget
        reorganize       = False

        if feature.target_pane == "new":
            new_pw = pg.GraphicsLayoutWidget()
            new_pw.setBackground(BG_MAIN)
            new_pw.setMinimumHeight(100)
            new_pw.setContentsMargins(0, 0, 0, 0)
            self._plot_splitter.addWidget(new_pw)
            container_widget = new_pw

            new_plot = UnifiedPlot()
            if feature.y_range:
                new_plot.set_fixed_y_range(
                    feature.y_range[0], feature.y_range[1], padding=feature.y_padding)
            new_plot.setXLink(self._main_plot)
            new_plot.getAxis("left").setWidth(40)
            new_pw.addItem(new_plot)
            plot_target = new_plot
            self.sub_plots[feat_name] = new_plot

            v_line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen(CHART_CROSSHAIR, width=1, style=Qt.PenStyle.DashLine))
            new_plot.addItem(v_line, ignoreBounds=True)
            self._v_lines.append(v_line)
            reorganize = True

        self.active_features[feat_name] = {
            "instance":   feature,
            "inputs":     input_widgets,
            "visibility": visibility_widgets,
            "overlays":   [],
            "plot":       plot_target,
            "v_line":     v_line,
            "widget":     group_widget,
            "container":  container_widget,
        }

        if self._df is not None:
            self.update(feat_name)

        if reorganize:
            self._reorganize_subplots()

        self.refresh_source_selectors()

    def remove(self, feat_name: str, widget, reorganize: bool = True,
               sync: bool = True) -> None:
        """Remove a feature and clean up its overlays, sub-plot, and v-line."""
        if feat_name in self.active_features:
            data = self.active_features[feat_name]
            plot = data["plot"]

            for o in data["overlays"]:
                plot.remove_overlay(o)
            for item in data.get("temp_items", []):
                plot.removeItem(item)

            if data.get("v_line") in self._v_lines:
                self._v_lines.remove(data["v_line"])

            if data["instance"].target_pane == "new":
                data["container"].deleteLater()
                self.sub_plots.pop(feat_name, None)
                if reorganize:
                    self._reorganize_subplots()

            del self.active_features[feat_name]

        widget.deleteLater()
        self.refresh_source_selectors()
        if sync:
            self.sync_needed.emit()

    def update(self, feat_name: str) -> None:
        """Recompute and re-render a single feature using the current df."""
        df = self._df
        if df is None or df.empty:
            return
        data = self.active_features.get(feat_name)
        if not data:
            return

        feat, plot = data["instance"], data["plot"]
        params = self._read_params(data)
        aug_df = self._augment_df(df, feat_name, feat, params)

        try:
            result = feat.compute(aug_df, params)
            if isinstance(result, FeatureResult):
                results = result.visuals
                data["raw_data"] = result.data
            else:
                results = result
        except Exception as e:
            logging.error(f"Error computing {feat_name}: {e}")
            return

        for o in data["overlays"]:
            plot.remove_overlay(o)
        data["overlays"] = []

        for item in data.get("temp_items", []):
            plot.removeItem(item)
        data["temp_items"] = []

        visibility = data.get("visibility", {})
        for res in results:
            sname = getattr(res, "schema_name", "") or res.name
            if sname in visibility and not visibility[sname].isChecked():
                continue
            if isinstance(res, LineOutput):
                o = LineOverlay({res.name: res.data}, color=res.color, width=res.width)
                plot.add_overlay(o)
                data["overlays"].append(o)
            elif isinstance(res, LevelOutput):
                o = LevelOverlay(res.min_price, color=res.color)
                plot.add_overlay(o)
                data["overlays"].append(o)
            elif isinstance(res, MarkerOutput):
                sym = {"o": "o", "d": "d", "t": "t1", "s": "s", "x": "x", "+": "+"}.get(
                    res.shape, "o")
                item = pg.ScatterPlotItem(
                    x=res.indices, y=res.values,
                    brush=pg.mkBrush(res.color), symbol=sym, size=10)
                plot.addItem(item)
                data.setdefault("temp_items", []).append(item)

        plot.update_all(df)

    def update_all(self) -> None:
        """Recompute and re-render every active feature."""
        for name in list(self.active_features.keys()):
            self.update(name)

    def clear_all(self) -> None:
        """Remove all active features without emitting sync_needed."""
        for name in list(self.active_features.keys()):
            data = self.active_features[name]
            self.remove(name, data["widget"], reorganize=False, sync=False)
        self._reorganize_subplots()

    # -----------------------------------------------------------------------
    # Manifest serialization helpers
    # -----------------------------------------------------------------------

    def build_features_list(self) -> list:
        """Build a serializable features list from current widget values.

        Coerces widget text against engine defaults to prevent corrupt values
        (e.g. a leftover dict-as-string) from being written to the manifest.
        """
        features_list = []
        for data in self.active_features.values():
            feat_adapted   = data["instance"]
            engine_defaults = feat_adapted._engine.parameters
            entry_params   = {}
            for k, w in data["inputs"].items():
                if k == "color" or k.startswith("color_"):
                    continue   # GUI-only; not stored in manifest
                default_val = engine_defaults.get(k)
                if isinstance(w, QComboBox):
                    entry_params[k] = (w.currentData() or ""
                                       if w.property("is_source_selector")
                                       else w.currentText())
                elif isinstance(w, QCheckBox):
                    entry_params[k] = w.isChecked()
                else:
                    raw = w.text().strip()
                    try:
                        if isinstance(default_val, bool):
                            entry_params[k] = raw.lower() in ("true", "1", "yes")
                        elif isinstance(default_val, int):
                            entry_params[k] = int(float(raw))
                        elif isinstance(default_val, float):
                            entry_params[k] = float(raw)
                        else:
                            entry_params[k] = raw
                    except (ValueError, TypeError):
                        entry_params[k] = default_val
            features_list.append({"id": feat_adapted._id, "params": entry_params})
        return features_list

    def get_color_prefs(self) -> dict:
        """Return a color preferences dict for all active features."""
        prefs = {}
        for feat_name, data in self.active_features.items():
            feat_colors = {
                k: w.text()
                for k, w in data["inputs"].items()
                if k == "color" or k.startswith("color_")
            }
            if feat_colors:
                prefs[feat_name] = feat_colors
        return prefs

    # -----------------------------------------------------------------------
    # Source selectors
    # -----------------------------------------------------------------------

    def _get_column_options(self) -> list:
        """Return (label, col_name) tuples for source-column selectors."""
        options = [("Price (High/Low)", "")]
        for data in self.active_features.values():
            for col_name in data.get("raw_data", {}):
                options.append((col_name, col_name))
        return options

    def refresh_source_selectors(self) -> None:
        """Update column-selector dropdowns on all active features.

        Each feature's own outputs are excluded from its own source options
        so a feature cannot feed into itself.
        """
        all_options = self._get_column_options()
        for feat_name, data in self.active_features.items():
            feature     = data["instance"]
            source_keys = getattr(feature, "source_param_keys", [])
            if not source_keys:
                continue
            own_cols = set(data.get("raw_data", {}).keys())
            filtered = [opt for opt in all_options if opt[1] not in own_cols]
            for key in source_keys:
                combo = data["inputs"].get(key)
                if combo is not None:
                    self._feature_panel.refresh_column_options(combo, filtered)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _on_param_changed(self, feat_name: str) -> None:
        """Called by each feature widget when any parameter changes."""
        self.update(feat_name)
        self.sync_needed.emit()

    def _read_params(self, data: dict) -> dict:
        return {
            k: (w.currentData()
                if isinstance(w, QComboBox) and w.property("is_source_selector")
                else w.currentText() if isinstance(w, QComboBox)
                else w.isChecked()   if isinstance(w, QCheckBox)
                else w.text())
            for k, w in data["inputs"].items()
        }

    def _augment_df(self, df, feat_name: str, feat, params: dict):
        """Extend df with any source columns that live in other features' raw_data."""
        source_keys = getattr(feat, "source_param_keys", [])
        if not source_keys:
            return df
        extra = {}
        for k in source_keys:
            col = params.get(k) or None
            if col and col not in df.columns:
                for other_name, other_data in self.active_features.items():
                    if other_name != feat_name and col in other_data.get("raw_data", {}):
                        extra[col] = other_data["raw_data"][col]
                        break
        if not extra:
            return df
        aug = df.copy()
        for col, series in extra.items():
            aug[col] = series
        return aug

    def _reorganize_subplots(self) -> None:
        """Distribute vertical space evenly between the main plot and sub-plots."""
        count = self._plot_splitter.count()
        if count <= 1:
            return
        num_features = count - 1
        total = 10000
        if num_features * 2000 <= 6000:
            feat_size = 2000
            main_size = total - num_features * feat_size
        else:
            main_size = 4000
            feat_size = 6000 // num_features
        self._plot_splitter.setSizes([main_size] + [feat_size] * num_features)
