import sys
import os
import json
import logging
import random

import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QComboBox, QCheckBox, QSplitter, QTabWidget, QScrollArea, QFrame,
    QGroupBox, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import pyqtgraph as pg

from engine import ModelEngine

from .features.loader import load_features
from .features.base import LineOutput, LevelOutput, MarkerOutput, FeatureResult

# Modular Components
from .gui_components.styling import setup_app_style
from .gui_components.controls import ControlBar
from .gui_components.feature_panel import FeaturePanel
from .gui_components.plots import (
    UnifiedPlot, CandleOverlay, VolumeOverlay, LineOverlay, LevelOverlay,
)
from .gui_components.backtest_panel import BacktestPanel


class SignalPreviewWorker(QThread):
    """Runs LocalBacktester.run_batch in a background thread for signal preview."""
    finished = pyqtSignal(object)   # emits pd.Series or None
    error    = pyqtSignal(str)

    def __init__(self, strategy_dir: str, ticker: str, df: pd.DataFrame):
        super().__init__()
        self._strategy_dir = strategy_dir
        self._ticker = ticker
        self._df = df.copy()

    def run(self):
        try:
            from engine.core.backtester import LocalBacktester
            bt = LocalBacktester(self._strategy_dir)
            # Backtester handles both Title-case and lowercase columns
            batch = bt.run_batch({self._ticker: self._df})
            signals = batch.get(self._ticker)
            self.finished.emit(signals)
        except KeyError as e:
            self.error.emit(
                f"Column not found: {e}\n"
                "Check that context.py matches the manifest features."
            )
        except Exception as e:
            import traceback
            self.error.emit(
                f"{type(e).__name__}: {e}\n\n"
                f"{traceback.format_exc().strip()}"
            )


class ChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Bot Pro")
        self.resize(1400, 900)
        setup_app_style(self)

        self.engine = ModelEngine(workspace_dir="./strategies", db_path="data/stocks.db")
        self.available_features = load_features()
        self.active_features = {}   # {name: {instance, inputs, overlays, plot, v_line, widget, container}}

        # Chart state
        self.df = None
        self.timestamps = []
        self._tick_intraday: list = []   # '%H:%M\n%m-%d' per bar
        self._tick_daily: list    = []   # '%Y-%m-%d'     per bar
        self._mouse_ts_labels: list  = []   # '%Y-%m-%d %H:%M' per bar
        self._mouse_vol_labels: list = []   # 'Vol: 1.23M'   per bar
        self.sub_plots = {}    # {feat_name: UnifiedPlot}
        self.v_lines = []

        self._init_ui()
        self.load_random()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top-level tab switcher
        self.top_tabs = QTabWidget()
        self.top_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.top_tabs.setStyleSheet("""
            QTabBar::tab           { padding: 8px 20px; font-size: 13px; }
            QTabBar::tab:selected  { font-weight: bold; color: #aaff00; border-bottom: 2px solid #aaff00; }
        """)
        self.top_tabs.currentChanged.connect(self._on_tab_changed)
        root_layout.addWidget(self.top_tabs)

        # Tab 0: Chart
        self.top_tabs.addTab(self._build_chart_tab(), "Chart")

        # Tab 1: Backtest
        self.backtest_panel = BacktestPanel(self.engine, self)
        self.top_tabs.addTab(self.backtest_panel, "Backtest")

        self.showMaximized()

    # --- Chart tab --------------------------------------------------------

    def _build_chart_tab(self) -> QWidget:
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)

        # Debounce timer — fires load_chart 800 ms after the user stops typing
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self.load_chart)

        # Control bar
        self.controls = ControlBar(self)
        self.controls.ticker_input.returnPressed.connect(self.load_chart)
        self.controls.ticker_input.textChanged.connect(lambda: self._load_timer.start(800))
        self.controls.ticker_history.currentIndexChanged.connect(self._load_from_history)
        self.controls.interval_combo.currentIndexChanged.connect(self.load_chart)
        self.controls.period_combo.currentIndexChanged.connect(self._apply_period_view)
        self.controls.btn_random.clicked.connect(self.load_random)
        chart_layout.addWidget(self.controls)

        # Horizontal splitter: chart area | features sidebar
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.setHandleWidth(6)
        chart_layout.addWidget(h_splitter)

        # Chart area (vertical splitter of main plot + sub-plots)
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot_splitter.setHandleWidth(6)

        self.main_plot_widget = pg.GraphicsLayoutWidget()
        self.main_plot_widget.setBackground('#1e1e1e')
        self.main_plot_widget.setContentsMargins(0, 0, 0, 0)
        self.plot_splitter.addWidget(self.main_plot_widget)

        self.main_plot = UnifiedPlot()
        self.main_plot_widget.addItem(self.main_plot)

        self.price_overlay  = CandleOverlay()
        self.volume_overlay = VolumeOverlay()
        self.main_plot.add_overlay(self.price_overlay)
        self.main_plot.add_overlay(self.volume_overlay)

        self.main_plot.getAxis('left').setPen('#444')
        self.main_plot.getAxis('left').setWidth(40)
        self.main_plot.getAxis('bottom').setPen('#444')

        h_splitter.addWidget(self.plot_splitter)

        # Right sidebar: strategy selector on top, features below
        sidebar = QWidget()
        sidebar.setMinimumWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # --- Strategy selector ---
        strat_box = QGroupBox("Strategy")
        strat_box.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 6px; }")
        strat_layout = QVBoxLayout(strat_box)
        strat_layout.setContentsMargins(6, 6, 6, 6)
        strat_layout.setSpacing(4)

        self.chart_strategy_combo = QComboBox()
        self.chart_strategy_combo.addItem("(none)")
        self.chart_strategy_combo.currentIndexChanged.connect(self._load_strategy_features)
        strat_layout.addWidget(self.chart_strategy_combo)

        self.btn_gen_signals = QPushButton("Generate Signals")
        self.btn_gen_signals.setFixedHeight(26)
        self.btn_gen_signals.setToolTip("Run strategy signals on current chart data")
        strat_layout.addWidget(self.btn_gen_signals)

        self.lbl_strat_status = QLabel("")
        self.lbl_strat_status.setStyleSheet("color: #888; font-size: 10px; padding: 0 2px;")
        self.lbl_strat_status.setWordWrap(True)
        strat_layout.addWidget(self.lbl_strat_status)

        sidebar_layout.addWidget(strat_box)

        self.btn_gen_signals.clicked.connect(self._generate_signals_preview)
        self._signal_preview_worker = None
        self._signal_preview_items = []  # scatter items on main_plot
        self._suppress_sync = False

        # --- Feature panel ---
        self.feature_panel = FeaturePanel(self.available_features, self)
        self.feature_panel.btn_add_feat.clicked.connect(self.add_feature_ui)

        feat_scroll = QScrollArea()
        feat_scroll.setWidgetResizable(True)
        feat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        feat_scroll.setWidget(self.feature_panel)
        sidebar_layout.addWidget(feat_scroll)

        h_splitter.addWidget(sidebar)

        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 0)

        self._setup_crosshair()
        self._refresh_strategy_combo()
        return chart_widget

    # -----------------------------------------------------------------------
    # Tab switching
    # -----------------------------------------------------------------------

    def _on_tab_changed(self, index: int):
        if index == 1:
            self.backtest_panel.refresh_strategies()
        elif index == 0:
            self._refresh_strategy_combo()

    def _refresh_strategy_combo(self):
        """Repopulate the chart-tab strategy combo from the workspace directory."""
        current = self.chart_strategy_combo.currentText()
        self.chart_strategy_combo.blockSignals(True)
        self.chart_strategy_combo.clear()
        self.chart_strategy_combo.addItem("(none)")
        try:
            for name in self.engine.list_strategies():
                self.chart_strategy_combo.addItem(name)
        except Exception:
            pass
        idx = self.chart_strategy_combo.findText(current)
        self.chart_strategy_combo.setCurrentIndex(max(0, idx))
        self.chart_strategy_combo.blockSignals(False)

    # -----------------------------------------------------------------------
    # Strategy feature loading
    # -----------------------------------------------------------------------

    def _load_strategy_features(self):
        """Read the selected strategy's manifest and add its features to the chart."""
        name = self.chart_strategy_combo.currentText()
        self._clear_all_features()
        if not name or name == "(none)":
            return

        manifest_path = os.path.join(self.engine.workspace_dir, name, "manifest.json")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as e:
            self.lbl_strat_status.setText(f"Could not read manifest: {e}")
            return

        features_cfg = manifest.get("features", [])
        if not features_cfg:
            self.lbl_strat_status.setText("No features in manifest.")
            return

        # Build reverse lookup: engine_id → display_name
        # available_features is keyed by display name; each value has ._id = engine registry key
        by_engine_id = {v._id: k for k, v in self.available_features.items()}
        color_prefs = self._load_color_prefs(name)

        loaded, skipped = [], []
        for entry in features_cfg:
            feat_id = entry.get("id", "")
            params  = entry.get("params", {})

            display_name = by_engine_id.get(feat_id)
            if display_name is None:
                skipped.append(feat_id)
                continue

            initial_values = {k: str(v) for k, v in params.items()}
            initial_values.update(color_prefs.get(display_name, {}))
            self._add_feature_by_name(display_name, initial_values=initial_values)
            loaded.append(display_name)

        parts = []
        if loaded:
            parts.append(f"Loaded: {', '.join(loaded)}")
        if skipped:
            parts.append(f"Skipped: {', '.join(skipped)}")
        self.lbl_strat_status.setText(" | ".join(parts))

    def _clear_all_features(self):
        """Remove every active feature from the chart without triggering manifest sync."""
        self._suppress_sync = True
        for name in list(self.active_features.keys()):
            data = self.active_features[name]
            self.remove_feature(name, data["widget"], reorganize=False)
        self._suppress_sync = False
        self._reorganize_subplots()
        self.lbl_strat_status.setText("")

    def _color_prefs_path(self, strategy_name: str) -> str:
        return os.path.join(self.engine.workspace_dir, strategy_name, "gui_prefs.json")

    def _load_color_prefs(self, strategy_name: str) -> dict:
        try:
            with open(self._color_prefs_path(strategy_name)) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_color_prefs(self):
        name = self.chart_strategy_combo.currentText()
        if not name or name == "(none)":
            return
        prefs = {}
        for feat_name, data in self.active_features.items():
            feat_colors = {k: w.text() for k, w in data["inputs"].items()
                           if k == "color" or k.startswith("color_")}
            if feat_colors:
                prefs[feat_name] = feat_colors
        try:
            with open(self._color_prefs_path(name), "w") as f:
                json.dump(prefs, f, indent=4)
        except Exception:
            pass

    def _sync_strategy_manifest(self):
        """Write active features back to manifest.json and regenerate context.py."""
        if self._suppress_sync:
            return
        strat_name = self.chart_strategy_combo.currentText()
        if not strat_name or strat_name == "(none)":
            return

        strategy_dir = os.path.join(self.engine.workspace_dir, strat_name)
        manifest_path = os.path.join(strategy_dir, "manifest.json")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

        # Build feature list from active widgets.
        # Params are coerced against the engine feature's defaults to prevent
        # corrupt widget values (e.g. leftover dict-as-string) being written back.
        features_list = []
        for data in self.active_features.values():
            feat_adapted = data["instance"]
            engine_defaults = feat_adapted._engine.parameters
            entry_params = {}
            for k, w in data["inputs"].items():
                if k == "color" or k.startswith("color_"):
                    continue  # GUI-only display param — not saved to manifest
                default_val = engine_defaults.get(k)
                if isinstance(w, QComboBox):
                    if w.property("is_source_selector"):
                        entry_params[k] = w.currentData() or ""
                    else:
                        entry_params[k] = w.currentText()
                elif isinstance(w, QCheckBox):
                    entry_params[k] = w.isChecked()
                else:
                    raw = w.text().strip()
                    try:
                        if isinstance(default_val, bool):
                            entry_params[k] = raw.lower() in ('true', '1', 'yes')
                        elif isinstance(default_val, int):
                            entry_params[k] = int(float(raw))
                        elif isinstance(default_val, float):
                            entry_params[k] = float(raw)
                        else:
                            entry_params[k] = raw
                    except (ValueError, TypeError):
                        # Widget value is corrupt — fall back to feature default
                        entry_params[k] = default_val
            features_list.append({"id": feat_adapted._id, "params": entry_params})

        # Preserve everything else in the manifest
        manifest["features"] = features_list
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
        except Exception as e:
            self.lbl_strat_status.setText(f"Manifest write failed: {e}")
            return

        # Regenerate context.py with short, stable attribute names
        try:
            self._write_context_py(strategy_dir, features_list,
                                   manifest.get("hyperparameters", {}))
            self.lbl_strat_status.setText("Synced.")
        except Exception as e:
            self.lbl_strat_status.setText(f"context.py update failed: {e}")
            logging.error(f"context.py write failed: {e}")

    @staticmethod
    def _write_context_py(strategy_dir: str, features_list: list, hparams: dict):
        """Write a clean context.py using short, stable attribute names.

        Attribute naming rules:
          - Single unnamed output (e.g. RSI):  attr = col_name  (e.g. RSI_14)
          - Named output (e.g. fractal cols):  attr = OUTPUT_NAME_UPPER
          - Collision (two features share an output name): prefix with FEATUREID_
        """
        from engine.core.features.base import FEATURE_REGISTRY, OutputType

        DATA_TYPES = (OutputType.LINE, OutputType.HISTOGRAM, OutputType.MARKER)

        lines = [
            "# AUTO-GENERATED. Do not edit — updated by the GUI when features change.",
            "from dataclasses import dataclass, field",
            "",
            "",
            "@dataclass(frozen=True)",
            "class FeaturesContext:",
            '    """Typed mapping from strategy features to DataFrame column names."""',
        ]

        seen_attrs: set = set()
        for entry in features_list:
            fid    = entry["id"]
            params = entry.get("params", {})
            if fid not in FEATURE_REGISTRY:
                continue
            feat          = FEATURE_REGISTRY[fid]()
            data_outputs  = [s for s in feat.output_schema if s.output_type in DATA_TYPES]
            source_keys   = set(getattr(feat, 'source_param_keys', []))
            name_params   = {k: v for k, v in params.items() if k not in source_keys}

            for schema in data_outputs:
                col_name = feat.generate_column_name(fid, name_params, schema.name)
                if schema.name:
                    attr = schema.name.upper().replace(" ", "_").replace("-", "_")
                else:
                    attr = col_name   # e.g. "RSI_14"

                if attr in seen_attrs:
                    attr = f"{fid.upper().replace(' ', '_')}_{attr}"
                seen_attrs.add(attr)
                lines.append(f"    {attr}: str = '{col_name}'")

        lines += [
            "",
            "",
            "@dataclass(frozen=True)",
            "class ParamsContext:",
            '    """Typed strategy hyperparameters."""',
        ]
        for k, v in hparams.items():
            hint = "float" if isinstance(v, float) else "int" if isinstance(v, int) else "str"
            lines.append(f"    {k}: {hint} = {repr(v)}")

        lines += [
            "",
            "",
            "@dataclass(frozen=True)",
            "class Context:",
            '    """Master context object."""',
            "    features: FeaturesContext = field(default_factory=FeaturesContext)",
            "    params:   ParamsContext   = field(default_factory=ParamsContext)",
            "",
        ]

        with open(os.path.join(strategy_dir, "context.py"), "w") as f:
            f.write("\n".join(lines))

    def _generate_signals_preview(self):
        """Run the selected strategy on current chart data and overlay entry/exit markers."""
        strat_name = self.chart_strategy_combo.currentText()
        if not strat_name or strat_name == "(none)":
            self.lbl_strat_status.setText("Select a strategy first.")
            return
        if self.df is None or self.df.empty:
            self.lbl_strat_status.setText("Load chart data first.")
            return
        if self._signal_preview_worker and self._signal_preview_worker.isRunning():
            return

        ticker = self.controls.ticker_input.text().strip().upper()
        strategy_dir = os.path.join(self.engine.workspace_dir, strat_name)

        # df in ChartWindow is Title-cased; backtester accepts both
        self._clear_signal_markers()
        self.lbl_strat_status.setText("Generating signals…")
        self.btn_gen_signals.setEnabled(False)

        self._signal_preview_worker = SignalPreviewWorker(strategy_dir, ticker, self.df)
        self._signal_preview_worker.finished.connect(self._on_signals_ready)
        self._signal_preview_worker.error.connect(self._on_signals_error)
        self._signal_preview_worker.start()

    def _on_signals_ready(self, signals):
        self.btn_gen_signals.setEnabled(True)
        if not isinstance(signals, pd.Series) or signals.empty:
            self.lbl_strat_status.setText("No signals generated.")
            return

        # Align signals to chart df index
        signals = signals.reindex(self.df.index).fillna(0.0)  # type: ignore[union-attr]

        prev = signals.shift(1).fillna(0.0)
        long_entries  = (prev == 0) & (signals > 0)
        short_entries = (prev == 0) & (signals < 0)
        exits         = (prev != 0) & (signals == 0)

        high_arr = self.df['High'].values if 'High' in self.df.columns else self.df['high'].values
        low_arr  = self.df['Low'].values  if 'Low'  in self.df.columns else self.df['low'].values

        def _scatter(mask, y_arr, color, symbol):
            xs = [i for i, v in enumerate(mask) if v]
            ys = [float(y_arr[i]) for i in xs]
            if not xs:
                return None
            return pg.ScatterPlotItem(
                x=xs, y=ys,
                brush=pg.mkBrush(color), pen=pg.mkPen(None),
                symbol=symbol, size=12)

        for item in filter(None, [
            _scatter(long_entries,  low_arr,  '#00ff88', 't1'),   # green up-triangle below bar
            _scatter(short_entries, high_arr, '#ff4444', 't'),    # red down-triangle above bar
            _scatter(exits,         low_arr,  '#aaaaaa', 'o'),    # grey circle
        ]):
            self.main_plot.addItem(item)
            self._signal_preview_items.append(item)

        n_long  = int(long_entries.sum())
        n_short = int(short_entries.sum())
        n_exit  = int(exits.sum())
        self.lbl_strat_status.setText(
            f"Signals: +{n_long}L  -{n_short}S  {n_exit}X")

    def _on_signals_error(self, msg: str):
        self.btn_gen_signals.setEnabled(True)
        first_line = msg.splitlines()[0] if msg else "Unknown error"
        self.lbl_strat_status.setText(f"Signal error: {first_line}")
        self.lbl_strat_status.setToolTip(msg)   # full traceback on hover
        logging.error(f"Signal preview error:\n{msg}")

    def _clear_signal_markers(self):
        for item in self._signal_preview_items:
            self.main_plot.removeItem(item)
        self._signal_preview_items = []

    # -----------------------------------------------------------------------
    # Crosshair
    # -----------------------------------------------------------------------

    def _setup_crosshair(self):
        for v in self.v_lines:
            if v.scene():
                v.scene().removeItem(v)
        self.v_lines = []

        v_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.main_plot.addItem(v_line, ignoreBounds=True)
        self.main_plot.addItem(self.h_line, ignoreBounds=True)
        self.v_lines.append(v_line)

        for data in self.active_features.values():
            if data.get("v_line") and data.get("plot") != self.main_plot:
                v_sub = pg.InfiniteLine(
                    angle=90, movable=False,
                    pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
                data["plot"].addItem(v_sub, ignoreBounds=True)
                data["v_line"] = v_sub
                self.v_lines.append(v_sub)

        self.price_label = pg.TextItem(
            anchor=(0, 1), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.time_label = pg.TextItem(
            anchor=(1, 0), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.vol_label = pg.TextItem(
            anchor=(0, 0), color='#aaa', fill=pg.mkBrush(0, 0, 0, 150))

        self.main_plot.addItem(self.price_label, ignoreBounds=True)
        self.main_plot.addItem(self.time_label,  ignoreBounds=True)
        self.main_plot.addItem(self.vol_label,   ignoreBounds=True)

        self.proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def mouse_moved(self, evt):
        pos = evt[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return

        mp = self.main_plot.getViewBox().mapSceneToView(pos)
        x, y = mp.x(), mp.y()

        idx = int(round(x))
        for v in self.v_lines:
            v.setPos(idx)
        self.h_line.setPos(y)

        vr = self.main_plot.getViewBox().viewRange()
        x_max, x_min = vr[0][1], vr[0][0]

        self.price_label.setPos(x_max, y)
        self.price_label.setText(f"{y:.2f}")
        self.price_label.setAnchor((1, 0.5))

        # Use pre-computed label lists — O(1) index, no object construction
        if self.df is not None and 0 <= idx < len(self._mouse_ts_labels):
            self.time_label.setPos(idx, vr[1][0])
            self.time_label.setText(self._mouse_ts_labels[idx])
            self.time_label.setAnchor((0.5, 1))

            self.vol_label.setPos(x_min, vr[1][1])
            self.vol_label.setText(self._mouse_vol_labels[idx])

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_random(self):
        try:
            tickers = self.engine.list_cached_tickers()
        except Exception:
            tickers = []
        ticker = (random.choice(tickers) if tickers
                  else random.choice(["AAPL", "MSFT", "GOOGL"]))
        self.controls.ticker_input.setText(ticker)
        self.load_chart()

    def _load_from_history(self, index: int):
        if index <= 0:
            return
        self.controls.ticker_input.setText(self.controls.ticker_history.currentText())
        self.load_chart()

    _PERIOD_BARS = {
        "1mo":  21,
        "3mo":  63,
        "6mo":  126,
        "1y":   252,
        "2y":   504,
        "5y":   1260,
        "10y":  2520,
        "All":  None,
    }

    def _apply_period_view(self):
        if self.df is None or self.df.empty:
            return
        period = self.controls.period_combo.currentText()
        n = self._PERIOD_BARS.get(period)
        total = len(self.df)
        if n is None or n >= total:
            self.main_plot.getViewBox().setXRange(0, total)
        else:
            self.main_plot.getViewBox().setXRange(total - n, total)

    def load_chart(self):
        self._load_timer.stop()
        ticker = self.controls.ticker_input.text().strip().upper()
        if not ticker:
            return
        interval = self.controls.interval_combo.currentText()

        try:
            df = self.engine.get_historical_data(ticker, interval)
        except Exception as e:
            logging.exception(f"Error fetching {ticker} ({interval}): {e}")
            return

        if df is None or df.empty:
            return

        # DataBroker returns lowercase; chart overlays expect Title-case
        df.columns = [c.capitalize() for c in df.columns]
        self.df = df
        self.controls.add_to_history(ticker)

        # Pre-compute all per-bar strings once so tick labels and the crosshair
        # tooltip are O(1) lookups during pan/zoom and mouse move.
        dti = pd.DatetimeIndex(self.df.index)
        self._tick_intraday  = dti.strftime('%H:%M\n%m-%d').tolist()
        self._tick_daily     = dti.strftime('%Y-%m-%d').tolist()
        self._mouse_ts_labels = dti.strftime('%Y-%m-%d %H:%M').tolist()
        vol_arr = self.df['Volume'].values
        self._mouse_vol_labels = [
            f"Vol: {v/1e6:.2f}M" if v > 1e6
            else f"Vol: {v/1e3:.1f}K" if v > 1e3
            else f"Vol: {int(v)}"
            for v in vol_arr
        ]

        # Keep raw string list for legacy compatibility
        self.timestamps = self._tick_daily

        # Tick label renderer: plain list-index lookup, no Timestamp construction
        _intra = self._tick_intraday
        _daily = self._tick_daily
        _n     = len(_daily)
        self.main_plot.getAxis('bottom').tickStrings = (
            lambda values, _scale, spacing,
                   intra=_intra, daily=_daily, n=_n: [
                (intra if spacing < 5 else daily)[int(v)]
                if 0 <= int(v) < n else ''
                for v in values
            ]
        )

        self.main_plot.update_all(self.df)

        for name in list(self.active_features.keys()):
            self.update_feature(name)

        self._refresh_source_selectors()
        self._apply_period_view()

    # -----------------------------------------------------------------------
    # Feature management
    # -----------------------------------------------------------------------

    def _get_column_options(self) -> list:
        """Return (label, col_name) tuples for source-column selectors.

        Always starts with ("Price (High/Low)", "") — the default price columns.
        Then one entry per series output from every active feature that has raw data.
        """
        options = [("Price (High/Low)", "")]
        for data in self.active_features.values():
            for col_name in data.get("raw_data", {}):
                options.append((col_name, col_name))
        return options

    def _refresh_source_selectors(self):
        """Update column-selector dropdowns on all active features after any change.

        Each feature's own outputs are excluded from its own source options so a
        feature cannot be configured to feed into itself.
        """
        all_options = self._get_column_options()
        for feat_name, data in self.active_features.items():
            feature = data["instance"]
            source_keys = getattr(feature, "source_param_keys", [])
            if not source_keys:
                continue
            own_cols = set(data.get("raw_data", {}).keys())
            filtered = [opt for opt in all_options if opt[1] not in own_cols]
            for key in source_keys:
                combo = data["inputs"].get(key)
                if combo is not None:
                    self.feature_panel.refresh_column_options(combo, filtered)

    def add_feature_ui(self, initial_values=None):
        feat_name = self.feature_panel.feat_combo.currentData()
        if not feat_name:
            return
        self._add_feature_by_name(feat_name, initial_values)

    def _add_feature_by_name(self, feat_name, initial_values=None):
        if feat_name in self.active_features:
            return

        # Always apply saved color prefs from the active strategy, merging under any
        # explicit initial_values so caller-provided values take precedence.
        strat_name = self.chart_strategy_combo.currentText()
        if strat_name and strat_name != "(none)":
            saved_colors = self._load_color_prefs(strat_name).get(feat_name, {})
            if saved_colors:
                merged = dict(saved_colors)
                if initial_values:
                    merged.update(initial_values)
                initial_values = merged

        feature = self.available_features[feat_name]
        output_names = getattr(feature, "output_names", None)

        # Build column_options for any source-selector parameters
        source_keys = getattr(feature, "source_param_keys", [])
        col_options = (
            {key: self._get_column_options() for key in source_keys}
            if source_keys else None
        )

        input_widgets, visibility_widgets, group_widget = self.feature_panel.create_feature_widget(
            feat_name, feature.parameters,
            lambda: self.update_and_save(feat_name),
            self.remove_feature,
            initial_values=initial_values,
            output_names=output_names if output_names and len(output_names) > 1 else None,
            column_options=col_options,
        )

        plot_target      = self.main_plot
        v_line           = None
        container_widget = self.main_plot_widget
        reorganize       = False

        if feature.target_pane == "new":
            new_pw = pg.GraphicsLayoutWidget()
            new_pw.setBackground('#1e1e1e')
            new_pw.setMinimumHeight(100)
            new_pw.setContentsMargins(0, 0, 0, 0)
            self.plot_splitter.addWidget(new_pw)
            container_widget = new_pw

            new_plot = UnifiedPlot()
            if feature.y_range:
                new_plot.set_fixed_y_range(
                    feature.y_range[0], feature.y_range[1], padding=feature.y_padding)
            new_plot.setXLink(self.main_plot)
            new_plot.getAxis('left').setWidth(40)
            new_pw.addItem(new_plot)
            plot_target = new_plot
            self.sub_plots[feat_name] = new_plot

            v_line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
            new_plot.addItem(v_line, ignoreBounds=True)
            self.v_lines.append(v_line)
            reorganize = True

        self.active_features[feat_name] = {
            "instance": feature, "inputs": input_widgets,
            "visibility": visibility_widgets, "overlays": [],
            "plot": plot_target, "v_line": v_line,
            "widget": group_widget, "container": container_widget,
        }
        self.update_feature(feat_name)
        if reorganize:
            self._reorganize_subplots()
        self._refresh_source_selectors()
        self._sync_strategy_manifest()

    def update_and_save(self, feat_name):
        self.update_feature(feat_name)
        self._save_color_prefs()
        self._sync_strategy_manifest()

    def update_feature(self, feat_name):
        if self.df is None or self.df.empty:
            return
        data = self.active_features[feat_name]
        feat, plot = data["instance"], data["plot"]

        params = {
            k: (w.currentData() if isinstance(w, QComboBox) and w.property("is_source_selector")
                else w.currentText() if isinstance(w, QComboBox)
                else w.isChecked() if isinstance(w, QCheckBox)
                else w.text())
            for k, w in data["inputs"].items()
        }

        # Augment df with any source columns that live in other features' raw_data
        # (e.g. RSI series selected as fractal source).
        aug_df = self.df
        source_keys = getattr(feat, "source_param_keys", [])
        if source_keys:
            extra = {}
            for k in source_keys:
                col = params.get(k) or None
                if col and col not in self.df.columns:
                    for other_name, other_data in self.active_features.items():
                        if other_name != feat_name and col in other_data.get("raw_data", {}):
                            extra[col] = other_data["raw_data"][col]
                            break
            if extra:
                aug_df = self.df.copy()
                for col, series in extra.items():
                    aug_df[col] = series

        try:
            result = feat.compute(aug_df, params)
            if isinstance(result, FeatureResult):
                results = result.visuals
                data["raw_data"] = result.data
            else:
                results = result
        except Exception as e:
            print(f"Error computing {feat_name}: {e}")
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
                sym = {'o': 'o', 'd': 'd', 't': 't1', 's': 's', 'x': 'x', '+': '+'}.get(
                    res.shape, 'o')
                item = pg.ScatterPlotItem(
                    x=res.indices, y=res.values,
                    brush=pg.mkBrush(res.color), symbol=sym, size=10)
                plot.addItem(item)
                data.setdefault("temp_items", []).append(item)

        plot.update_all(self.df)

    def remove_feature(self, feat_name, widget, reorganize=True, sync=True):
        if feat_name in self.active_features:
            data = self.active_features[feat_name]
            plot = data["plot"]

            for o in data["overlays"]:
                plot.remove_overlay(o)

            if "temp_items" in data:
                for item in data["temp_items"]:
                    plot.removeItem(item)

            if data.get("v_line") in self.v_lines:
                self.v_lines.remove(data["v_line"])

            if data["instance"].target_pane == "new":
                data["container"].deleteLater()
                if feat_name in self.sub_plots:
                    del self.sub_plots[feat_name]
                if reorganize:
                    self._reorganize_subplots()

            del self.active_features[feat_name]

        widget.deleteLater()
        self._refresh_source_selectors()
        if sync:
            self._sync_strategy_manifest()

    def _reorganize_subplots(self):
        count = self.plot_splitter.count()
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
        self.plot_splitter.setSizes([main_size] + [feat_size] * num_features)

    # -----------------------------------------------------------------------
    # Window lifecycle
    # -----------------------------------------------------------------------

    def closeEvent(self, event):
        if (hasattr(self.backtest_panel, '_worker')
                and self.backtest_panel._worker
                and self.backtest_panel._worker.isRunning()):
            self.backtest_panel._worker.cancel()
            self.backtest_panel._worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ChartWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
