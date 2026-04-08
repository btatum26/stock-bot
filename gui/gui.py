import os
import logging
import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGroupBox, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from engine import ModelEngine

from .features.loader import load_features
from .gui_components.backtest_panel import BacktestPanel
from .gui_components.controls import ControlBar
from .gui_components.feature_panel import FeaturePanel
from .gui_components.plots import CandleOverlay, UnifiedPlot, VolumeOverlay
from .gui_components.styling import setup_app_style

from .chart.data_manager import ChartDataManager
from .chart.feature_manager import FeatureManager
from .chart.manifest_manager import ManifestManager

from .colors import ACCENT_GREEN, BG_MAIN, CHART_AXIS, CHART_CROSSHAIR, SIGNAL_EXIT, SIGNAL_LONG, SIGNAL_SHORT
from .config import LOAD_DEBOUNCE_MS, MAX_CHART_BARS, PERIOD_BARS, SIDEBAR_MIN_WIDTH, WORKSPACE_DIR, DB_PATH


# ---------------------------------------------------------------------------
# Background worker for signal preview
# ---------------------------------------------------------------------------

class SignalPreviewWorker(QThread):
    """Runs LocalBacktester.run_batch in a background thread for signal preview."""
    finished = pyqtSignal(object)   # emits pd.Series or None
    error    = pyqtSignal(str)

    def __init__(self, strategy_dir: str, ticker: str, df: pd.DataFrame):
        super().__init__()
        self._strategy_dir = strategy_dir
        self._ticker = ticker
        self._df = df

    def run(self):
        try:
            from engine.core.backtester import LocalBacktester
            bt = LocalBacktester(self._strategy_dir)
            batch = bt.run_batch({self._ticker: self._df})
            self.finished.emit(batch.get(self._ticker))
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


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Bot Pro")
        self.resize(1400, 900)
        setup_app_style(self)

        self.engine = ModelEngine(workspace_dir=WORKSPACE_DIR, db_path=DB_PATH)

        # Manager classes — each owns a single concern
        self._data_manager     = ChartDataManager(self.engine)
        self._manifest_manager = ManifestManager(self.engine)
        self._available_features = load_features()

        # Chart state
        self.df: pd.DataFrame | None = None
        self.timestamps: list = []
        self._tick_intraday: list  = []
        self._tick_daily: list     = []
        self._mouse_ts_labels: list  = []
        self._mouse_vol_labels: list = []
        self.v_lines: list = []

        # Signal preview state
        self._signal_preview_worker: SignalPreviewWorker | None = None
        self._signal_preview_items: list = []

        # Suppress manifest sync while batch-loading features from a strategy
        self._suppress_sync = False

        self._init_ui()
        self.load_random()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.top_tabs = QTabWidget()
        self.top_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.top_tabs.setStyleSheet(f"""
            QTabBar::tab          {{ padding: 8px 20px; font-size: 13px; }}
            QTabBar::tab:selected {{ font-weight: bold; color: {ACCENT_GREEN};
                                     border-bottom: 2px solid {ACCENT_GREEN}; }}
        """)
        self.top_tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.top_tabs)

        self.top_tabs.addTab(self._build_chart_tab(), "Chart")

        self.backtest_panel = BacktestPanel(self.engine, self)
        self.top_tabs.addTab(self.backtest_panel, "Backtest")

        self.showMaximized()

    def _build_chart_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Debounce timer — fires load_chart 800 ms after the user stops typing
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self.load_chart)

        # Control bar
        self.controls = ControlBar(self)
        self.controls.ticker_input.returnPressed.connect(self.load_chart)
        self.controls.ticker_input.textChanged.connect(
            lambda: self._load_timer.start(LOAD_DEBOUNCE_MS))
        self.controls.ticker_history.currentIndexChanged.connect(self._load_from_history)
        self.controls.interval_combo.currentIndexChanged.connect(self.load_chart)
        self.controls.period_combo.currentIndexChanged.connect(self._apply_period_view)
        self.controls.btn_random.clicked.connect(self.load_random)
        layout.addWidget(self.controls)

        # Horizontal splitter: chart area | sidebar
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.setHandleWidth(6)
        layout.addWidget(h_splitter)

        # --- Chart area (vertical splitter: main plot + sub-plots) ---
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot_splitter.setHandleWidth(6)

        self.main_plot_widget = pg.GraphicsLayoutWidget()
        self.main_plot_widget.setBackground(BG_MAIN)
        self.main_plot_widget.setContentsMargins(0, 0, 0, 0)
        self.plot_splitter.addWidget(self.main_plot_widget)

        self.main_plot = UnifiedPlot()
        self.main_plot_widget.addItem(self.main_plot)

        self.price_overlay  = CandleOverlay()
        self.volume_overlay = VolumeOverlay()
        self.main_plot.add_overlay(self.price_overlay)
        self.main_plot.add_overlay(self.volume_overlay)

        self.main_plot.getAxis("left").setPen(CHART_AXIS)
        self.main_plot.getAxis("left").setWidth(40)
        self.main_plot.getAxis("bottom").setPen(CHART_AXIS)

        h_splitter.addWidget(self.plot_splitter)

        # --- Right sidebar ---
        sidebar = self._build_sidebar()
        h_splitter.addWidget(sidebar)
        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 0)

        # Feature panel (created here so FeatureManager can reference it)
        self.feature_panel = FeaturePanel(self._available_features, self)
        self.feature_panel.btn_add_feat.clicked.connect(self._add_feature_from_combo)
        self._sidebar_feat_scroll.setWidget(self.feature_panel)

        # Feature manager — owns active_features, sub_plots, overlay rendering
        self._feature_manager = FeatureManager(
            available_features=self._available_features,
            main_plot=self.main_plot,
            main_plot_widget=self.main_plot_widget,
            plot_splitter=self.plot_splitter,
            feature_panel=self.feature_panel,
            v_lines=self.v_lines,
            parent=self,
        )
        self._feature_manager.sync_needed.connect(self._on_feature_sync_needed)

        self._setup_crosshair()
        self._refresh_strategy_combo()
        return widget

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Strategy selector group
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
        self.btn_gen_signals.clicked.connect(self._generate_signals_preview)
        strat_layout.addWidget(self.btn_gen_signals)

        self.lbl_strat_status = QLabel("")
        self.lbl_strat_status.setStyleSheet("color: #888; font-size: 10px; padding: 0 2px;")
        self.lbl_strat_status.setWordWrap(True)
        strat_layout.addWidget(self.lbl_strat_status)

        sidebar_layout.addWidget(strat_box)

        # Scroll area for the feature panel (widget assigned later)
        self._sidebar_feat_scroll = QScrollArea()
        self._sidebar_feat_scroll.setWidgetResizable(True)
        self._sidebar_feat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_layout.addWidget(self._sidebar_feat_scroll)

        return sidebar

    # -----------------------------------------------------------------------
    # Tab management
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
        except Exception as e:
            logging.warning(f"Could not list strategies: {e}")
        idx = self.chart_strategy_combo.findText(current)
        self.chart_strategy_combo.setCurrentIndex(max(0, idx))
        self.chart_strategy_combo.blockSignals(False)

    # -----------------------------------------------------------------------
    # Strategy feature loading
    # -----------------------------------------------------------------------

    def _load_strategy_features(self):
        """Read the selected strategy's manifest and add its features to the chart."""
        name = self.chart_strategy_combo.currentText()

        self._suppress_sync = True
        self._feature_manager.clear_all()
        self._suppress_sync = False

        self.lbl_strat_status.setText("")
        if not name or name == "(none)":
            return

        manifest = self._manifest_manager.load_manifest(name)
        if not manifest:
            self.lbl_strat_status.setText("Could not read manifest.")
            return

        features_cfg = manifest.get("features", [])
        if not features_cfg:
            self.lbl_strat_status.setText("No features in manifest.")
            return

        by_engine_id = {v._id: k for k, v in self._available_features.items()}
        color_prefs  = self._manifest_manager.load_color_prefs(name)

        loaded, skipped = [], []
        self._suppress_sync = True
        for entry in features_cfg:
            feat_id      = entry.get("id", "")
            params       = entry.get("params", {})
            display_name = by_engine_id.get(feat_id)
            if display_name is None:
                skipped.append(feat_id)
                continue
            initial_values = {k: str(v) for k, v in params.items()}
            initial_values.update(color_prefs.get(display_name, {}))
            self._feature_manager.add(display_name, initial_values=initial_values)
            loaded.append(display_name)
        self._suppress_sync = False

        parts = []
        if loaded:
            parts.append(f"Loaded: {', '.join(loaded)}")
        if skipped:
            parts.append(f"Skipped: {', '.join(skipped)}")
        self.lbl_strat_status.setText(" | ".join(parts))

    def _on_feature_sync_needed(self):
        """Write active features to manifest.json and regenerate context.py."""
        if self._suppress_sync:
            return
        name = self.chart_strategy_combo.currentText()
        if not name or name == "(none)":
            return
        self._manifest_manager.save_color_prefs(
            name, self._feature_manager.get_color_prefs())
        features_list = self._feature_manager.build_features_list()
        status = self._manifest_manager.sync(name, features_list)
        self.lbl_strat_status.setText(status)

    def _add_feature_from_combo(self):
        """Add the feature currently selected in the combo box."""
        feat_name = self.feature_panel.feat_combo.currentData()
        if not feat_name:
            return
        name = self.chart_strategy_combo.currentText()
        saved_colors = {}
        if name and name != "(none)":
            saved_colors = self._manifest_manager.load_color_prefs(name).get(feat_name, {})
        self._feature_manager.add(feat_name, initial_values=saved_colors or None)
        self._on_feature_sync_needed()

    # -----------------------------------------------------------------------
    # Signal preview
    # -----------------------------------------------------------------------

    def _generate_signals_preview(self):
        """Run the selected strategy on current chart data and overlay markers."""
        strat_name = self.chart_strategy_combo.currentText()
        if not strat_name or strat_name == "(none)":
            self.lbl_strat_status.setText("Select a strategy first.")
            return
        if self.df is None or self.df.empty:
            self.lbl_strat_status.setText("Load chart data first.")
            return

        # Cancel any still-running worker before starting a new one
        if self._signal_preview_worker and self._signal_preview_worker.isRunning():
            self._signal_preview_worker.finished.disconnect()
            self._signal_preview_worker.error.disconnect()
            self._signal_preview_worker.quit()
            self._signal_preview_worker.wait(500)

        ticker       = self.controls.ticker_input.text().strip().upper()
        strategy_dir = os.path.join(self.engine.workspace_dir, strat_name)

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

        if self.df is None:
            return
        signals = signals.reindex(self.df.index).fillna(0.0)
        prev          = signals.shift(1).fillna(0.0)
        long_entries  = (prev == 0) & (signals > 0)
        short_entries = (prev == 0) & (signals < 0)
        exits         = (prev != 0) & (signals == 0)

        high_arr = self.df["High"].values if "High" in self.df.columns else self.df["high"].values
        low_arr  = self.df["Low"].values  if "Low"  in self.df.columns else self.df["low"].values

        def _scatter(mask, y_arr, color, symbol):
            xs = np.where(mask)[0]
            if len(xs) == 0:
                return None
            return pg.ScatterPlotItem(
                x=xs, y=y_arr[xs],
                brush=pg.mkBrush(color), pen=pg.mkPen(None),
                symbol=symbol, size=12)

        for item in filter(None, [
            _scatter(long_entries,  low_arr,  SIGNAL_LONG,  "t1"),
            _scatter(short_entries, high_arr, SIGNAL_SHORT, "t"),
            _scatter(exits,         low_arr,  SIGNAL_EXIT,  "o"),
        ]):
            self.main_plot.addItem(item)
            self._signal_preview_items.append(item)

        self.lbl_strat_status.setText(
            f"Signals: +{int(long_entries.sum())}L  "
            f"-{int(short_entries.sum())}S  "
            f"{int(exits.sum())}X")

    def _on_signals_error(self, msg: str):
        self.btn_gen_signals.setEnabled(True)
        first_line = msg.splitlines()[0] if msg else "Unknown error"
        self.lbl_strat_status.setText(f"Signal error: {first_line}")
        self.lbl_strat_status.setToolTip(msg)
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
        self.v_lines.clear()

        v_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(CHART_CROSSHAIR, width=1, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(CHART_CROSSHAIR, width=1, style=Qt.PenStyle.DashLine))
        self.main_plot.addItem(v_line, ignoreBounds=True)
        self.main_plot.addItem(self.h_line, ignoreBounds=True)
        self.v_lines.append(v_line)

        self.price_label = pg.TextItem(anchor=(0, 1), color="#ddd", fill=pg.mkBrush(0, 0, 0, 200))
        self.time_label  = pg.TextItem(anchor=(1, 0), color="#ddd", fill=pg.mkBrush(0, 0, 0, 200))
        self.vol_label   = pg.TextItem(anchor=(0, 0), color="#aaa", fill=pg.mkBrush(0, 0, 0, 150))
        self.main_plot.addItem(self.price_label, ignoreBounds=True)
        self.main_plot.addItem(self.time_label,  ignoreBounds=True)
        self.main_plot.addItem(self.vol_label,   ignoreBounds=True)

        self.proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def mouse_moved(self, evt):
        pos = evt[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        mp  = self.main_plot.getViewBox().mapSceneToView(pos)
        x, y = mp.x(), mp.y()
        idx = int(round(x))

        for v in self.v_lines:
            v.setPos(idx)
        self.h_line.setPos(y)

        vr    = self.main_plot.getViewBox().viewRange()
        x_max = vr[0][1]
        x_min = vr[0][0]

        self.price_label.setPos(x_max, y)
        self.price_label.setText(f"{y:.2f}")
        self.price_label.setAnchor((1, 0.5))

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
        ticker = self._data_manager.load_random_ticker()
        self.controls.ticker_input.setText(ticker)
        self.load_chart()

    def _load_from_history(self, index: int):
        if index <= 0:
            return
        self.controls.ticker_input.setText(self.controls.ticker_history.currentText())
        self.load_chart()

    def _apply_period_view(self):
        if self.df is None or self.df.empty:
            return
        period = self.controls.period_combo.currentText()
        n      = PERIOD_BARS.get(period)
        total  = len(self.df)
        if n is None:
            # "All" — show up to MAX_CHART_BARS from the end
            n = min(total, MAX_CHART_BARS)
        n = min(n, total)
        self.main_plot.getViewBox().setXRange(total - n, total)

    def load_chart(self):
        self._load_timer.stop()
        ticker = self.controls.ticker_input.text().strip().upper()
        if not ticker:
            return
        interval = self.controls.interval_combo.currentText()

        df, labels = self._data_manager.load(ticker, interval)
        if df is None:
            return

        self.df                 = df
        self._tick_intraday     = labels["tick_intraday"]
        self._tick_daily        = labels["tick_daily"]
        self._mouse_ts_labels   = labels["mouse_ts_labels"]
        self._mouse_vol_labels  = labels["mouse_vol_labels"]
        self.timestamps         = self._tick_daily

        self.controls.add_to_history(ticker)

        # Tick label renderer: plain list-index lookup, no Timestamp construction
        _intra = self._tick_intraday
        _daily = self._tick_daily
        _n     = len(_daily)
        self.main_plot.getAxis("bottom").tickStrings = (
            lambda values, _scale, spacing,
                   intra=_intra, daily=_daily, n=_n: [
                (intra if spacing < 5 else daily)[int(v)]
                if 0 <= int(v) < n else ""
                for v in values
            ]
        )

        self.main_plot.update_all(self.df)

        # Constrain the ViewBox: full pan range, but max zoom-out = MAX_CHART_BARS
        total = len(self.df)
        vb = self.main_plot.getViewBox()
        vb.setLimits(xMin=-0.5, xMax=total + 0.5, maxXRange=MAX_CHART_BARS)

        # Update feature manager with the new df before re-rendering
        self._feature_manager.set_df(self.df)
        self._feature_manager.update_all()
        self._feature_manager.refresh_source_selectors()

        self._apply_period_view()

    # -----------------------------------------------------------------------
    # Window lifecycle
    # -----------------------------------------------------------------------

    def closeEvent(self, event):
        if (hasattr(self.backtest_panel, "_worker")
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
