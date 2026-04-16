"""
PortfolioPanel — multi-asset portfolio backtest UI.

Layout
------
Left  (fixed ~300 px scroll sidebar):
    Strategy selector, ticker picker (manual + universe dropdown),
    date/interval controls, all PortfolioConfig knobs, run/cancel,
    progress + log.

Right (takes all remaining space):
    Top  — equity curve plot (strategy vs starting-capital baseline)
    Bottom tabs:
        "Heatmap"      — position weight heatmap (time × ticker, colour-coded)
        "Contribution" — per-ticker P&L bar chart
        "Trade Log"    — sortable table, one row per round-trip

The worker thread calls ModelEngine.run_portfolio_backtest() and emits
finished/progress/error back to the UI thread exactly like BacktestPanel.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFormLayout, QLabel, QComboBox, QPushButton,
    QTextEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit,
    QProgressBar, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from engine import ModelEngine


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _PortfolioWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    log_msg  = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, engine_method, *args):
        super().__init__()
        self._method    = engine_method
        self._args      = args
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        callbacks = {
            "on_progress":  lambda p, m: self.progress.emit(p, m),
            "on_log":       lambda m:    self.log_msg.emit(m),
            "is_cancelled": lambda:      self._cancelled,
        }
        try:
            result = self._method(*self._args, callbacks=callbacks)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Summary metrics bar
# ---------------------------------------------------------------------------

_METRIC_DISPLAY = [
    ("Total Return", "Total Return (%)"),
    ("CAGR",         "CAGR (%)"),
    ("Sharpe",       "Sharpe Ratio"),
    ("Sortino",      "Sortino Ratio"),
    ("Max DD",       "Max Drawdown (%)"),
    ("Win Rate",     "Win Rate (%)"),
    ("Trades",       "Total Trades"),
    ("Turnover",     "Turnover (%)"),
]

_TABLE_STYLE = """
    QTableWidget          { background-color: #1a1a1a; gridline-color: #333; color: #ddd; }
    QTableWidget::item:selected { background-color: #2a4a7a; }
    QHeaderView::section  { background-color: #252525; color: #aaa;
                            padding: 4px; border: 1px solid #444; }
    QTableWidget::item:alternate { background-color: #1e1e1e; }
"""


# ---------------------------------------------------------------------------
# PortfolioPanel
# ---------------------------------------------------------------------------

class PortfolioPanel(QWidget):

    def __init__(self, engine: ModelEngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._worker = None
        self._result = {}

        self._init_ui()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        root.addWidget(splitter)

        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_main_area())
        splitter.setSizes([300, 9999])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ---- Sidebar ----------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFrameShape(QFrame.Shape.NoFrame)
        outer.setMaximumWidth(320)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # -- Strategy -------------------------------------------------------
        sg = QGroupBox("Strategy")
        sl = QVBoxLayout(sg)
        self.strategy_combo = QComboBox()
        sl.addWidget(self.strategy_combo)
        layout.addWidget(sg)

        # -- Tickers --------------------------------------------------------
        tg = QGroupBox("Tickers")
        tl = QVBoxLayout(tg)

        univ_row = QWidget()
        ul = QHBoxLayout(univ_row)
        ul.setContentsMargins(0, 0, 0, 0)
        ul.setSpacing(4)
        self.universe_combo = QComboBox()
        self.universe_combo.addItem("(manual)")
        try:
            from engine.core.universes import list_universes
            for u in list_universes():
                self.universe_combo.addItem(u)
        except Exception:
            pass
        btn_load_univ = QPushButton("Load")
        btn_load_univ.setFixedWidth(42)
        btn_load_univ.setToolTip("Populate ticker box from universe")
        btn_load_univ.clicked.connect(self._load_universe)
        ul.addWidget(self.universe_combo, 1)
        ul.addWidget(btn_load_univ)
        tl.addWidget(univ_row)

        self.tickers_input = QTextEdit()
        self.tickers_input.setPlaceholderText("AAPL, MSFT, GOOGL, ...")
        self.tickers_input.setMaximumHeight(80)
        tl.addWidget(self.tickers_input)
        layout.addWidget(tg)

        # -- Timeframe & interval -------------------------------------------
        dfg = QGroupBox("Timeframe")
        dfl = QFormLayout(dfg)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1d", "1wk", "4h", "1h", "15m"])
        dfl.addRow("Interval:", self.interval_combo)

        self.start_edit = QLineEdit("2018-01-01")
        self.end_edit   = QLineEdit(pd.Timestamp.now().strftime("%Y-%m-%d"))
        dfl.addRow("Start:", self.start_edit)
        dfl.addRow("End:",   self.end_edit)
        layout.addWidget(dfg)

        # -- Risk / sizing --------------------------------------------------
        rg = QGroupBox("Risk & Sizing")
        rl = QFormLayout(rg)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1_000, 100_000_000)
        self.capital_spin.setValue(100_000)
        self.capital_spin.setSingleStep(10_000)
        self.capital_spin.setDecimals(0)
        self.capital_spin.setPrefix("$")
        rl.addRow("Starting Capital:", self.capital_spin)

        self.max_pos_spin = QSpinBox()
        self.max_pos_spin.setRange(1, 200)
        self.max_pos_spin.setValue(10)
        rl.addRow("Max Positions:", self.max_pos_spin)

        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.001, 0.50)
        self.risk_spin.setValue(0.02)
        self.risk_spin.setSingleStep(0.005)
        self.risk_spin.setDecimals(3)
        self.risk_spin.setSuffix(" (2% = 0.02)")
        rl.addRow("Risk/Trade:", self.risk_spin)

        self.stop_spin = QDoubleSpinBox()
        self.stop_spin.setRange(0.001, 0.99)
        self.stop_spin.setValue(0.05)
        self.stop_spin.setSingleStep(0.005)
        self.stop_spin.setDecimals(3)
        self.stop_spin.setSuffix(" (5% = 0.05)")
        rl.addRow("Stop Loss:", self.stop_spin)

        self.max_pos_pct_spin = QDoubleSpinBox()
        self.max_pos_pct_spin.setRange(0.01, 1.0)
        self.max_pos_pct_spin.setValue(0.20)
        self.max_pos_pct_spin.setSingleStep(0.05)
        self.max_pos_pct_spin.setDecimals(2)
        self.max_pos_pct_spin.setSuffix(" (20% = 0.20)")
        rl.addRow("Max Pos Size:", self.max_pos_pct_spin)
        layout.addWidget(rg)

        # -- Entry / eviction -----------------------------------------------
        eg = QGroupBox("Entry & Eviction")
        el = QFormLayout(eg)

        self.entry_thresh_spin = QDoubleSpinBox()
        self.entry_thresh_spin.setRange(0.01, 1.0)
        self.entry_thresh_spin.setValue(0.10)
        self.entry_thresh_spin.setSingleStep(0.05)
        self.entry_thresh_spin.setDecimals(2)
        el.addRow("Entry Threshold:", self.entry_thresh_spin)

        self.eviction_spin = QDoubleSpinBox()
        self.eviction_spin.setRange(0.0, 1.0)
        self.eviction_spin.setValue(0.15)
        self.eviction_spin.setSingleStep(0.05)
        self.eviction_spin.setDecimals(2)
        el.addRow("Eviction Margin:", self.eviction_spin)

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 0.05)
        self.friction_spin.setValue(0.001)
        self.friction_spin.setSingleStep(0.0005)
        self.friction_spin.setDecimals(4)
        el.addRow("Friction (1-way):", self.friction_spin)
        layout.addWidget(eg)

        # -- Options --------------------------------------------------------
        og = QGroupBox("Options")
        ol = QFormLayout(og)

        self.allow_short_cb = QCheckBox()
        self.allow_short_cb.setChecked(True)
        ol.addRow("Allow Short:", self.allow_short_cb)

        self.rebalance_cb = QCheckBox()
        self.rebalance_cb.setChecked(False)
        self.rebalance_cb.stateChanged.connect(self._on_rebalance_toggled)
        ol.addRow("Rebalance on Strength:", self.rebalance_cb)

        self.rebalance_delta_spin = QDoubleSpinBox()
        self.rebalance_delta_spin.setRange(0.01, 1.0)
        self.rebalance_delta_spin.setValue(0.10)
        self.rebalance_delta_spin.setSingleStep(0.05)
        self.rebalance_delta_spin.setDecimals(2)
        self.rebalance_delta_spin.setEnabled(False)
        ol.addRow("Rebalance Delta:", self.rebalance_delta_spin)
        layout.addWidget(og)

        # -- Run / Cancel ---------------------------------------------------
        self.btn_run = QPushButton("Run Portfolio Backtest")
        self.btn_run.setStyleSheet(
            "background-color: #1a4a1a; color: #aaff00; font-weight: bold; "
            "height: 40px; font-size: 13px; border-radius: 4px;"
        )
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "background-color: #4a1a1a; color: #ff8888; height: 28px; border-radius: 4px;")
        self.btn_cancel.clicked.connect(self._cancel)
        layout.addWidget(self.btn_cancel)

        # -- Status ---------------------------------------------------------
        stat_g = QGroupBox("Status")
        stat_l = QVBoxLayout(stat_g)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        stat_l.addWidget(self.progress_bar)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(180)
        self.log_console.setStyleSheet(
            "background-color: #0a0a0a; color: #00ff00; "
            "font-family: monospace; font-size: 10px;"
        )
        stat_l.addWidget(self.log_console)
        layout.addWidget(stat_g)

        layout.addStretch()
        outer.setWidget(inner)
        return outer

    # ---- Main area --------------------------------------------------------

    def _build_main_area(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # -- Summary metrics bar -------------------------------------------
        self._metrics_bar = QWidget()
        mb_layout = QHBoxLayout(self._metrics_bar)
        mb_layout.setContentsMargins(8, 4, 8, 4)
        mb_layout.setSpacing(20)
        self._metric_labels: dict = {}
        for short, _ in _METRIC_DISPLAY:
            lbl = QLabel(f"{short}: —")
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            mb_layout.addWidget(lbl)
            self._metric_labels[short] = lbl
        mb_layout.addStretch()
        vl.addWidget(self._metrics_bar)

        # -- Vertical splitter: equity curve + tabs ------------------------
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setHandleWidth(6)
        vl.addWidget(vsplit)

        # Equity curve
        self.equity_plot = pg.PlotWidget(background="#1a1a1a")
        self.equity_plot.setLabel("left", "Portfolio Value ($)")
        self.equity_plot.setLabel("bottom", "Bar")
        self.equity_plot.showGrid(x=True, y=True, alpha=0.2)
        self.equity_plot.addLegend(offset=(10, 10))
        vsplit.addWidget(self.equity_plot)

        # Bottom tabs
        self.detail_tabs = QTabWidget()
        vsplit.addWidget(self.detail_tabs)
        vsplit.setSizes([400, 350])

        self.detail_tabs.addTab(self._build_heatmap_tab(),     "Position Heatmap")
        self.detail_tabs.addTab(self._build_contribution_tab(), "Contribution")
        self.detail_tabs.addTab(self._build_trade_log_tab(),   "Trade Log")

        return w

    def _build_heatmap_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)

        label = QLabel(
            "Green = Long position · Red = Short position · Darkness = Weight size"
        )
        label.setStyleSheet("color: #888; font-size: 10px; padding: 4px 8px;")
        vl.addWidget(label)

        self.heatmap_widget = pg.GraphicsLayoutWidget()
        self.heatmap_widget.setBackground("#1a1a1a")
        self.heatmap_plot   = self.heatmap_widget.addPlot()
        self.heatmap_plot.getAxis("left").setPen("#444")
        self.heatmap_plot.getAxis("bottom").setPen("#444")
        self.heatmap_plot.setMouseEnabled(x=True, y=False)
        self._heatmap_img = pg.ImageItem(border=None)
        self.heatmap_plot.addItem(self._heatmap_img)

        # Custom colormap: red → near-black → green
        _cmap_colors = np.array([
            [180,  30,  30, 255],   # -1.0  short (red)
            [ 20,  20,  20, 255],   #  0.0  flat  (near-black)
            [ 30, 180,  80, 255],   # +1.0  long  (green)
        ], dtype=np.uint8)
        self._heatmap_cmap = pg.ColorMap(
            pos=np.array([0.0, 0.5, 1.0]),
            color=_cmap_colors,
        )
        self._heatmap_img.setColorMap(self._heatmap_cmap)

        vl.addWidget(self.heatmap_widget)
        return w

    def _build_contribution_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Net realised P&L per ticker (after friction).")
        label.setStyleSheet("color: #888; font-size: 10px; padding: 4px 8px;")
        vl.addWidget(label)

        self.contrib_plot = pg.PlotWidget(background="#1a1a1a")
        self.contrib_plot.showGrid(x=False, y=True, alpha=0.3)
        self.contrib_plot.setLabel("left", "P&L ($)")
        vl.addWidget(self.contrib_plot)
        return w

    def _build_trade_log_tab(self) -> QWidget:
        self.trade_table = QTableWidget(0, 11)
        self.trade_table.setHorizontalHeaderLabels([
            "Ticker", "Dir", "Entry Date", "Exit Date",
            "Entry $", "Exit $", "Shares", "P&L ($)", "Ret %",
            "Bars", "Reason",
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.trade_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.trade_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.setSortingEnabled(True)
        self.trade_table.setStyleSheet(_TABLE_STYLE)
        return self.trade_table

    # -----------------------------------------------------------------------
    # Refresh (called when tab becomes visible)
    # -----------------------------------------------------------------------

    def refresh_strategies(self):
        current = self.strategy_combo.currentText()
        self.strategy_combo.blockSignals(True)
        self.strategy_combo.clear()
        try:
            for name in self._engine.list_strategies():
                self.strategy_combo.addItem(name)
            if current:
                idx = self.strategy_combo.findText(current)
                if idx >= 0:
                    self.strategy_combo.setCurrentIndex(idx)
        except Exception as e:
            self._log(f"Could not list strategies: {e}")
        finally:
            self.strategy_combo.blockSignals(False)

    # -----------------------------------------------------------------------
    # Config helpers
    # -----------------------------------------------------------------------

    def _on_rebalance_toggled(self, state):
        self.rebalance_delta_spin.setEnabled(bool(state))

    def _load_universe(self):
        name = self.universe_combo.currentText()
        if name == "(manual)":
            return
        try:
            from engine.core.universes import get_universe
            tickers = get_universe(name)
            self.tickers_input.setPlainText(", ".join(tickers))
        except Exception as e:
            self._log(f"Could not load universe '{name}': {e}")

    def _build_config_dict(self) -> dict:
        return {
            "starting_capital":      self.capital_spin.value(),
            "max_positions":         self.max_pos_spin.value(),
            "risk_per_trade_pct":    self.risk_spin.value(),
            "stop_loss_pct":         self.stop_spin.value(),
            "max_position_pct":      self.max_pos_pct_spin.value(),
            "entry_threshold":       self.entry_thresh_spin.value(),
            "eviction_margin":       self.eviction_spin.value(),
            "friction":              self.friction_spin.value(),
            "rebalance_on_strength": self.rebalance_cb.isChecked(),
            "rebalance_delta":       self.rebalance_delta_spin.value(),
            "allow_short":           self.allow_short_cb.isChecked(),
        }

    # -----------------------------------------------------------------------
    # Run / cancel
    # -----------------------------------------------------------------------

    def _run(self):
        strategy = self.strategy_combo.currentText()
        if not strategy:
            self._log("Select a strategy first.")
            return

        raw = self.tickers_input.toPlainText().strip()
        tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
        if not tickers:
            self._log("Enter at least one ticker.")
            return

        timeframe = {
            "start":    self.start_edit.text().strip() or "2010-01-01",
            "end":      self.end_edit.text().strip()   or pd.Timestamp.now().strftime("%Y-%m-%d"),
            "interval": self.interval_combo.currentText(),
        }
        config_dict = self._build_config_dict()

        self._result = {}
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._log(f"[Portfolio] {strategy} × {len(tickers)} tickers "
                  f"({timeframe['interval']}  {timeframe['start']} → {timeframe['end']})")

        def _call(*args, callbacks):
            return self._engine.run_portfolio_backtest(
                *args, callbacks=callbacks, config_dict=config_dict
            )

        self._worker = _PortfolioWorker(_call, strategy, tickers, timeframe)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_complete)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._log("Cancellation requested…")
        self.btn_cancel.setEnabled(False)

    # -----------------------------------------------------------------------
    # Worker callbacks
    # -----------------------------------------------------------------------

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self._log(msg)

    def _on_error(self, err: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._log(f"ERROR: {err}")

    def _on_complete(self, result: dict):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        if result.get("cancelled"):
            self._log("Run cancelled.")
            return

        self._result = result
        self._populate_metrics(result.get("metrics", {}))
        self._populate_equity_curve(result)
        self._populate_heatmap(result.get("position_weights", {}))
        self._populate_contribution(result.get("per_ticker_contribution", {}))
        self._populate_trade_log(result.get("trade_log", []))
        self._log(
            f"Done — {result.get('metrics', {}).get('Total Trades', 0)} trades, "
            f"CAGR {result.get('metrics', {}).get('CAGR (%)', 0):.2f}%"
        )

    # -----------------------------------------------------------------------
    # Populate: metrics bar
    # -----------------------------------------------------------------------

    def _populate_metrics(self, metrics: dict):
        for short, key in _METRIC_DISPLAY:
            lbl   = self._metric_labels.get(short)
            if lbl is None:
                continue
            raw   = metrics.get(key)
            if raw is None:
                lbl.setText(f"{short}: —")
                lbl.setStyleSheet("color: #aaa; font-size: 11px;")
                continue

            if isinstance(raw, float):
                text = f"{short}: {raw:.2f}"
            else:
                text = f"{short}: {raw}"
            lbl.setText(text)

            # Colour hints
            color = "#aaa"
            if key == "CAGR (%)":
                color = "#44ff88" if raw >= 10 else "#dddd44" if raw > 0 else "#ff5555"
            elif key == "Sharpe Ratio":
                color = "#44ff88" if raw >= 1.5 else "#dddd44" if raw >= 0.5 else "#ff5555"
            elif key == "Max Drawdown (%)":
                color = "#ff5555" if raw <= -30 else "#dddd44" if raw <= -15 else "#44ff88"
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

    # -----------------------------------------------------------------------
    # Populate: equity curve
    # -----------------------------------------------------------------------

    def _populate_equity_curve(self, result: dict):
        self.equity_plot.clear()
        self.equity_plot.addLegend(offset=(10, 10))

        eq_points = result.get("equity_curve", [])
        if eq_points:
            xs = list(range(len(eq_points)))
            ys = [p["v"] for p in eq_points]
            self.equity_plot.plot(
                xs, ys,
                pen=pg.mkPen("#44aaff", width=2),
                name="Portfolio",
            )

        capital = result.get("starting_capital", self.capital_spin.value())
        self.equity_plot.addLine(
            y=capital,
            pen=pg.mkPen("#555", style=Qt.PenStyle.DashLine),
        )
        # Label the baseline
        text = pg.TextItem(
            f"Start: ${capital:,.0f}",
            color="#666",
            anchor=(0, 1),
        )
        self.equity_plot.addItem(text)
        if eq_points:
            text.setPos(0, capital)

        self.equity_plot.setTitle("Portfolio Equity Curve", color="#ddd")

    # -----------------------------------------------------------------------
    # Populate: position heatmap
    # -----------------------------------------------------------------------

    def _populate_heatmap(self, position_weights: dict):
        self.heatmap_plot.clear()
        self.heatmap_plot.addItem(self._heatmap_img)

        if not position_weights:
            return

        tickers = sorted(position_weights.keys())
        n_t     = len(tickers)

        # Build a common time index (union of all series)
        all_series = [
            s for s in position_weights.values()
            if isinstance(s, pd.Series) and not s.empty
        ]
        if not all_series:
            return

        time_idx = all_series[0].index
        for s in all_series[1:]:
            time_idx = time_idx.union(s.index)
        time_idx = time_idx.sort_values()

        # Downsample time axis for display (max 800 bars)
        max_bars = 800
        if len(time_idx) > max_bars:
            step     = len(time_idx) // max_bars
            time_idx = time_idx[::step]

        n_bars = len(time_idx)

        # Weight matrix: shape (n_bars, n_tickers)
        matrix = np.zeros((n_bars, n_t), dtype=np.float32)
        for j, t in enumerate(tickers):
            s = position_weights[t]
            if isinstance(s, pd.Series) and not s.empty:
                aligned = s.reindex(time_idx).fillna(0.0).values
                matrix[:, j] = aligned.astype(np.float32)

        self._heatmap_img.setImage(matrix, levels=(-1.0, 1.0))

        # Scale image to fill the plot
        from PyQt6.QtCore import QRectF
        self._heatmap_img.setRect(QRectF(0, 0, n_bars, n_t))

        # Y axis: ticker names
        y_ticks = [(j + 0.5, tickers[j]) for j in range(n_t)]
        if n_t > 40:
            step   = max(1, n_t // 20)
            y_ticks = y_ticks[::step]
        self.heatmap_plot.getAxis("left").setTicks([y_ticks])

        # X axis: date labels at ~10 points
        step    = max(1, n_bars // 10)
        x_ticks = [
            (i, str(time_idx[i])[:10])
            for i in range(0, n_bars, step)
        ]
        self.heatmap_plot.getAxis("bottom").setTicks([x_ticks])

        self.heatmap_plot.setXRange(0, n_bars, padding=0.01)
        self.heatmap_plot.setYRange(0, n_t, padding=0.01)
        self.heatmap_plot.setTitle("Position Heatmap", color="#ddd")

    # -----------------------------------------------------------------------
    # Populate: per-ticker contribution bar chart
    # -----------------------------------------------------------------------

    def _populate_contribution(self, contribution: dict):
        self.contrib_plot.clear()

        if not contribution:
            return

        # Sort by contribution descending
        items   = sorted(contribution.items(), key=lambda x: x[1], reverse=True)
        tickers = [it[0] for it in items]
        values  = [it[1] for it in items]
        n       = len(tickers)

        brushes = [
            pg.mkBrush("#44ff88") if v >= 0 else pg.mkBrush("#ff5555")
            for v in values
        ]
        bar_item = pg.BarGraphItem(
            x=np.arange(n),
            height=values,
            width=0.7,
            brushes=brushes,
        )
        self.contrib_plot.addItem(bar_item)

        # Zero line
        self.contrib_plot.addLine(
            y=0, pen=pg.mkPen("#555", style=Qt.PenStyle.DashLine)
        )

        # X axis tick labels (show every Nth if too many tickers)
        step    = max(1, n // 30)
        x_ticks = [(i, tickers[i]) for i in range(0, n, step)]
        self.contrib_plot.getAxis("bottom").setTicks([x_ticks])
        self.contrib_plot.setTitle("Per-Ticker P&L Contribution ($)", color="#ddd")
        self.contrib_plot.setXRange(-0.5, n - 0.5, padding=0.01)

    # -----------------------------------------------------------------------
    # Populate: trade log table
    # -----------------------------------------------------------------------

    def _populate_trade_log(self, trades: list):
        self.trade_table.setSortingEnabled(False)
        self.trade_table.setRowCount(0)

        for t in trades:
            row = self.trade_table.rowCount()
            self.trade_table.insertRow(row)

            pnl       = float(t.get("pnl", 0.0))
            ret       = float(t.get("return_pct", 0.0))
            direction = str(t.get("direction", ""))
            reason    = str(t.get("exit_reason", ""))
            row_bg    = QColor("#142814") if pnl > 0 else QColor("#281414") if pnl < 0 else None

            cells = [
                str(t.get("ticker",       "")),
                direction,
                str(t.get("entry_date",   "")),
                str(t.get("exit_date",    "")),
                f"{t.get('entry_price', 0):.4f}",
                f"{t.get('exit_price',  0):.4f}",
                f"{t.get('shares',      0):.2f}",
                f"${pnl:,.2f}",
                f"{ret:.2f}%",
                str(t.get("bars_held",    "")),
                reason,
            ]
            for col_idx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if col_idx == 1:  # Direction
                    item.setForeground(
                        QColor("#44ff88") if direction == "LONG"
                        else QColor("#ff5555")
                    )
                if col_idx == 10:  # Reason
                    if reason == "STOP":
                        item.setForeground(QColor("#ff9900"))
                    elif reason == "EVICTED":
                        item.setForeground(QColor("#aa88ff"))
                    elif reason == "END_OF_DATA":
                        item.setForeground(QColor("#888888"))
                if row_bg:
                    item.setBackground(row_bg)

                self.trade_table.setItem(row, col_idx, item)

        self.trade_table.setSortingEnabled(True)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_console.append(msg)
        self.log_console.ensureCursorVisible()
