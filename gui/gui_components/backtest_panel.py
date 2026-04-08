"""
BacktestPanel — three-panel batch backtest UI.

Layout:
  Left  (splitter vertical):
    Panel B (top)  — sortable leaderboard table, one row per ticker
    Panel C (bottom) — drill-down tabs: Portfolio chart | Trade Log | Signal Chart
  Right (fixed 350 px):
    Panel A — strategy selector, hyperparams, batch config, run/cancel, progress/log
"""
import os
import json

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFormLayout, QLabel, QComboBox, QPushButton,
    QTextEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit,
    QProgressBar, QScrollArea, QFrame, QInputDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from engine import ModelEngine
from .plots import UnifiedPlot, CandleOverlay


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class EngineWorker(QThread):
    """Runs a ModelEngine method off the UI thread."""
    finished = pyqtSignal(object)
    progress  = pyqtSignal(int, str)
    log_msg   = pyqtSignal(str)
    error     = pyqtSignal(str)

    def __init__(self, engine_method, *args):
        super().__init__()
        self._method = engine_method
        self._args   = args
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        callbacks = {
            "on_progress": lambda p, m: self.progress.emit(p, m),
            "on_log":      lambda m:    self.log_msg.emit(m),
            "is_cancelled": lambda:     self._cancelled,
        }
        try:
            result = self._method(*self._args, callbacks=callbacks)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Leaderboard column spec: (header, metrics-dict-key, format-string)
# ---------------------------------------------------------------------------
_COLS = [
    ("Ticker",        "ticker",                "{}"),
    ("CAGR (%)",      "CAGR (%)",              "{:.2f}"),
    ("Total Ret (%)", "Total Return (%)",       "{:.2f}"),
    ("Sharpe",        "Sharpe Ratio",           "{:.2f}"),
    ("Sortino",       "Sortino Ratio",          "{:.2f}"),
    ("Calmar",        "Calmar Ratio",           "{:.2f}"),
    ("Max DD (%)",    "Max Drawdown (%)",       "{:.2f}"),
    ("Win Rate (%)",  "Win Rate (%)",           "{:.2f}"),
    ("Prof. Factor",  "Profit Factor",          "{:.2f}"),
    ("Trades",        "Discrete Trades",        "{:d}"),
]


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class BacktestPanel(QWidget):
    def __init__(self, engine: ModelEngine, parent=None):
        super().__init__(parent)
        self._engine  = engine
        self._worker  = None
        self._result  = {}   # last completed run_backtest() result

        self._init_ui()

    # -----------------------------------------------------------------------
    # Layout construction
    # -----------------------------------------------------------------------

    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        root.addWidget(splitter)

        # Left: B (leaderboard) over C (drill-down)
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setHandleWidth(6)
        left_splitter.addWidget(self._build_panel_b())
        left_splitter.addWidget(self._build_panel_c())
        left_splitter.setSizes([350, 450])

        splitter.addWidget(left_splitter)
        splitter.addWidget(self._build_panel_a())
        splitter.setSizes([9999, 350])   # left takes all extra space
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

    # --- Panel B: Leaderboard ------------------------------------------

    def _build_panel_b(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 4)

        hdr = QLabel("Batch Results")
        hdr.setStyleSheet("font-size: 13px; font-weight: bold; color: #ddd; padding-bottom: 4px;")
        layout.addWidget(hdr)

        self.leaderboard = QTableWidget(0, len(_COLS))
        self.leaderboard.setHorizontalHeaderLabels([c[0] for c in _COLS])
        self.leaderboard.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.leaderboard.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.leaderboard.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.leaderboard.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.leaderboard.setAlternatingRowColors(True)
        self.leaderboard.setSortingEnabled(True)
        self.leaderboard.setStyleSheet("""
            QTableWidget          { background-color: #1a1a1a; gridline-color: #333; color: #ddd; }
            QTableWidget::item:selected { background-color: #2a4a7a; }
            QHeaderView::section  { background-color: #252525; color: #aaa;
                                    padding: 4px; border: 1px solid #444; }
            QTableWidget::item:alternate { background-color: #1e1e1e; }
        """)
        self.leaderboard.cellClicked.connect(self._on_ticker_selected)
        layout.addWidget(self.leaderboard)
        return w

    # --- Panel C: Drill-down ------------------------------------------

    def _build_panel_c(self) -> QWidget:
        self.drill_tabs = QTabWidget()

        # Tab 1 — Portfolio chart
        self.portfolio_plot = pg.PlotWidget(background='#1a1a1a')
        self.portfolio_plot.setLabel('left', 'Portfolio Value ($)')
        self.portfolio_plot.showGrid(x=True, y=True, alpha=0.2)
        self.portfolio_plot.addLegend(offset=(10, 10))
        self.drill_tabs.addTab(self.portfolio_plot, "Portfolio")

        # Tab 2 — Trade log
        self.trade_table = QTableWidget(0, 7)
        self.trade_table.setHorizontalHeaderLabels(
            ["Entry Date", "Exit Date", "Direction", "Entry $", "Exit $", "Return %", "Bars"])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.setSortingEnabled(True)
        self.trade_table.setStyleSheet("""
            QTableWidget          { background-color: #1a1a1a; gridline-color: #333; color: #ddd; }
            QHeaderView::section  { background-color: #252525; color: #aaa;
                                    padding: 4px; border: 1px solid #444; }
            QTableWidget::item:alternate { background-color: #1e1e1e; }
        """)
        self.drill_tabs.addTab(self.trade_table, "Trade Log")

        # Tab 3 — Signal chart (candles + entry/exit markers)
        # Use GraphicsLayoutWidget + UnifiedPlot for auto-Y and horizontal-only scroll,
        # identical behaviour to the main chart tab.
        self.signal_plot_widget = pg.GraphicsLayoutWidget()
        self.signal_plot_widget.setBackground('#1a1a1a')
        self.signal_plot = UnifiedPlot()
        self.signal_plot.getAxis('left').setPen('#444')
        self.signal_plot.getAxis('left').setWidth(50)
        self.signal_plot.getAxis('bottom').setPen('#444')
        self.signal_plot_widget.addItem(self.signal_plot)
        self._signal_candle_overlay = None   # rebuilt per ticker
        self._signal_df = None               # kept for auto-scale context
        self.drill_tabs.addTab(self.signal_plot_widget, "Signal Chart")

        return self.drill_tabs

    # --- Panel A: Config sidebar --------------------------------------

    def _build_panel_a(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFrameShape(QFrame.Shape.NoFrame)
        outer.setMinimumWidth(220)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Strategy
        strat_group = QGroupBox("Strategy")
        strat_layout = QVBoxLayout(strat_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strat_layout.addWidget(self.strategy_combo)
        layout.addWidget(strat_group)

        # Hyperparameters (rebuilt dynamically when strategy changes)
        self.params_group = QGroupBox("Hyperparameters")
        params_group_layout = QVBoxLayout(self.params_group)
        params_group_layout.setContentsMargins(6, 6, 6, 6)
        params_group_layout.setSpacing(4)

        self._params_rows_widget = QWidget()
        self.params_layout = QVBoxLayout(self._params_rows_widget)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(2)
        params_group_layout.addWidget(self._params_rows_widget)

        self._btn_add_hparam = QPushButton("+ Add Hyperparameter")
        self._btn_add_hparam.setStyleSheet(
            "background-color: #1a3a1a; color: #aaff00; "
            "height: 24px; border-radius: 3px; font-size: 11px;"
        )
        self._btn_add_hparam.clicked.connect(self._add_hyperparam_dialog)
        params_group_layout.addWidget(self._btn_add_hparam)

        self._param_widgets: dict = {}
        layout.addWidget(self.params_group)

        # Batch config
        batch_group = QGroupBox("Batch Configuration")
        batch_form = QFormLayout(batch_group)

        self.tickers_input = QTextEdit()
        self.tickers_input.setPlaceholderText("AAPL, MSFT, GOOGL, ...")
        self.tickers_input.setMaximumHeight(70)
        batch_form.addRow("Tickers:", self.tickers_input)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1d", "1w", "4h", "1h", "15m"])
        batch_form.addRow("Interval:", self.interval_combo)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100.0, 10_000_000.0)
        self.capital_spin.setValue(10_000.0)
        self.capital_spin.setSingleStep(1_000.0)
        self.capital_spin.setPrefix("$")
        self.capital_spin.setDecimals(0)
        batch_form.addRow("Starting Capital:", self.capital_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 1.0)
        self.threshold_spin.setValue(0.2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        batch_form.addRow("Entry Threshold:", self.threshold_spin)

        layout.addWidget(batch_group)

        # Run / Cancel
        self.btn_run = QPushButton("Run Batch")
        self.btn_run.setStyleSheet(
            "background-color: #1a4a1a; color: #aaff00; font-weight: bold; "
            "height: 38px; font-size: 13px; border-radius: 4px;"
        )
        self.btn_run.clicked.connect(self._run_batch)
        layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "background-color: #4a1a1a; color: #ff8888; height: 28px; border-radius: 4px;")
        self.btn_cancel.clicked.connect(self._cancel_batch)
        layout.addWidget(self.btn_cancel)

        # Status / log
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(200)
        self.log_console.setStyleSheet(
            "background-color: #0a0a0a; color: #00ff00; "
            "font-family: monospace; font-size: 10px;")
        status_layout.addWidget(self.log_console)
        layout.addWidget(status_group)

        layout.addStretch()
        outer.setWidget(inner)
        return outer

    # -----------------------------------------------------------------------
    # Called when the Backtest tab becomes visible
    # -----------------------------------------------------------------------

    def refresh_strategies(self):
        """Repopulate strategy combo from the workspace directory."""
        current = self.strategy_combo.currentText()
        self.strategy_combo.blockSignals(True)
        self.strategy_combo.clear()
        try:
            strategies = self._engine.list_strategies()
            self.strategy_combo.addItems(strategies)
            if current in strategies:
                self.strategy_combo.setCurrentText(current)
        except Exception as e:
            self._log(f"Could not list strategies: {e}")
        finally:
            self.strategy_combo.blockSignals(False)
        self._on_strategy_changed()

    # -----------------------------------------------------------------------
    # Strategy / param helpers
    # -----------------------------------------------------------------------

    def _on_strategy_changed(self):
        name = self.strategy_combo.currentText()
        if not name:
            return
        manifest_path = os.path.join(self._engine.workspace_dir, name, "manifest.json")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            return
        self._build_param_widgets(manifest.get("hyperparameters", {}))

    def _build_param_widgets(self, params: dict):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._param_widgets = {}
        for key, val in params.items():
            self._add_param_row(key, val)

    def _add_param_row(self, key: str, val):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        name_edit = QLineEdit(key)
        name_edit.setFixedWidth(110)
        name_edit.setToolTip("Parameter name — edit to rename")
        name_edit.setStyleSheet("color: #aaa; font-style: italic;")
        name_edit.editingFinished.connect(lambda: self._inline_rename_hyperparam(name_edit, key))
        row_layout.addWidget(name_edit)

        if isinstance(val, bool):
            w = QCheckBox()
            w.setChecked(val)
            w.stateChanged.connect(self._save_hparams)
        elif isinstance(val, int):
            w = QSpinBox()
            w.setRange(-100_000, 100_000)
            w.setValue(val)
            w.valueChanged.connect(self._save_hparams)
        elif isinstance(val, float):
            w = QDoubleSpinBox()
            w.setRange(-1e6, 1e6)
            w.setDecimals(4)
            w.setValue(val)
            w.valueChanged.connect(self._save_hparams)
        else:
            w = QLineEdit(str(val))
            w.editingFinished.connect(self._save_hparams)
        row_layout.addWidget(w, 1)
        self._param_widgets[key] = w

        btn_remove = QPushButton("x")
        btn_remove.setFixedWidth(22)
        btn_remove.setStyleSheet("color: #ff6666; font-weight: bold; padding: 2px;")
        btn_remove.setToolTip(f"Remove '{key}'")
        btn_remove.clicked.connect(lambda _, k=key: self._remove_hyperparam(k))
        row_layout.addWidget(btn_remove)

        self.params_layout.addWidget(row_widget)

    def _get_params(self) -> dict:
        out = {}
        for key, w in self._param_widgets.items():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[key] = w.value()
            elif isinstance(w, QCheckBox):
                out[key] = w.isChecked()
            elif isinstance(w, QLineEdit):
                out[key] = w.text()
        return out

    def _save_hparams(self):
        """Persist current widget values to manifest.json and regenerate context.py."""
        self._write_hparams_to_manifest(self._get_params())

    def _write_hparams_to_manifest(self, hparams: dict, remove_bounds_key: str = None):
        name = self.strategy_combo.currentText()
        if not name:
            return
        strategy_dir = os.path.join(self._engine.workspace_dir, name)
        manifest_path = os.path.join(strategy_dir, "manifest.json")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
        if remove_bounds_key:
            manifest.get("parameter_bounds", {}).pop(remove_bounds_key, None)
        manifest["hyperparameters"] = hparams
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
            self._engine.write_context_py(name, manifest.get("features", []), hparams)
        except Exception as e:
            self._log(f"Manifest save failed: {e}")

    def _add_hyperparam_dialog(self):
        param_name, ok = QInputDialog.getText(
            self, "Add Hyperparameter", "Parameter name (valid Python identifier):")
        if not ok or not param_name.strip():
            return
        param_name = param_name.strip()
        if not param_name.isidentifier():
            self._log(f"Invalid parameter name: '{param_name}' — must be a valid Python identifier.")
            return

        current = self._get_params()
        if param_name in current:
            self._log(f"Hyperparameter '{param_name}' already exists.")
            return

        type_str, ok = QInputDialog.getItem(
            self, "Parameter Type", "Select type:", ["float", "int", "str"], 0, False)
        if not ok:
            return

        if type_str == "float":
            val, ok = QInputDialog.getDouble(
                self, "Initial Value", f"Value for '{param_name}':", value=0.0, decimals=4)
        elif type_str == "int":
            val, ok = QInputDialog.getInt(
                self, "Initial Value", f"Value for '{param_name}':", value=0)
        else:
            val, ok = QInputDialog.getText(
                self, "Initial Value", f"Value for '{param_name}':")
            val = val.strip()
        if not ok:
            return

        current[param_name] = val
        self._write_hparams_to_manifest(current)
        self._build_param_widgets(current)

    def _inline_rename_hyperparam(self, name_edit: QLineEdit, old_key: str):
        new_key = name_edit.text().strip()
        if new_key == old_key:
            return
        if not new_key.isidentifier():
            self._log(f"Invalid name: '{new_key}' — must be a valid Python identifier.")
            name_edit.setText(old_key)
            return
        current = self._get_params()
        if new_key in current:
            self._log(f"Hyperparameter '{new_key}' already exists.")
            name_edit.setText(old_key)
            return
        new_params = {(new_key if k == old_key else k): v for k, v in current.items()}
        self._write_hparams_to_manifest(new_params)
        self._build_param_widgets(new_params)

    def _remove_hyperparam(self, key: str):
        current = self._get_params()
        current.pop(key, None)
        self._write_hparams_to_manifest(current, remove_bounds_key=key)
        self._build_param_widgets(current)

    # -----------------------------------------------------------------------
    # Batch execution
    # -----------------------------------------------------------------------

    def _run_batch(self):
        strategy = self.strategy_combo.currentText()
        if not strategy:
            self._log("Select a strategy first.")
            return

        raw = self.tickers_input.toPlainText().strip()
        tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
        if not tickers:
            self._log("Enter at least one ticker.")
            return

        capital   = self.capital_spin.value()
        interval  = self.interval_combo.currentText()
        timeframe = {
            "start":    "1900-01-01",
            "end":      pd.Timestamp.now().strftime("%Y-%m-%d"),
            "interval": interval,
        }

        # Wrap run_backtest so we can inject starting_capital via closure
        def _call(*args, callbacks):
            return self._engine.run_backtest(
                *args, callbacks=callbacks, starting_capital=capital
            )

        self._result = {}
        self.leaderboard.setRowCount(0)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._log(f"[Backtest] {strategy} × {len(tickers)} tickers ({interval})")

        self._worker = EngineWorker(_call, strategy, tickers, timeframe)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_batch_complete)
        self._worker.error.connect(self._on_batch_error)
        self._worker.start()

    def _cancel_batch(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._log("Cancellation requested…")
        self.btn_cancel.setEnabled(False)

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self._log(msg)

    def _on_batch_error(self, err: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._log(f"ERROR: {err}")

    def _on_batch_complete(self, result: dict):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        if result.get("cancelled"):
            self._log("Batch cancelled.")
            return

        self._result = result
        self._populate_leaderboard(result.get("metrics", {}))
        self._log(f"Done — {len(result.get('metrics', {}))} ticker(s) scored.")

    # -----------------------------------------------------------------------
    # Leaderboard population
    # -----------------------------------------------------------------------

    def _populate_leaderboard(self, metrics_map: dict):
        self.leaderboard.setSortingEnabled(False)
        self.leaderboard.setRowCount(0)

        for ticker, m in metrics_map.items():
            row = self.leaderboard.rowCount()
            self.leaderboard.insertRow(row)

            if "error" in m:
                ti = QTableWidgetItem(ticker)
                ti.setData(Qt.ItemDataRole.UserRole, ticker)
                ti.setForeground(QColor("#888"))
                self.leaderboard.setItem(row, 0, ti)
                ei = QTableWidgetItem(str(m["error"]))
                ei.setForeground(QColor("#ff6666"))
                self.leaderboard.setItem(row, 1, ei)
                continue

            for col_idx, (_, key, fmt) in enumerate(_COLS):
                if key == "ticker":
                    val_str = ticker
                    raw_val = None
                else:
                    raw = m.get(key)
                    if raw is None:
                        val_str, raw_val = "—", None
                    elif isinstance(raw, float) and raw == float('inf'):
                        val_str, raw_val = "∞", float('inf')
                    else:
                        try:
                            val_str = fmt.format(raw)
                            raw_val = raw
                        except (ValueError, TypeError):
                            val_str, raw_val = str(raw), None

                item = QTableWidgetItem(val_str)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col_idx == 0:
                    item.setData(Qt.ItemDataRole.UserRole, ticker)

                # Colour coding per column
                if key == "CAGR (%)":
                    if raw_val is not None:
                        if raw_val >= 15:
                            item.setForeground(QColor("#44ff88"))
                        elif raw_val > 0:
                            item.setForeground(QColor("#dddd44"))
                        else:
                            item.setForeground(QColor("#ff5555"))
                elif key == "Sharpe Ratio":
                    if raw_val is not None:
                        if raw_val >= 1.5:
                            item.setForeground(QColor("#44ff88"))
                        elif raw_val >= 0.5:
                            item.setForeground(QColor("#dddd44"))
                        elif raw_val < 0:
                            item.setForeground(QColor("#ff5555"))
                elif key == "Max Drawdown (%)":
                    if raw_val is not None:
                        if raw_val <= -30:
                            item.setForeground(QColor("#ff5555"))
                        elif raw_val <= -15:
                            item.setForeground(QColor("#dddd44"))

                self.leaderboard.setItem(row, col_idx, item)

        self.leaderboard.setSortingEnabled(True)

    # -----------------------------------------------------------------------
    # Drill-down: called when a leaderboard row is clicked
    # -----------------------------------------------------------------------

    def _on_ticker_selected(self, row: int, _col: int):
        item = self.leaderboard.item(row, 0)
        if not item:
            return
        ticker = item.data(Qt.ItemDataRole.UserRole)
        if not ticker:
            return
        self._populate_portfolio(ticker)
        self._populate_trade_log(ticker)
        self._populate_signal_chart(ticker)

    # --- Tab 1: Portfolio -----------------------------------------------

    def _populate_portfolio(self, ticker: str):
        self.portfolio_plot.clear()
        # Re-add legend after clear
        self.portfolio_plot.addLegend(offset=(10, 10))

        def _to_xy(series_list: list):
            xs = list(range(len(series_list)))
            ys = [p["v"] for p in series_list]
            return xs, ys

        portfolio  = self._result.get("portfolios", {}).get(ticker, [])
        bh_portf   = self._result.get("bh_portfolios", {}).get(ticker, [])

        if portfolio:
            xs, ys = _to_xy(portfolio)
            self.portfolio_plot.plot(xs, ys, pen=pg.mkPen("#44aaff", width=2), name="Strategy")

        if bh_portf:
            xs, ys = _to_xy(bh_portf)
            self.portfolio_plot.plot(xs, ys, pen=pg.mkPen("#888888", width=1.5), name="Buy & Hold")

        capital = self.capital_spin.value()
        self.portfolio_plot.addLine(y=capital, pen=pg.mkPen("#555", style=Qt.PenStyle.DashLine))
        self.portfolio_plot.setTitle(f"{ticker} — Portfolio vs Buy & Hold", color="#ddd")

    # --- Tab 2: Trade log -----------------------------------------------

    def _populate_trade_log(self, ticker: str):
        self.trade_table.setSortingEnabled(False)
        self.trade_table.setRowCount(0)
        trades = self._result.get("trade_logs", {}).get(ticker, [])

        for t in trades:
            row = self.trade_table.rowCount()
            self.trade_table.insertRow(row)

            ret = t.get("return_pct", 0.0)
            row_bg = QColor("#142814") if ret > 0 else QColor("#281414") if ret < 0 else None
            direction = str(t.get("direction", ""))

            cells = [
                str(t.get("entry_date", "")),
                str(t.get("exit_date", "")),
                direction,
                f"{t.get('entry_price', 0):.4f}",
                f"{t.get('exit_price', 0):.4f}",
                f"{ret:.2f}%",
                str(t.get("bars_held", "")),
            ]
            for col_idx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col_idx == 2:
                    item.setForeground(
                        QColor("#44ff88") if direction == "LONG" else QColor("#ff5555"))
                if row_bg:
                    item.setBackground(row_bg)
                self.trade_table.setItem(row, col_idx, item)

        self.trade_table.setSortingEnabled(True)

    # --- Tab 3: Signal chart -------------------------------------------

    def _populate_signal_chart(self, ticker: str):
        # Remove previous candle overlay and scatter items
        if self._signal_candle_overlay is not None:
            self.signal_plot.remove_overlay(self._signal_candle_overlay)
            self._signal_candle_overlay = None
        # Remove any leftover scatter items added directly
        for item in list(self.signal_plot.items):
            if isinstance(item, pg.ScatterPlotItem):
                self.signal_plot.removeItem(item)

        interval = self.interval_combo.currentText()
        try:
            df = self._engine.get_historical_data(ticker, interval)
            if df is None or df.empty:
                return
            df.columns = [c.capitalize() for c in df.columns]
        except Exception as e:
            self._log(f"Could not load chart data for {ticker}: {e}")
            return

        self._signal_df = df

        # Candles via CandleOverlay so auto_scale_y works correctly
        self._signal_candle_overlay = CandleOverlay()
        self.signal_plot.add_overlay(self._signal_candle_overlay)
        self.signal_plot.update_all(df)

        # Date → integer index lookup for trade markers
        ts_index = {str(ts): i for i, ts in enumerate(df.index)}

        trades = self._result.get("trade_logs", {}).get(ticker, [])
        entry_x, entry_y = [], []
        exit_x,  exit_y  = [], []

        price_range = float(df["High"].max() - df["Low"].min())
        offset = price_range * 0.025 if price_range > 0 else 0.5

        for t in trades:
            direction = t.get("direction", "LONG")
            for date_key, xs, ys, is_entry in (
                ("entry_date", entry_x, entry_y, True),
                ("exit_date",  exit_x,  exit_y,  False),
            ):
                # Try exact string match first, then nearest Timestamp
                dt_str = str(t.get(date_key, ""))
                loc = ts_index.get(dt_str)
                if loc is None:
                    try:
                        loc = df.index.get_loc(pd.Timestamp(dt_str))
                    except Exception:
                        continue

                if direction == "LONG":
                    y = float(df["Low"].iloc[loc]) - offset if is_entry \
                        else float(df["High"].iloc[loc]) + offset
                else:
                    y = float(df["High"].iloc[loc]) + offset if is_entry \
                        else float(df["Low"].iloc[loc]) - offset
                xs.append(loc)
                ys.append(y)

        if entry_x:
            self.signal_plot.addItem(pg.ScatterPlotItem(
                x=entry_x, y=entry_y,
                symbol="t1", size=14,
                brush=pg.mkBrush("#44ff88"), pen=pg.mkPen("#44ff88"),
            ))
        if exit_x:
            self.signal_plot.addItem(pg.ScatterPlotItem(
                x=exit_x, y=exit_y,
                symbol="t", size=14,
                brush=pg.mkBrush("#ff5555"), pen=pg.mkPen("#ff5555"),
            ))

        self.signal_plot.setTitle(f"{ticker} — Signals", color="#ddd")
        # Pan to the end of the data, showing roughly 252 bars (1 year)
        n = len(df)
        view_bars = min(252, n)
        self.signal_plot.getViewBox().setXRange(n - view_bars, n, padding=0.02)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_console.append(msg)
        self.log_console.ensureCursorVisible()
