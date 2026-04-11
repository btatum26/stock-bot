"""
TrainingPanel — single-ticker training UI with results display.

Layout (horizontal splitter):
  Left:
    Results table (train vs val metrics side-by-side)
  Right (fixed ~350 px sidebar):
    Strategy selector, ticker/interval/timeframe, hyperparameters,
    Train button, progress bar, log console
"""

import os
import json

import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFormLayout, QLabel, QComboBox, QPushButton,
    QTextEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit,
    QProgressBar, QScrollArea, QFrame, QInputDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from engine import ModelEngine
from engine.core.universes import UNIVERSES, list_universes


# ---------------------------------------------------------------------------
# Background worker (same pattern as backtest)
# ---------------------------------------------------------------------------

class TrainWorker(QThread):
    """Runs ModelEngine.run_training off the UI thread."""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    log_msg  = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, engine: ModelEngine, strategy_name: str,
                 assets: list, timeframe: dict):
        super().__init__()
        self._engine = engine
        self._strategy = strategy_name
        self._assets = assets
        self._timeframe = timeframe
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
            result = self._engine.run_training(
                self._strategy, self._assets, self._timeframe,
                callbacks=callbacks,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Metrics display columns
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    ("Metric",    "metric"),
    ("Train",     "train"),
    ("Validation", "val"),
]


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class TrainingPanel(QWidget):
    def __init__(self, engine: ModelEngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._worker = None

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

        # Left: results
        splitter.addWidget(self._build_results_panel())
        # Right: config sidebar
        splitter.addWidget(self._build_sidebar())
        splitter.setSizes([9999, 350])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

    # --- Left: Results table -----------------------------------------------

    def _build_results_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 4)

        hdr = QLabel("Training Results")
        hdr.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #ddd; padding-bottom: 4px;"
        )
        layout.addWidget(hdr)

        self.split_info_label = QLabel("")
        self.split_info_label.setStyleSheet(
            "color: #aaa; font-size: 11px; padding: 2px 0 6px 0;"
        )
        self.split_info_label.setWordWrap(True)
        layout.addWidget(self.split_info_label)

        self.results_table = QTableWidget(0, len(_METRIC_COLS))
        self.results_table.setHorizontalHeaderLabels([c[0] for c in _METRIC_COLS])
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet("""
            QTableWidget          { background-color: #1a1a1a; gridline-color: #333; color: #ddd; }
            QTableWidget::item:selected { background-color: #2a4a7a; }
            QHeaderView::section  { background-color: #252525; color: #aaa;
                                    padding: 4px; border: 1px solid #444; }
            QTableWidget::item:alternate { background-color: #1e1e1e; }
        """)
        layout.addWidget(self.results_table)
        return w

    # --- Right: Config sidebar ---------------------------------------------

    def _build_sidebar(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFrameShape(QFrame.Shape.NoFrame)
        outer.setMinimumWidth(220)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Strategy selector
        strat_group = QGroupBox("Strategy")
        strat_layout = QVBoxLayout(strat_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strat_layout.addWidget(self.strategy_combo)
        layout.addWidget(strat_group)

        # Data scope
        data_group = QGroupBox("Data")
        data_form = QFormLayout(data_group)

        # TODO: Decide long-term shape. For now expose BOTH a typed list
        # and a universe dropdown — typed tickers take precedence, and
        # if empty we fall back to the selected universe. Revisit once
        # usage pattern is clearer (e.g. always-universe vs always-list
        # vs tag-based picker).
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("AAPL, MSFT, ...")
        self.ticker_input.setToolTip(
            "Comma-separated list of tickers. Overrides the universe "
            "dropdown if non-empty."
        )
        data_form.addRow("Tickers:", self.ticker_input)

        self.universe_combo = QComboBox()
        self.universe_combo.addItem("(none)")
        self.universe_combo.addItems(list_universes())
        self.universe_combo.setToolTip(
            "Preset ticker universe. Used when the Tickers field is empty."
        )
        data_form.addRow("Universe:", self.universe_combo)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1d", "1wk", "4h", "1h", "15m"])
        data_form.addRow("Interval:", self.interval_combo)

        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("1900-01-01")
        self.start_input.setText("1900-01-01")
        data_form.addRow("Start:", self.start_input)

        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText("2026-04-09")
        self.end_input.setText(pd.Timestamp.now().strftime("%Y-%m-%d"))
        data_form.addRow("End:", self.end_input)

        self.price_norm_combo = QComboBox()
        self.price_norm_combo.addItems(["none", "log_returns"])
        data_form.addRow("Price Norm:", self.price_norm_combo)

        layout.addWidget(data_group)

        # Hyperparameters (rebuilt when strategy changes)
        self.params_group = QGroupBox("Hyperparameters")
        params_group_layout = QVBoxLayout(self.params_group)
        params_group_layout.setContentsMargins(6, 6, 6, 6)
        params_group_layout.setSpacing(4)

        self._params_rows_widget = QWidget()
        self.params_layout = QVBoxLayout(self._params_rows_widget)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(2)
        params_group_layout.addWidget(self._params_rows_widget)

        self._param_widgets: dict = {}
        layout.addWidget(self.params_group)

        # Train button
        self.btn_train = QPushButton("Train")
        self.btn_train.setStyleSheet(
            "background-color: #1a3a5a; color: #88ccff; font-weight: bold; "
            "height: 38px; font-size: 13px; border-radius: 4px;"
        )
        self.btn_train.clicked.connect(self._run_training)
        layout.addWidget(self.btn_train)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "background-color: #4a1a1a; color: #ff8888; height: 28px; border-radius: 4px;"
        )
        self.btn_cancel.clicked.connect(self._cancel_training)
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
            "font-family: monospace; font-size: 10px;"
        )
        status_layout.addWidget(self.log_console)
        layout.addWidget(status_group)

        layout.addStretch()
        outer.setWidget(inner)
        return outer

    # -----------------------------------------------------------------------
    # Called when Training tab becomes visible
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
        manifest_path = os.path.join(
            self._engine.workspace_dir, name, "manifest.json"
        )
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            return
        self._build_param_widgets(manifest.get("hyperparameters", {}))

        # Sync price norm dropdown with whatever is stored in the manifest
        saved_norm = manifest.get("training", {}).get("price_normalization", "none")
        idx = self.price_norm_combo.findText(saved_norm)
        if idx >= 0:
            self.price_norm_combo.setCurrentIndex(idx)

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

        label = QLabel(key)
        label.setFixedWidth(110)
        label.setStyleSheet("color: #aaa;")
        row_layout.addWidget(label)

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
        name = self.strategy_combo.currentText()
        if not name:
            return
        manifest_path = os.path.join(
            self._engine.workspace_dir, name, "manifest.json"
        )
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
        hparams = self._get_params()
        manifest["hyperparameters"] = hparams
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
            self._engine.write_context_py(name, manifest.get("features", []), hparams)
        except Exception as e:
            self._log(f"Manifest save failed: {e}")

    # -----------------------------------------------------------------------
    # Training execution
    # -----------------------------------------------------------------------

    def _run_training(self):
        strategy = self.strategy_combo.currentText()
        if not strategy:
            self._log("Select a strategy first.")
            return

        assets = self._resolve_assets()
        if not assets:
            self._log("Enter at least one ticker or pick a universe.")
            return

        interval = self.interval_combo.currentText()
        start = self.start_input.text().strip() or "1900-01-01"
        end = self.end_input.text().strip() or pd.Timestamp.now().strftime("%Y-%m-%d")

        timeframe = {
            "start": start,
            "end": end,
            "interval": interval,
        }

        # Persist price normalization choice to the manifest so the backtester
        # applies the same transform during inference.
        price_norm = self.price_norm_combo.currentText()
        try:
            self._engine.save_training_config(strategy, {"price_normalization": price_norm})
        except Exception as e:
            self._log(f"Warning: could not save training config: {e}")

        self.results_table.setRowCount(0)
        self.split_info_label.setText("")
        self.progress_bar.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        ticker_summary = ", ".join(assets) if len(assets) <= 5 else f"{len(assets)} tickers"
        self._log(f"[Train] {strategy} on {ticker_summary} ({interval})")

        self._worker = TrainWorker(
            self._engine, strategy, assets, timeframe
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_training_complete)
        self._worker.error.connect(self._on_training_error)
        self._worker.start()

    def _resolve_assets(self) -> list:
        """Combines the typed ticker field and the universe dropdown.

        Typed tickers take precedence. If the typed field is empty, the
        selected universe (if any) is used. Returns an empty list if
        neither source yields tickers.
        """
        raw = self.ticker_input.text().strip()
        if raw:
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            # De-dupe while preserving order
            seen = set()
            out = []
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out

        universe_name = self.universe_combo.currentText()
        if universe_name and universe_name != "(none)":
            return list(UNIVERSES.get(universe_name, []))

        return []

    def _cancel_training(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._log("Cancellation requested...")
        self.btn_cancel.setEnabled(False)

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self._log(msg)

    def _on_training_error(self, err: str):
        self.btn_train.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._log(f"ERROR: {err}")

    def _on_training_complete(self, result: dict):
        self.btn_train.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        if result.get("cancelled"):
            self._log("Training cancelled.")
            return

        self._log("Training complete.")
        self._populate_results(result)

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------

    def _populate_results(self, result: dict):
        train_metrics = result.get("train_metrics", {})
        val_metrics = result.get("val_metrics", {})
        split_info = result.get("split_info", {})
        params = result.get("params", {})
        optimal = result.get("optimal_params", {})

        # Split info label
        parts = []
        method = split_info.get("method", "unknown").upper()
        parts.append(f"Split: {method}")
        if method == "TEMPORAL":
            parts.append(
                f"Train: {split_info.get('train_size', '?')} bars  |  "
                f"Val: {split_info.get('val_size', '?')} bars"
            )
            tr = split_info.get("train_range", [])
            vr = split_info.get("val_range", [])
            if tr:
                parts.append(f"Train range: {tr[0]} -> {tr[1]}")
            if vr:
                parts.append(f"Val range: {vr[0]} -> {vr[1]}")
        elif method == "CPCV":
            parts.append(
                f"Folds: {split_info.get('n_folds', '?')}  |  "
                f"Groups: {split_info.get('n_groups', '?')} "
                f"(k={split_info.get('k_test_groups', '?')})"
            )
        elif method == "WALK_FORWARD":
            parts.append(
                f"Folds: {split_info.get('n_folds', '?')}  |  "
                f"Tickers: {split_info.get('n_tickers', '?')}  |  "
                f"Rows: {split_info.get('n_rows', '?')}"
            )
            tickers = split_info.get("tickers", [])
            if tickers:
                summary = ", ".join(tickers) if len(tickers) <= 8 else f"{len(tickers)} tickers"
                parts.append(f"Tickers: {summary}")
        if optimal:
            parts.append(f"Optimal params: {optimal}")

        self.split_info_label.setText("\n".join(parts))

        # Populate metrics table
        skip_keys = {"equity_curve", "portfolio", "bh_portfolio", "trade_log"}
        all_keys = [
            k for k in val_metrics if k not in skip_keys
        ]

        is_cpcv = all_keys and isinstance(val_metrics.get(all_keys[0]), dict)

        self.results_table.setRowCount(len(all_keys))
        for row, key in enumerate(all_keys):
            # Metric name
            name_item = QTableWidgetItem(key)
            name_item.setForeground(QColor("#ddd"))
            self.results_table.setItem(row, 0, name_item)

            if is_cpcv:
                t_val = train_metrics.get(key, {})
                v_val = val_metrics.get(key, {})
                t_str = f"{t_val.get('mean', 0):.4f} +/- {t_val.get('std', 0):.4f}"
                v_str = f"{v_val.get('mean', 0):.4f} +/- {v_val.get('std', 0):.4f}"
            else:
                t_raw = train_metrics.get(key)
                v_raw = val_metrics.get(key)
                t_str = self._fmt_metric(t_raw)
                v_str = self._fmt_metric(v_raw)

            train_item = QTableWidgetItem(t_str)
            train_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 1, train_item)

            val_item = QTableWidgetItem(v_str)
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 2, val_item)

            # Color-code: compare train vs val for overfitting hints
            self._color_metric_row(key, train_metrics.get(key), val_metrics.get(key),
                                   train_item, val_item, is_cpcv)

    @staticmethod
    def _fmt_metric(val) -> str:
        if val is None:
            return "-"
        if isinstance(val, float):
            if val == float("inf"):
                return "inf"
            return f"{val:.4f}"
        if isinstance(val, int):
            return str(val)
        return str(val)

    @staticmethod
    def _color_metric_row(key: str, t_val, v_val,
                          train_item: QTableWidgetItem,
                          val_item: QTableWidgetItem,
                          is_cpcv: bool):
        """Apply color hints based on metric quality."""
        if is_cpcv:
            v_num = v_val.get("mean", 0) if isinstance(v_val, dict) else 0
        else:
            v_num = v_val if isinstance(v_val, (int, float)) else 0

        lower_key = key.lower()
        if "sharpe" in lower_key or "sortino" in lower_key or "calmar" in lower_key:
            if v_num >= 1.5:
                val_item.setForeground(QColor("#44ff88"))
            elif v_num >= 0.5:
                val_item.setForeground(QColor("#dddd44"))
            elif v_num < 0:
                val_item.setForeground(QColor("#ff5555"))
        elif "drawdown" in lower_key:
            if v_num <= -30:
                val_item.setForeground(QColor("#ff5555"))
            elif v_num <= -15:
                val_item.setForeground(QColor("#dddd44"))
        elif "return" in lower_key or "cagr" in lower_key:
            if v_num > 0:
                val_item.setForeground(QColor("#44ff88"))
            elif v_num < 0:
                val_item.setForeground(QColor("#ff5555"))

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_console.append(msg)
        self.log_console.ensureCursorVisible()
