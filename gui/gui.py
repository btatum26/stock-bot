import sys
import os
import logging
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QCheckBox, QInputDialog, QMessageBox, QSplitter, QTabWidget, QScrollArea, QFrame, QHBoxLayout)
from PyQt6.QtCore import Qt, QRectF, QThread, QTimer, pyqtSignal
import pyqtgraph as pg

from engine import ModelEngine

from .features.loader import load_features
from .features.base import LineOutput, LevelOutput, MarkerOutput, HeatmapOutput, FeatureResult
from .signals.base import SignalEvent

# Modular Components
from .gui_components.styling import setup_app_style
from .gui_components.axes import DateAxis
from .gui_components.controls import ControlBar
from .gui_components.feature_panel import FeaturePanel
from .gui_components.models_panel import ModelsPanel
from .gui_components.score_panel import ScorePanel
from .gui_components.plots import UnifiedPlot, CandleOverlay, VolumeOverlay, LineOverlay, LevelOverlay, ScoreOverlay


class EngineWorker(QThread):
    """QThread wrapper that runs a ModelEngine method off the UI thread."""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    log_msg  = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, engine_method, *args):
        super().__init__()
        self.engine_method = engine_method
        self.args = args
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        callbacks = {
            "on_progress": lambda p, m: self.progress.emit(p, m),
            "on_log":       lambda m:    self.log_msg.emit(m),
            "is_cancelled": lambda:      self._cancelled,
        }
        try:
            result = self.engine_method(*self.args, callbacks=callbacks)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class ChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Bot Pro")
        self.resize(1400, 900)
        setup_app_style(self)
        
        self.engine = ModelEngine(workspace_dir="./strategies", db_path="data/stocks.db")
        self.available_features = load_features()
        self.active_features = {}  # {name: {params, overlays, plot_item_ref}}

        # Strategy state
        self.current_strategy = None  # name of selected strategy workspace folder
        self._worker = None
        
        # State
        self.df = None
        self.timestamps = []
        self.sub_plots = {} # {feat_name: UnifiedPlot}
        self.v_lines = []
        self.signal_items = []
        self.score_plot = None
        self.score_plot_widget = None
        self.score_overlay = None
        self.score_cache = {} # {func_name: scores_series}

        self._init_ui()
        self.load_random()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. Main Horizontal Splitter (Left side Plot/Controls vs Right side Sidebar)
        self.main_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_h_splitter.setHandleWidth(6)
        main_layout.addWidget(self.main_h_splitter)
        
        # --- Left Side Container ---
        self.left_side = QWidget()
        left_layout = QVBoxLayout(self.left_side)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # Debounce timer — fires load_chart 400 ms after the user stops typing
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self.load_chart)

        # Header Controls
        self.controls = ControlBar(self)
        self.controls.ticker_input.returnPressed.connect(self.load_chart)
        self.controls.ticker_input.textChanged.connect(lambda: self._load_timer.start(400))
        self.controls.ticker_history.currentIndexChanged.connect(self.load_from_history)
        self.controls.interval_combo.currentIndexChanged.connect(self.load_chart)
        # Period combo only adjusts the visible window, no re-fetch
        self.controls.period_combo.currentIndexChanged.connect(self._apply_period_view)
        self.controls.btn_random.clicked.connect(self.load_random)
        
        # Connect Strategy Controls in Header
        self.controls.btn_save_strategy.clicked.connect(self.save_strategy)
        self.controls.btn_load_strategy.clicked.connect(self.load_strategy)
        self.controls.btn_rename_strategy.clicked.connect(self.rename_strategy)
        self.controls.btn_new_strategy.clicked.connect(self.new_strategy)
        self.controls.lbl_strategy_name.setText("Strategy: (none)")

        left_layout.addWidget(self.controls)
        
        # Plot Area Splitter
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot_splitter.setHandleWidth(6)
        left_layout.addWidget(self.plot_splitter)
        
        # Container for main plot
        self.main_plot_widget = pg.GraphicsLayoutWidget()
        self.main_plot_widget.setBackground('#1e1e1e')
        self.main_plot_widget.setContentsMargins(0, 0, 0, 0)
        self.plot_splitter.addWidget(self.main_plot_widget)
        
        self.main_plot = UnifiedPlot()
        self.main_plot_widget.addItem(self.main_plot)
        
        # Add basic overlays to main plot
        self.price_overlay = CandleOverlay()
        self.volume_overlay = VolumeOverlay()
        self.main_plot.add_overlay(self.price_overlay)
        self.main_plot.add_overlay(self.volume_overlay)
        
        self.main_plot.getAxis('left').setPen('#444')
        self.main_plot.getAxis('left').setWidth(40) # Keep axes aligned
        self.main_plot.getAxis('bottom').setPen('#444')
        
        self._setup_crosshair()
        self.main_h_splitter.addWidget(self.left_side)
        
        # --- Right Side: Sidebar Tabs (Full Height) ---
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setFixedWidth(350)
        
        # Features Tab (Wrapped in scroll area)
        self.feature_panel = FeaturePanel(self.available_features, self)
        self.feature_panel.btn_add_feat.clicked.connect(self.add_feature_ui)
        
        feature_scroll = QScrollArea()
        feature_scroll.setWidgetResizable(True)
        feature_scroll.setFrameShape(QFrame.Shape.NoFrame)
        feature_scroll.setWidget(self.feature_panel)
        self.sidebar_tabs.addTab(feature_scroll, "Features")
        
        # Combined Models & Training Tab
        self.models_panel = ModelsPanel(self)
        self.signals_panel = self.models_panel.signals_panel
        self.training_panel = self.models_panel.training_panel
        
        self.signals_panel.generate_requested.connect(self.preview_signals)
        self.signals_panel.set_active_requested.connect(self.set_active_model)
        self.signals_panel.delete_model_requested.connect(self.delete_model)
        self.signals_panel.rename_model_requested.connect(self.rename_model)
        self.training_panel.train_requested.connect(self.train_model)
        self.models_panel.parameters_changed.connect(self._autosave_strategy)
        
        self.sidebar_tabs.addTab(self.models_panel, "Models")
        
        # Scoring Tab
        self.score_panel = ScorePanel(self)
        self.score_panel.settings_changed.connect(self.update_score_visualization)
        self.sidebar_tabs.addTab(self.score_panel, "Scoring")
        
        self.main_h_splitter.addWidget(self.sidebar_tabs)
        
        # Initial Sync
        self._sync_score_panel()

        
        # Set stretch factors: Plot Area (0) gets all extra space
        self.main_h_splitter.setStretchFactor(0, 1)
        self.main_h_splitter.setStretchFactor(1, 0)
        
        self.showMaximized()

    def _sync_score_panel(self):
        self.score_panel.set_available_functions([])

    def _update_score_params(self):
        pass

    def _setup_crosshair(self):
        for v in self.v_lines:
            if v.scene(): v.scene().removeItem(v)
        self.v_lines = []
        
        v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.main_plot.addItem(v_line, ignoreBounds=True)
        self.main_plot.addItem(self.h_line, ignoreBounds=True)
        self.v_lines.append(v_line)

        # Restore lines for existing sub-plots
        for feat_name, data in self.active_features.items():
            if data.get("v_line") and data.get("plot") != self.main_plot:
                # Re-create the line since it was removed from scene
                v_sub = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
                data["plot"].addItem(v_sub, ignoreBounds=True)
                data["v_line"] = v_sub
                self.v_lines.append(v_sub)
        
        self.price_label = pg.TextItem(anchor=(0, 1), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.time_label = pg.TextItem(anchor=(1, 0), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.vol_label = pg.TextItem(anchor=(0, 0), color='#aaa', fill=pg.mkBrush(0, 0, 0, 150))
        self.score_label = pg.TextItem(text="Score: N/A", anchor=(0, 0), color='#00ff00', fill=pg.mkBrush(0, 0, 0, 200))
        self.score_label.setZValue(2000)
        self.score_label.hide()
        
        self.main_plot.addItem(self.price_label, ignoreBounds=True)
        self.main_plot.addItem(self.time_label, ignoreBounds=True)
        self.main_plot.addItem(self.vol_label, ignoreBounds=True)
        self.main_plot.addItem(self.score_label, ignoreBounds=True)
        
        self.proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        self.main_plot.sigPlotClicked.connect(self.on_plot_clicked)

    def on_plot_clicked(self, event):
        print(f"DEBUG: Plot Clicked at {event.scenePos()}")
        if self.df is None or self.df.empty: return
        
        pos = event.scenePos()
        if self.main_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.main_plot.vb.mapSceneToView(pos)
            idx = int(round(mouse_point.x()))
            print(f"DEBUG: Index {idx}")
            
            if 0 <= idx < len(self.df):
                settings = self.score_panel.get_settings()
                func_name = settings.get("function")
                print(f"DEBUG: Func {func_name}")
                if not func_name: return
                
                if func_name not in self.score_cache:
                    self.calculate_scores(func_name, settings.get("parameters", {}))
                
                if func_name in self.score_cache:
                    scores = self.score_cache[func_name]
                    if scores is not None and idx < len(scores):
                        score = scores.iloc[idx]
                        print(f"DEBUG: Score {score}")
                        if not np.isnan(score):
                            self.score_label.show()
                            color = '#00ff00' if score > 0 else '#ff4444' if score < 0 else '#aaaaaa'
                            self.score_label.setColor(color)
                            self.score_label.setText(f"Score: {score:.4f}")
                            
                            view_range = self.main_plot.vb.viewRange()
                            x_min = view_range[0][0]
                            y_max = view_range[1][1]
                            
                            offset_x = (view_range[0][1] - view_range[0][0]) * 0.2
                            self.score_label.setPos(x_min + offset_x, y_max)
                        else:
                            self.score_label.hide()
                    else:
                        print("DEBUG: Index out of scores range")
                        self.score_label.hide()

    def calculate_scores(self, func_name, params):
        pass

    def mouse_moved(self, evt):
        pos = evt[0]
        if self.main_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.main_plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            idx = int(round(x))
            for v_line in self.v_lines: v_line.setPos(idx)
            self.h_line.setPos(y)
            
            view_range = self.main_plot.vb.viewRange()
            x_max, x_min = view_range[0][1], view_range[0][0]
            
            self.price_label.setPos(x_max, y)
            self.price_label.setText(f"{y:.2f}")
            self.price_label.setAnchor((1, 0.5))
            
            if self.df is not None and 0 <= idx < len(self.df):
                ts = pd.Timestamp(self.df.index[idx])
                self.time_label.setPos(idx, view_range[1][0])
                self.time_label.setText(ts.strftime('%Y-%m-%d %H:%M'))
                self.time_label.setAnchor((0.5, 1))
                
                vol = self.df['Volume'].iloc[idx]
                vol_str = f"{vol/1e6:.2f}M" if vol > 1e6 else f"{vol/1e3:.1f}K" if vol > 1e3 else str(int(vol))
                self.vol_label.setPos(x_min, view_range[1][1])
                self.vol_label.setText(f"Vol: {vol_str}")
                
                # Show score under crosshair
                settings = self.score_panel.get_settings()
                func_name = settings.get("function")
                if func_name:
                    if func_name not in self.score_cache:
                        # Optional: only calculate if "Active" is checked to avoid lag
                        if settings.get("active"):
                            self.calculate_scores(func_name, settings.get("parameters", {}))
                    
                    if func_name in self.score_cache:
                        scores = self.score_cache[func_name]
                        if scores is not None and idx < len(scores):
                            score = scores.iloc[idx]
                            if not np.isnan(score):
                                self.score_label.show()
                                color = '#00ff00' if score > 0 else '#ff4444' if score < 0 else '#aaaaaa'
                                self.score_label.setColor(color)
                                self.score_label.setText(f"Score: {score:.4f}")
                                
                            # Position to the right of Vol label and slightly lower
                                offset_x = (view_range[0][1] - view_range[0][0]) * 0.15
                                y_pos = view_range[1][1] - (view_range[1][1] - view_range[1][0]) * 0.05
                                self.score_label.setPos(x_min + offset_x, y_pos)
                            else:
                                self.score_label.hide()
                        else:
                            self.score_label.hide()
                    else:
                        self.score_label.hide()
                else:
                    self.score_label.hide()

    def add_feature_ui(self, initial_values=None):
        feat_name = self.feature_panel.feat_combo.currentData()
        if not feat_name: return
        self._add_feature_by_name(feat_name, initial_values)

    def _add_feature_by_name(self, feat_name, initial_values=None):
        if feat_name in self.active_features: return
        
        feature = self.available_features[feat_name]
        input_widgets, group_widget = self.feature_panel.create_feature_widget(
            feat_name, feature.parameters, 
            lambda: self.update_and_save(feat_name),
            self.remove_feature,
            initial_values=initial_values
        )

        plot_target = self.main_plot
        v_line = None
        container_widget = self.main_plot_widget
        reorganize_needed = False
        
        if feature.target_pane == "new":
            new_plot_widget = pg.GraphicsLayoutWidget()
            new_plot_widget.setBackground('#1e1e1e')
            new_plot_widget.setMinimumHeight(100)
            new_plot_widget.setContentsMargins(0, 0, 0, 0)
            self.plot_splitter.addWidget(new_plot_widget)
            container_widget = new_plot_widget
            
            new_plot = UnifiedPlot()
            if feature.y_range:
                new_plot.set_fixed_y_range(feature.y_range[0], feature.y_range[1], padding=feature.y_padding)
            
            new_plot.setXLink(self.main_plot)
            new_plot.getAxis('left').setWidth(40) # Axis alignment
            new_plot_widget.addItem(new_plot)
            plot_target = new_plot
            self.sub_plots[feat_name] = new_plot
            
            # Sub-plot cursor
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
            new_plot.addItem(v_line, ignoreBounds=True)
            self.v_lines.append(v_line)
            reorganize_needed = True

        self.active_features[feat_name] = {
            "instance": feature, "inputs": input_widgets, "overlays": [], "plot": plot_target, 
            "v_line": v_line, "widget": group_widget, "container": container_widget
        }
        self.update_feature(feat_name)
        if reorganize_needed:
            self._reorganize_subplots()
            
        self._autosave_strategy()

    def update_and_save(self, feat_name):
        self.update_feature(feat_name)
        self._autosave_strategy()

    def update_feature(self, feat_name):
        if self.df is None or self.df.empty: return
        data = self.active_features[feat_name]
        feat, plot = data["instance"], data["plot"]
        
        params = {k: w.currentText() if isinstance(w, QComboBox) else w.isChecked() if isinstance(w, QCheckBox) else w.text() 
                  for k, w in data["inputs"].items()}
        
        try: 
            result = feat.compute(self.df, params)
            if isinstance(result, FeatureResult):
                results = result.visuals
                data["raw_data"] = result.data
            else:
                results = result
        except Exception as e: 
            print(f"Error computing {feat_name}: {e}")
            return

        # Clear existing overlays for this feature
        for o in data["overlays"]:
            plot.remove_overlay(o)
        data["overlays"] = []
        
        for res in results:
            if isinstance(res, LineOutput):
                o = LineOverlay({res.name: res.data}, color=res.color, width=res.width)
                plot.add_overlay(o)
                data["overlays"].append(o)
            elif isinstance(res, LevelOutput):
                # Handle LevelOutput with LevelOverlay
                o = LevelOverlay(res.min_price, color=res.color)
                plot.add_overlay(o)
                data["overlays"].append(o)
            elif isinstance(res, MarkerOutput):
                # Fallback for MarkerOutput if no overlay exists yet
                sym = {'o':'o','d':'d','t':'t1','s':'s','x':'x','+':'+'}.get(res.shape, 'o')
                item = pg.ScatterPlotItem(x=res.indices, y=res.values, brush=pg.mkBrush(res.color), symbol=sym, size=10)
                plot.addItem(item)
                # Store in a temporary list to allow removal later (this is a bit hacky until we have MarkerOverlay)
                data.setdefault("temp_items", []).append(item)

        self.clear_score_cache()
        plot.update_all(self.df)
        self._refresh_score_underlay()

    def remove_feature(self, feat_name, widget, reorganize=True):
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
                # Destroy the plot widget entirely
                data["container"].deleteLater()
                if feat_name in self.sub_plots:
                    del self.sub_plots[feat_name]
                if reorganize:
                    self._reorganize_subplots()
            
            self.clear_score_cache()
            del self.active_features[feat_name]
            self._refresh_score_underlay()
        widget.deleteLater()
        self._autosave_strategy()

    def _reorganize_subplots(self):
        """Manages splitter sizes based on the 20%/40% rules."""
        count = self.plot_splitter.count()
        if count <= 1: return
        
        # Use a large base (10000) to set sizes as relative percentages
        num_features = count - 1
        total_units = 10000
        
        # Rule: Each feature initially takes 20% (2000 units).
        # IF the total for features exceeds 60% (6000 units), 
        # THEN main chart gets 40% (4000 units) and features share the remaining 6000.
        
        if num_features * 2000 <= 6000:
            feat_size = 2000
            main_size = total_units - (num_features * feat_size)
        else:
            main_size = 4000
            feat_size = 6000 // num_features
            
        sizes = [main_size] + [feat_size] * num_features
        self.plot_splitter.setSizes(sizes)

    def rename_strategy(self):
        QMessageBox.information(self, "Rename Strategy",
                                "Strategies are managed as workspace directories.\n"
                                "Rename the folder in the 'strategies/' directory directly.")

    def _sync_active_strategy(self):
        pass  # Feature overlays are UI-only; strategy definition lives in manifest.json

    def _autosave_strategy(self):
        self.signals_panel.refresh_models({}, None, self.active_features.keys())

    def save_strategy(self):
        QMessageBox.information(self, "Save Strategy",
                                "Strategies are workspace directories in 'strategies/'.\n"
                                "Use 'Load Strategy' to select a workspace, then edit\n"
                                "manifest.json and model.py directly.")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        event.accept()

    def new_strategy(self):
        name, ok = QInputDialog.getText(self, "New Strategy", "Strategy name (letters, digits, underscores):")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            self.engine.create_strategy(name)
        except Exception as e:
            QMessageBox.critical(self, "New Strategy", f"Failed to create strategy: {e}")
            return

        self.current_strategy = name
        self.controls.lbl_strategy_name.setText(f"Strategy: {name}")
        self.signals_panel.refresh_models({}, None, self.active_features.keys())
        self._sync_score_panel()
        QMessageBox.information(self, "New Strategy", f"Strategy '{name}' created in strategies/{name}/")

    def load_strategy(self):
        try:
            available = self.engine.list_strategies()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not list strategies: {e}")
            return

        if not available:
            QMessageBox.warning(self, "Load Strategy",
                                "No strategy workspaces found in 'strategies/'.\n"
                                "Create one with: uv run python main.py INIT --strategy <name>")
            return

        name, ok = QInputDialog.getItem(self, "Load Strategy", "Select a strategy workspace:", available, 0, False)
        if not ok or not name:
            return

        self.current_strategy = name
        self.controls.lbl_strategy_name.setText(f"Strategy: {name}")
        self.signals_panel.refresh_models({}, None, self.active_features.keys())
        self._sync_score_panel()

    def set_active_model(self, model_id):
        pass

    def rename_model(self, model_id):
        pass

    def delete_model(self, model_id):
        pass

    def train_model(self, settings):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Train", "Load data first.")
            return
        if not self.current_strategy:
            QMessageBox.warning(self, "Train", "Load a strategy workspace first.")
            return

        ticker = self.controls.ticker_input.text().upper()
        interval = self.controls.interval_combo.currentText()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")  # 5y

        self.training_panel.btn_train.setEnabled(False)
        self.training_panel.progress_bar.setValue(0)
        self.training_panel.log(f"Starting training for '{self.current_strategy}'...")

        self._worker = EngineWorker(
            self.engine.run_training,
            self.current_strategy,
            [ticker],
            {"start": start, "end": end, "interval": interval},
        )
        self._worker.progress.connect(lambda p, m: self.training_panel.progress_bar.setValue(p))
        self._worker.log_msg.connect(self.training_panel.log)
        self._worker.finished.connect(self._on_training_finished)
        self._worker.error.connect(self._on_training_error)
        self._worker.start()

    def _on_training_finished(self, result):
        self.training_panel.btn_train.setEnabled(True)
        self.training_panel.progress_bar.setValue(100)
        self.training_panel.log(f"Training complete: {result}")
        self.signals_panel.refresh_models({}, None, self.active_features.keys())

    def _on_training_error(self, err_msg):
        self.training_panel.btn_train.setEnabled(True)
        self.training_panel.progress_bar.setValue(0)
        self.training_panel.log(f"ERROR: {err_msg}")
        QMessageBox.critical(self, "Training Error", err_msg)

    def preview_signals(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Signals", "Load data first.")
            return
        if not self.current_strategy:
            QMessageBox.warning(self, "Signals", "Load a strategy workspace first.")
            return

        ticker = self.controls.ticker_input.text().upper()
        interval = self.controls.interval_combo.currentText()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")  # 5y

        self._worker = EngineWorker(
            self.engine.run_backtest,
            self.current_strategy,
            [ticker],
            {"start": start, "end": end, "interval": interval},
        )
        self._worker.finished.connect(lambda result: self._on_backtest_finished(result, ticker))
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Error", f"Signal generation failed: {e}"))
        self._worker.start()

    def _on_backtest_finished(self, result, ticker):
        if result.get("cancelled"):
            return

        raw_signals = result.get("signals", {}).get(ticker)
        if raw_signals is None or raw_signals.empty:
            QMessageBox.information(self, "Signals", "No signals generated.")
            return

        # Align signals onto self.df index
        raw_signals = raw_signals.reindex(self.df.index, fill_value=0.0)
        self.clear_score_cache()

        # Convert Series values to SignalEvents
        events = []
        for idx in raw_signals[raw_signals != 0].index:
            iloc = self.df.index.get_loc(idx)
            val = raw_signals[idx]
            side = 'buy' if val > 0 else 'sell'
            close_col = 'Close' if 'Close' in self.df.columns else 'close'
            events.append(SignalEvent(
                name=f"{self.current_strategy}_Signal",
                index=iloc, timestamp=idx,
                value=self.df[close_col].iloc[iloc],
                side=side,
                description=f"Strategy Signal ({side})",
            ))

        # Clear old signals
        for item in self.signal_items:
            self.main_plot.removeItem(item)
        self.signal_items = []

        if self.score_plot_widget:
            self.score_plot_widget.deleteLater()
            self.score_plot = None
            self.score_plot_widget = None
            self._reorganize_subplots()

        scatter_configs = {
            'buy':  {'color': '#00ff00', 'sym': 't1', 'size': 15},
            'sell': {'color': '#ff4444', 'sym': 't',  'size': 15},
        }
        scatters = {}
        for side, cfg in scatter_configs.items():
            s = pg.ScatterPlotItem(
                symbol=cfg['sym'], size=cfg['size'],
                brush=pg.mkBrush(cfg['color']),
                pen=pg.mkPen(cfg['color'], width=1),
                hoverable=True, tip=None,
            )
            s.sigHovered.connect(self.on_signal_hovered)
            self.main_plot.addItem(s)
            self.signal_items.append(s)
            scatters[side] = s

        high_col = 'High' if 'High' in self.df.columns else 'high'
        low_col  = 'Low'  if 'Low'  in self.df.columns else 'low'
        price_range = self.df[high_col].max() - self.df[low_col].min()
        offset = price_range * 0.03

        for e in events:
            if e.side in scatters:
                row = self.df.iloc[e.index]
                pos_y = row[low_col] - offset if e.side == 'buy' else row[high_col] + offset
                scatters[e.side].addPoints([{'pos': (e.index, pos_y), 'data': e}])

        QMessageBox.information(self, "Signals",
                                f"Generated {len(events)} signals for {ticker} "
                                f"(strategy: {self.current_strategy}).")
        self._refresh_score_underlay()

    def clear_score_cache(self):
        """Invalidates the score cache when data or features change."""
        self.score_cache = {}

    def _refresh_score_underlay(self):
        """Refreshes the score underlay if it's currently active in the panel."""
        settings = self.score_panel.get_settings()
        if settings.get("active"):
            self.update_score_visualization(settings)

    def update_score_visualization(self, settings):
        if not settings.get("active"):
            if self.score_overlay:
                self.main_plot.remove_overlay(self.score_overlay)
                self.score_overlay = None
                self.score_label.hide()
                if self.df is not None:
                    self.main_plot.update_all(self.df)

    def on_signal_hovered(self, scatter, points):
        from PyQt6.QtWidgets import QToolTip
        if len(points) > 0 and points:
            p = points[0]
            e = p.data() # Get the SignalEvent object
            if e:
                date_str = pd.Timestamp(e.timestamp).strftime('%Y-%m-%d %H:%M')
                text = (f"<b>{e.side.upper()} SIGNAL</b><br>"
                        f"Date: {date_str}<br>"
                        f"Price: {e.value:.2f}<br>"
                        f"Desc: {e.description}")
                
                # Show tooltip at mouse position
                QToolTip.showText(self.cursor().pos(), text)
        else:
            QToolTip.hideText()

    def load_from_history(self, index):
        if index <= 0: return 
        self.controls.ticker_input.setText(self.controls.ticker_history.currentText())
        self.load_chart()

    def load_random(self):
        try:
            tickers = self.engine.list_cached_tickers()
        except Exception:
            tickers = []
        ticker = random.choice(tickers) if tickers else random.choice(["AAPL", "MSFT", "GOOGL"])
        self.controls.ticker_input.setText(ticker)
        self.load_chart()

    # Maps period label → number of bars to show (approximate, used for view only)
    _PERIOD_BARS = {
        "1mo":  21,
        "3mo":  63,
        "6mo":  126,
        "1y":   252,
        "2y":   504,
        "5y":   1260,
        "10y":  2520,
        "All":  None,  # show everything
    }

    def _apply_period_view(self):
        """Pan/zoom the chart to the selected view period without re-fetching."""
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
        self._load_timer.stop()  # cancel any pending debounce
        ticker = self.controls.ticker_input.text().strip().upper()
        if not ticker:
            return
        interval = self.controls.interval_combo.currentText()

        try:
            df = self.engine.get_historical_data(ticker, interval)
        except Exception as e:
            QMessageBox.warning(self, "Load Data", f"Failed to fetch {ticker}: {e}")
            logging.exception(f"Error fetching data for {ticker} ({interval}): {e}")
            raise e

        if df is None or df.empty:
            QMessageBox.warning(self, "Load Data", f"No data found for {ticker} ({interval}).")
            return

        # Normalize column names: DataBroker returns lowercase, chart code expects Title-case
        df.columns = [c.capitalize() for c in df.columns]
        self.df = df
            
        self.controls.add_to_history(ticker)

        self.clear_score_cache()

        # Clear signal markers
        for item in self.signal_items:
            self.main_plot.removeItem(item)
        self.signal_items = []
        
        # Clear score plot
        if self.score_plot_widget:
            self.score_plot_widget.deleteLater()
            self.score_plot = None
            self.score_plot_widget = None
            self._reorganize_subplots()

        self.timestamps = self.df.index.astype(str).tolist()
        ts_list = self.timestamps
        self.main_plot.getAxis('bottom').tickStrings = lambda values, scale, spacing: [
            pd.Timestamp(ts_list[int(v)]).strftime('%H:%M\n%m-%d' if spacing < 5 else '%Y-%m-%d') 
            if 0 <= int(v) < len(ts_list) else '' for v in values]
        
        # Update main plot overlays
        self.main_plot.update_all(self.df)
        
        # Update all active feature overlays
        for name in list(self.active_features.keys()):
            self.update_feature(name)
            
        self._sync_score_panel()
            
        # Apply the current period view after loading
        self._apply_period_view()
        self._refresh_score_underlay()

def main():
    app = QApplication(sys.argv)
    window = ChartWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
