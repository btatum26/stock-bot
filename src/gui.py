import sys
import os
import pandas as pd
import numpy as np
import random
import shutil
import uuid
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QCheckBox, QInputDialog, QMessageBox, QSplitter, QTabWidget, QScrollArea, QFrame, QHBoxLayout)
from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal
import pyqtgraph as pg

from .database import Database
from .engine import TradingEngine, SignalEvaluation
from .features.loader import load_features
from .features.feature_set import FeatureSet
from .features.base import LineOutput, LevelOutput, MarkerOutput, HeatmapOutput, FeatureResult
from .strategy import Strategy
from .signals.base import SignalEvent

# Signal Models
from .signals.ml_models import MLSignalModel
from .signals.rule_based import DivergenceSignalModel

# Modular Components
from .gui_components.styling import setup_app_style
from .gui_components.axes import DateAxis
from .gui_components.controls import ControlBar
from .gui_components.feature_panel import FeaturePanel
from .gui_components.signals_panel import SignalsPanel
from .gui_components.training_panel import TrainingPanel
from .gui_components.score_panel import ScorePanel
from .gui_components.plots import UnifiedPlot, CandleOverlay, VolumeOverlay, LineOverlay, LevelOverlay, ScoreOverlay

class TrainingThread(QThread):
    finished = pyqtSignal(object, object) # model, metrics
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, script_instance, engine, active_features, settings, current_ticker, current_interval):
        super().__init__()
        self.script_instance = script_instance
        self.engine = engine
        self.active_features = active_features
        self.settings = settings
        self.current_ticker = current_ticker
        self.current_interval = current_interval

    def run(self):
        try:
            self.log.emit(f"Starting training session...")
            
            # 1. Resolve Tickers
            tickers = []
            mode = self.settings.get("ticker_mode")
            if mode == "Current Ticker":
                tickers = [self.current_ticker]
            elif mode == "Custom Basket":
                raw = self.settings.get("ticker_list", "")
                tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            elif mode == "Random Selection":
                count = self.settings.get("random_ticker_count", 5)
                all_available = self.engine.db.get_all_tickers()
                if len(all_available) > count:
                    import random
                    tickers = random.sample(all_available, count)
                else:
                    tickers = all_available
            
            if not tickers:
                raise Exception("No tickers selected for training.")

            # 2. Collect and Prepare Data
            combined_df_list = []
            combined_features_list = []
            
            for ticker in tickers:
                self.log.emit(f"Processing {ticker}...")
                df = self.engine.db.get_data(ticker, self.current_interval)
                if df.empty:
                    self.log.emit(f"  Warning: No data for {ticker}. Skipping.")
                    continue
                
                # Apply Range Mode (Slicing)
                ranges = []
                if self.settings.get("range_mode") == "Random Slices":
                    num_slices = self.settings.get("slice_count", 5)
                    slice_size = self.settings.get("slice_size", 500)
                    if len(df) > slice_size:
                        import random
                        for _ in range(num_slices):
                            start_idx = random.randint(0, len(df) - slice_size)
                            ranges.append(df.iloc[start_idx : start_idx + slice_size])
                    else:
                        ranges.append(df)
                else:
                    ranges.append(df)

                for sub_df in ranges:
                    # Compute features for this slice
                    feat_data = {"Volume": sub_df['Volume']}
                    for feat_name, data in self.active_features.items():
                        feat = data["instance"]
                        params = {k: w.currentText() if isinstance(w, QComboBox) else w.isChecked() if isinstance(w, QCheckBox) else w.text() 
                                  for k, w in data["inputs"].items()}
                        res = feat.compute(sub_df, params)
                        if isinstance(res, FeatureResult):
                            feat_data.update(res.data)
                        elif hasattr(res, 'data'):
                            feat_data.update(res.data)
                    
                    combined_df_list.append(sub_df)
                    combined_features_list.append(pd.DataFrame(feat_data, index=sub_df.index))

            if not combined_df_list:
                raise Exception("Failed to collect any data for training.")

            # 3. Concatenate all data
            final_df = pd.concat(combined_df_list).reset_index(drop=True)
            # For features, we can't easily concat if keys differ, but they should be consistent across tickers
            final_features_df = pd.concat(combined_features_list).reset_index(drop=True)
            
            # Convert back to dict for the script
            final_feature_dict = {col: final_features_df[col] for col in final_features_df.columns}

            # 4. Train
            if not hasattr(self.script_instance, 'train'):
                raise Exception("The strategy script does not have a 'train' method implemented.")
                
            model, metrics = self.script_instance.train(
                final_df, 
                final_feature_dict, 
                self.settings
            )
            
            self.log.emit("Training complete.")
            self.finished.emit(model, metrics)
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
        
        self.db = Database("data/stocks.db")
        self.engine = TradingEngine()
        self.available_features = load_features()
        self.active_features = {} # {name: {params, overlays, plot_item_ref}}
        
        # Initial strategy is saved in a hidden location
        self.strategy = Strategy("Default", directory="data/.internal_strategies")
        
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
        
        # Header Controls
        self.controls = ControlBar(self)
        self.controls.ticker_input.returnPressed.connect(self.load_chart)
        self.controls.ticker_history.currentIndexChanged.connect(self.load_from_history)
        self.controls.interval_combo.currentIndexChanged.connect(self.load_chart)
        self.controls.btn_load.clicked.connect(self.load_chart)
        self.controls.btn_random.clicked.connect(self.load_random)
        
        # Connect Strategy Controls in Header
        self.controls.btn_save_strategy.clicked.connect(self.save_strategy)
        self.controls.btn_load_strategy.clicked.connect(self.load_strategy)
        self.controls.btn_rename_strategy.clicked.connect(self.rename_strategy)
        self.controls.lbl_strategy_name.setText(f"Strategy: {self.strategy.name}")
        
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
        
        # Training Tab
        self.training_panel = TrainingPanel(self)
        self.training_panel.train_requested.connect(self.train_model)
        self.sidebar_tabs.addTab(self.training_panel, "Training")
        
        # Models Tab (formerly Signals)
        self.signals_panel = SignalsPanel(self)
        self.signals_panel.generate_requested.connect(self.preview_signals)
        self.signals_panel.set_active_requested.connect(self.set_active_model)
        self.signals_panel.delete_model_requested.connect(self.delete_model)
        self.signals_panel.rename_model_requested.connect(self.rename_model)
        self.sidebar_tabs.addTab(self.signals_panel, "Models")
        
        # Scoring Tab
        self.score_panel = ScorePanel(self)
        self.score_panel.settings_changed.connect(self.update_score_visualization)
        self.sidebar_tabs.addTab(self.score_panel, "Scoring")
        
        self.main_h_splitter.addWidget(self.sidebar_tabs)
        
        # Initial Sync
        self._sync_score_panel()
        script_instance = self.strategy.get_script_instance()
        if script_instance and hasattr(script_instance, 'parameters'):
            self.training_panel.set_parameters(script_instance.parameters)
        
        # Set stretch factors: Plot Area (0) gets all extra space
        self.main_h_splitter.setStretchFactor(0, 1)
        self.main_h_splitter.setStretchFactor(1, 0)
        
        self.showMaximized()

    def _sync_score_panel(self):
        """Populates the score panel with available functions from the strategy script."""
        script_instance = self.strategy.get_script_instance()
        if script_instance:
            score_funcs = [m for m in dir(script_instance) if m.startswith('calculate_') and m.endswith('_scores')]
            self.score_panel.set_available_functions(score_funcs)
            # Auto-select first if none selected
            if score_funcs and not self.score_panel.func_combo.currentText():
                self.score_panel.func_combo.setCurrentIndex(0)

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
        
        self.main_plot.addItem(self.price_label, ignoreBounds=True)
        self.main_plot.addItem(self.time_label, ignoreBounds=True)
        self.main_plot.addItem(self.vol_label, ignoreBounds=True)
        
        self.proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

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
        name, ok = QInputDialog.getText(self, "Rename Strategy", "Enter new name:", text=self.strategy.name)
        if ok and name:
            self.strategy.name = name
            self.controls.lbl_strategy_name.setText(f"Strategy: {name}")
            self._autosave_strategy()

    def _sync_active_strategy(self):
        """Syncs GUI feature state into self.strategy object."""
        feature_config = {}
        for feat_name, data in self.active_features.items():
            params = {}
            for k, w in data["inputs"].items():
                if isinstance(w, QComboBox): params[k] = w.currentText()
                elif isinstance(w, QCheckBox): params[k] = w.isChecked()
                else: params[k] = w.text()
            feature_config[feat_name] = params
        self.strategy.feature_config = feature_config

    def _autosave_strategy(self):
        """Saves current state without prompting."""
        self._sync_active_strategy()
        try:
            self.strategy.save()
            # Update models list to reflect feature compatibility changes
            self.signals_panel.refresh_models(self.strategy.model_instances, self.strategy.active_model_id, self.active_features.keys())
        except Exception as e:
            print(f"Autosave failed: {e}")

    def save_strategy(self):
        name, ok = QInputDialog.getText(self, "Save Strategy", "Enter a name for this strategy:", text=self.strategy.name)
        if not ok or not name: return
        
        old_script_path = self.strategy.script_path
        
        self.strategy.name = name
        self.strategy.directory = "strategies" # Move from internal to main if it was internal
        self._sync_active_strategy()
        self.controls.lbl_strategy_name.setText(f"Strategy: {name}")
        
        # Ensure the physical .py file follows the name change
        new_script_path = self.strategy.script_path
        if os.path.exists(old_script_path) and old_script_path != new_script_path:
            try:
                os.makedirs(os.path.dirname(new_script_path), exist_ok=True)
                shutil.copy(old_script_path, new_script_path)
            except Exception as e:
                print(f"Error moving script file: {e}")

        try:
            self.strategy.save()
            QMessageBox.information(self, "Save Strategy", f"Strategy '{name}' saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save strategy: {e}")

    def closeEvent(self, event):
        """Cleanup working scripts on exit."""
        # 1. Handle main strategies
        working_dir = os.path.join("strategies", "working_scripts")
        self._cleanup_working_dir(working_dir, "strategies")
        
        # 2. Handle internal strategies
        internal_working_dir = os.path.join("data", ".internal_strategies", "working_scripts")
        self._cleanup_working_dir(internal_working_dir, "data/.internal_strategies")
        
        event.accept()

    def _cleanup_working_dir(self, working_dir, strat_dir):
        if os.path.exists(working_dir):
            for f in os.listdir(working_dir):
                if f.endswith(".py"):
                    name = f[:-3]
                    try:
                        strat = Strategy.load(name, directory=strat_dir)
                        strat.save()
                    except Exception as e:
                        print(f"Error syncing strategy {name}: {e}")
            try:
                shutil.rmtree(working_dir, ignore_errors=True)
                os.makedirs(working_dir, exist_ok=True)
            except Exception as e:
                print(f"Error wiping {working_dir}: {e}")

    def load_strategy(self):
        available = Strategy.list_available()
        if not available:
            QMessageBox.warning(self, "Load Strategy", "No saved strategies found.")
            return

        name, ok = QInputDialog.getItem(self, "Load Strategy", "Select a strategy:", available, 0, False)
        if not ok or not name: return

        try:
            # 1. Clear current features
            for feat_name in list(self.active_features.keys()):
                data = self.active_features[feat_name]
                self.remove_feature(feat_name, data["widget"], reorganize=False)
            self._reorganize_subplots()

            # 2. Load strategy
            self.strategy = Strategy.load(name)
            self.controls.lbl_strategy_name.setText(f"Strategy: {self.strategy.name}")
            
            # Update Training Parameters from Script
            script_instance = self.strategy.get_script_instance()
            if script_instance and hasattr(script_instance, 'parameters'):
                self.training_panel.set_parameters(script_instance.parameters)
            else:
                self.training_panel._setup_default_params()
            
            # 3. Add features from config
            for feat_name, params in self.strategy.feature_config.items():
                if feat_name in self.available_features:
                    self._add_feature_by_name(feat_name, initial_values=params)
            
            # 4. Refresh models list
            self.signals_panel.refresh_models(self.strategy.model_instances, self.strategy.active_model_id, self.active_features.keys())
            
            # 5. Sync score panel functions
            self._sync_score_panel()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load strategy: {e}")
            import traceback
            traceback.print_exc()

    def set_active_model(self, model_id):
        self.strategy.active_model_id = model_id
        self.strategy.save()
        self.signals_panel.refresh_models(self.strategy.model_instances, self.strategy.active_model_id, self.active_features.keys())
        self._refresh_score_underlay() # New active model changes signals/scores

    def rename_model(self, model_id):
        if model_id not in self.strategy.model_instances: return
        info = self.strategy.model_instances[model_id]
        current_name = info.get('comment', '')
        name, ok = QInputDialog.getText(self, "Rename Model", "Enter new name/comment:", text=current_name)
        if ok:
            info['comment'] = name
            self.strategy.save()
            self.signals_panel.refresh_models(self.strategy.model_instances, self.strategy.active_model_id, self.active_features.keys())

    def delete_model(self, model_id):
        if model_id in self.strategy.model_instances:
            reply = QMessageBox.question(self, "Delete Model", "Are you sure you want to delete this trained model?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.strategy.model_instances.pop(model_id)
                if self.strategy.active_model_id == model_id:
                    self.strategy.active_model_id = None
                self.strategy.save()
                self.signals_panel.refresh_models(self.strategy.model_instances, self.strategy.active_model_id, self.active_features.keys())
                self._refresh_score_underlay()

    def train_model(self, settings):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Train", "Load data first.")
            return
            
        features_to_use = list(self.active_features.keys())
        if not features_to_use:
            QMessageBox.warning(self, "Train", "Add at least one feature (e.g. RSI) to train the model on.")
            return

        script_instance = self.strategy.get_script_instance()
        if not script_instance:
            QMessageBox.warning(self, "Train", "Could not load the strategy script.")
            return
            
        if hasattr(script_instance, 'features_to_use'):
            script_instance.features_to_use = features_to_use
        
        self.training_panel.btn_train.setEnabled(False)
        self.training_panel.progress_bar.setValue(10)
        self.training_panel.log("Initializing multi-ticker training thread...")
        
        # Start Thread with new signature
        self.training_thread = TrainingThread(
            script_instance, 
            self.engine, 
            self.active_features, 
            settings,
            self.controls.ticker_input.text().upper(),
            self.controls.interval_combo.currentText()
        )
        self.training_thread.finished.connect(lambda w, m: self._on_training_finished(w, m, settings, features_to_use))
        self.training_thread.error.connect(self._on_training_error)
        self.training_thread.log.connect(self.training_panel.log)
        self.training_thread.start()

    def _on_training_finished(self, model_weights, metrics, settings, features_to_use):
        self.training_panel.btn_train.setEnabled(True)
        self.training_panel.progress_bar.setValue(100)
        self.training_panel.log(f"Final Accuracy: {metrics.get('accuracy',0):.2%}, Samples: {metrics.get('samples', 0)}")
        
        # Save to strategy
        model_id = str(uuid.uuid4())[:8]
        self.strategy.model_instances[model_id] = {
            "weights": model_weights,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "comment": f"{settings.get('model_type')} ({metrics.get('accuracy',0):.0%})",
            "settings": settings,
            "training_scope": {
                "mode": settings.get("ticker_mode"),
                "tickers": settings.get("ticker_list") if settings.get("ticker_mode") == "Custom Basket" else "Random" if settings.get("ticker_mode") == "Random Selection" else self.controls.ticker_input.text().upper(),
                "interval": self.controls.interval_combo.currentText(),
                "features": features_to_use,
                "samples": metrics.get('samples', 0)
            },
            "metrics": metrics
        }
        
        # Auto-set active if it's the first one
        if not self.strategy.active_model_id:
            self.strategy.active_model_id = model_id
            
        self.strategy.save()
        self.signals_panel.refresh_models(self.strategy.model_instances, self.strategy.active_model_id, self.active_features.keys())
        self._refresh_score_underlay()

    def _on_training_error(self, err_msg):
        self.training_panel.btn_train.setEnabled(True)
        self.training_panel.progress_bar.setValue(0)
        self.training_panel.log(f"ERROR: {err_msg}")
        QMessageBox.critical(self, "Training Error", err_msg)

    def preview_signals(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Signals", "Load data first.")
            return

        try:
            # preview the ACTIVE strategy (one in the dock), sync it first
            self._sync_active_strategy()
            self.strategy.save()
            strategy = self.strategy
            
            # 1. Compute all features required by this strategy
            all_feature_data = {"Volume": self.df['Volume']}
            if not strategy.feature_config:
                QMessageBox.warning(self, "Signals", f"Strategy '{strategy.name}' has no saved features. Add RSI/ATR to the Features panel and save the strategy.")
                return

            for feat_name, params in strategy.feature_config.items():
                if feat_name in self.available_features:
                    feat = self.available_features[feat_name]
                    result = feat.compute(self.df, params)
                    if isinstance(result, FeatureResult):
                         if result.data: all_feature_data.update(result.data)
                    elif hasattr(result, 'data'):
                         if result.data: all_feature_data.update(result.data)

            # 2. Run the signal script
            model_instance = strategy.get_script_instance()
            if not model_instance:
                raise Exception("Could not load script instance.")
            
            model_instance.model_instances = strategy.model_instances
            model_instance.active_model_id = strategy.active_model_id
            
            # Sync score panel with available functions
            self._sync_score_panel()
            
            raw_signals = model_instance.generate_signals(self.df, all_feature_data)
            self.clear_score_cache() # New signals invalidate scores

            # Convert Series to Events for the legacy logic
            events = []
            signal_indices = raw_signals[raw_signals != 0].index
            for idx in signal_indices:
                iloc = self.df.index.get_loc(idx)
                val = raw_signals[idx]
                side = 'buy' if val == 1 else 'sell' if val == -1 else 'neutral' if val == 2 else 'unknown'
                if side == 'unknown': continue
                events.append(SignalEvent(name=f"{strategy.name}_Signal", index=iloc, timestamp=idx, value=self.df['Close'].iloc[iloc], side=side, description=f"Strategy Signal ({side})"))

            # 3. Display on charts
            # Clear old signals from main plot
            for item in self.signal_items:
                self.main_plot.removeItem(item)
            self.signal_items = []

            # Clear existing score subplot if it exists to keep UI clean
            if self.score_plot_widget:
                self.score_plot_widget.deleteLater()
                self.score_plot = None
                self.score_plot_widget = None
                self._reorganize_subplots()

            # Create grouped scatter items for better performance and hover handling
            scatter_configs = {
                'buy': {'color': '#00ff00', 'sym': 't1', 'size': 15},
                'sell': {'color': '#ff4444', 'sym': 't', 'size': 15},
                'neutral': {'color': '#ffff00', 'sym': 'o', 'size': 10}
            }
            
            main_scatters = {}
            for side, cfg in scatter_configs.items():
                s = pg.ScatterPlotItem(
                    symbol=cfg['sym'], size=cfg['size'], 
                    brush=pg.mkBrush(cfg['color']),
                    pen=pg.mkPen(cfg['color'], width=1),
                    hoverable=True,
                    tip=None # We will use custom tooltip
                )
                s.sigHovered.connect(self.on_signal_hovered)
                self.main_plot.addItem(s)
                self.signal_items.append(s)
                main_scatters[side] = s

            # Calculate a sensible vertical offset (e.g., 2% of the visible range or average volatility)
            price_range = self.df['High'].max() - self.df['Low'].min()
            offset = price_range * 0.03 # 3% of the total price range

            for e in events:
                if e.side in main_scatters:
                    # Determine vertical position based on side
                    row = self.df.iloc[e.index]
                    if e.side == 'buy':
                        pos_y = row['Low'] - offset
                    elif e.side == 'sell':
                        pos_y = row['High'] + offset
                    else:
                        pos_y = e.value # Neutral/Unknown

                    # Add to main plot scatter
                    main_scatters[e.side].addPoints([{
                        'pos': (e.index, pos_y),
                        'data': e # Attach the event object to the point
                    }])
                
            QMessageBox.information(self, "Signals", f"Generated {len(events)} signals from strategy '{strategy.name}'.")
            self._refresh_score_underlay()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run signals: {e}")
            import traceback
            traceback.print_exc()

    def clear_score_cache(self):
        """Invalidates the score cache when data or features change."""
        self.score_cache = {}

    def _refresh_score_underlay(self):
        """Refreshes the score underlay if it's currently active in the panel."""
        settings = self.score_panel.get_settings()
        if settings.get("active"):
            self.update_score_visualization(settings)

    def update_score_visualization(self, settings):
        if self.df is None or self.df.empty: return
        
        if not settings.get("active"):
            if self.score_overlay:
                self.main_plot.remove_overlay(self.score_overlay)
                self.score_overlay = None
                self.main_plot.update_all(self.df)
            return

        func_name = settings.get("function")
        if not func_name: return

        force_refresh = settings.get("force_refresh", False)

        # PERFORMANCE OPTIMIZATION:
        # If we already have an overlay for THIS function, AND it is in cache, 
        # just update its visuals (alpha/colors) without re-calculating anything.
        if not force_refresh and self.score_overlay and \
           getattr(self.score_overlay, 'func_name', None) == func_name and \
           func_name in self.score_cache:
            self.score_overlay.set_visuals(
                pos_color=settings.get("pos_color"),
                neg_color=settings.get("neg_color"),
                alpha=settings.get("alpha")
            )
            return

        # If it's a different function, forced refresh, or not in cache, we might need to calculate
        try:
            if not force_refresh and func_name in self.score_cache:
                scores = self.score_cache[func_name]
            else:
                strategy = self.strategy
                model_instance = strategy.get_script_instance()
                
                if not hasattr(model_instance, func_name):
                    print(f"Strategy script has no function: {func_name}")
                    return

                # Pass model metadata to the script instance
                model_instance.model_instances = strategy.model_instances
                model_instance.active_model_id = strategy.active_model_id

                # Re-calculating signals for the current state to get fresh scores
                all_feature_data = {"Volume": self.df['Volume']}
                for feat_name, data in self.active_features.items():
                    feat = data["instance"]
                    params = {k: w.currentText() if isinstance(w, QComboBox) else w.isChecked() if isinstance(w, QCheckBox) else w.text() 
                              for k, w in data["inputs"].items()}
                    res = feat.compute(self.df, params)
                    if isinstance(res, FeatureResult): all_feature_data.update(res.data)
                    elif hasattr(res, 'data'): all_feature_data.update(res.data)

                raw_signals = model_instance.generate_signals(self.df, all_feature_data)
                
                score_func = getattr(model_instance, func_name)
                scores = score_func(self.df, raw_signals)
                self.score_cache[func_name] = scores

            if self.score_overlay:
                self.main_plot.remove_overlay(self.score_overlay)

            self.score_overlay = ScoreOverlay(
                scores, 
                pos_color=settings.get("pos_color"),
                neg_color=settings.get("neg_color"),
                alpha=settings.get("alpha")
            )
            self.score_overlay.func_name = func_name # Tag it for the optimization above
            self.main_plot.add_overlay(self.score_overlay)
            self.main_plot.update_all(self.df)
            print("DEBUG: Overlay added and plot updated")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error updating score visualization: {e}")

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
        tickers = self.db.get_all_tickers() or ["AAPL", "MSFT", "GOOGL"]
        ticker = random.choice(tickers)
        self.controls.ticker_input.setText(ticker)
        self.load_chart()

    def load_chart(self):
        ticker = self.controls.ticker_input.text().upper()
        interval = self.controls.interval_combo.currentText()

        self.df = self.db.get_data(ticker, interval)
        if self.df.empty:
            try: self.engine.sync_data(ticker, interval, period="10y"); self.df = self.db.get_data(ticker, interval)
            except Exception as e: print(e)
        if self.df.empty: return
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
            
        # Initial X range
        self.main_plot.setXRange(max(0, len(self.df)-100), len(self.df))
        self._refresh_score_underlay()

def main():
    app = QApplication(sys.argv)
    window = ChartWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
