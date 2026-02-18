import sys
import pandas as pd
import numpy as np
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QCheckBox, QInputDialog, QMessageBox, QSplitter)
from PyQt6.QtCore import Qt, QRectF
import pyqtgraph as pg

from .database import Database
from .engine import TradingEngine
from .features.loader import load_features
from .features.feature_set import FeatureSet
from .features.signals import SignalEngine
from .features.base import LineOutput, LevelOutput, MarkerOutput, HeatmapOutput, FeatureResult

# Modular Components
from .gui_components.styling import setup_app_style
from .gui_components.axes import DateAxis
from .gui_components.controls import ControlBar
from .gui_components.feature_dock import FeatureDock
from .gui_components.plots import UnifiedPlot, CandleOverlay, VolumeOverlay, LineOverlay, LevelOverlay

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
        self.signal_engine = SignalEngine()
        self.signal_items = []

        # State
        self.df = None
        self.timestamps = []
        self.sub_plots = {} # {feat_name: UnifiedPlot}
        self.v_lines = []

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Controls
        self.controls = ControlBar(self)
        self.controls.ticker_input.returnPressed.connect(self.load_chart)
        self.controls.ticker_history.currentIndexChanged.connect(self.load_from_history)
        self.controls.btn_load.clicked.connect(self.load_chart)
        self.controls.btn_random.clicked.connect(self.load_random)
        self.controls.btn_signals.clicked.connect(self.detect_signals)
        layout.addWidget(self.controls)
        
        # 2. Main Plot Area
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot_splitter.setHandleWidth(6)
        layout.addWidget(self.plot_splitter)
        
        # Container for main plot
        self.main_plot_widget = pg.GraphicsLayoutWidget()
        self.main_plot_widget.setBackground('#1e1e1e')
        self.plot_splitter.addWidget(self.main_plot_widget)
        
        self.main_plot = UnifiedPlot()
        self.main_plot_widget.addItem(self.main_plot)
        
        # Add basic overlays to main plot
        self.price_overlay = CandleOverlay()
        self.volume_overlay = VolumeOverlay()
        self.main_plot.add_overlay(self.price_overlay)
        self.main_plot.add_overlay(self.volume_overlay)
        
        self.main_plot.getAxis('left').setPen('#444')
        self.main_plot.getAxis('left').setWidth(60) # Keep axes aligned
        self.main_plot.getAxis('bottom').setPen('#444')
        
        self._setup_crosshair()
        
        self.feature_dock = FeatureDock(self.available_features, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.feature_dock)
        self.feature_dock.btn_add_feat.clicked.connect(self.add_feature_ui)
        self.feature_dock.btn_save_set.clicked.connect(self.save_feature_set)
        self.feature_dock.btn_load_set.clicked.connect(self.load_feature_set)
        
        self.showMaximized()

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
        feat_name = self.feature_dock.feat_combo.currentData()
        if not feat_name: return
        self._add_feature_by_name(feat_name, initial_values)

    def _add_feature_by_name(self, feat_name, initial_values=None):
        if feat_name in self.active_features: return
        
        feature = self.available_features[feat_name]
        input_widgets, group_widget = self.feature_dock.create_feature_widget(
            feat_name, feature.parameters, 
            lambda: self.update_feature(feat_name),
            self.remove_feature,
            initial_values=initial_values
        )

        plot_target = self.main_plot
        v_line = None
        container_widget = self.main_plot_widget
        
        if feature.target_pane == "new":
            new_plot_widget = pg.GraphicsLayoutWidget()
            new_plot_widget.setBackground('#1e1e1e')
            new_plot_widget.setMinimumHeight(100)
            self.plot_splitter.addWidget(new_plot_widget)
            container_widget = new_plot_widget
            
            new_plot = UnifiedPlot()
            new_plot.setXLink(self.main_plot)
            new_plot.getAxis('left').setWidth(60) # Axis alignment
            new_plot_widget.addItem(new_plot)
            plot_target = new_plot
            self.sub_plots[feat_name] = new_plot
            
            # Sub-plot cursor
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
            new_plot.addItem(v_line, ignoreBounds=True)
            self.v_lines.append(v_line)

        self.active_features[feat_name] = {
            "instance": feature, "inputs": input_widgets, "overlays": [], "plot": plot_target, 
            "v_line": v_line, "widget": group_widget, "container": container_widget
        }
        self.update_feature(feat_name)

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

        plot.update_all(self.df)

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
            del self.active_features[feat_name]
        widget.deleteLater()

    def _reorganize_subplots(self):
        """Manages splitter stretch factors."""
        # Main plot gets more initial space
        self.plot_splitter.setStretchFactor(0, 10)
        
        # Subplots get less initial space but remain adjustable
        for i in range(1, self.plot_splitter.count()):
            self.plot_splitter.setStretchFactor(i, 2)

    def save_feature_set(self):
        if not self.active_features:
            QMessageBox.warning(self, "Save Feature Set", "No active features to save.")
            return

        name, ok = QInputDialog.getText(self, "Save Feature Set", "Enter a name for this feature set:")
        if not ok or not name:
            return

        fs = FeatureSet(name)
        for feat_name, data in self.active_features.items():
            params = {}
            for k, w in data["inputs"].items():
                if isinstance(w, QComboBox):
                    params[k] = w.currentText()
                elif isinstance(w, QCheckBox):
                    params[k] = w.isChecked()
                else: # QLineEdit
                    params[k] = w.text()
            fs.add_feature(feat_name, params)
        
        try:
            fs.save()
            QMessageBox.information(self, "Save Feature Set", f"Feature set '{name}' saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save feature set: {e}")

    def load_feature_set(self):
        available = FeatureSet.list_available()
        if not available:
            QMessageBox.warning(self, "Load Feature Set", "No saved feature sets found.")
            return

        name, ok = QInputDialog.getItem(self, "Load Feature Set", "Select a feature set:", available, 0, False)
        if not ok or not name:
            return

        try:
            # Clear current features
            for feat_name in list(self.active_features.keys()):
                data = self.active_features[feat_name]
                self.remove_feature(feat_name, data["widget"], reorganize=False)
            
            # Load new set
            QMessageBox.warning(self, "load feature set", "loading feature set")
            fs = FeatureSet.load(name)
            for feat_name, params in fs.features.items():
                if feat_name in self.available_features:
                    self._add_feature_by_name(feat_name, initial_values=params)
                else:
                    print(f"Warning: Feature '{feat_name}' not found.")
            
            self._reorganize_subplots()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load feature set: {e}")

    def detect_signals(self):
        if self.df is None or self.df.empty: return
        
        # Collect data from all active features
        all_feature_data = {}
        for feat_name, data in self.active_features.items():
            if "raw_data" in data and data["raw_data"]:
                all_feature_data.update(data["raw_data"])
        
        if not all_feature_data:
            QMessageBox.warning(self, "Signal Detection", "No feature data available. Add features like SMA or RSI first.")
            return

        # Detect
        signals = self.signal_engine.extract_signals(self.df, all_feature_data)
        
        # Clear old
        for item in self.signal_items:
            self.main_plot.removeItem(item)
        self.signal_items = []
        
        if not signals:
            QMessageBox.information(self, "Signal Detection", "No signals detected with current settings.")
            return

        # Plot Buy/Sell markers
        buy_indices = [s.index for s in signals if s.side == 'buy']
        buy_prices = [s.value for s in signals if s.side == 'buy']
        sell_indices = [s.index for s in signals if s.side == 'sell']
        sell_prices = [s.value for s in signals if s.side == 'sell']
        
        if buy_indices:
            item = pg.ScatterPlotItem(x=buy_indices, y=[p * 0.98 for p in buy_prices], 
                                       symbol='t1', size=15, brush=pg.mkBrush('#00ff00'))
            self.main_plot.addItem(item)
            self.signal_items.append(item)
            
        if sell_indices:
            item = pg.ScatterPlotItem(x=sell_indices, y=[p * 1.02 for p in sell_prices], 
                                       symbol='t', size=15, brush=pg.mkBrush('#ff0000'))
            self.main_plot.addItem(item)
            self.signal_items.append(item)
            
        QMessageBox.information(self, "Signal Detection", f"Detected {len(signals)} signals!")

    def load_from_history(self, index):
        if index <= 0: return 
        self.controls.ticker_input.setText(self.controls.ticker_history.currentText())
        self.load_chart()

    def load_random(self):
        tickers = self.db.get_all_tickers() or ["AAPL", "MSFT", "GOOGL"]
        self.controls.ticker_input.setText(random.choice(tickers))
        self.load_chart()

    def load_chart(self):
        ticker, interval = self.controls.ticker_input.text().upper(), self.controls.interval_combo.currentText()
        self.df = self.db.get_data(ticker, interval)
        if self.df.empty:
            try: self.engine.sync_data(ticker, interval, period="10y"); self.df = self.db.get_data(ticker, interval)
            except Exception as e: print(e)
        if self.df.empty: return
        self.controls.add_to_history(ticker)

        # Clear signal markers
        for item in self.signal_items:
            self.main_plot.removeItem(item)
        self.signal_items = []
        
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

def main():
    app = QApplication(sys.argv)
    window = ChartWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
