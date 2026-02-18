import sys
import pandas as pd
import numpy as np
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt, QRectF
import pyqtgraph as pg

from .database import Database
from .engine import TradingEngine
from .features.loader import load_features
from .features.base import LineOutput, LevelOutput, MarkerOutput, HeatmapOutput

# Modular Components
from .gui_components.styling import setup_app_style
from .gui_components.axes import DateAxis
from .gui_components.candles import CandlestickItem, SimpleCandleItem
from .gui_components.controls import ControlBar
from .gui_components.feature_dock import FeatureDock
from .gui_components.volume import VolumeItem

class ChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Bot Pro")
        self.resize(1400, 900)
        setup_app_style(self)
        
        self.db = Database("data/stocks.db")
        self.engine = TradingEngine()
        self.available_features = load_features()
        self.active_features = {} # {name: {params, items, plot_item_ref}}

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
        layout.addWidget(self.controls)
        
        # 2. Main Plot Area
        self.layout_widget = pg.GraphicsLayoutWidget()
        self.layout_widget.setBackground('#1e1e1e')
        layout.addWidget(self.layout_widget)
        
        self.main_plot = self.layout_widget.addPlot(row=0, col=0)
        self.main_plot.showGrid(x=False, y=True, alpha=0.15)
        self.main_plot.setMouseEnabled(x=True, y=False)
        self.main_plot.getAxis('left').setPen('#444')
        self.main_plot.getAxis('bottom').setPen('#444')
        self.main_plot.vb.setZValue(10)
        
        # Volume Overlay
        self.vol_view = pg.ViewBox()
        self.main_plot.scene().addItem(self.vol_view)
        self.vol_view.setXLink(self.main_plot.vb)
        self.vol_view.setZValue(-10)
        self.vol_view.setMouseEnabled(x=False, y=False)
        self.vol_view.setAcceptHoverEvents(False)
        self.vol_view.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        
        self.main_plot.vb.sigResized.connect(self.update_volume_geometry)
        self.main_plot.vb.sigXRangeChanged.connect(self.update_view)
        
        self._setup_crosshair()
        
        # 3. Feature Dock
        self.feature_dock = FeatureDock(self.available_features, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.feature_dock)
        self.feature_dock.btn_add_feat.clicked.connect(self.add_feature_ui)

        # State
        self.df = None
        self.timestamps = []
        self.candle_item = None
        self.simple_candle_item = None
        self.sub_plots = {} # {feat_name: PlotItem}

    def update_volume_geometry(self):
        if not hasattr(self, 'vol_view') or not hasattr(self, 'main_plot'): return
        rect = self.main_plot.vb.sceneBoundingRect()
        if rect.isValid() and rect.width() > 0:
            self.vol_view.setGeometry(rect)

    def _setup_crosshair(self):
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.main_plot.addItem(self.v_line, ignoreBounds=True)
        self.main_plot.addItem(self.h_line, ignoreBounds=True)
        
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
            self.v_line.setPos(idx)
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

    def add_feature_ui(self):
        feat_name = self.feature_dock.feat_combo.currentData()
        if feat_name in self.active_features: return
        
        feature = self.available_features[feat_name]
        input_widgets = self.feature_dock.create_feature_widget(
            feat_name, feature.parameters, 
            lambda: self.update_feature(feat_name),
            self.remove_feature
        )

        plot_target = self.main_plot
        if feature.target_pane == "new":
            self.layout_widget.nextRow()
            new_plot = self.layout_widget.addPlot()
            new_plot.setMaximumHeight(150)
            new_plot.setXLink(self.main_plot)
            new_plot.showGrid(x=False, y=True, alpha=0.15)
            new_plot.setMouseEnabled(x=True, y=False)
            if feature.y_range: new_plot.setYRange(*feature.y_range, padding=0)
            else: new_plot.vb.setAutoVisible(y=True)
            plot_target = new_plot
            self.sub_plots[feat_name] = new_plot

        self.active_features[feat_name] = {
            "instance": feature, "inputs": input_widgets, "items": [], "plot": plot_target
        }
        self.update_feature(feat_name)

    def update_feature(self, feat_name):
        if self.df is None or self.df.empty: return
        data = self.active_features[feat_name]
        feat, plot_item = data["instance"], data["plot"]
        
        params = {k: w.currentText() if isinstance(w, QComboBox) else w.isChecked() if isinstance(w, QCheckBox) else w.text() 
                  for k, w in data["inputs"].items()}
        
        try: results = feat.compute(self.df, params)
        except Exception as e: print(f"Error computing {feat_name}: {e}"); return

        for item in data["items"]: plot_item.removeItem(item)
        data["items"] = []
        
        for res in results:
            if isinstance(res, LineOutput):
                item = plot_item.plot(np.arange(len(res.data)), 
                                      np.array([float(v) if v is not None else np.nan for v in res.data]), 
                                      pen=pg.mkPen(res.color, width=res.width))
                data["items"].append(item)
            elif isinstance(res, LevelOutput):
                region = pg.LinearRegionItem(values=[res.min_price, res.max_price], orientation=pg.LinearRegionItem.Horizontal, 
                                             movable=False, brush=pg.mkBrush(res.color + "40"), pen=None)
                for line in region.lines: line.setPen(None); line.setHoverPen(None)
                plot_item.addItem(region); data["items"].append(region)
            elif isinstance(res, MarkerOutput):
                sym = {'o':'o','d':'d','t':'t1','s':'s','x':'x','+':'+'}.get(res.shape, 'o')
                item = pg.ScatterPlotItem(x=res.indices, y=res.values, brush=pg.mkBrush(res.color), symbol=sym, size=10)
                plot_item.addItem(item); data["items"].append(item)
            elif isinstance(res, HeatmapOutput):
                img = pg.ImageItem(np.array(res.density).reshape(1, -1))
                color = np.array([[0,0,0,0], [255,255,0,100]] if res.color_map == "viridis" else [[0,0,0,0], [0,100,255,100]], dtype=np.ubyte)
                img.setLookupTable(pg.ColorMap([0,1], color).getLookupTable(0,1,256))
                img.setRect(QRectF(-500, res.price_grid[0], len(self.df)+1000, res.price_grid[-1]-res.price_grid[0]))
                img.setZValue(-10); plot_item.addItem(img); data["items"].append(img)

    def remove_feature(self, feat_name, widget):
        if feat_name in self.active_features:
            data = self.active_features[feat_name]
            for item in data["items"]: data["plot"].removeItem(item)
            if data["instance"].target_pane == "new":
                self.layout_widget.removeItem(data["plot"])
                del self.sub_plots[feat_name]
            del self.active_features[feat_name]
        widget.deleteLater()

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

        self.main_plot.clear(); self._setup_crosshair()
        self.timestamps = self.df.index.astype(str).tolist()
        
        ts_list = self.timestamps
        self.main_plot.getAxis('bottom').tickStrings = lambda values, scale, spacing: [
            pd.Timestamp(ts_list[int(v)]).strftime('%H:%M\n%m-%d' if spacing < 5 else '%Y-%m-%d') 
            if 0 <= int(v) < len(ts_list) else '' for v in values]
        
        data = [(float(i), r['Open'], r['Close'], r['Low'], r['High']) for i, r in enumerate(self.df.to_dict('records'))]
        self.candle_item, self.simple_candle_item = CandlestickItem(data), SimpleCandleItem(data)
        self.main_plot.addItem(self.candle_item); self.main_plot.addItem(self.simple_candle_item)
        self.simple_candle_item.setVisible(False)
        
        self.vol_view.clear()
        vol_data = [(float(i), r['Open'], r['Close'], r['Volume']) for i, r in enumerate(self.df.to_dict('records'))]
        self.vol_item = VolumeItem(vol_data)
        self.vol_view.addItem(self.vol_item)
        
        for name in self.active_features: self.update_feature(name)
        self.main_plot.setXRange(max(0, len(data)-100), len(data))
        self.update_view()

    def update_view(self):
        if self.df is None or self.df.empty: return
        self.update_volume_geometry()
        x_min, x_max = self.main_plot.vb.viewRange()[0]
        
        self.candle_item.setVisible(x_max - x_min <= 500)
        self.simple_candle_item.setVisible(x_max - x_min > 500)
        
        idx_min, idx_max = max(0, int(x_min)), min(len(self.df), int(x_max) + 1)
        if idx_min >= idx_max: return
        sub = self.df.iloc[idx_min:idx_max]
        
        y_min, y_max = np.nanmin(sub['Low'].values), np.nanmax(sub['High'].values)
        if not np.isnan(y_min):
            pad = (y_max - y_min) * 0.1 or 1.0
            self.main_plot.vb.setYRange(y_min - pad, y_max + pad, padding=0)
        
        v_max = np.nanmax(sub['Volume'].values)
        self.vol_view.setYRange(0, v_max * 4 if v_max > 0 else 1, padding=0)

        for name, data in self.active_features.items():
            feat, plot = data["instance"], data["plot"]
            if feat.target_pane == "main" or feat.y_range is not None: continue
            all_y = [val for item in data["items"] if hasattr(item, 'getData') for val in (item.getData()[1][idx_min:idx_max] if item.getData()[1] is not None else [])]
            if all_y:
                y_min_f, y_max_f = np.nanmin(all_y), np.nanmax(all_y)
                if not np.isnan(y_min_f):
                    h = y_max_f - y_min_f
                    pad = h * feat.y_padding if h > 0 else 1.0
                    plot.setYRange(y_min_f - pad, y_max_f + pad, padding=0)

def main():
    app = QApplication(sys.argv)
    window = ChartWindow()
    window.showMaximized()
    sys.exit(app.exec())
