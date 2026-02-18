import sys
import pandas as pd
import numpy as np
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QComboBox, QLabel, QPushButton, QLineEdit, 
                             QDockWidget, QFormLayout, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPicture, QPainter, QColor, QPen, QBrush
import pyqtgraph as pg
from .database import Database
from .engine import TradingEngine
from .features.loader import load_features
from .features.base import LineOutput, LevelOutput, MarkerOutput

# --- Custom Items (Candle, DateAxis) ---
class DateAxis(pg.AxisItem):
    def __init__(self, timestamps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamps = timestamps

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            idx = int(v)
            if 0 <= idx < len(self.timestamps):
                ts = pd.Timestamp(self.timestamps[idx])
                if spacing < 5:
                    strings.append(ts.strftime('%H:%M\n%m-%d'))
                else:
                    strings.append(ts.strftime('%Y-%m-%d'))
            else:
                strings.append('')
        return strings

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data 
        self.picture = QPicture()
        self.generatePicture()

    def generatePicture(self):
        p = QPainter(self.picture)
        p.setPen(pg.mkPen('w'))
        w = 0.4
        for (t, open, close, low, high) in self.data:
            if close > open:
                p.setPen(pg.mkPen('g'))
                p.setBrush(pg.mkBrush('g'))
            else:
                p.setPen(pg.mkPen('r'))
                p.setBrush(pg.mkBrush('r'))
            p.drawLine(QPointF(t, low), QPointF(t, high))
            p.drawRect(QRectF(t - w, open, w * 2, close - open))
        p.end()

    def paint(self, p, *args):
        self.picture.play(p)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())

class SimpleCandleItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.picture = QPicture()
        self.generatePicture()

    def generatePicture(self):
        p = QPainter(self.picture)
        pen_green = pg.mkPen('g', width=1)
        pen_red = pg.mkPen('r', width=1)
        for (t, open, close, low, high) in self.data:
            if close > open:
                p.setPen(pen_green)
            else:
                p.setPen(pen_red)
            p.drawLine(QPointF(t, low), QPointF(t, high))
        p.end()

    def paint(self, p, *args):
        self.picture.play(p)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())

# --- Main Window ---
class ChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Bot Pro (Native PyQt6)")
        self.resize(1400, 900)
        self._setup_style()
        
        self.db = Database("data/stocks.db")
        self.engine = TradingEngine()
        
        # Load Features
        self.available_features = load_features()
        self.active_features = {} # {name: {params: {}, item: plot_item}}

        self._init_ui()

    def _setup_style(self):
        # 1. Application-wide Dark Palette (Fixes menus and title bars on some OSs)
        app = QApplication.instance()
        palette = app.palette()
        palette.setColor(palette.ColorRole.Window, QColor("#1e1e1e"))
        palette.setColor(palette.ColorRole.WindowText, QColor("#dddddd"))
        palette.setColor(palette.ColorRole.Base, QColor("#2d2d2d"))
        palette.setColor(palette.ColorRole.AlternateBase, QColor("#1e1e1e"))
        palette.setColor(palette.ColorRole.ToolTipBase, QColor("#dddddd"))
        palette.setColor(palette.ColorRole.ToolTipText, QColor("#dddddd"))
        palette.setColor(palette.ColorRole.Text, QColor("#dddddd"))
        palette.setColor(palette.ColorRole.Button, QColor("#1e1e1e"))
        palette.setColor(palette.ColorRole.ButtonText, QColor("#dddddd"))
        palette.setColor(palette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(palette.ColorRole.Link, QColor("#007acc"))
        palette.setColor(palette.ColorRole.Highlight, QColor("#007acc"))
        palette.setColor(palette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        app.setPalette(palette)

        # 2. Specific Stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #dddddd; font-family: "Segoe UI", Arial; font-size: 10pt; }
            QLineEdit, QComboBox { background-color: #2d2d2d; border: 1px solid #3d3d3d; padding: 4px; border-radius: 4px; color: #fff; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #fff; margin-right: 5px; }
            QComboBox QAbstractItemView { background-color: #2d2d2d; color: #fff; selection-background-color: #007acc; selection-color: #fff; border: 1px solid #3d3d3d; }
            QPushButton { background-color: #007acc; border: none; padding: 6px 12px; border-radius: 4px; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #005a9e; }
            QDockWidget { border: 1px solid #333; titlebar-close-icon: url(close.png); titlebar-normal-icon: url(undock.png); }
            QDockWidget::title { background: #121212; padding-left: 5px; }
            QGroupBox { border: 1px solid #444; margin-top: 6px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QMenu { background-color: #2d2d2d; border: 1px solid #3d3d3d; }
            QMenu::item { padding: 5px 20px; }
            QMenu::item:selected { background-color: #007acc; }
        """)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Top Controls
        controls = QWidget()
        controls.setStyleSheet("background-color: #121212; border-bottom: 1px solid #333;")
        c_layout = QHBoxLayout(controls)
        
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setFixedWidth(80)
        self.ticker_input.returnPressed.connect(self.load_chart) # Enter key to load
        
        # Ticker History
        self.ticker_history = QComboBox()
        self.ticker_history.setFixedWidth(100)
        self.ticker_history.addItem("History...")
        self.ticker_history.currentIndexChanged.connect(self.load_from_history)
        
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1d", "4h", "1h", "15m", "1w"])
        
        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self.load_chart)
        
        self.btn_random = QPushButton("Random")
        self.btn_random.clicked.connect(self.load_random)
        self.btn_random.setStyleSheet("background-color: #444; margin-left: 5px;")
        
        c_layout.addWidget(QLabel("Ticker:"))
        c_layout.addWidget(self.ticker_input)
        c_layout.addWidget(self.ticker_history)
        c_layout.addWidget(self.btn_random)
        c_layout.addSpacing(15)
        c_layout.addWidget(QLabel("Interval:"))
        c_layout.addWidget(self.interval_combo)
        c_layout.addSpacing(15)
        c_layout.addWidget(self.btn_load)
        c_layout.addStretch()
        layout.addWidget(controls)
        
        # 2. Plot Area
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getAxis('left').setPen('#444')
        self.plot_widget.getAxis('bottom').setPen('#444')
        layout.addWidget(self.plot_widget)
        
        self._setup_crosshair()
        
        # 3. Feature Dock
        self.feature_dock = QDockWidget("Features", self)
        self.feature_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.feature_dock)
        
        dock_content = QWidget()
        self.feature_layout = QVBoxLayout(dock_content)
        self.feature_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Feature Selector
        self.feat_combo = QComboBox()
        self.feat_combo.addItems(sorted(self.available_features.keys()))
        self.btn_add_feat = QPushButton("Add Feature")
        self.btn_add_feat.clicked.connect(self.add_feature_ui)
        
        self.feature_layout.addWidget(QLabel("Add Feature:"))
        self.feature_layout.addWidget(self.feat_combo)
        self.feature_layout.addWidget(self.btn_add_feat)
        self.feature_layout.addSpacing(20)
        self.feature_layout.addWidget(QLabel("Active Features:"))
        
        # Container for active feature widgets
        self.active_feats_container = QVBoxLayout()
        self.feature_layout.addLayout(self.active_feats_container)
        
        self.feature_dock.setWidget(dock_content)

        # Internal State
        self.df = None
        self.timestamps = []
        self.candle_item = None
        self.simple_candle_item = None

    def _setup_crosshair(self):
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        
        self.price_label = pg.TextItem(anchor=(0, 1), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.time_label = pg.TextItem(anchor=(1, 0), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.plot_widget.addItem(self.price_label, ignoreBounds=True)
        self.plot_widget.addItem(self.time_label, ignoreBounds=True)
        
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def mouse_moved(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getPlotItem().getViewBox().mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            idx = int(round(x))
            self.v_line.setPos(idx)
            self.h_line.setPos(y)
            
            view_range = self.plot_widget.getPlotItem().getViewBox().viewRange()
            x_max = view_range[0][1]
            y_min = view_range[1][0]
            
            self.price_label.setPos(x_max, y)
            self.price_label.setText(f"{y:.2f}")
            self.price_label.setAnchor((1, 0.5))
            
            if 0 <= idx < len(self.timestamps):
                ts = pd.Timestamp(self.timestamps[idx])
                self.time_label.setPos(idx, y_min)
                self.time_label.setText(ts.strftime('%Y-%m-%d %H:%M'))
                self.time_label.setAnchor((0.5, 1))
            else:
                self.time_label.setText("")

    def add_feature_ui(self):
        feat_name = self.feat_combo.currentText()
        if feat_name in self.active_features:
            return # Already active
            
        feature = self.available_features[feat_name]
        default_params = feature.parameters
        
        # Create UI Group
        group = QGroupBox(feat_name)
        form = QFormLayout(group)
        
        input_widgets = {}
        
        for key, val in default_params.items():
            if isinstance(val, list): # Dropdown
                inp = QComboBox()
                inp.addItems([str(x) for x in val])
            elif isinstance(val, bool):
                inp = QCheckBox()
                inp.setChecked(val)
            else:
                inp = QLineEdit(str(val))
            
            input_widgets[key] = inp
            form.addRow(key, inp)
            
            # Connect change signal
            if isinstance(inp, QComboBox):
                inp.currentTextChanged.connect(lambda _, n=feat_name: self.update_feature(n))
            elif isinstance(inp, QCheckBox):
                inp.stateChanged.connect(lambda _, n=feat_name: self.update_feature(n))
            else:
                inp.editingFinished.connect(lambda n=feat_name: self.update_feature(n))

        # Remove Button
        btn_rem = QPushButton("Remove")
        btn_rem.setStyleSheet("background-color: #cc3300;")
        btn_rem.clicked.connect(lambda: self.remove_feature(feat_name, group))
        form.addRow(btn_rem)
        
        self.active_feats_container.addWidget(group)
        
        self.active_features[feat_name] = {
            "instance": feature,
            "inputs": input_widgets,
            "items": [] # List of pyqtgraph items
        }
        
        # Initial Render
        self.update_feature(feat_name)

    def update_feature(self, feat_name):
        if self.df is None or self.df.empty:
            return
            
        feat_data = self.active_features[feat_name]
        feature = feat_data["instance"]
        inputs = feat_data["inputs"]
        
        # 1. Parse Parameters
        params = {}
        for key, widget in inputs.items():
            if isinstance(widget, QComboBox):
                params[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            else:
                params[key] = widget.text() # Keep as string, feature class parses it
        
        # 2. Compute
        try:
            results = feature.compute(self.df, params)
        except Exception as e:
            print(f"Error computing feature {feat_name}: {e}")
            return

        # 3. Clear Old Items
        for item in feat_data["items"]:
            self.plot_widget.removeItem(item)
        feat_data["items"] = []
        
        # 4. Render New Items
        for res in results:
            if isinstance(res, LineOutput):
                # Cleanse data (None/NaN cannot be plotted directly by all plot calls, but pg handles NaNs usually)
                # Need valid X indices
                x = np.arange(len(res.data))
                # Convert None to NaN
                y = np.array([float(v) if v is not None else np.nan for v in res.data])
                
                item = self.plot_widget.plot(x, y, pen=pg.mkPen(res.color, width=res.width), name=res.name)
                feat_data["items"].append(item)
                
            elif isinstance(res, LevelOutput):
                # Draw Region
                region = pg.LinearRegionItem(values=[res.min_price, res.max_price], orientation=pg.LinearRegionItem.Horizontal, 
                                             movable=False, brush=pg.mkBrush(res.color + "40"), pen=None)
                # Remove interaction lines
                for line in region.lines: line.setPen(None); line.setHoverPen(None)
                
                self.plot_widget.addItem(region)
                feat_data["items"].append(region)
                
                # Optional: Text Label
                # text = pg.TextItem(res.name, color=res.color, anchor=(0, 1))
                # text.setPos(len(self.df), res.price)
                # self.plot_widget.addItem(text)
                # feat_data["items"].append(text)

    def remove_feature(self, feat_name, widget):
        if feat_name in self.active_features:
            # Remove items from plot
            for item in self.active_features[feat_name]["items"]:
                self.plot_widget.removeItem(item)
            del self.active_features[feat_name]
        
        # Remove UI
        widget.deleteLater()

    def load_from_history(self, index):
        if index <= 0: return 
        ticker = self.ticker_history.currentText()
        self.ticker_input.setText(ticker)
        self.load_chart()

    def load_random(self):
        tickers = self.db.get_all_tickers()
        if not tickers:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
        
        random_ticker = random.choice(tickers)
        self.ticker_input.setText(random_ticker)
        self.load_chart()

    def add_to_history(self, ticker):
        if self.ticker_history.findText(ticker) == -1:
            self.ticker_history.addItem(ticker)

    def load_chart(self):
        ticker = self.ticker_input.text().upper()
        interval = self.interval_combo.currentText()
        
        # DB & Sync Logic
        self.df = self.db.get_data(ticker, interval)
        if self.df.empty or len(self.df) < 50:
            print("Fetching data...")
            period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d", "4h": "730d", "1d": "10y", "1w": "10y"}
            try:
                self.engine.sync_data(ticker, interval, period=period_map.get(interval, "1y"))
                self.df = self.db.get_data(ticker, interval)
            except Exception as e:
                print(e)

        if self.df.empty: return

        self.add_to_history(ticker)
        self.plot_widget.clear()
        
        # Re-add persistent items
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        self.plot_widget.addItem(self.price_label, ignoreBounds=True)
        self.plot_widget.addItem(self.time_label, ignoreBounds=True)
        
        self.timestamps = self.df.index.astype(str).tolist()
        
        # Axis Tick Logic (Closure)
        ts_list = self.timestamps
        def tickStrings(values, scale, spacing):
            strings = []
            for v in values:
                idx = int(v)
                if 0 <= idx < len(ts_list):
                    ts = pd.Timestamp(ts_list[idx])
                    if spacing < 5: strings.append(ts.strftime('%H:%M\n%m-%d'))
                    else: strings.append(ts.strftime('%Y-%m-%d'))
                else: strings.append('')
            return strings
        self.plot_widget.getPlotItem().getAxis('bottom').tickStrings = tickStrings
        
        # Candles
        data = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            data.append((float(i), row['Open'], row['Close'], row['Low'], row['High']))
            
        self.candle_item = CandlestickItem(data)
        self.simple_candle_item = SimpleCandleItem(data)
        
        self.plot_widget.addItem(self.candle_item)
        self.plot_widget.addItem(self.simple_candle_item)
        self.simple_candle_item.setVisible(False)
        
        # Update Features (re-calculate on new data)
        for name in self.active_features:
            self.update_feature(name)
        
        # Zoom & AutoScale
        self.plot_widget.getPlotItem().getViewBox().sigXRangeChanged.connect(self.update_view)
        self.plot_widget.setXRange(max(0, len(data)-100), len(data))
        self.update_view()

    def update_view(self):
        if self.df is None: return
        vb = self.plot_widget.getPlotItem().getViewBox()
        x_min, x_max = vb.viewRange()[0]
        
        # LOD
        if (x_max - x_min) > 500:
            if self.candle_item.isVisible():
                self.candle_item.setVisible(False)
                self.simple_candle_item.setVisible(True)
        else:
            if not self.candle_item.isVisible():
                self.candle_item.setVisible(True)
                self.simple_candle_item.setVisible(False)
        
        # Auto Scale Y
        idx_min = max(0, int(x_min))
        idx_max = min(len(self.df), int(x_max) + 1)
        if idx_min >= idx_max: return
        
        sub = self.df.iloc[idx_min:idx_max]
        if sub.empty: return
        
        # Get Min/Max of candles
        y_min = sub['Low'].min()
        y_max = sub['High'].max()
        
        if pd.isna(y_min): return
        
        pad = (y_max - y_min) * 0.1
        if pad == 0: pad = 1
        vb.setYRange(y_min - pad, y_max + pad, padding=0)

def main():
    app = QApplication(sys.argv)
    window = ChartWindow()
    window.show()
    sys.exit(app.exec())
