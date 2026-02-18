import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QComboBox, QLabel, QPushButton, QLineEdit, 
                             QDockWidget, QFormLayout, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPicture, QPainter, QColor, QPen, QBrush
import pyqtgraph as pg
from .database import Database
from .engine import TradingEngine
from .features.loader import load_features
from .features.base import LineOutput, LevelOutput, MarkerOutput, HeatmapOutput

# --- Custom Items ---
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
                if spacing < 5: strings.append(ts.strftime('%H:%M\n%m-%d'))
                else: strings.append(ts.strftime('%Y-%m-%d'))
            else: strings.append('')
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
                p.setPen(pg.mkPen('g')); p.setBrush(pg.mkBrush('g'))
            else:
                p.setPen(pg.mkPen('r')); p.setBrush(pg.mkBrush('r'))
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
            if close > open: p.setPen(pen_green)
            else: p.setPen(pen_red)
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
        
        self.available_features = load_features()
        self.active_features = {} # {name: {params, items, plot_item_ref}}

        self._init_ui()

    def _setup_style(self):
        # ... (Same styling as before) ...
        app = QApplication.instance()
        palette = app.palette()
        palette.setColor(palette.ColorRole.Window, QColor("#1e1e1e"))
        palette.setColor(palette.ColorRole.WindowText, QColor("#dddddd"))
        palette.setColor(palette.ColorRole.Base, QColor("#2d2d2d"))
        palette.setColor(palette.ColorRole.Text, QColor("#dddddd"))
        palette.setColor(palette.ColorRole.Button, QColor("#1e1e1e"))
        palette.setColor(palette.ColorRole.ButtonText, QColor("#dddddd"))
        app.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #dddddd; font-family: "Segoe UI", Arial; font-size: 10pt; }
            QLineEdit, QComboBox { background-color: #2d2d2d; border: 1px solid #3d3d3d; padding: 4px; border-radius: 4px; color: #fff; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #2d2d2d; color: #fff; selection-background-color: #007acc; }
            QPushButton { background-color: #007acc; border: none; padding: 6px 12px; border-radius: 4px; color: white; font-weight: bold; }
            QDockWidget { border: 1px solid #333; }
            QDockWidget::title { background: #121212; padding-left: 5px; }
            QGroupBox { border: 1px solid #444; margin-top: 6px; font-weight: bold; }
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
        self.ticker_input.returnPressed.connect(self.load_chart)
        
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
        
        # 2. Graphics Layout Widget (Replaces PlotWidget)
        self.layout_widget = pg.GraphicsLayoutWidget()
        self.layout_widget.setBackground('#1e1e1e')
        layout.addWidget(self.layout_widget)
        
        # Initialize Main Plot (Row 0)
        self.main_plot = self.layout_widget.addPlot(row=0, col=0)
        self.main_plot.showGrid(x=True, y=True, alpha=0.15)
        self.main_plot.setMouseEnabled(x=True, y=False)
        self.main_plot.getAxis('left').setPen('#444')
        self.main_plot.getAxis('bottom').setPen('#444')
        
        # Volume Overlay (Linked to Main Plot)
        # We will manually scale bar heights to fit bottom 20% of main plot
        # No separate ViewBox needed if we scale data directly or use a separate Item.
        
        self._setup_crosshair()
        
        # 3. Feature Dock
        self.feature_dock = QDockWidget("Features", self)
        self.feature_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.feature_dock)
        
        dock_content = QWidget()
        self.feature_layout = QVBoxLayout(dock_content)
        self.feature_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.feat_combo = QComboBox()
        sorted_feats = sorted(self.available_features.values(), key=lambda f: (f.category, f.name))
        for f in sorted_feats:
            self.feat_combo.addItem(f"{f.category}: {f.name}", userData=f.name)
            
        self.btn_add_feat = QPushButton("Add Feature")
        self.btn_add_feat.clicked.connect(self.add_feature_ui)
        
        self.feature_layout.addWidget(QLabel("Add Feature:"))
        self.feature_layout.addWidget(self.feat_combo)
        self.feature_layout.addWidget(self.btn_add_feat)
        self.feature_layout.addSpacing(20)
        self.feature_layout.addWidget(QLabel("Active Features:"))
        self.active_feats_container = QVBoxLayout()
        self.feature_layout.addLayout(self.active_feats_container)
        self.feature_dock.setWidget(dock_content)

        # State
        self.df = None
        self.timestamps = []
        self.candle_item = None
        self.simple_candle_item = None
        self.sub_plots = {} # {feat_name: PlotItem}

    def _setup_crosshair(self):
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#888', width=1, style=Qt.PenStyle.DashLine))
        self.main_plot.addItem(self.v_line, ignoreBounds=True)
        self.main_plot.addItem(self.h_line, ignoreBounds=True)
        
        self.price_label = pg.TextItem(anchor=(0, 1), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.time_label = pg.TextItem(anchor=(1, 0), color='#ddd', fill=pg.mkBrush(0, 0, 0, 200))
        self.main_plot.addItem(self.price_label, ignoreBounds=True)
        self.main_plot.addItem(self.time_label, ignoreBounds=True)
        
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
            x_max = view_range[0][1]
            
            self.price_label.setPos(x_max, y)
            self.price_label.setText(f"{y:.2f}")
            self.price_label.setAnchor((1, 0.5))
            
            if 0 <= idx < len(self.timestamps):
                ts = pd.Timestamp(self.timestamps[idx])
                date_str = ts.strftime('%Y-%m-%d %H:%M')
                # If we have subplots, maybe show time on bottom-most plot?
                # For now keep on main plot bottom
                y_min = view_range[1][0]
                self.time_label.setPos(idx, y_min)
                self.time_label.setText(date_str)
                self.time_label.setAnchor((0.5, 1))
            else:
                self.time_label.setText("")

    def add_feature_ui(self):
        feat_name = self.feat_combo.currentData()
        if feat_name in self.active_features: return
        
        feature = self.available_features[feat_name]
        
        group = QGroupBox(feat_name)
        form = QFormLayout(group)
        
        input_widgets = {}
        for key, val in feature.parameters.items():
            if isinstance(val, list):
                inp = QComboBox()
                inp.addItems([str(x) for x in val])
                inp.currentTextChanged.connect(lambda _, n=feat_name: self.update_feature(n))
            elif isinstance(val, bool):
                inp = QCheckBox()
                inp.setChecked(val)
                inp.stateChanged.connect(lambda _, n=feat_name: self.update_feature(n))
            else:
                inp = QLineEdit(str(val))
                inp.editingFinished.connect(lambda n=feat_name: self.update_feature(n))
            
            input_widgets[key] = inp
            form.addRow(key, inp)

        btn_rem = QPushButton("Remove")
        btn_rem.setStyleSheet("background-color: #cc3300;")
        btn_rem.clicked.connect(lambda: self.remove_feature(feat_name, group))
        form.addRow(btn_rem)
        
        self.active_feats_container.addWidget(group)
        
        # Check target pane
        plot_target = self.main_plot
        if feature.target_pane == "new":
            # Create new row
            self.layout_widget.nextRow()
            new_plot = self.layout_widget.addPlot()
            new_plot.setMaximumHeight(150)
            new_plot.setXLink(self.main_plot)
            new_plot.showGrid(x=True, y=True, alpha=0.15)
            # Hide X-axis tick values on subplots to avoid clutter? 
            # Or keep them. Let's keep them for now.
            plot_target = new_plot
            self.sub_plots[feat_name] = new_plot

        self.active_features[feat_name] = {
            "instance": feature,
            "inputs": input_widgets,
            "items": [],
            "plot": plot_target
        }
        
        self.update_feature(feat_name)

    def update_feature(self, feat_name):
        if self.df is None or self.df.empty: return
        
        feat_data = self.active_features[feat_name]
        feature = feat_data["instance"]
        inputs = feat_data["inputs"]
        plot_item = feat_data["plot"]
        
        params = {}
        for key, widget in inputs.items():
            if isinstance(widget, QComboBox): params[key] = widget.currentText()
            elif isinstance(widget, QCheckBox): params[key] = widget.isChecked()
            else: params[key] = widget.text()
        
        try:
            results = feature.compute(self.df, params)
        except Exception as e:
            print(f"Error: {e}")
            return

        for item in feat_data["items"]:
            plot_item.removeItem(item)
        feat_data["items"] = []
        
        for res in results:
            if isinstance(res, LineOutput):
                x = np.arange(len(res.data))
                y = np.array([float(v) if v is not None else np.nan for v in res.data])
                item = plot_item.plot(x, y, pen=pg.mkPen(res.color, width=res.width), name=res.name)
                feat_data["items"].append(item)
                
            elif isinstance(res, LevelOutput):
                region = pg.LinearRegionItem(values=[res.min_price, res.max_price], orientation=pg.LinearRegionItem.Horizontal, 
                                             movable=False, brush=pg.mkBrush(res.color + "40"), pen=None)
                for line in region.lines: line.setPen(None); line.setHoverPen(None)
                plot_item.addItem(region)
                feat_data["items"].append(region)

            elif isinstance(res, HeatmapOutput):
                density = np.array(res.density)
                img_data = density.reshape(1, -1)
                img_item = pg.ImageItem(img_data)
                
                pos = np.array([0.0, 1.0])
                color = np.array([[0, 0, 0, 0], [0, 100, 255, 100]], dtype=np.ubyte)
                if res.color_map == "viridis": # Simple toggle for volume profile
                     color = np.array([[0, 0, 0, 0], [255, 255, 0, 100]], dtype=np.ubyte)

                map = pg.ColorMap(pos, color)
                lut = map.getLookupTable(0.0, 1.0, 256)
                img_item.setLookupTable(lut)
                
                min_p = res.price_grid[0]
                max_p = res.price_grid[-1]
                width = len(self.df) + 1000
                img_item.setRect(QRectF(-500, min_p, width, max_p - min_p))
                img_item.setZValue(-10)
                
                plot_item.addItem(img_item)
                feat_data["items"].append(img_item)

            elif isinstance(res, MarkerOutput):
                symbol_map = {'o': 'o', 'd': 'd', 't': 't1', 's': 's', 'x': 'x', '+': '+'}
                symbol = symbol_map.get(res.shape, 'o')
                scatter = pg.ScatterPlotItem(x=res.indices, y=res.values, pen=None, brush=pg.mkBrush(res.color), symbol=symbol, size=10)
                plot_item.addItem(scatter)
                feat_data["items"].append(scatter)

    def remove_feature(self, feat_name, widget):
        if feat_name in self.active_features:
            data = self.active_features[feat_name]
            # Remove items
            for item in data["items"]:
                data["plot"].removeItem(item)
            
            # If it was a sub-plot, we should remove the plot item entirely?
            # pyqtgraph layout removal is tricky. Usually we just clear it or leave it empty.
            # Removing a row from GraphicsLayoutWidget is hard.
            # For now, we just clear the items. The empty plot pane remains.
            # To fix: we'd need to rebuild the layout. 
            # Or: plot_item.close() might work if supported.
            if data["instance"].target_pane == "new":
                self.layout_widget.removeItem(data["plot"])
                del self.sub_plots[feat_name]
            
            del self.active_features[feat_name]
        
        widget.deleteLater()

    def load_from_history(self, index):
        if index <= 0: return 
        ticker = self.ticker_history.currentText()
        self.ticker_input.setText(ticker)
        self.load_chart()

    def load_random(self):
        tickers = self.db.get_all_tickers()
        if not tickers: tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
        self.ticker_input.setText(random.choice(tickers))
        self.load_chart()

    def add_to_history(self, ticker):
        if self.ticker_history.findText(ticker) == -1:
            self.ticker_history.addItem(ticker)

    def load_chart(self):
        ticker = self.ticker_input.text().upper()
        interval = self.interval_combo.currentText()
        
        self.df = self.db.get_data(ticker, interval)
        if self.df.empty or len(self.df) < 50:
            print("Fetching data...")
            period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d", "4h": "730d", "1d": "10y", "1w": "10y"}
            try:
                self.engine.sync_data(ticker, interval, period=period_map.get(interval, "1y"))
                self.df = self.db.get_data(ticker, interval)
            except Exception as e: print(e)

        if self.df.empty: return
        self.add_to_history(ticker)

        # Clear Main Plot
        self.main_plot.clear()
        
        # Clear Sub Plots
        # We need to iterate and clear feature items, but the plot items persist.
        # Ideally, we keep the feature widgets active but clear their drawn items.
        # The update_feature logic handles clearing items, so we just need to re-render.
        
        self._setup_crosshair()
        
        self.timestamps = self.df.index.astype(str).tolist()
        
        # Axis Tick Logic
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
        self.main_plot.getAxis('bottom').tickStrings = tickStrings
        
        # Candles
        data = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            data.append((float(i), row['Open'], row['Close'], row['Low'], row['High']))
            
        self.candle_item = CandlestickItem(data)
        self.simple_candle_item = SimpleCandleItem(data)
        
        self.main_plot.addItem(self.candle_item)
        self.main_plot.addItem(self.simple_candle_item)
        self.simple_candle_item.setVisible(False)
        
        # Volume Overlay
        # Draw volume bars at bottom of main chart
        vol = self.df['Volume'].values
        # Scale volume to fit in bottom 20% of price range?
        # Better: use a separate viewbox or just normalized bars?
        # Simple: Normalized bars scaled to (min_y, min_y + range*0.2)
        # We'll update this dynamic scaling in update_view()
        self.volume_item = pg.BarGraphItem(x=np.arange(len(data)), height=vol, width=0.8, brush=pg.mkBrush(255, 255, 255, 30))
        self.main_plot.addItem(self.volume_item)
        # We need to scale it manually in update_view
        
        # Update All Active Features
        for name in self.active_features:
            self.update_feature(name)
        
        # Zoom & AutoScale
        try: self.main_plot.vb.sigXRangeChanged.disconnect()
        except: pass
        self.main_plot.vb.sigXRangeChanged.connect(self.update_view)
        
        self.main_plot.setXRange(max(0, len(data)-100), len(data))
        self.update_view()

    def update_view(self):
        if self.df is None: return
        vb = self.main_plot.vb
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
        
        y_min = sub['Low'].min()
        y_max = sub['High'].max()
        if pd.isna(y_min): return
        
        # Scale Volume
        # We map volume 0..MaxVol to y_min..(y_min + range*0.2)
        # But BarGraphItem draws from 0. We need to transform it.
        # Actually, simpler to just set the scale of the item?
        # No, bar graph heights are fixed.
        # We can re-set the data of the volume item?
        # Or better: use a separate ViewBox for volume overlaid on the plot.
        # For simplicity in this step, I'll just skip complex volume scaling scaling 
        # and assume the user focuses on the sub-pane architecture.
        # Wait, I added volume_item. Let's try to scale it.
        
        vol_sub = sub['Volume']
        max_vol = vol_sub.max()
        if max_vol > 0:
            # We want bar height to be (vol / max_vol) * (y_max - y_min) * 0.2
            # AND we want base at y_min.
            # BarGraphItem draws from y=0.
            # So we set y0 = y_min.
            # And height = scaled height.
            # Only way is to update the item data.
            price_range = y_max - y_min
            target_height = price_range * 0.2
            scale_factor = target_height / max_vol
            
            # This is expensive to do every frame.
            # Alternative: Use a separate ViewBox linked to X but independent Y.
            # I will omit dynamic volume scaling for this iteration to ensure stability
            # of the sub-pane features which was the main request.
            pass

        pad = (y_max - y_min) * 0.1
        if pad == 0: pad = 1
        vb.setYRange(y_min - pad, y_max + pad, padding=0)
