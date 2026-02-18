import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from .candles import CandlestickItem, SimpleCandleItem
from .volume import VolumeItem

class BaseOverlay:
    def __init__(self):
        self.items = []
        self.plot_item = None

    def add_to_plot(self, plot_item):
        self.plot_item = plot_item
        for item in self.items:
            plot_item.addItem(item)

    def clear_items(self):
        """Removes items from plot but keeps the plot_item reference."""
        if self.plot_item:
            for item in self.items:
                try:
                    self.plot_item.removeItem(item)
                except:
                    pass
        self.items = []

    def remove_from_plot(self):
        """Full detachment."""
        self.clear_items()
        self.plot_item = None

    def update(self, df):
        pass

    def get_y_range(self, x_min, x_max):
        return None, None

class CandleOverlay(BaseOverlay):
    def __init__(self):
        super().__init__()
        self.candle_full = None
        self.candle_simple = None
        self.df = None

    def update(self, df):
        self.df = df
        self.clear_items()
        
        data = [(float(i), r['Open'], r['Close'], r['Low'], r['High']) for i, r in enumerate(df.to_dict('records'))]
        self.candle_full = CandlestickItem(data)
        self.candle_simple = SimpleCandleItem(data)
        self.items = [self.candle_full, self.candle_simple]
        
        if self.plot_item:
            self.add_to_plot(self.plot_item)
            # Use current view range if available
            try:
                x_range = self.plot_item.vb.viewRange()[0]
                self.update_lod(x_range[0], x_range[1])
            except:
                self.update_lod(0, 100)

    def update_lod(self, x_min, x_max):
        if not self.candle_full: return
        width = x_max - x_min
        if width < 450:
            self.candle_full.setVisible(True)
            self.candle_simple.setVisible(False)
        else:
            self.candle_full.setVisible(False)
            self.candle_simple.setVisible(True)

    def get_y_range(self, x_min, x_max):
        if self.df is None or self.df.empty: return None, None
        idx_min, idx_max = max(0, int(x_min)), min(len(self.df), int(x_max) + 1)
        if idx_min >= idx_max: return None, None
        sub = self.df.iloc[idx_min:idx_max]
        return np.nanmin(sub['Low'].values), np.nanmax(sub['High'].values)

class LineOverlay(BaseOverlay):
    def __init__(self, data_dict, color='#fff', width=1):
        """
        data_dict: {name: array_like}
        """
        super().__init__()
        self.data_dict = data_dict
        self.color = color
        self.width = width
        self.line_items = {} # {name: PlotDataItem}

    def update(self, df):
        self.clear_items()
        for name, data in self.data_dict.items():
            # Handle potential None/NaN values
            clean_data = np.array([float(v) if v is not None else np.nan for v in data])
            item = pg.PlotDataItem(np.arange(len(clean_data)), clean_data, pen=pg.mkPen(self.color, width=self.width))
            self.line_items[name] = item
            self.items.append(item)
        
        if self.plot_item:
            self.add_to_plot(self.plot_item)

    def get_y_range(self, x_min, x_max):
        y_min, y_max = np.inf, -np.inf
        for item in self.line_items.values():
            x_data, y_data = item.getData()
            if y_data is not None:
                idx_min, idx_max = max(0, int(x_min)), min(len(y_data), int(x_max) + 1)
                visible_y = y_data[idx_min:idx_max]
                valid_y = visible_y[~np.isnan(visible_y)]
                if len(valid_y) > 0:
                    y_min = min(y_min, np.nanmin(valid_y))
                    y_max = max(y_max, np.nanmax(valid_y))
        
        return (y_min, y_max) if y_min != np.inf else (None, None)

class VolumeOverlay(BaseOverlay):
    def __init__(self):
        super().__init__()
        self.vol_view = pg.ViewBox()
        self.vol_item = None
        self.df = None

    def add_to_plot(self, plot_item):
        self.plot_item = plot_item
        if plot_item.scene():
            plot_item.scene().addItem(self.vol_view)
        self.vol_view.setXLink(plot_item.vb)
        self.vol_view.setZValue(-10)
        self.vol_view.setMouseEnabled(x=False, y=False)
        self.vol_view.setAcceptHoverEvents(False)
        # Ensure volume view doesn't block mouse events for the main plot
        self.vol_view.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        
        # Connect resize
        plot_item.vb.sigResized.connect(self.update_geometry)
        self.update_geometry()

    def update_geometry(self):
        if not self.plot_item: return
        rect = self.plot_item.vb.sceneBoundingRect()
        if rect.isValid():
            self.vol_view.setGeometry(rect)

    def remove_from_plot(self):
        if self.plot_item:
            if self.plot_item.scene():
                self.plot_item.scene().removeItem(self.vol_view)
            self.plot_item.vb.sigResized.disconnect(self.update_geometry)
        self.vol_view.clear()
        self.plot_item = None

    def update(self, df):
        self.df = df
        self.vol_view.clear()
        data = [(float(i), r['Open'], r['Close'], r['Volume']) for i, r in enumerate(df.to_dict('records'))]
        self.vol_item = VolumeItem(data)
        self.vol_view.addItem(self.vol_item)

    def update_y_range(self, x_min, x_max):
        if self.df is None or self.df.empty: return
        idx_min, idx_max = max(0, int(x_min)), min(len(self.df), int(x_max) + 1)
        if idx_min >= idx_max: return
        v_max = np.nanmax(self.df['Volume'].iloc[idx_min:idx_max].values)
        self.vol_view.setYRange(0, v_max * 4 if v_max > 0 else 1, padding=0)

class LevelOverlay(BaseOverlay):
    def __init__(self, price, color='#888', style=Qt.PenStyle.DashLine):
        super().__init__()
        self.price = price
        self.color = color
        self.style = style

    def update(self, df):
        self.clear_items()
        item = pg.InfiniteLine(pos=self.price, angle=0, pen=pg.mkPen(self.color, width=1, style=self.style))
        self.items = [item]
        if self.plot_item:
            self.add_to_plot(self.plot_item)

    def get_y_range(self, x_min, x_max):
        return self.price, self.price

class UnifiedPlot(pg.PlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlays = []
        self.showGrid(x=False, y=True, alpha=0.15)
        # Explicitly enable mouse on the ViewBox
        self.vb.setMouseEnabled(x=True, y=False)
        self.vb.disableAutoRange(pg.ViewBox.YAxis)
        # Ensure ViewBox is high enough in Z-order to capture events
        self.vb.setZValue(10)
        self.vb.sigXRangeChanged.connect(self.auto_scale_y)
        
    def add_overlay(self, overlay):
        self.overlays.append(overlay)
        overlay.add_to_plot(self)

    def remove_overlay(self, overlay):
        if overlay in self.overlays:
            overlay.remove_from_plot()
            self.overlays.remove(overlay)

    def clear_overlays(self):
        for o in self.overlays[:]:
            self.remove_overlay(o)

    def update_all(self, df):
        for o in self.overlays:
            o.update(df)
        self.auto_scale_y()

    def auto_scale_y(self):
        x_range = self.vb.viewRange()[0]
        x_min, x_max = x_range
        
        y_min, y_max = np.inf, -np.inf
        
        for o in self.overlays:
            # Handle volume separately as it has its own ViewBox
            if isinstance(o, VolumeOverlay):
                o.update_y_range(x_min, x_max)
                continue
                
            # Handle LOD for candles
            if isinstance(o, CandleOverlay):
                o.update_lod(x_min, x_max)

            omin, omax = o.get_y_range(x_min, x_max)
            if omin is not None:
                y_min = min(y_min, omin)
                y_max = max(y_max, omax)

        if y_min != np.inf:
            pad = (y_max - y_min) * 0.1 or 1.0
            self.setYRange(y_min - pad, y_max + pad, padding=0)
