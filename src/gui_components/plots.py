import pyqtgraph as pg
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QRectF
from .candles import CandlestickItem, SimpleCandleItem
from .volume import VolumeItem

class BaseOverlay:
    def __init__(self, z_value=0):
        self.items = []
        self.plot_item = None
        self.z_value = z_value

    def add_to_plot(self, plot_item):
        self.plot_item = plot_item
        for item in self.items:
            item.setZValue(self.z_value)
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
        # Maximum Z-value so it stays on top of everything
        super().__init__(z_value=1000)
        self.candle_full = None
        self.candle_simple = None
        self.df = None

    def update(self, df):
        self.df = df
        self.clear_items()
        
        data = [(float(i), r['Open'], r['Close'], r['Low'], r['High']) for i, r in enumerate(df.to_dict('records'))]
        self.candle_full = CandlestickItem(data)
        self.candle_simple = SimpleCandleItem(data)
        
        # Apply Z-value to new items
        self.candle_full.setZValue(self.z_value)
        self.candle_simple.setZValue(self.z_value)
        
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
    def __init__(self, data_dict, color='#fff', width=1, z_value=10):
        """
        data_dict: {name: array_like}
        """
        super().__init__(z_value=z_value)
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
            item.setZValue(self.z_value)
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
        super().__init__(z_value=-10)
        self.vol_view = pg.ViewBox()
        self.vol_item = None
        self.df = None

    def add_to_plot(self, plot_item):
        self.plot_item = plot_item
        if plot_item.scene():
            plot_item.scene().addItem(self.vol_view)
        self.vol_view.setXLink(plot_item.vb)
        self.vol_view.setZValue(self.z_value)
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
        self.vol_item.setZValue(self.z_value)
        self.vol_view.addItem(self.vol_item)

    def update_y_range(self, x_min, x_max):
        if self.df is None or self.df.empty: return
        idx_min, idx_max = max(0, int(x_min)), min(len(self.df), int(x_max) + 1)
        if idx_min >= idx_max: return
        v_max = np.nanmax(self.df['Volume'].iloc[idx_min:idx_max].values)
        self.vol_view.setYRange(0, v_max * 4 if v_max > 0 else 1, padding=0)

class LevelOverlay(BaseOverlay):
    def __init__(self, price, color='#888', style=Qt.PenStyle.DashLine, z_value=5):
        super().__init__(z_value=z_value)
        self.price = price
        self.color = color
        self.style = style

    def update(self, df):
        self.clear_items()
        item = pg.InfiniteLine(pos=self.price, angle=0, pen=pg.mkPen(self.color, width=1, style=self.style))
        item.setZValue(self.z_value)
        self.items = [item]
        if self.plot_item:
            self.add_to_plot(self.plot_item)

    def get_y_range(self, x_min, x_max):
        return self.price, self.price

class ScoreOverlay(BaseOverlay):
    def __init__(self, scores, pos_color='#00ff00', neg_color='#ff0000', alpha=0.3, z_value=-100):
        """
        Background coloring based on scores using ImageItem for performance.
        z_value is very low to stay behind everything.
        """
        super().__init__(z_value=z_value)
        self.scores = np.array(scores) if scores is not None else None
        self.pos_color = pg.mkColor(pos_color)
        self.neg_color = pg.mkColor(neg_color)
        self.alpha = alpha
        self.img_item = pg.ImageItem()
        self.img_item.setZValue(self.z_value)
        self.img_item.setOpacity(1.0)
        # Completely disable mouse events so panning works on the main chart
        self.img_item.setEnabled(False)
        self.items = [self.img_item]
        self._connected = False
        
        # Initial data update
        self._update_image_data()

    def set_visuals(self, pos_color=None, neg_color=None, alpha=None):
        """Update only the colors/alpha of the image for performance."""
        if pos_color: self.pos_color = pg.mkColor(pos_color)
        if neg_color: self.neg_color = pg.mkColor(neg_color)
        if alpha is not None: self.alpha = alpha
        self._update_image_data()

    def _update_image_data(self):
        if self.scores is None or len(self.scores) == 0:
            self.img_item.setImage(np.zeros((1, 1, 4), dtype=np.uint8))
            return
            
        # Create an RGBA image array (N, 1, 4)
        n = len(self.scores)
        img_data = np.zeros((n, 1, 4), dtype=np.uint8)
        
        valid_mask = ~np.isnan(self.scores)
        if not np.any(valid_mask):
            self.img_item.setImage(img_data)
            return
            
        max_val = np.nanmax(np.abs(self.scores[valid_mask]))
        if max_val == 0: max_val = 1.0

        p = self.pos_color
        n_col = self.neg_color
        
        for i, val in enumerate(self.scores):
            if np.isnan(val) or val == 0:
                continue
            
            intensity = min(1.0, abs(val) / max_val)
            # Ensure even small scores are visible (min 10% of alpha)
            effective_alpha = max(0.1, intensity) * self.alpha
            alpha_val = int(255 * effective_alpha)
            
            if val > 0:
                img_data[i, 0] = [p.red(), p.green(), p.blue(), alpha_val]
            else:
                img_data[i, 0] = [n_col.red(), n_col.green(), n_col.blue(), alpha_val]
        
        self.img_item.setImage(img_data)

    def update(self, df):
        # Image data doesn't change with df unless scores are re-calculated outside
        # We just need to ensure the geometry is correct
        self._update_img_geometry()

    def add_to_plot(self, plot_item):
        super().add_to_plot(plot_item)
        if not self._connected:
            plot_item.vb.sigRangeChanged.connect(self._update_img_geometry)
            self._connected = True
        self._update_img_geometry()

    def remove_from_plot(self):
        if self.plot_item and self._connected:
            try: self.plot_item.vb.sigRangeChanged.disconnect(self._update_img_geometry)
            except: pass
            self._connected = False
        super().remove_from_plot()

    def _update_img_geometry(self):
        if self.img_item and self.plot_item and self.scores is not None:
            try:
                y_range = self.plot_item.vb.viewRange()[1]
                height = y_range[1] - y_range[0]
                if height > 0:
                    # Map the N pixels to X indices [0, N] and Y range
                    self.img_item.setRect(QRectF(-0.5, y_range[0], len(self.scores), height))
            except:
                pass

class UnifiedPlot(pg.PlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlays = []
        self.fixed_y_range = None
        self.y_padding = 0.1
        self.showGrid(x=False, y=True, alpha=0.15)
        # Explicitly enable mouse on the ViewBox
        self.vb.setMouseEnabled(x=True, y=False)
        self.vb.disableAutoRange(pg.ViewBox.YAxis)
        # ViewBox Z-order - Ensure ViewBox is above background items for mouse events
        self.vb.setZValue(100)
        # ViewBox signal for auto scaling
        self.vb.sigXRangeChanged.connect(self.auto_scale_y)
        
    def set_fixed_y_range(self, y_min, y_max, padding=0.1):
        self.fixed_y_range = (y_min, y_max)
        self.y_padding = padding
        self.setYRange(y_min, y_max, padding=padding)

    def add_overlay(self, overlay):
        if overlay in self.overlays: return
        self.overlays.append(overlay)
        self.overlays.sort(key=lambda x: x.z_value)
        overlay.add_to_plot(self)

    def remove_overlay(self, overlay):
        if overlay in self.overlays:
            overlay.remove_from_plot()
            self.overlays.remove(overlay)

    def clear_overlays(self):
        for o in self.overlays[:]:
            self.remove_overlay(o)

    def update_all(self, df):
        # Update in Z-order
        for o in sorted(self.overlays, key=lambda x: x.z_value):
            o.update(df)
        self.auto_scale_y()

    def auto_scale_y(self):
        if self.fixed_y_range:
            return

        x_range = self.vb.viewRange()[0]
        x_min, x_max = x_range
        
        y_min, y_max = np.inf, -np.inf
        
        # Priority 1: Check if we have a CandleOverlay (main price chart)
        has_candles = any(isinstance(o, CandleOverlay) for o in self.overlays)
        
        for o in self.overlays:
            if isinstance(o, VolumeOverlay):
                o.update_y_range(x_min, x_max)
                continue
            
            # Update candles LOD if applicable
            if hasattr(o, 'update_lod'):
                o.update_lod(x_min, x_max)
                
            # Use general get_y_range if available
            if hasattr(o, 'get_y_range'):
                # If we have candles, ONLY let the CandleOverlay determine the scale
                if has_candles and not isinstance(o, CandleOverlay):
                    continue
                    
                omin, omax = o.get_y_range(x_min, x_max)
                if omin is not None:
                    y_min = min(y_min, omin)
                    y_max = max(y_max, omax)

        if y_min != np.inf:
            pad = (y_max - y_min) * self.y_padding or 1.0
            self.setYRange(y_min - pad, y_max + pad, padding=0)
