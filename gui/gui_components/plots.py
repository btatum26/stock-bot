import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal
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
        super().__init__(z_value=1000)
        self.candle_full = None
        self.candle_simple = None
        self.df = None
        self._low_arr = np.array([])
        self._high_arr = np.array([])

    def update(self, df):
        self.df = df
        self._low_arr = df['Low'].values
        self._high_arr = df['High'].values

        self.clear_items()

        # Build numpy arrays once — no df.to_dict('records')
        n = len(df)
        t = np.arange(n, dtype=np.float64)
        op = df['Open'].values.astype(np.float64)
        cl = df['Close'].values.astype(np.float64)
        lo = df['Low'].values.astype(np.float64)
        hi = df['High'].values.astype(np.float64)

        self.candle_full = CandlestickItem(t, op, cl, lo, hi)
        self.candle_simple = SimpleCandleItem(t, op, cl, lo, hi)

        self.candle_full.setZValue(self.z_value)
        self.candle_simple.setZValue(self.z_value)

        self.items = [self.candle_full, self.candle_simple]

        if self.plot_item:
            self.add_to_plot(self.plot_item)
            try:
                x_range = self.plot_item.vb.viewRange()[0]
                self.update_lod(x_range[0], x_range[1])
            except:
                self.update_lod(0, min(n, 300))

    def update_lod(self, x_min, x_max):
        if not self.candle_full:
            return
        width = x_max - x_min
        if width < 450:
            self.candle_full.setVisible(True)
            self.candle_simple.setVisible(False)
            self.candle_full.rebuild_visible(x_min, x_max)
        else:
            self.candle_full.setVisible(False)
            self.candle_simple.setVisible(True)
            self.candle_simple.rebuild_visible(x_min, x_max)

    def get_y_range(self, x_min, x_max):
        if self.df is None or len(self._low_arr) == 0:
            return None, None
        idx_min, idx_max = max(0, int(x_min)), min(len(self._low_arr), int(x_max) + 1)
        if idx_min >= idx_max:
            return None, None
        return np.nanmin(self._low_arr[idx_min:idx_max]), np.nanmax(self._high_arr[idx_min:idx_max])

class LineOverlay(BaseOverlay):
    def __init__(self, data_dict, color='#fff', width=1, z_value=10):
        """
        data_dict: {name: numpy_array_or_list}
        """
        super().__init__(z_value=z_value)
        self.data_dict = data_dict
        self.color = color
        self.width = width
        self._pen = pg.mkPen(color, width=width)
        self.line_items = {}  # {name: PlotDataItem}
        self._y_arrays = {}  # {name: np.ndarray} — cached for get_y_range

    def set_color(self, color):
        self.color = color
        self._pen = pg.mkPen(color, width=self.width)
        for item in self.line_items.values():
            item.setPen(self._pen)

    def update(self, df):
        for name, data in self.data_dict.items():
            arr = np.asarray(data, dtype=np.float64)
            self._y_arrays[name] = arr
            x = np.arange(len(arr))

            if name in self.line_items:
                # Reuse existing PlotDataItem
                self.line_items[name].setData(x, arr)
            else:
                item = pg.PlotDataItem(x, arr, pen=self._pen)
                item.setZValue(self.z_value)
                self.line_items[name] = item
                self.items.append(item)

        if self.plot_item:
            # Only add items not yet on the plot
            for item in self.items:
                if item.scene() is None:
                    item.setZValue(self.z_value)
                    self.plot_item.addItem(item)

    def get_y_range(self, x_min, x_max):
        y_min, y_max = np.inf, -np.inf
        for arr in self._y_arrays.values():
            idx_min = max(0, int(x_min))
            idx_max = min(len(arr), int(x_max) + 1)
            if idx_min >= idx_max:
                continue
            sl = arr[idx_min:idx_max]
            lo = np.nanmin(sl)
            hi = np.nanmax(sl)
            if not np.isnan(lo):
                y_min = min(y_min, lo)
                y_max = max(y_max, hi)

        return (y_min, y_max) if y_min != np.inf else (None, None)

class VolumeOverlay(BaseOverlay):
    def __init__(self):
        super().__init__(z_value=-10)
        self.vol_view = pg.ViewBox()
        self.vol_item = None
        self.df = None
        self._vol_arr = None

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
        self._vol_arr = df['Volume'].values.astype(np.float64)
        self.vol_view.clear()

        n = len(df)
        t = np.arange(n, dtype=np.float64)
        op = df['Open'].values.astype(np.float64)
        cl = df['Close'].values.astype(np.float64)

        self.vol_item = VolumeItem(t, op, cl, self._vol_arr)
        self.vol_item.setZValue(self.z_value)
        self.vol_view.addItem(self.vol_item)

    def update_y_range(self, x_min, x_max):
        """Rescale volume Y-axis. Viewport culling (rebuild_visible) is handled
        by UnifiedPlot._on_x_range_changed on every frame already."""
        if self._vol_arr is None or len(self._vol_arr) == 0:
            return
        idx_min, idx_max = max(0, int(x_min)), min(len(self._vol_arr), int(x_max) + 1)
        if idx_min >= idx_max:
            return
        v_max = np.nanmax(self._vol_arr[idx_min:idx_max])
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

        n = len(self.scores)
        img_data = np.zeros((n, 1, 4), dtype=np.uint8)

        valid_mask = ~np.isnan(self.scores)
        nonzero_mask = valid_mask & (self.scores != 0)
        if not np.any(nonzero_mask):
            self.img_item.setImage(img_data)
            return

        max_val = np.nanmax(np.abs(self.scores[valid_mask]))
        if max_val == 0:
            max_val = 1.0

        # Vectorized intensity and alpha
        intensity = np.clip(np.abs(self.scores) / max_val, 0.0, 1.0)
        effective_alpha = np.maximum(0.1, intensity) * self.alpha
        alpha_vals = (255 * effective_alpha).astype(np.uint8)

        p = self.pos_color
        n_col = self.neg_color

        pos_mask = nonzero_mask & (self.scores > 0)
        neg_mask = nonzero_mask & (self.scores < 0)

        img_data[pos_mask, 0, 0] = p.red()
        img_data[pos_mask, 0, 1] = p.green()
        img_data[pos_mask, 0, 2] = p.blue()
        img_data[pos_mask, 0, 3] = alpha_vals[pos_mask]

        img_data[neg_mask, 0, 0] = n_col.red()
        img_data[neg_mask, 0, 1] = n_col.green()
        img_data[neg_mask, 0, 2] = n_col.blue()
        img_data[neg_mask, 0, 3] = alpha_vals[neg_mask]

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
    sigPlotClicked = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlays = []
        self._viewport_overlays = []  # cached: only CandleOverlay/VolumeOverlay
        self._has_viewport_items = False
        self._has_candles = False
        self.fixed_y_range = None
        self.y_padding = 0.1
        self.showGrid(x=False, y=True, alpha=0.15)
        # Explicitly enable mouse on the ViewBox
        self.vb.setMouseEnabled(x=True, y=False)
        self.vb.disableAutoRange(pg.ViewBox.YAxis)
        # ViewBox Z-order - Ensure ViewBox is above background items for mouse events
        self.vb.setZValue(100)

        # Debounce Y-scaling: fire at most once per 24 ms instead of on every
        # pixel of pan/zoom.
        self._y_scale_timer = QTimer()
        self._y_scale_timer.setSingleShot(True)
        self._y_scale_timer.setInterval(24)
        self._y_scale_timer.timeout.connect(self._run_auto_scale_y)

        # Cache the last integer-rounded viewport range so we can skip
        # redundant rebuild_visible calls when panning sub-pixel amounts.
        self._last_vp: tuple = (0, 0)

        # Viewport culling runs immediately on every range change so there are
        # no blank regions while dragging.  Y-scaling is still debounced.
        self.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.sigPlotClicked.emit(ev)
        super().mousePressEvent(ev)

    def set_fixed_y_range(self, y_min, y_max, padding=0.1):
        self.fixed_y_range = (y_min, y_max)
        self.y_padding = padding
        self.setYRange(y_min, y_max, padding=padding)

    def add_overlay(self, overlay):
        if overlay in self.overlays: return
        self.overlays.append(overlay)
        self.overlays.sort(key=lambda x: x.z_value)
        overlay.add_to_plot(self)
        self._rebuild_viewport_cache()

    def remove_overlay(self, overlay):
        if overlay in self.overlays:
            overlay.remove_from_plot()
            self.overlays.remove(overlay)
            self._rebuild_viewport_cache()

    def _rebuild_viewport_cache(self):
        self._viewport_overlays = [
            o for o in self.overlays
            if isinstance(o, (CandleOverlay, VolumeOverlay))
        ]
        self._has_viewport_items = len(self._viewport_overlays) > 0
        self._has_candles = any(isinstance(o, CandleOverlay) for o in self.overlays)

    def clear_overlays(self):
        for o in self.overlays[:]:
            self.remove_overlay(o)

    def update_all(self, df):
        # Update in Z-order
        for o in sorted(self.overlays, key=lambda x: x.z_value):
            o.update(df)
        # Run immediately (not deferred) — this is a programmatic data load.
        self._y_scale_timer.stop()
        self._run_auto_scale_y()

    def auto_scale_y(self):
        """Public entry point: run immediately (for programmatic calls)."""
        self._y_scale_timer.stop()
        self._run_auto_scale_y()

    def _on_x_range_changed(self):
        """Fires on every pan/zoom pixel — rebuild visible items immediately,
        then schedule the heavier Y-range computation."""
        if not self._has_viewport_items:
            # Sub-plots with only LineOverlays — nothing to cull, just rescale Y
            self._y_scale_timer.start()
            return

        x_range = self.vb.viewRange()[0]
        x_min, x_max = x_range

        # Skip viewport rebuild if the integer-rounded range hasn't changed —
        # sub-pixel panning doesn't affect which candles/bars are visible.
        vp = (int(x_min), int(x_max))
        if vp != self._last_vp:
            self._last_vp = vp
            for o in self._viewport_overlays:
                if isinstance(o, VolumeOverlay):
                    if o.vol_item is not None:
                        o.vol_item.rebuild_visible(x_min, x_max)
                elif hasattr(o, 'update_lod'):
                    o.update_lod(x_min, x_max)

        self._y_scale_timer.start()

    def _run_auto_scale_y(self):
        """Debounced Y-range computation. Viewport culling already happened
        in _on_x_range_changed, so this only does the math + setYRange."""
        if self.fixed_y_range:
            return

        x_range = self.vb.viewRange()[0]
        x_min, x_max = x_range

        y_min, y_max = np.inf, -np.inf

        for o in self.overlays:
            if isinstance(o, VolumeOverlay):
                o.update_y_range(x_min, x_max)
                continue

            if hasattr(o, 'get_y_range'):
                if self._has_candles and not isinstance(o, CandleOverlay):
                    continue
                omin, omax = o.get_y_range(x_min, x_max)
                if omin is not None:
                    y_min = min(y_min, omin)
                    y_max = max(y_max, omax)

        if y_min != np.inf:
            pad = (y_max - y_min) * self.y_padding or 1.0
            self.setYRange(y_min - pad, y_max + pad, padding=0)
