import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QPicture, QPainter
from PyQt6.QtCore import QPointF, QRectF


class CandlestickItem(pg.GraphicsObject):
    """Full-body candlestick renderer with viewport culling.

    Stores OHLC as numpy arrays and only paints the visible slice (+buffer)
    into the QPicture, rather than baking the entire dataset up front.
    """

    _pen_g = None
    _pen_r = None
    _brush_g = None
    _brush_r = None

    def __init__(self, t, open_, close, low, high):
        super().__init__()
        self._t = np.asarray(t, dtype=np.float64)
        self._open = np.asarray(open_, dtype=np.float64)
        self._close = np.asarray(close, dtype=np.float64)
        self._low = np.asarray(low, dtype=np.float64)
        self._high = np.asarray(high, dtype=np.float64)
        self._n = len(self._t)

        self._picture = QPicture()
        self._vis_start = 0
        self._vis_end = 0
        self._dirty = True

        # Class-level pen/brush cache (created once, shared across instances)
        if CandlestickItem._pen_g is None:
            CandlestickItem._pen_g = pg.mkPen('g')
            CandlestickItem._pen_r = pg.mkPen('r')
            CandlestickItem._brush_g = pg.mkBrush('g')
            CandlestickItem._brush_r = pg.mkBrush('r')

    def rebuild_visible(self, x_min: float, x_max: float, buffer: int = 50):
        """Regenerate the QPicture for indices in [x_min-buffer, x_max+buffer]."""
        lo = max(0, int(x_min) - buffer)
        hi = min(self._n, int(x_max) + 1 + buffer)

        if lo == self._vis_start and hi == self._vis_end and not self._dirty:
            return  # nothing changed

        self._vis_start = lo
        self._vis_end = hi
        self._dirty = False

        pic = QPicture()
        p = QPainter(pic)
        w = 0.4

        t = self._t[lo:hi]
        op = self._open[lo:hi]
        cl = self._close[lo:hi]
        lo_arr = self._low[lo:hi]
        hi_arr = self._high[lo:hi]

        pen_g = self._pen_g
        pen_r = self._pen_r
        brush_g = self._brush_g
        brush_r = self._brush_r

        for i in range(len(t)):
            ti, oi, ci, li, hi_val = t[i], op[i], cl[i], lo_arr[i], hi_arr[i]
            if ci > oi:
                p.setPen(pen_g)
                p.setBrush(brush_g)
            else:
                p.setPen(pen_r)
                p.setBrush(brush_r)
            p.drawLine(QPointF(ti, li), QPointF(ti, hi_val))
            p.drawRect(QRectF(ti - w, oi, w * 2, ci - oi))
        p.end()
        self._picture = pic
        self.prepareGeometryChange()
        self.update()

    def paint(self, p, *args):
        self._picture.play(p)

    def boundingRect(self):
        return QRectF(self._picture.boundingRect())


class SimpleCandleItem(pg.GraphicsObject):
    """Wick-only candlestick renderer with viewport culling."""

    _pen_g = None
    _pen_r = None

    def __init__(self, t, open_, close, low, high):
        super().__init__()
        self._t = np.asarray(t, dtype=np.float64)
        self._open = np.asarray(open_, dtype=np.float64)
        self._close = np.asarray(close, dtype=np.float64)
        self._low = np.asarray(low, dtype=np.float64)
        self._high = np.asarray(high, dtype=np.float64)
        self._n = len(self._t)

        self._picture = QPicture()
        self._vis_start = 0
        self._vis_end = 0
        self._dirty = True

        if SimpleCandleItem._pen_g is None:
            SimpleCandleItem._pen_g = pg.mkPen('g', width=1)
            SimpleCandleItem._pen_r = pg.mkPen('r', width=1)

    def rebuild_visible(self, x_min: float, x_max: float, buffer: int = 50):
        lo = max(0, int(x_min) - buffer)
        hi = min(self._n, int(x_max) + 1 + buffer)

        if lo == self._vis_start and hi == self._vis_end and not self._dirty:
            return

        self._vis_start = lo
        self._vis_end = hi
        self._dirty = False

        pic = QPicture()
        p = QPainter(pic)
        pen_g = self._pen_g
        pen_r = self._pen_r

        t = self._t[lo:hi]
        op = self._open[lo:hi]
        cl = self._close[lo:hi]
        lo_arr = self._low[lo:hi]
        hi_arr = self._high[lo:hi]

        for i in range(len(t)):
            ti, oi, ci, li, hi_val = t[i], op[i], cl[i], lo_arr[i], hi_arr[i]
            if ci > oi:
                p.setPen(pen_g)
            else:
                p.setPen(pen_r)
            p.drawLine(QPointF(ti, li), QPointF(ti, hi_val))
        p.end()
        self._picture = pic
        self.prepareGeometryChange()
        self.update()

    def paint(self, p, *args):
        self._picture.play(p)

    def boundingRect(self):
        return QRectF(self._picture.boundingRect())
