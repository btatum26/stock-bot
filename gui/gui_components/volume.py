import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QPicture, QPainter
from PyQt6.QtCore import QRectF


class VolumeItem(pg.GraphicsObject):
    """Volume bar renderer with viewport culling.

    Stores data as numpy arrays and only paints the visible slice (+buffer).
    """

    _green_brush = None
    _red_brush = None
    _no_pen = None

    def __init__(self, t, open_, close, volume):
        super().__init__()
        self._t = np.asarray(t, dtype=np.float64)
        self._open = np.asarray(open_, dtype=np.float64)
        self._close = np.asarray(close, dtype=np.float64)
        self._vol = np.asarray(volume, dtype=np.float64)
        self._n = len(self._t)

        self._picture = QPicture()
        self._vis_start = 0
        self._vis_end = 0
        self._dirty = True

        if VolumeItem._green_brush is None:
            VolumeItem._green_brush = pg.mkBrush('#00ff0050')
            VolumeItem._red_brush = pg.mkBrush('#ff000050')
            VolumeItem._no_pen = pg.mkPen(None)

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
        w = 0.4

        p.setPen(self._no_pen)
        green_brush = self._green_brush
        red_brush = self._red_brush

        t = self._t[lo:hi]
        op = self._open[lo:hi]
        cl = self._close[lo:hi]
        vol = self._vol[lo:hi]

        for i in range(len(t)):
            if cl[i] >= op[i]:
                p.setBrush(green_brush)
            else:
                p.setBrush(red_brush)
            p.drawRect(QRectF(t[i] - w, 0, w * 2, vol[i]))
        p.end()
        self._picture = pic
        self.prepareGeometryChange()
        self.update()

    def paint(self, p, *args):
        self._picture.play(p)

    def boundingRect(self):
        return QRectF(self._picture.boundingRect())
