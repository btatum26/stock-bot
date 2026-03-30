import pyqtgraph as pg
from PyQt6.QtGui import QPicture, QPainter
from PyQt6.QtCore import QPointF, QRectF

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
