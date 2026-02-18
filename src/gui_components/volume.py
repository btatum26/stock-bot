import pyqtgraph as pg
from PyQt6.QtGui import QPicture, QPainter
from PyQt6.QtCore import QPointF, QRectF
import numpy as np

class VolumeItem(pg.GraphicsObject):
    def __init__(self, data):
        """
        data: list of (t, open, close, volume)
        """
        pg.GraphicsObject.__init__(self)
        self.data = data 
        self.picture = QPicture()
        self.generatePicture()

    def generatePicture(self):
        p = QPainter(self.picture)
        w = 0.4
        
        # Pre-create pens and brushes to avoid overhead in loop
        green_brush = pg.mkBrush('#00ff0050')
        red_brush = pg.mkBrush('#ff000050')
        
        p.setPen(pg.mkPen(None)) # No border for volume bars
        
        for (t, open, close, vol) in self.data:
            if close >= open:
                p.setBrush(green_brush)
            else:
                p.setBrush(red_brush)
            
            # Draw bar from 0 to vol
            p.drawRect(QRectF(t - w, 0, w * 2, vol))
        p.end()

    def paint(self, p, *args):
        self.picture.play(p)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())
