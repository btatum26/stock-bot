import pandas as pd
import pyqtgraph as pg

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
