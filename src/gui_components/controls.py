from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton)

class ControlBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #121212; border-bottom: 1px solid #333;")
        layout = QHBoxLayout(self)
        
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setFixedWidth(80)
        
        self.ticker_history = QComboBox()
        self.ticker_history.setFixedWidth(100)
        self.ticker_history.addItem("History...")

        self.interval_combo = QComboBox()
        self.interval_combo.addItems([ "1w", "1d", "4h", "1h", "15m"])
        
        self.btn_load = QPushButton("Load Data")
        self.btn_random = QPushButton("Random")
        self.btn_random.setStyleSheet("background-color: #444; margin-left: 5px;")
        
        self.btn_signals = QPushButton("Detect Signals")
        self.btn_signals.setStyleSheet("background-color: #5500aa; color: white; font-weight: bold;")
        
        layout.addWidget(QLabel("Ticker:"))
        layout.addWidget(self.ticker_input)
        layout.addWidget(self.ticker_history)
        layout.addWidget(self.btn_random)
        layout.addSpacing(15)
        layout.addWidget(QLabel("Interval:"))
        layout.addWidget(self.interval_combo)
        layout.addSpacing(15)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_signals)
        layout.addStretch()

    def add_to_history(self, ticker):
        if self.ticker_history.findText(ticker) == -1:
            self.ticker_history.addItem(ticker)
