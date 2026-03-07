from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFrame)

class ControlBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet("background-color: #121212; border-bottom: 1px solid #333;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # --- Section 1: Ticker & Data ---
        layout.addWidget(QLabel("Ticker:"))
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setFixedWidth(80)
        layout.addWidget(self.ticker_input)
        
        self.ticker_history = QComboBox()
        self.ticker_history.setFixedWidth(100)
        self.ticker_history.addItem("History...")
        layout.addWidget(self.ticker_history)
        
        self.btn_random = QPushButton("Random")
        self.btn_random.setStyleSheet("background-color: #444; margin-left: 5px;")
        layout.addWidget(self.btn_random)
        
        layout.addSpacing(15)
        layout.addWidget(QLabel("Interval:"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems([ "1w", "1d", "4h", "1h", "15m"])
        layout.addWidget(self.interval_combo)
        
        self.btn_load = QPushButton("Load Data")
        layout.addWidget(self.btn_load)
        
        # --- Separator ---
        v_line = QFrame()
        v_line.setFrameShape(QFrame.Shape.VLine)
        v_line.setFrameShadow(QFrame.Shadow.Sunken)
        v_line.setStyleSheet("background-color: #444; margin: 5px 15px;")
        layout.addWidget(v_line)
        
        # --- Section 2: Strategy Management ---
        layout.addWidget(QLabel("Strategy:"))
        self.lbl_strategy_name = QLabel("Default")
        self.lbl_strategy_name.setStyleSheet("font-weight: bold; color: #aaff00; font-size: 14px;")
        layout.addWidget(self.lbl_strategy_name)
        
        self.btn_rename_strategy = QPushButton("Rename")
        self.btn_load_strategy = QPushButton("Load Strat")
        self.btn_save_strategy = QPushButton("Save As")
        
        for b in [self.btn_rename_strategy, self.btn_load_strategy, self.btn_save_strategy]:
            b.setFixedWidth(80)
            layout.addWidget(b)
            
        layout.addStretch()

    def add_to_history(self, ticker):
        if self.ticker_history.findText(ticker) == -1:
            self.ticker_history.addItem(ticker)
