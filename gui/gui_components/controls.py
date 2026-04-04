from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit,
                             QComboBox, QPushButton, QFrame)


class ControlBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet("background-color: #121212; border-bottom: 1px solid #333;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # --- Ticker & Data ---
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
        self.interval_combo.addItems(["1w", "1d", "4h", "1h", "15m"])
        layout.addWidget(self.interval_combo)

        layout.addSpacing(15)
        layout.addWidget(QLabel("View:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "All"])
        self.period_combo.setCurrentText("1y")
        layout.addWidget(self.period_combo)

        layout.addStretch()

    def add_to_history(self, ticker: str):
        if self.ticker_history.findText(ticker) == -1:
            self.ticker_history.addItem(ticker)
