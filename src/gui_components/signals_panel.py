from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt

class SignalsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.btn_generate = QPushButton("Generate Signals")
        self.btn_generate.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; height: 50px;")
        
        layout.addWidget(QLabel("Signal Generation"))
        layout.addWidget(self.btn_generate)
        layout.addStretch()
