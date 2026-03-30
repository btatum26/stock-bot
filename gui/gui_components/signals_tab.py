import os
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QGroupBox, QLineEdit)
from PyQt6.QtCore import Qt
from ..strategy import Strategy
from .plots import UnifiedPlot, LineOverlay

class SignalsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)

        # Top Config Area
        config_group = QGroupBox("Strategy Signal Logic")
        config_layout = QVBoxLayout(config_group)
        top_controls = QHBoxLayout()
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setFixedWidth(80)
        self.btn_random = QPushButton("Random")
        self.btn_random.setStyleSheet("background-color: #444; margin-right: 10px;")
        
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1w", "1d", "4h", "1h", "15m"])
        
        self.strategy_combo = QComboBox()
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_open_vscode = QPushButton("Open Script in VS Code")
        self.btn_open_vscode.setStyleSheet("background-color: #007acc; color: white; font-weight: bold;")
        
        top_controls.addWidget(QLabel("Ticker:"))
        top_controls.addWidget(self.ticker_input)
        top_controls.addWidget(self.btn_random)
        top_controls.addSpacing(10)
        top_controls.addWidget(QLabel("Interval:"))
        top_controls.addWidget(self.interval_combo)
        top_controls.addSpacing(20)
        top_controls.addWidget(QLabel("Strategy:"))
        top_controls.addWidget(self.strategy_combo)
        top_controls.addWidget(self.btn_refresh)
        top_controls.addStretch()
        top_controls.addWidget(self.btn_open_vscode)
        config_layout.addLayout(top_controls)
        
        # Info Area
        self.script_info = QLabel("Script: None")
        self.script_info.setStyleSheet("color: #aaa; font-style: italic;")
        config_layout.addWidget(self.script_info)
        
        config_layout.addSpacing(10)
        
        # Preview Chart
        self.preview_widget = pg.GraphicsLayoutWidget()
        self.preview_widget.setBackground('#1e1e1e')
        self.preview_plot = UnifiedPlot()
        self.preview_widget.addItem(self.preview_plot)
        
        config_layout.addWidget(self.preview_widget)
        
        self.main_layout.addWidget(config_group)
        
        # Execution Area (Preview)
        self.btn_run_signals = QPushButton("Run & Preview Signals on Chart")
        self.btn_run_signals.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; height: 50px;")
        self.main_layout.addWidget(self.btn_run_signals)

        self.btn_refresh.clicked.connect(self.refresh_strategies)
        self.strategy_combo.currentIndexChanged.connect(self.update_script_info)
        self.btn_open_vscode.clicked.connect(self.open_in_vscode)
        
        self.refresh_strategies()

    def refresh_strategies(self):
        s_sets = Strategy.list_available()
        curr_s = self.strategy_combo.currentText()
        
        self.strategy_combo.clear()
        self.strategy_combo.addItems(s_sets)
        if curr_s in s_sets: self.strategy_combo.setCurrentText(curr_s)
        self.update_script_info()

    def update_script_info(self):
        name = self.strategy_combo.currentText()
        if not name: 
            self.script_info.setText("Script: None")
            return
        
        strat = Strategy(name) # This ensures script exists
        self.script_info.setText(f"Script: {strat.script_path}")

    def open_in_vscode(self):
        name = self.strategy_combo.currentText()
        if not name: return
        
        # Load ensures we have the feature_config etc.
        strat = Strategy.load(name)
        # Save ensures the .strat file is synced with the .py before opening
        strat.save()
        
        abs_path = os.path.abspath(strat.script_path)
        
        import subprocess
        try:
            # Try to open with 'code' command
            subprocess.run(["code", abs_path], shell=True)
        except Exception as e:
            print(f"Error opening VS Code: {e}")
