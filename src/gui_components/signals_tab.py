from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt
from ..strategy import Strategy

class SignalsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Top Config Area
        config_group = QGroupBox("Strategy Configuration")
        config_layout = QHBoxLayout(config_group)
        
        self.strategy_combo = QComboBox()
        self.btn_refresh = QPushButton("Refresh Strategies")
        self.btn_run_backtest = QPushButton("Run Backtest")
        self.btn_run_backtest.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; min-width: 120px;")
        
        config_layout.addWidget(QLabel("Strategy:"))
        config_layout.addWidget(self.strategy_combo)
        config_layout.addWidget(self.btn_refresh)
        config_layout.addStretch()
        config_layout.addWidget(self.btn_run_backtest)
        
        self.layout.addWidget(config_group)
        
        # Results Summary
        summary_group = QGroupBox("Backtest Results")
        summary_layout = QHBoxLayout(summary_group)
        
        self.lbl_total = QLabel("Total Signals: 0")
        self.lbl_win_rate = QLabel("Win Rate: 0.0%")
        self.lbl_correct = QLabel("Correct: 0")
        self.lbl_incorrect = QLabel("Incorrect: 0")
        
        for lbl in [self.lbl_total, self.lbl_win_rate, self.lbl_correct, self.lbl_incorrect]:
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; margin-right: 20px;")
            summary_layout.addWidget(lbl)
        
        summary_layout.addStretch()
        self.layout.addWidget(summary_group)
        
        # Trade Log Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Model", "Side", "Entry", "Max Fwd PnL", "Success"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("QTableWidget { background-color: #1e1e1e; color: #ddd; }")
        
        self.layout.addWidget(QLabel("Detailed Signal Log:"))
        self.layout.addWidget(self.table)
        
        self.btn_refresh.clicked.connect(self.refresh_strategies)
        self.refresh_strategies()

    def refresh_strategies(self):
        s_sets = Strategy.list_available()
        curr_s = self.strategy_combo.currentText()
        
        self.strategy_combo.clear()
        self.strategy_combo.addItems(s_sets)
        if curr_s in s_sets: self.strategy_combo.setCurrentText(curr_s)

    def update_results(self, evaluation):
        self.lbl_total.setText(f"Total Signals: {evaluation.total_signals}")
        win_rate = (evaluation.correct_calls / evaluation.total_signals * 100) if evaluation.total_signals > 0 else 0
        self.lbl_win_rate.setText(f"Win Rate: {win_rate:.1f}%")
        self.lbl_win_rate.setStyleSheet(f"color: {'#00ff00' if win_rate >= 50 else '#ff4444'}; font-size: 14px; font-weight: bold;")
        self.lbl_correct.setText(f"Correct: {evaluation.correct_calls}")
        self.lbl_incorrect.setText(f"Incorrect: {evaluation.incorrect_calls}")
        
        self.table.setRowCount(0)
        for res in reversed(evaluation.results): # Show newest first
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            self.table.setItem(row, 0, QTableWidgetItem(str(res["timestamp"])))
            self.table.setItem(row, 1, QTableWidgetItem(res["model"]))
            self.table.setItem(row, 2, QTableWidgetItem(res["side"].upper()))
            self.table.setItem(row, 3, QTableWidgetItem(f"{res['entry']:.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{res['max_fwd_pnl']*100:.2f}%"))
            
            success_item = QTableWidgetItem("YES" if res["success"] else "NO")
            success_item.setForeground(Qt.GlobalColor.green if res["success"] else Qt.GlobalColor.red)
            self.table.setItem(row, 5, success_item)
