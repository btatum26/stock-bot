from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, 
                             QGroupBox, QComboBox, QFormLayout, QSpinBox, QDoubleSpinBox, 
                             QSlider, QProgressBar, QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal

class TrainingPanel(QWidget):
    train_requested = pyqtSignal(dict) # settings

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Wrap in Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        main_layout.addWidget(scroll)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(container)
        
        # --- Data Scope Group ---
        scope_group = QGroupBox("Data Scope")
        scope_layout = QFormLayout(scope_group)
        
        self.ticker_mode_combo = QComboBox()
        self.ticker_mode_combo.addItems(["Current Ticker", "Custom Basket", "Random Selection"])
        scope_layout.addRow("Ticker Mode:", self.ticker_mode_combo)
        
        self.ticker_input = QTextEdit()
        self.ticker_input.setPlaceholderText("AAPL,MSFT,GOOGL...")
        self.ticker_input.setMaximumHeight(60)
        self.ticker_input.setVisible(False)
        scope_layout.addRow("Ticker List:", self.ticker_input)
        
        self.random_ticker_count = QSpinBox()
        self.random_ticker_count.setRange(1, 100)
        self.random_ticker_count.setValue(5)
        self.random_ticker_count.setVisible(False)
        scope_layout.addRow("Random Count:", self.random_ticker_count)
        
        self.ticker_mode_combo.currentIndexChanged.connect(self._on_ticker_mode_changed)
        
        self.range_mode_combo = QComboBox()
        self.range_mode_combo.addItems(["Full Available", "Random Slices"])
        scope_layout.addRow("Range Mode:", self.range_mode_combo)
        
        self.slice_count_spin = QSpinBox()
        self.slice_count_spin.setRange(1, 100)
        self.slice_count_spin.setValue(10)
        self.slice_count_spin.setVisible(False)
        scope_layout.addRow("Num Slices:", self.slice_count_spin)
        
        self.slice_size_spin = QSpinBox()
        self.slice_size_spin.setRange(50, 10000)
        self.slice_size_spin.setValue(500)
        self.slice_size_spin.setVisible(False)
        scope_layout.addRow("Slice Size (bars):", self.slice_size_spin)
        
        self.range_mode_combo.currentIndexChanged.connect(self._on_range_mode_changed)
        layout.addWidget(scope_group)
        
        # --- Model Group ---
        model_group = QGroupBox("Model Parameters")
        model_layout = QFormLayout(model_group)
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["RandomForest", "XGBoost"])
        model_layout.addRow("Model Type:", self.model_type_combo)
        
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 2000)
        self.n_estimators_spin.setValue(100)
        model_layout.addRow("n_estimators:", self.n_estimators_spin)
        
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 100)
        self.max_depth_spin.setValue(10)
        model_layout.addRow("max_depth:", self.max_depth_spin)
        
        self.val_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.val_split_slider.setRange(5, 50)
        self.val_split_slider.setValue(20)
        self.lbl_val_split = QLabel("Val Split: 20%")
        self.val_split_slider.valueChanged.connect(lambda v: self.lbl_val_split.setText(f"Val Split: {v}%"))
        
        val_layout = QHBoxLayout()
        val_layout.addWidget(self.val_split_slider)
        val_layout.addWidget(self.lbl_val_split)
        model_layout.addRow(val_layout)

        self.target_window_spin = QSpinBox()
        self.target_window_spin.setRange(1, 100)
        self.target_window_spin.setValue(5)
        model_layout.addRow("Target Window:", self.target_window_spin)

        self.target_threshold_spin = QDoubleSpinBox()
        self.target_threshold_spin.setRange(0.01, 20.0)
        self.target_threshold_spin.setValue(1.0)
        self.target_threshold_spin.setSuffix("%")
        model_layout.addRow("Threshold:", self.target_threshold_spin)
        
        layout.addWidget(model_group)
        
        self.btn_train = QPushButton("Start Multi-Ticker Training")
        self.btn_train.setStyleSheet("background-color: #444; height: 40px; font-weight: bold;")
        self.btn_train.clicked.connect(self._on_train_clicked)
        layout.addWidget(self.btn_train)
        
        # --- Feedback Group ---
        feedback_group = QGroupBox("Training Status")
        feedback_layout = QVBoxLayout(feedback_group)
        
        self.progress_bar = QProgressBar()
        feedback_layout.addWidget(self.progress_bar)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(150)
        self.log_console.setStyleSheet("background-color: #000; color: #0f0; font-family: monospace; font-size: 10px;")
        feedback_layout.addWidget(self.log_console)
        
        layout.addWidget(feedback_group)
        layout.addStretch()

    def _on_ticker_mode_changed(self, index):
        self.ticker_input.setVisible(index == 1)
        self.random_ticker_count.setVisible(index == 2)

    def _on_range_mode_changed(self, index):
        self.slice_count_spin.setVisible(index == 1)
        self.slice_size_spin.setVisible(index == 1)

    def _on_train_clicked(self):
        settings = {
            "ticker_mode": self.ticker_mode_combo.currentText(),
            "ticker_list": self.ticker_input.toPlainText().strip(),
            "random_ticker_count": self.random_ticker_count.value(),
            "range_mode": self.range_mode_combo.currentText(),
            "slice_count": self.slice_count_spin.value(),
            "slice_size": self.slice_size_spin.value(),
            "model_type": self.model_type_combo.currentText(),
            "n_estimators": self.n_estimators_spin.value(),
            "max_depth": self.max_depth_spin.value(),
            "validation_split": self.val_split_slider.value() / 100.0,
            "target_window": self.target_window_spin.value(),
            "target_threshold": self.target_threshold_spin.value() / 100.0
        }
        self.train_requested.emit(settings)

    def log(self, message):
        self.log_console.append(message)
        self.log_console.ensureCursorVisible()

from PyQt6.QtWidgets import QFrame # Required for QFrame.Shape.NoFrame
