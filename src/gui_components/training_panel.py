from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, 
                             QGroupBox, QComboBox, QFormLayout, QSpinBox, QDoubleSpinBox, 
                             QSlider, QProgressBar, QTextEdit, QScrollArea, QFrame, QCheckBox)
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
        self.model_group = QGroupBox("Model Parameters")
        self.model_layout = QFormLayout(self.model_group)
        layout.addWidget(self.model_group)
        
        self.param_widgets = {}
        self._setup_default_params()
        
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

    def _setup_default_params(self):
        # Default parameters if the strategy doesn't provide any
        defaults = {
            "n_estimators": 100,
            "max_depth": 10,
            "target_window": 5,
            "target_threshold": 0.01
        }
        self.set_parameters(defaults)

    def set_parameters(self, parameters: dict):
        """Dynamically builds the parameter UI."""
        # Clear existing
        while self.model_layout.count():
            child = self.model_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.param_widgets = {}
        
        for key, val in parameters.items():
            if isinstance(val, bool):
                inp = QCheckBox()
                inp.setChecked(val)
            elif isinstance(val, int):
                inp = QSpinBox()
                inp.setRange(0, 10000)
                inp.setValue(val)
            elif isinstance(val, float):
                inp = QDoubleSpinBox()
                inp.setRange(0.0, 1000.0)
                inp.setDecimals(3)
                inp.setValue(val)
            else:
                inp = QLineEdit(str(val)) # Using QLineEdit for strings instead of QTextEdit
                inp.setMaximumHeight(30)
            
            self.param_widgets[key] = inp
            self.model_layout.addRow(f"{key}:", inp)

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
            "model_type": "XGBoost", # Defaulting to XGBoost
        }
        
        # Add dynamic parameters
        for key, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                settings[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                settings[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                settings[key] = widget.isChecked()
            elif hasattr(widget, 'text'):
                settings[key] = widget.text()
                
        self.train_requested.emit(settings)

    def log(self, message):
        self.log_console.append(message)
        self.log_console.ensureCursorVisible()

from PyQt6.QtWidgets import QFrame, QLineEdit # Required imports
