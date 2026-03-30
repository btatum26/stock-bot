from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QFrame, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal
from .signals_panel import SignalsPanel
from .training_panel import TrainingPanel

class ModelsPanel(QWidget):
    parameters_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        main_layout.addWidget(self.scroll_area)
        
        # Container for all panels
        self.container = QWidget()
        self.content_layout = QVBoxLayout(self.container)
        self.scroll_area.setWidget(self.container)

        # 1. Signal Instance Browser
        self.signals_panel = SignalsPanel(self)
        self.content_layout.addWidget(self.signals_panel)

        # 2. Strategy Parameters (Now integrated)
        self.params_group = QGroupBox("Signal Parameters")
        self.params_layout = QFormLayout(self.params_group)
        self.content_layout.addWidget(self.params_group)
        self.widgets = {}

        # 3. Training Group
        self.training_group = QGroupBox("Training")
        self.training_layout = QVBoxLayout(self.training_group)
        self.training_panel = TrainingPanel(self)
        self.training_layout.addWidget(self.training_panel)
        self.content_layout.addWidget(self.training_group)

        self.content_layout.addStretch()

    def set_parameters(self, parameters: dict):
        """Dynamically builds the Parameters UI within the ModelsPanel."""
        # Clear existing
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child:
                w = child.widget()
                if w:
                    w.deleteLater()
        
        self.widgets = {}
        
        for key, val in parameters.items():
            if isinstance(val, bool):
                inp = QCheckBox()
                inp.setChecked(val)
                inp.stateChanged.connect(self.parameters_changed.emit)
            elif isinstance(val, int):
                inp = QSpinBox()
                inp.setRange(0, 10000)
                inp.setValue(val)
                inp.valueChanged.connect(self.parameters_changed.emit)
            elif isinstance(val, float):
                inp = QDoubleSpinBox()
                inp.setRange(0.0, 1000.0)
                inp.setDecimals(3)
                inp.setValue(val)
                inp.valueChanged.connect(self.parameters_changed.emit)
            else:
                inp = QLineEdit(str(val))
                inp.setMaximumHeight(30)
                inp.textChanged.connect(self.parameters_changed.emit)
            
            self.widgets[key] = inp
            self.params_layout.addRow(f"{key}:", inp)

    def get_values(self) -> dict:
        """Returns the current parameter values from the UI."""
        values = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif hasattr(widget, 'text'):
                values[key] = widget.text()
        return values
