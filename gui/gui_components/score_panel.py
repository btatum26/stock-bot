from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, 
                             QGroupBox, QComboBox, QSlider, QFrame, QColorDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

class ScorePanel(QWidget):
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_parent = parent
        self.scoring_functions = []
        self.current_function = None
        self.pos_color = "#00ff00"
        self.neg_color = "#ff0000"
        self.alpha = 0.3
        self.active = False
        self.param_widgets = {} # {param_name: widget}
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # 1. Function Selection
        func_group = QGroupBox("Scoring Method")
        func_layout = QVBoxLayout(func_group)
        self.func_combo = QComboBox()
        self.func_combo.currentIndexChanged.connect(self._on_function_changed)
        func_layout.addWidget(self.func_combo)
        layout.addWidget(func_group)

        # 2. Dynamic Parameters
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout(self.params_group)
        layout.addWidget(self.params_group)
        self.params_group.hide() # Hide until parameters are added

        # 3. Appearance Control
        app_group = QGroupBox("Appearance")
        app_layout = QVBoxLayout(app_group)

        # Positive Color
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Positive:"))
        self.pos_btn = QPushButton()
        self.pos_btn.setFixedWidth(40)
        self.pos_btn.setStyleSheet(f"background-color: {self.pos_color}; border: 1px solid #555;")
        self.pos_btn.clicked.connect(lambda: self._pick_color('pos'))
        pos_layout.addWidget(self.pos_btn)
        app_layout.addLayout(pos_layout)

        # Negative Color
        neg_layout = QHBoxLayout()
        neg_layout.addWidget(QLabel("Negative:"))
        self.neg_btn = QPushButton()
        self.neg_btn.setFixedWidth(40)
        self.neg_btn.setStyleSheet(f"background-color: {self.neg_color}; border: 1px solid #555;")
        self.neg_btn.clicked.connect(lambda: self._pick_color('neg'))
        neg_layout.addWidget(self.neg_btn)
        app_layout.addLayout(neg_layout)

        # Alpha Slider
        alpha_layout = QVBoxLayout()
        self.alpha_label = QLabel(f"Transparency: {self.alpha:.1f}")
        alpha_layout.addWidget(self.alpha_label)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 10)
        self.alpha_slider.setValue(int(self.alpha * 10))
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        alpha_layout.addWidget(self.alpha_slider)
        app_layout.addLayout(alpha_layout)

        layout.addWidget(app_group)

        # 3. Actions
        self.btn_apply = QPushButton("Apply Visualization")
        self.btn_apply.setStyleSheet("background-color: #2a5a2a; padding: 8px; font-weight: bold;")
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.btn_apply)

        self.btn_clear = QPushButton("Clear Visualization")
        self.btn_clear.clicked.connect(self._on_clear_requested)
        layout.addWidget(self.btn_clear)

        layout.addStretch()

    def set_available_functions(self, functions):
        """Sets the list of available scoring function names."""
        self.scoring_functions = functions
        self.func_combo.blockSignals(True)
        self.func_combo.clear()
        self.func_combo.addItems(functions)
        self.func_combo.blockSignals(False)

    def set_parameters(self, params_info):
        """
        Sets dynamic parameter widgets.
        params_info: list of (name, default_value, type)
        """
        # Clear existing
        while self.params_layout.count() > 0:
            child = self.params_layout.takeAt(0)
            if child:
                w = child.widget()
                if w:
                    w.deleteLater()
        self.param_widgets = {}

        if not params_info:
            self.params_group.hide()
            return

        for name, default, ptype in params_info:
            if ptype == float:
                widget = QDoubleSpinBox()
                widget.setRange(-1000, 1000)
                widget.setValue(float(default))
                widget.setSingleStep(0.1)
            elif ptype == int:
                widget = QSpinBox()
                widget.setRange(-10000, 10000)
                widget.setValue(int(default))
            elif ptype == bool:
                widget = QCheckBox()
                widget.setChecked(bool(default))
            else:
                continue

            widget.valueChanged.connect(self._on_settings_changed) if not isinstance(widget, QCheckBox) else widget.stateChanged.connect(self._on_settings_changed)
            self.params_layout.addRow(f"{name.replace('_', ' ').title()}:", widget)
            self.param_widgets[name] = widget

        self.params_group.show()

    def get_settings(self):
        """Returns current settings dict."""
        params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()

        return {
            "function": self.func_combo.currentText(),
            "parameters": params,
            "pos_color": self.pos_color,
            "neg_color": self.neg_color,
            "alpha": self.alpha,
            "active": self.active
        }

    def _on_function_changed(self):
        """Called when function selection changes in combo box."""
        # This will be handled by ChartWindow to fetch and set parameters
        self._on_settings_changed()

    def _pick_color(self, target):
        initial = QColor(self.pos_color if target == 'pos' else self.neg_color)
        color = QColorDialog.getColor(initial, self, "Select Color")
        if color.isValid():
            hex_color = color.name()
            if target == 'pos':
                self.pos_color = hex_color
                self.pos_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")
            else:
                self.neg_color = hex_color
                self.neg_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")
            if self.active:
                self._on_settings_changed()

    def _on_alpha_changed(self, value):
        self.alpha = value / 10.0
        self.alpha_label.setText(f"Transparency: {self.alpha:.1f}")
        if self.active:
            self._on_settings_changed()

    def _on_apply_clicked(self):
        """Forces a re-activation/re-calculation."""
        self.active = True
        settings = self.get_settings()
        # Tag it as a forced refresh
        settings["force_refresh"] = True
        self.settings_changed.emit(settings)

    def _on_settings_changed(self):
        """Standard update, only propagates if already active."""
        if self.active:
            settings = self.get_settings()
            self.settings_changed.emit(settings)

    def _on_clear_requested(self):
        self.active = False
        self.settings_changed.emit({"active": False})
