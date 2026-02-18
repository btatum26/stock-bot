from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit, QCheckBox, QScrollArea, QFrame, QListWidget, QAbstractItemView, QMessageBox)
from PyQt6.QtCore import Qt
from ..signals.ml_models import MLSignalModel
from ..signals.rule_based import DivergenceSignalModel

class SignalDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Signal Models", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(350)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 1. Controls (Add/Save/Load)
        self.type_combo = QComboBox()
        self.type_combo.addItem("ML Model", "MLSignalModel")
        self.type_combo.addItem("RSI Divergence", "DivergenceSignalModel")
        
        self.btn_add_model = QPushButton("Add Signal Model")
        self.btn_save_set = QPushButton("Save Set")
        self.btn_load_set = QPushButton("Load Set")
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btn_save_set)
        h_layout.addWidget(self.btn_load_set)

        main_layout.addWidget(QLabel("Add Signal Model:"))
        main_layout.addWidget(self.type_combo)
        main_layout.addWidget(self.btn_add_model)
        main_layout.addLayout(h_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(QLabel("Active Signal Models:"))

        # 2. Scrollable Area for Active Models
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.scroll_content = QWidget()
        self.active_layout = QVBoxLayout(self.scroll_content)
        self.active_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.active_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scroll.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll)
        
        self.setWidget(main_widget)
        self.active_data_widgets = {} # {model_name: list_widget or combo}

    def refresh_features(self, active_data_keys):
        """
        Updates the available features in all active model widgets.
        """
        for model_name, widget in self.active_data_widgets.items():
            if isinstance(widget, QListWidget):
                # Save selection
                selected = [item.text() for item in widget.selectedItems()]
                widget.blockSignals(True)
                widget.clear()
                widget.addItems(active_data_keys)
                # Restore selection
                for i in range(widget.count()):
                    if widget.item(i).text() in selected:
                        widget.item(i).setSelected(True)
                widget.blockSignals(False)
            elif isinstance(widget, QComboBox):
                current = widget.currentText()
                widget.blockSignals(True)
                widget.clear()
                widget.addItems(active_data_keys)
                widget.setCurrentText(current)
                widget.blockSignals(False)

    def create_model_widget(self, model, active_data_keys, on_update, on_remove, on_train=None):
        group = QGroupBox(model.name)
        form = QFormLayout(group)
        
        input_widgets = {}
        
        # Specific fields based on model type
        if isinstance(model, MLSignalModel):
            # Model Type
            m_type = QComboBox()
            m_type.addItems(["RandomForest"])
            m_type.setCurrentText(model.model_type)
            m_type.currentTextChanged.connect(on_update)
            form.addRow("Model Type", m_type)
            input_widgets["model_type"] = m_type
            
            # Feature Selection (Multi-select list)
            feat_list = QListWidget()
            feat_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            feat_list.setMaximumHeight(100)
            feat_list.addItems(active_data_keys)
            for i in range(feat_list.count()):
                if feat_list.item(i).text() in model.features_to_use:
                    feat_list.item(i).setSelected(True)
            feat_list.itemSelectionChanged.connect(on_update)
            form.addRow("Input Features", feat_list)
            input_widgets["features_to_use"] = feat_list
            self.active_data_widgets[model.name] = feat_list
            
            # Target Window
            target_w = QLineEdit(str(model.target_window))
            target_w.editingFinished.connect(on_update)
            form.addRow("Target Window (bars)", target_w)
            input_widgets["target_window"] = target_w
            
            # Target Threshold
            target_t = QLineEdit(str(model.target_threshold))
            target_t.editingFinished.connect(on_update)
            form.addRow("Target Threshold (0.01 = 1%)", target_t)
            input_widgets["target_threshold"] = target_t
            
            # Train Button
            if on_train:
                btn_train = QPushButton(f"Train Model {'(Trained)' if model.trained else ''}")
                btn_train.clicked.connect(lambda: on_train(model, group))
                form.addRow(btn_train)
                input_widgets["btn_train"] = btn_train

        elif isinstance(model, DivergenceSignalModel):
            # Indicator
            ind = QComboBox()
            # Find RSI features in active_data_keys
            rsi_feats = [f for f in active_data_keys if "RSI" in f]
            ind.addItems(rsi_feats if rsi_feats else active_data_keys)
            ind.setCurrentText(model.indicator)
            ind.currentTextChanged.connect(on_update)
            form.addRow("Indicator", ind)
            input_widgets["indicator"] = ind
            self.active_data_widgets[model.name] = ind
            
            # Lookback
            lb = QLineEdit(str(model.lookback))
            lb.editingFinished.connect(on_update)
            form.addRow("Lookback", lb)
            input_widgets["lookback"] = lb
            
            # Order
            order = QLineEdit(str(model.order))
            order.editingFinished.connect(on_update)
            form.addRow("Order", order)
            input_widgets["order"] = order

        # Common: Remove Button
        btn_rem = QPushButton("Remove")
        btn_rem.setStyleSheet("background-color: #cc3300; color: white; font-weight: bold;")
        btn_rem.clicked.connect(lambda: on_remove(model.name, group))
        form.addRow(btn_rem)
        
        self.active_layout.addWidget(group)
        return input_widgets, group
