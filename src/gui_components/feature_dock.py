from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit, QCheckBox, QScrollArea, QFrame)
from PyQt6.QtCore import Qt

class FeatureDock(QDockWidget):
    def __init__(self, available_features, parent=None):
        super().__init__("Features", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(300)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 1. Controls (Add/Save/Load)
        self.feat_combo = QComboBox()
        self.btn_add_feat = QPushButton("Add Feature")
        self.btn_save_set = QPushButton("Save Set")
        self.btn_load_set = QPushButton("Load Set")
        
        sorted_feats = sorted(available_features.values(), key=lambda f: (f.category, f.name))
        for f in sorted_feats:
            self.feat_combo.addItem(f"{f.category}: {f.name}", userData=f.name)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btn_save_set)
        h_layout.addWidget(self.btn_load_set)

        main_layout.addWidget(QLabel("Add Feature:"))
        main_layout.addWidget(self.feat_combo)
        main_layout.addWidget(self.btn_add_feat)
        main_layout.addLayout(h_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(QLabel("Active Features:"))

        # 2. Scrollable Area for Active Features
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

    def create_feature_widget(self, feat_name, parameters, on_update, on_remove, initial_values=None):
        group = QGroupBox(feat_name)
        form = QFormLayout(group)
        
        input_widgets = {}
        for key, val in parameters.items():
            initial_val = initial_values.get(key) if initial_values else None
            
            if isinstance(val, list):
                inp = QComboBox()
                inp.addItems([str(x) for x in val])
                if initial_val is not None:
                    inp.setCurrentText(str(initial_val))
                inp.currentTextChanged.connect(on_update)
            elif isinstance(val, bool):
                inp = QCheckBox()
                inp.setChecked(initial_val if initial_val is not None else val)
                inp.stateChanged.connect(on_update)
            else:
                inp = QLineEdit(str(initial_val if initial_val is not None else val))
                inp.editingFinished.connect(on_update)
            
            input_widgets[key] = inp
            form.addRow(key, inp)

        btn_rem = QPushButton("Remove")
        btn_rem.setStyleSheet("background-color: #cc3300; color: white; font-weight: bold;")
        btn_rem.clicked.connect(lambda: on_remove(feat_name, group))
        form.addRow(btn_rem)
        
        self.active_layout.addWidget(group)
        return input_widgets, group
