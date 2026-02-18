from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit, QCheckBox)
from PyQt6.QtCore import Qt

class FeatureDock(QDockWidget):
    def __init__(self, available_features, parent=None):
        super().__init__("Features", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        
        dock_content = QWidget()
        self.main_layout = QVBoxLayout(dock_content)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.feat_combo = QComboBox()
        sorted_feats = sorted(available_features.values(), key=lambda f: (f.category, f.name))
        for f in sorted_feats:
            self.feat_combo.addItem(f"{f.category}: {f.name}", userData=f.name)
            
        self.btn_add_feat = QPushButton("Add Feature")
        
        self.main_layout.addWidget(QLabel("Add Feature:"))
        self.main_layout.addWidget(self.feat_combo)
        self.main_layout.addWidget(self.btn_add_feat)
        self.main_layout.addSpacing(20)
        self.main_layout.addWidget(QLabel("Active Features:"))
        
        self.active_container = QVBoxLayout()
        self.main_layout.addLayout(self.active_container)
        
        self.setWidget(dock_content)

    def create_feature_widget(self, feat_name, parameters, on_update, on_remove):
        group = QGroupBox(feat_name)
        form = QFormLayout(group)
        
        input_widgets = {}
        for key, val in parameters.items():
            if isinstance(val, list):
                inp = QComboBox()
                inp.addItems([str(x) for x in val])
                inp.currentTextChanged.connect(on_update)
            elif isinstance(val, bool):
                inp = QCheckBox()
                inp.setChecked(val)
                inp.stateChanged.connect(on_update)
            else:
                inp = QLineEdit(str(val))
                inp.editingFinished.connect(on_update)
            
            input_widgets[key] = inp
            form.addRow(key, inp)

        btn_rem = QPushButton("Remove")
        btn_rem.setStyleSheet("background-color: #cc3300;")
        btn_rem.clicked.connect(lambda: on_remove(feat_name, group))
        form.addRow(btn_rem)
        
        self.active_container.addWidget(group)
        return input_widgets
