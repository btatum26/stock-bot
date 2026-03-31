from collections import defaultdict
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit, QCheckBox, QScrollArea, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItem, QColor, QFont

class FeaturePanel(QWidget):
    def __init__(self, available_features, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 1. Controls (Add Feature)
        self.feat_combo = QComboBox()
        self.btn_add_feat = QPushButton("Add Feature")

        # Group features by category, insert styled separator headers
        by_category = defaultdict(list)
        for f in available_features.values():
            by_category[f.category].append(f)

        model = self.feat_combo.model()
        for category in sorted(by_category):
            # --- Category header (non-selectable) ---
            header = QStandardItem(f"  {category}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setForeground(QColor("#888888"))
            font = QFont()
            font.setBold(True)
            font.setItalic(True)
            header.setFont(font)
            header.setBackground(QColor("#2a2a2a"))
            model.appendRow(header)

            for f in sorted(by_category[category], key=lambda x: x.name):
                item = QStandardItem(f"    {f.name}")
                item.setData(f.name)
                model.appendRow(item)
        
        main_layout.addWidget(QLabel("Add Feature:"))
        main_layout.addWidget(self.feat_combo)
        main_layout.addWidget(self.btn_add_feat)
        main_layout.addSpacing(10)
        main_layout.addWidget(QLabel("Active Features:"))

        # 2. Scrollable Area for Active Features
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.scroll_content = QWidget()
        self.active_layout = QVBoxLayout(self.scroll_content)
        self.active_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.active_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

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
