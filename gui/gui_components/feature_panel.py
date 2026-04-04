from collections import defaultdict
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit, QCheckBox, QScrollArea, QFrame, QColorDialog)
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
                item.setData(f.name, Qt.ItemDataRole.UserRole)
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

    @staticmethod
    def _make_color_button(hex_color: str, on_update) -> QPushButton:
        """Creates a button that shows its color and opens a color picker on click."""
        btn = QPushButton()
        btn.setFixedSize(22, 22)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Click to pick a color")

        def _apply_color(color_hex: str):
            btn.setProperty("color_value", color_hex)
            btn.setStyleSheet(
                f"background-color: {color_hex}; "
                f"border: 1px solid #555; border-radius: 3px;"
            )

        def _open_picker():
            current = btn.property("color_value") or "#ffffff"
            chosen = QColorDialog.getColor(QColor(current), btn, "Pick line color")
            if chosen.isValid():
                _apply_color(chosen.name())
                on_update()

        _apply_color(hex_color)
        btn.clicked.connect(_open_picker)

        # Expose .text() so the rest of gui.py can read it the same way as QLineEdit
        btn.text = lambda: btn.property("color_value") or "#ffffff"
        return btn

    def create_feature_widget(self, feat_name, parameters, on_update, on_remove,
                              initial_values=None, output_names=None, column_options=None):
        """
        column_options: optional dict mapping param_key → list of (label, col_name) tuples.
        When provided for a key, that parameter is rendered as a source-column QComboBox
        whose userData holds the actual column name (empty string = default price data).
        """
        group = QGroupBox(feat_name)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(3)

        input_widgets = {}
        visibility_widgets = {}

        # Split params into color vs. everything else
        color_params = {k: v for k, v in parameters.items()
                        if k == "color" or k.startswith("color_")}
        other_params = {k: v for k, v in parameters.items()
                        if k != "color" and not k.startswith("color_")}

        # Non-color params as a compact form
        if other_params:
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(3)
            for key, val in other_params.items():
                initial_val = (initial_values or {}).get(key)
                if column_options and key in column_options:
                    # Dynamic column selector — userData holds the real column name
                    inp = QComboBox()
                    inp.setProperty("is_source_selector", True)
                    for label, col_name in column_options[key]:
                        inp.addItem(label, userData=col_name)
                    target = initial_val if initial_val is not None else val
                    idx = inp.findData(target)
                    inp.setCurrentIndex(max(0, idx))
                    inp.currentIndexChanged.connect(on_update)
                elif isinstance(val, list):
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
            group_layout.addLayout(form)

        # Output rows: [toggle checkbox] [name label] [color swatch]
        if output_names:
            for oname in output_names:
                color_key = f"color_{oname}" if f"color_{oname}" in color_params else None

                row = QHBoxLayout()
                row.setSpacing(4)
                row.setContentsMargins(0, 0, 0, 0)

                cb = QCheckBox()
                cb.setChecked(True)
                cb.setFixedWidth(18)
                cb.stateChanged.connect(on_update)
                visibility_widgets[oname] = cb
                row.addWidget(cb)

                lbl = QLabel(oname.replace("_", " "))
                row.addWidget(lbl, 1)

                if color_key and color_key in color_params:
                    initial_val = (initial_values or {}).get(color_key)
                    hex_val = str(initial_val if initial_val is not None else color_params[color_key])
                    color_btn = self._make_color_button(hex_val, on_update)
                    input_widgets[color_key] = color_btn
                    row.addWidget(color_btn)

                group_layout.addLayout(row)
        elif color_params:
            # Single-line feature: one row with just the color swatch
            for key, val in color_params.items():
                initial_val = (initial_values or {}).get(key)
                hex_val = str(initial_val if initial_val is not None else val)
                inp = self._make_color_button(hex_val, on_update)
                input_widgets[key] = inp
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(QLabel("color"))
                row.addStretch(1)
                row.addWidget(inp)
                group_layout.addLayout(row)

        btn_rem = QPushButton("Remove")
        btn_rem.setStyleSheet("background-color: #cc3300; color: white; font-weight: bold;")
        btn_rem.clicked.connect(lambda: on_remove(feat_name, group))
        group_layout.addWidget(btn_rem)

        self.active_layout.addWidget(group)
        return input_widgets, visibility_widgets, group

    @staticmethod
    def refresh_column_options(combo: QComboBox, options: list):
        """Repopulate a source-selector QComboBox without changing the current selection.

        options: list of (label, col_name) tuples — same format as column_options values.
        """
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, col_name in options:
            combo.addItem(label, userData=col_name)
        idx = combo.findData(current)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)
