from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, 
                             QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QScrollArea, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal

class SignalsPanel(QWidget):
    # Signals
    generate_requested = pyqtSignal()
    set_active_requested = pyqtSignal(str) # model_id
    delete_model_requested = pyqtSignal(str) # model_id
    rename_model_requested = pyqtSignal(str) # model_id

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
        
        # 1. Signal Generation (Quick Action)
        self.btn_generate = QPushButton("Generate Signals")
        self.btn_generate.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; height: 35px;")
        self.btn_generate.clicked.connect(self.generate_requested.emit)
        layout.addWidget(self.btn_generate)
        layout.addSpacing(10)
        
        # 2. Model Instance Browser
        instances_group = QGroupBox("Model Instances")
        instances_layout = QVBoxLayout(instances_group)
        
        self.instance_table = QTableWidget(0, 4)
        self.instance_table.setHorizontalHeaderLabels(["Name/Date", "Acc", "Prec", "Status"])
        self.instance_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.instance_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.instance_table.setMinimumHeight(300)
        self.instance_table.setStyleSheet("QTableWidget { background-color: #1e1e1e; gridline-color: #333; }")
        
        instances_layout.addWidget(self.instance_table)
        
        actions_layout = QHBoxLayout()
        self.btn_set_active = QPushButton("Set Active")
        self.btn_delete = QPushButton("Delete")
        self.btn_rename = QPushButton("Rename")
        
        self.btn_set_active.clicked.connect(self._on_set_active_clicked)
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        self.btn_rename.clicked.connect(self._on_rename_clicked)
        
        for b in [self.btn_set_active, self.btn_rename, self.btn_delete]:
            actions_layout.addWidget(b)
        instances_layout.addLayout(actions_layout)
        
        layout.addWidget(instances_group)
        layout.addStretch()

    def _on_set_active_clicked(self):
        row = self.instance_table.currentRow()
        if row >= 0:
            item = self.instance_table.item(row, 0)
            if item:
                model_id = item.data(Qt.ItemDataRole.UserRole)
                self.set_active_requested.emit(model_id)

    def _on_delete_clicked(self):
        row = self.instance_table.currentRow()
        if row >= 0:
            item = self.instance_table.item(row, 0)
            if item:
                model_id = item.data(Qt.ItemDataRole.UserRole)
                self.delete_model_requested.emit(model_id)

    def _on_rename_clicked(self):
        row = self.instance_table.currentRow()
        if row >= 0:
            item = self.instance_table.item(row, 0)
            if item:
                model_id = item.data(Qt.ItemDataRole.UserRole)
                self.rename_model_requested.emit(model_id)

    def refresh_models(self, instances, active_id=None, active_features=None):
        """Updates the table of model instances. Filters by compatibility with active features."""
        self.instance_table.setRowCount(0)
        
        # Determine current available raw features (keys in the dict)
        current_features = set()
        if active_features:
            # We need to know which RAW keys are available. 
            # This is slightly tricky as features can produce multiple series.
            # But the 'training_scope' stores the names of the features (e.g., 'RSI').
            current_features = set(active_features)

        for model_id, info in sorted(instances.items(), key=lambda x: x[1].get('timestamp', ''), reverse=True):
            # Compatibility Check
            if active_features is not None:
                scope = info.get('training_scope', {})
                trained_features = scope.get('features', [])
                
                # Check if all features used during training are currently active
                # Using a set check: is trained_features a subset of current_features?
                is_compatible = all(f in current_features for f in trained_features)
                if not is_compatible:
                    continue

            row = self.instance_table.rowCount()
            self.instance_table.insertRow(row)
            
            comment = info.get('comment') or info.get('timestamp') or "Unnamed"
            name_item = QTableWidgetItem(str(comment))
            name_item.setData(Qt.ItemDataRole.UserRole, model_id)
            
            metrics = info.get('metrics', {})
            acc = QTableWidgetItem(f"{metrics.get('accuracy', 0):.2%}")
            prec = QTableWidgetItem(f"{metrics.get('precision', 0):.2%}")
            
            status_text = "ACTIVE" if model_id == active_id else "Ready"
            status = QTableWidgetItem(status_text)
            if model_id == active_id:
                status.setForeground(Qt.GlobalColor.green)
            
            self.instance_table.setItem(row, 0, name_item)
            self.instance_table.setItem(row, 1, acc)
            self.instance_table.setItem(row, 2, prec)
            self.instance_table.setItem(row, 3, status)
