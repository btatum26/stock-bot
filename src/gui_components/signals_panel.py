from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget, 
                             QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QSlider, QProgressBar, QTextEdit, QComboBox, 
                             QFormLayout, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal

class SignalsPanel(QWidget):
    # Signals
    generate_requested = pyqtSignal()
    train_requested = pyqtSignal(dict) # settings
    set_active_requested = pyqtSignal(str) # model_id
    delete_model_requested = pyqtSignal(str) # model_id
    rename_model_requested = pyqtSignal(str) # model_id

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 1. Signal Generation (Quick Action)
        self.btn_generate = QPushButton("Generate Signals")
        self.btn_generate.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; height: 35px;")
        self.btn_generate.clicked.connect(self.generate_requested.emit)
        layout.addWidget(self.btn_generate)
        layout.addSpacing(10)
        
        # 2. Training Configuration
        train_group = QGroupBox("Training Configuration")
        train_layout = QVBoxLayout(train_group)
        
        form = QFormLayout()
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["RandomForest", "XGBoost"])
        form.addRow("Model Type:", self.model_type_combo)
        
        # Hyperparameters (Example for RF)
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 1000)
        self.n_estimators_spin.setValue(100)
        form.addRow("n_estimators:", self.n_estimators_spin)
        
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 50)
        self.max_depth_spin.setValue(10)
        form.addRow("max_depth:", self.max_depth_spin)
        
        # Validation Split
        self.val_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.val_split_slider.setRange(5, 50)
        self.val_split_slider.setValue(20)
        self.lbl_val_split = QLabel("Val Split: 20%")
        self.val_split_slider.valueChanged.connect(lambda v: self.lbl_val_split.setText(f"Val Split: {v}%"))
        
        val_layout = QHBoxLayout()
        val_layout.addWidget(self.val_split_slider)
        val_layout.addWidget(self.lbl_val_split)
        form.addRow(val_layout)

        # Labeling Parameters (Target)
        self.target_window_spin = QSpinBox()
        self.target_window_spin.setRange(1, 100)
        self.target_window_spin.setValue(5)
        form.addRow("Target Window (bars):", self.target_window_spin)

        self.target_threshold_spin = QDoubleSpinBox()
        self.target_threshold_spin.setRange(0.01, 10.0)
        self.target_threshold_spin.setSingleStep(0.1)
        self.target_threshold_spin.setValue(1.0)
        self.target_threshold_spin.setSuffix("%")
        form.addRow("Target Threshold:", self.target_threshold_spin)
        
        train_layout.addLayout(form)
        
        self.btn_train = QPushButton("Start Training")
        self.btn_train.setStyleSheet("background-color: #444; height: 30px;")
        self.btn_train.clicked.connect(self._on_train_clicked)
        train_layout.addWidget(self.btn_train)
        
        layout.addWidget(train_group)
        layout.addSpacing(10)
        
        # 3. Model Instance Browser
        instances_group = QGroupBox("Model Instances")
        instances_layout = QVBoxLayout(instances_group)
        
        self.instance_table = QTableWidget(0, 4)
        self.instance_table.setHorizontalHeaderLabels(["Name/Date", "Acc", "Prec", "Status"])
        self.instance_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.instance_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.instance_table.setMinimumHeight(200)
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
        layout.addSpacing(10)
        
        # 4. Training Feedback
        feedback_group = QGroupBox("Training Feedback")
        feedback_layout = QVBoxLayout(feedback_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        feedback_layout.addWidget(self.progress_bar)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(100)
        self.log_console.setStyleSheet("background-color: #000; color: #0f0; font-family: monospace; font-size: 10px;")
        feedback_layout.addWidget(self.log_console)
        
        layout.addWidget(feedback_group)
        
        layout.addStretch()

    def _on_train_clicked(self):
        settings = {
            "model_type": self.model_type_combo.currentText(),
            "n_estimators": self.n_estimators_spin.value(),
            "max_depth": self.max_depth_spin.value(),
            "validation_split": self.val_split_slider.value() / 100.0,
            "target_window": self.target_window_spin.value(),
            "target_threshold": self.target_threshold_spin.value() / 100.0 # Convert from %
        }
        self.train_requested.emit(settings)

    def _on_set_active_clicked(self):
        row = self.instance_table.currentRow()
        if row >= 0:
            model_id = self.instance_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.set_active_requested.emit(model_id)

    def _on_delete_clicked(self):
        row = self.instance_table.currentRow()
        if row >= 0:
            model_id = self.instance_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.delete_model_requested.emit(model_id)

    def _on_rename_clicked(self):
        row = self.instance_table.currentRow()
        if row >= 0:
            model_id = self.instance_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.rename_model_requested.emit(model_id)

    def log(self, message):
        self.log_console.append(message)
        self.log_console.ensureCursorVisible()

    def refresh_models(self, instances, active_id=None):
        """Updates the table of model instances."""
        self.instance_table.setRowCount(0)
        for model_id, info in sorted(instances.items(), key=lambda x: x[1].get('timestamp', ''), reverse=True):
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
