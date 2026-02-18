from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor

def setup_app_style(window):
    # Palette
    app = QApplication.instance()
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor("#1e1e1e"))
    palette.setColor(palette.ColorRole.WindowText, QColor("#dddddd"))
    palette.setColor(palette.ColorRole.Base, QColor("#2d2d2d"))
    palette.setColor(palette.ColorRole.Text, QColor("#dddddd"))
    palette.setColor(palette.ColorRole.Button, QColor("#1e1e1e"))
    palette.setColor(palette.ColorRole.ButtonText, QColor("#dddddd"))
    app.setPalette(palette)

    # Stylesheet
    window.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; }
        QWidget { color: #dddddd; font-family: "Segoe UI", Arial; font-size: 10pt; }
        QLineEdit, QComboBox { background-color: #2d2d2d; border: 1px solid #3d3d3d; padding: 4px; border-radius: 4px; color: #fff; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #2d2d2d; color: #fff; selection-background-color: #007acc; }
        QPushButton { background-color: #007acc; border: none; padding: 6px 12px; border-radius: 4px; color: white; font-weight: bold; }
        QDockWidget { border: 1px solid #333; }
        QDockWidget::title { background: #121212; padding-left: 5px; }
        QGroupBox { border: 1px solid #444; margin-top: 6px; font-weight: bold; }
        QSplitter::handle:vertical {
            height: 4px;
            background-color: #333333;
            border-top: 1px solid #444444;
            border-bottom: 1px solid #444444;
        }
        QSplitter::handle:vertical:hover {
            background-color: #555555;
        }
    """)
