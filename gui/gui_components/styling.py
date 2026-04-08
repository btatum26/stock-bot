from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from ..colors import (
    ACCENT_BLUE, BG_DARK, BG_INPUT, BG_MAIN,
    TEXT_PRIMARY, TEXT_WHITE,
)


def setup_app_style(window):
    app = QApplication.instance()
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window,      QColor(BG_MAIN))
    palette.setColor(palette.ColorRole.WindowText,  QColor(TEXT_PRIMARY))
    palette.setColor(palette.ColorRole.Base,        QColor(BG_INPUT))
    palette.setColor(palette.ColorRole.Text,        QColor(TEXT_PRIMARY))
    palette.setColor(palette.ColorRole.Button,      QColor(BG_MAIN))
    palette.setColor(palette.ColorRole.ButtonText,  QColor(TEXT_PRIMARY))
    app.setPalette(palette)

    window.setStyleSheet(f"""
        QMainWindow {{ background-color: {BG_MAIN}; }}
        QWidget {{ color: {TEXT_PRIMARY}; font-family: "Segoe UI", Arial; font-size: 10pt; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
            background-color: {BG_INPUT};
            border: 1px solid #3d3d3d;
            padding: 4px;
            border-radius: 4px;
            color: {TEXT_WHITE};
        }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{
            background-color: {BG_INPUT}; color: {TEXT_WHITE};
            selection-background-color: {ACCENT_BLUE};
        }}
        QPushButton {{
            background-color: {ACCENT_BLUE}; border: none;
            padding: 6px 12px; border-radius: 4px;
            color: {TEXT_WHITE}; font-weight: bold;
        }}
        QDockWidget {{ border: 1px solid #333; }}
        QDockWidget::title {{ background: {BG_DARK}; padding-left: 5px; }}
        QGroupBox {{ border: 1px solid #444; margin-top: 6px; font-weight: bold; }}
        QSplitter::handle:vertical {{
            height: 4px;
            background-color: #333333;
            border-top: 1px solid #444444;
            border-bottom: 1px solid #444444;
        }}
        QSplitter::handle:vertical:hover {{ background-color: #555555; }}
    """)
