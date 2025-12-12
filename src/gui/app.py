"""Application entry point for the GUI.

Initializes Qt application and launches main window.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

if TYPE_CHECKING:
    from src.models.base import BaseModel


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def create_model(
    variant: str = "mobilenetv3",
    fp16: bool = False,
) -> "BaseModel":
    """Create background removal model.

    Args:
        variant: Model variant (mobilenetv3 or resnet50).
        fp16: Enable FP16 inference.

    Returns:
        RVMModel instance.
    """
    from src.models.rvm import RVMConfig, RVMModel

    config = RVMConfig(
        variant=variant,
        fp16=fp16,
    )

    return RVMModel(config)


def run_app(
    model: "BaseModel | None" = None,
    create_default_model: bool = True,
    log_level: int = logging.INFO,
) -> int:
    """Run the GUI application.

    Args:
        model: Optional pre-configured model.
        create_default_model: Create default model if none provided.
        log_level: Logging level.

    Returns:
        Application exit code.
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Background Remove application")

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Background Remove")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("BackgroundRemove")

    # Set application style
    app.setStyle("Fusion")

    # Apply dark theme
    _apply_dark_theme(app)

    # Create model if needed
    if model is None and create_default_model:
        model = create_model()

    # Create and show main window
    from src.gui.main_window import MainWindow

    window = MainWindow()
    if model is not None:
        window.set_model(model)
    window.show()

    logger.info("Application started")

    return app.exec()


def _apply_dark_theme(app: QApplication) -> None:
    """Apply dark theme to application.

    Args:
        app: QApplication instance.
    """
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()

    # Base colors
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))

    app.setPalette(palette)

    # Additional stylesheet
    app.setStyleSheet("""
        QToolTip {
            color: #ffffff;
            background-color: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
        }

        QGroupBox {
            font-weight: bold;
            border: 1px solid #444;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: #aaa;
        }

        QLineEdit {
            padding: 6px;
            border: 1px solid #444;
            border-radius: 4px;
            background-color: #2a2a2a;
        }

        QLineEdit:focus {
            border-color: #0078d4;
        }

        QPushButton {
            padding: 6px 16px;
            border: 1px solid #444;
            border-radius: 4px;
            background-color: #3a3a3a;
        }

        QPushButton:hover {
            background-color: #4a4a4a;
            border-color: #555;
        }

        QPushButton:pressed {
            background-color: #2a2a2a;
        }

        QPushButton:disabled {
            color: #666;
            background-color: #2a2a2a;
        }

        QComboBox {
            padding: 6px;
            border: 1px solid #444;
            border-radius: 4px;
            background-color: #2a2a2a;
        }

        QComboBox:hover {
            border-color: #555;
        }

        QComboBox::drop-down {
            border: none;
            width: 24px;
        }

        QComboBox QAbstractItemView {
            background-color: #2a2a2a;
            border: 1px solid #444;
            selection-background-color: #0078d4;
        }

        QProgressBar {
            border: 1px solid #444;
            border-radius: 4px;
            text-align: center;
            background-color: #2a2a2a;
        }

        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: #2a2a2a;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            width: 16px;
            height: 16px;
            margin: -5px 0;
            background: #0078d4;
            border-radius: 8px;
        }

        QSlider::handle:horizontal:hover {
            background: #106ebe;
        }

        QSpinBox {
            padding: 4px;
            border: 1px solid #444;
            border-radius: 4px;
            background-color: #2a2a2a;
        }

        QScrollBar:vertical {
            background: #2a2a2a;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background: #555;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background: #666;
        }
    """)


def main() -> int:
    """Main entry point.

    Returns:
        Application exit code.
    """
    return run_app()


if __name__ == "__main__":
    sys.exit(main())
