"""GUI module for Background Remove application.

Provides PySide6-based graphical user interface with:
- MainWindow: Main application window
- PreviewWidget: Video frame preview
- ProcessingWorker: Background processing thread
"""

from src.gui.main_window import MainWindow
from src.gui.preview import PreviewMode, PreviewWidget, ThumbnailWidget
from src.gui.worker import (
    ModelLoaderWorker,
    PreviewWorker,
    ProcessingResult,
    ProcessingSignals,
    ProcessingWorker,
)

__all__ = [
    "MainWindow",
    "PreviewWidget",
    "PreviewMode",
    "ThumbnailWidget",
    "ProcessingWorker",
    "ProcessingSignals",
    "ProcessingResult",
    "ModelLoaderWorker",
    "PreviewWorker",
]
