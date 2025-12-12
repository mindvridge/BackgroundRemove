"""Main application window for Background Remove.

Provides the main GUI interface with file selection, preview,
output options, and progress tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.gui.preview import PreviewWidget, ThumbnailWidget
from src.gui.worker import ModelLoaderWorker, PreviewWorker, ProcessingWorker
from src.pipeline.output import OutputConfig, OutputFormat

if TYPE_CHECKING:
    from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window.

    Provides interface for:
    - Input/output file selection
    - Output format options
    - Preview display
    - Processing progress
    """

    def __init__(
        self,
        model: BaseModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize main window.

        Args:
            model: Background removal model (created if None).
            parent: Parent widget.
        """
        super().__init__(parent)

        self._model = model
        self._input_path: Path | None = None
        self._output_path: Path | None = None
        self._background_path: Path | None = None

        self._processing_worker: ProcessingWorker | None = None
        self._preview_worker: PreviewWorker | None = None
        self._model_loader: ModelLoaderWorker | None = None

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

        # Load model in background if provided
        if self._model is not None and not self._model.is_loaded:
            self._load_model_async()

    def _setup_ui(self) -> None:
        """Set up the main UI components."""
        self.setWindowTitle("Background Remove")
        self.setMinimumSize(1000, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Left panel (controls)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel (preview)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=2)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _create_left_panel(self) -> QWidget:
        """Create left control panel.

        Returns:
            Left panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        # Input file section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        # Input file row
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Select input video...")
        self._input_edit.setReadOnly(True)
        self._input_browse_btn = QPushButton("Browse...")
        self._input_browse_btn.clicked.connect(self._browse_input)

        input_row.addWidget(self._input_edit)
        input_row.addWidget(self._input_browse_btn)
        input_layout.addLayout(input_row)

        # Input thumbnail
        self._input_thumbnail = ThumbnailWidget()
        input_layout.addWidget(self._input_thumbnail, alignment=Qt.AlignCenter)

        layout.addWidget(input_group)

        # Output file section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        # Output file row
        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select output path...")
        self._output_edit.setReadOnly(True)
        self._output_browse_btn = QPushButton("Browse...")
        self._output_browse_btn.clicked.connect(self._browse_output)

        output_row.addWidget(self._output_edit)
        output_row.addWidget(self._output_browse_btn)
        output_layout.addLayout(output_row)

        layout.addWidget(output_group)

        # Output options section
        options_group = QGroupBox("Output Options")
        options_layout = QFormLayout(options_group)

        # Format selector
        self._format_combo = QComboBox()
        format_items = [
            ("MP4 (H.264)", OutputFormat.MP4_H264),
            ("WebM (VP9 Alpha)", OutputFormat.WEBM_VP9),
            ("ProRes 4444", OutputFormat.PRORES_4444),
            ("Green Screen", OutputFormat.GREEN_SCREEN),
            ("Custom Background", OutputFormat.CUSTOM_BG),
        ]
        for text, fmt in format_items:
            self._format_combo.addItem(text, fmt)
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        options_layout.addRow("Format:", self._format_combo)

        # Quality slider
        self._quality_slider = QSlider(Qt.Horizontal)
        self._quality_slider.setRange(0, 51)
        self._quality_slider.setValue(23)
        self._quality_label = QLabel("23")
        self._quality_slider.valueChanged.connect(
            lambda v: self._quality_label.setText(str(v))
        )

        quality_row = QHBoxLayout()
        quality_row.addWidget(self._quality_slider)
        quality_row.addWidget(self._quality_label)
        options_layout.addRow("Quality (CRF):", quality_row)

        # Background file (for custom background)
        bg_row = QHBoxLayout()
        self._bg_edit = QLineEdit()
        self._bg_edit.setPlaceholderText("Background image/video...")
        self._bg_edit.setReadOnly(True)
        self._bg_browse_btn = QPushButton("...")
        self._bg_browse_btn.setMaximumWidth(40)
        self._bg_browse_btn.clicked.connect(self._browse_background)

        bg_row.addWidget(self._bg_edit)
        bg_row.addWidget(self._bg_browse_btn)

        self._bg_label = QLabel("Background:")
        options_layout.addRow(self._bg_label, bg_row)

        # Background blur
        self._blur_spin = QSpinBox()
        self._blur_spin.setRange(0, 50)
        self._blur_spin.setValue(0)
        self._blur_label = QLabel("Blur Amount:")
        options_layout.addRow(self._blur_label, self._blur_spin)

        # Initially hide background options
        self._set_background_options_visible(False)

        layout.addWidget(options_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        self._progress_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self._progress_label)

        # FPS indicator
        self._fps_label = QLabel("")
        self._fps_label.setAlignment(Qt.AlignCenter)
        self._fps_label.setStyleSheet("color: #888;")
        progress_layout.addWidget(self._fps_label)

        layout.addWidget(progress_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._generate_preview)
        self._preview_btn.setEnabled(False)

        self._process_btn = QPushButton("Start Processing")
        self._process_btn.clicked.connect(self._toggle_processing)
        self._process_btn.setEnabled(False)
        self._process_btn.setStyleSheet(
            "QPushButton { background-color: #0078d4; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #106ebe; }"
            "QPushButton:disabled { background-color: #555; }"
        )

        button_layout.addWidget(self._preview_btn)
        button_layout.addWidget(self._process_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right preview panel.

        Returns:
            Right panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Preview widget
        self._preview_widget = PreviewWidget()
        layout.addWidget(self._preview_widget)

        return panel

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._browse_input)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        pass

    def _set_background_options_visible(self, visible: bool) -> None:
        """Show/hide background-related options.

        Args:
            visible: Whether to show background options.
        """
        self._bg_label.setVisible(visible)
        self._bg_edit.setVisible(visible)
        self._bg_browse_btn.setVisible(visible)
        self._blur_label.setVisible(visible)
        self._blur_spin.setVisible(visible)

    @Slot()
    def _on_format_changed(self) -> None:
        """Handle output format change."""
        fmt = self._format_combo.currentData()

        # Show/hide background options for CUSTOM_BG format
        self._set_background_options_visible(fmt == OutputFormat.CUSTOM_BG)

    @Slot()
    def _browse_input(self) -> None:
        """Open file dialog for input video."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )

        if path:
            self._input_path = Path(path)
            self._input_edit.setText(str(path))
            self._input_thumbnail.set_video(path)

            # Auto-generate output path
            output_path = self._input_path.parent / f"{self._input_path.stem}_removed.mp4"
            self._output_path = output_path
            self._output_edit.setText(str(output_path))

            # Enable buttons
            self._preview_btn.setEnabled(True)
            self._process_btn.setEnabled(True)

            self._status_bar.showMessage(f"Loaded: {self._input_path.name}")

    @Slot()
    def _browse_output(self) -> None:
        """Open file dialog for output video."""
        fmt = self._format_combo.currentData()

        # Set filter based on format
        filters = {
            OutputFormat.MP4_H264: "MP4 Video (*.mp4)",
            OutputFormat.WEBM_VP9: "WebM Video (*.webm)",
            OutputFormat.PRORES_4444: "MOV Video (*.mov)",
            OutputFormat.GREEN_SCREEN: "MP4 Video (*.mp4)",
            OutputFormat.CUSTOM_BG: "MP4 Video (*.mp4)",
        }
        file_filter = filters.get(fmt, "Video Files (*.mp4 *.webm *.mov)")

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output Video",
            str(self._output_path) if self._output_path else "",
            file_filter,
        )

        if path:
            self._output_path = Path(path)
            self._output_edit.setText(str(path))

    @Slot()
    def _browse_background(self) -> None:
        """Open file dialog for background image/video."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background",
            "",
            "Media Files (*.jpg *.jpeg *.png *.mp4 *.mov *.webm);;All Files (*)",
        )

        if path:
            self._background_path = Path(path)
            self._bg_edit.setText(str(path))

    def _load_model_async(self) -> None:
        """Load model in background thread."""
        if self._model is None:
            return

        self._status_bar.showMessage("Loading model...")
        self._process_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)

        self._model_loader = ModelLoaderWorker(self._model)
        self._model_loader.finished.connect(self._on_model_loaded)
        self._model_loader.progress.connect(
            lambda msg: self._status_bar.showMessage(msg)
        )
        self._model_loader.start()

    @Slot(bool, str)
    def _on_model_loaded(self, success: bool, message: str) -> None:
        """Handle model loading completion."""
        if success:
            self._status_bar.showMessage("Model loaded - Ready")
            if self._input_path:
                self._preview_btn.setEnabled(True)
                self._process_btn.setEnabled(True)
        else:
            self._status_bar.showMessage(f"Model loading failed: {message}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load model:\n{message}",
            )

    @Slot()
    def _generate_preview(self) -> None:
        """Generate preview for current frame."""
        if self._input_path is None or self._model is None:
            return

        self._status_bar.showMessage("Generating preview...")
        self._preview_btn.setEnabled(False)

        self._preview_worker = PreviewWorker(self._model, self._input_path)
        self._preview_worker.frame_ready.connect(self._on_preview_ready)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    @Slot(object, object)
    def _on_preview_ready(self, foreground, alpha) -> None:
        """Handle preview frame ready."""
        self._preview_widget.set_frame(foreground, alpha)
        self._preview_btn.setEnabled(True)
        self._status_bar.showMessage("Preview generated")

    @Slot(str)
    def _on_preview_error(self, error: str) -> None:
        """Handle preview error."""
        self._preview_btn.setEnabled(True)
        self._status_bar.showMessage(f"Preview error: {error}")

    def _get_output_config(self) -> OutputConfig:
        """Get current output configuration.

        Returns:
            OutputConfig based on UI settings.
        """
        fmt = self._format_combo.currentData()
        crf = self._quality_slider.value()

        config = OutputConfig(
            format=fmt,
            crf=crf,
            background_path=self._background_path,
            background_blur=self._blur_spin.value(),
        )

        return config

    @Slot()
    def _toggle_processing(self) -> None:
        """Start or cancel processing."""
        if self._processing_worker is not None and self._processing_worker.isRunning():
            # Cancel processing
            self._processing_worker.cancel()
            self._process_btn.setText("Cancelling...")
            self._process_btn.setEnabled(False)
        else:
            # Start processing
            self._start_processing()

    def _start_processing(self) -> None:
        """Start video processing."""
        if self._input_path is None or self._output_path is None:
            QMessageBox.warning(
                self,
                "Missing Files",
                "Please select input and output files.",
            )
            return

        if self._model is None or not self._model.is_loaded:
            QMessageBox.warning(
                self,
                "Model Not Ready",
                "Please wait for the model to load.",
            )
            return

        output_config = self._get_output_config()

        # Validate custom background
        if (
            output_config.format == OutputFormat.CUSTOM_BG and
            not self._background_path
        ):
            QMessageBox.warning(
                self,
                "Missing Background",
                "Please select a background image or video.",
            )
            return

        self._progress_bar.setValue(0)
        self._process_btn.setText("Cancel")
        self._preview_btn.setEnabled(False)

        self._processing_worker = ProcessingWorker(
            self._model,
            self._input_path,
            self._output_path,
            output_config,
        )

        # Connect signals
        self._processing_worker.signals.started.connect(self._on_processing_started)
        self._processing_worker.signals.progress.connect(self._on_processing_progress)
        self._processing_worker.signals.frame_ready.connect(self._on_frame_ready)
        self._processing_worker.signals.finished.connect(self._on_processing_finished)
        self._processing_worker.signals.error.connect(self._on_processing_error)
        self._processing_worker.signals.status_changed.connect(
            lambda msg: self._progress_label.setText(msg)
        )

        self._processing_worker.start()

    @Slot()
    def _on_processing_started(self) -> None:
        """Handle processing start."""
        self._status_bar.showMessage("Processing started...")

    @Slot(int, int, float)
    def _on_processing_progress(
        self,
        current: int,
        total: int,
        fps: float,
    ) -> None:
        """Handle processing progress update."""
        if total > 0:
            percent = int((current / total) * 100)
            self._progress_bar.setValue(percent)
            self._progress_label.setText(f"Frame {current} / {total}")
        else:
            self._progress_label.setText(f"Frame {current}")

        self._fps_label.setText(f"{fps:.1f} fps")

    @Slot(object, object)
    def _on_frame_ready(self, foreground, alpha) -> None:
        """Handle preview frame during processing."""
        self._preview_widget.set_frame(foreground, alpha)

    @Slot(object)
    def _on_processing_finished(self, result) -> None:
        """Handle processing completion."""
        self._process_btn.setText("Start Processing")
        self._process_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)

        if result.success:
            self._progress_bar.setValue(100)
            self._status_bar.showMessage(
                f"Complete: {result.frames_processed} frames in "
                f"{result.duration_seconds:.1f}s"
            )
            QMessageBox.information(
                self,
                "Processing Complete",
                f"Video saved to:\n{result.output_path}\n\n"
                f"Frames: {result.frames_processed}\n"
                f"Time: {result.duration_seconds:.1f}s\n"
                f"Speed: {result.frames_processed / result.duration_seconds:.1f} fps",
            )
        else:
            self._status_bar.showMessage(f"Processing failed: {result.error_message}")

    @Slot(str)
    def _on_processing_error(self, error: str) -> None:
        """Handle processing error."""
        self._process_btn.setText("Start Processing")
        self._process_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)

        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred:\n{error}",
        )

    @Slot()
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Background Remove",
            "Background Remove v0.1.0\n\n"
            "A video background removal application using\n"
            "Robust Video Matting (RVM) deep learning model.\n\n"
            "Supports multiple output formats including\n"
            "WebM VP9 (alpha), ProRes 4444, and green screen.",
        )

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Cancel any running processing
        if self._processing_worker is not None and self._processing_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is in progress. Cancel and exit?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if reply == QMessageBox.No:
                event.ignore()
                return

            self._processing_worker.cancel()
            self._processing_worker.wait(5000)

        # Unload model
        if self._model is not None and self._model.is_loaded:
            self._model.unload()

        event.accept()

    def set_model(self, model: BaseModel) -> None:
        """Set the background removal model.

        Args:
            model: Background removal model.
        """
        self._model = model

        if not model.is_loaded:
            self._load_model_async()
        else:
            self._status_bar.showMessage("Model ready")
            if self._input_path:
                self._preview_btn.setEnabled(True)
                self._process_btn.setEnabled(True)
