"""Preview widget for displaying video frames.

Provides QLabel-based preview with QPixmap rendering
and alpha channel visualization support.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from numpy import ndarray


class PreviewMode(Enum):
    """Preview display modes."""

    COMPOSITE = "Composite (RGBA)"
    FOREGROUND = "Foreground Only"
    ALPHA = "Alpha Matte"
    SIDE_BY_SIDE = "Side by Side"
    CHECKERBOARD = "Checkerboard BG"


class PreviewWidget(QWidget):
    """Widget for previewing processed video frames.

    Displays foreground and alpha matte with various visualization modes.
    """

    mode_changed = Signal(PreviewMode)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize preview widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._foreground: ndarray | None = None
        self._alpha: ndarray | None = None
        self._mode = PreviewMode.COMPOSITE
        self._checkerboard: ndarray | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Preview Mode:")
        self._mode_combo = QComboBox()
        for mode in PreviewMode:
            self._mode_combo.addItem(mode.value, mode)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self._mode_combo)
        mode_layout.addStretch()

        layout.addLayout(mode_layout)

        # Preview frame
        self._preview_frame = QFrame()
        self._preview_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._preview_frame.setMinimumSize(640, 360)
        self._preview_frame.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        frame_layout = QVBoxLayout(self._preview_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        # Preview label
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumSize(320, 180)
        self._preview_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self._preview_label.setStyleSheet("background-color: #1a1a1a;")

        frame_layout.addWidget(self._preview_label)
        layout.addWidget(self._preview_frame)

        # Info label
        self._info_label = QLabel("No preview available")
        self._info_label.setAlignment(Qt.AlignCenter)
        self._info_label.setStyleSheet("color: #888;")
        layout.addWidget(self._info_label)

    def _on_mode_changed(self, index: int) -> None:
        """Handle mode selection change."""
        self._mode = self._mode_combo.itemData(index)
        self.mode_changed.emit(self._mode)
        self._update_preview()

    def _create_checkerboard(self, height: int, width: int, size: int = 16) -> ndarray:
        """Create checkerboard pattern for transparency visualization.

        Args:
            height: Image height.
            width: Image width.
            size: Checker square size.

        Returns:
            Checkerboard pattern as numpy array.
        """
        if (
            self._checkerboard is not None and
            self._checkerboard.shape[0] == height and
            self._checkerboard.shape[1] == width
        ):
            return self._checkerboard

        # Create checkerboard
        rows = (height + size - 1) // size
        cols = (width + size - 1) // size

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        pattern[::2, ::2] = 1
        pattern[1::2, 1::2] = 1

        # Scale up
        board = np.kron(pattern, np.ones((size, size), dtype=np.uint8))
        board = board[:height, :width]

        # Convert to RGB
        light = np.array([200, 200, 200], dtype=np.uint8)
        dark = np.array([150, 150, 150], dtype=np.uint8)

        checkerboard = np.where(
            board[:, :, np.newaxis] == 1,
            light,
            dark
        ).astype(np.uint8)

        self._checkerboard = checkerboard
        return checkerboard

    def _render_composite(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Render composite with alpha on checkerboard background.

        Args:
            foreground: RGB foreground.
            alpha: Alpha matte.

        Returns:
            Composited image.
        """
        h, w = foreground.shape[:2]
        checkerboard = self._create_checkerboard(h, w)

        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        composite = (
            foreground.astype(np.float32) * alpha_norm +
            checkerboard.astype(np.float32) * (1 - alpha_norm)
        ).astype(np.uint8)

        return composite

    def _render_side_by_side(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Render foreground and alpha side by side.

        Args:
            foreground: RGB foreground.
            alpha: Alpha matte.

        Returns:
            Side-by-side image.
        """
        h, w = foreground.shape[:2]

        # Convert alpha to RGB for display
        alpha_rgb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)

        # Concatenate horizontally
        combined = np.hstack([foreground, alpha_rgb])

        return combined

    def _numpy_to_qpixmap(self, image: ndarray) -> QPixmap:
        """Convert numpy array to QPixmap.

        Args:
            image: RGB image as numpy array.

        Returns:
            QPixmap for display.
        """
        h, w = image.shape[:2]

        if len(image.shape) == 2:
            # Grayscale
            bytes_per_line = w
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format_Grayscale8
            )
        elif image.shape[2] == 3:
            # RGB
            bytes_per_line = 3 * w
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
        elif image.shape[2] == 4:
            # RGBA
            bytes_per_line = 4 * w
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format_RGBA8888
            )
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return QPixmap.fromImage(q_image)

    def _update_preview(self) -> None:
        """Update the preview display based on current mode."""
        if self._foreground is None or self._alpha is None:
            return

        foreground = self._foreground
        alpha = self._alpha

        # Render based on mode
        if self._mode == PreviewMode.FOREGROUND:
            display_image = foreground
        elif self._mode == PreviewMode.ALPHA:
            display_image = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        elif self._mode == PreviewMode.SIDE_BY_SIDE:
            display_image = self._render_side_by_side(foreground, alpha)
        elif self._mode == PreviewMode.CHECKERBOARD:
            display_image = self._render_composite(foreground, alpha)
        else:  # COMPOSITE
            display_image = self._render_composite(foreground, alpha)

        # Convert to QPixmap and scale to fit
        pixmap = self._numpy_to_qpixmap(display_image)

        # Scale to fit label while maintaining aspect ratio
        label_size = self._preview_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self._preview_label.setPixmap(scaled_pixmap)

        # Update info
        h, w = foreground.shape[:2]
        self._info_label.setText(f"Resolution: {w}x{h}")

    def set_frame(self, foreground: ndarray, alpha: ndarray) -> None:
        """Set the frame to display.

        Args:
            foreground: RGB foreground image.
            alpha: Alpha matte.
        """
        self._foreground = foreground.copy()
        self._alpha = alpha.copy()
        self._update_preview()

    def clear(self) -> None:
        """Clear the preview."""
        self._foreground = None
        self._alpha = None
        self._preview_label.clear()
        self._preview_label.setStyleSheet("background-color: #1a1a1a;")
        self._info_label.setText("No preview available")

    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        self._update_preview()


class ThumbnailWidget(QLabel):
    """Simple thumbnail widget for video preview."""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize thumbnail widget."""
        super().__init__(parent)

        self.setFixedSize(160, 90)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #2a2a2a; border: 1px solid #444; border-radius: 4px;"
        )
        self.setCursor(Qt.PointingHandCursor)

    def set_image(self, image: ndarray) -> None:
        """Set thumbnail image.

        Args:
            image: RGB image as numpy array.
        """
        h, w = image.shape[:2]
        bytes_per_line = 3 * w

        q_image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)

        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def set_video(self, video_path: str) -> bool:
        """Set thumbnail from video file.

        Args:
            video_path: Path to video file.

        Returns:
            True if successful.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.set_image(frame_rgb)
        return True

    def mousePressEvent(self, event) -> None:
        """Handle mouse click."""
        self.clicked.emit()
        super().mousePressEvent(event)
