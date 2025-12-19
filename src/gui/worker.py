"""Worker thread for background video processing.

Provides QThread-based worker for non-blocking video processing
with signal-based progress reporting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal

if TYPE_CHECKING:
    from numpy import ndarray

    from src.models.base import BaseModel
    from src.pipeline.output import OutputConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of video processing."""

    success: bool
    input_path: Path
    output_path: Path
    frames_processed: int
    duration_seconds: float
    error_message: str | None = None


class ProcessingSignals(QObject):
    """Signals for processing worker."""

    # Progress signals
    started = Signal()  # Processing started
    progress = Signal(int, int, float)  # current_frame, total_frames, fps
    frame_ready = Signal(object, object)  # foreground, alpha (for preview)
    finished = Signal(object)  # ProcessingResult
    error = Signal(str)  # Error message

    # Status signals
    status_changed = Signal(str)  # Status message
    model_loaded = Signal()  # Model loading complete


class ProcessingWorker(QThread):
    """Worker thread for video processing.

    Runs video processing in a background thread to keep UI responsive.
    Emits signals for progress updates and preview frames.

    Example:
        >>> worker = ProcessingWorker(model, input_path, output_path, config)
        >>> worker.signals.progress.connect(update_progress)
        >>> worker.signals.frame_ready.connect(update_preview)
        >>> worker.start()
    """

    def __init__(
        self,
        model: BaseModel,
        input_path: str | Path,
        output_path: str | Path,
        output_config: OutputConfig,
        preview_interval: int = 5,
        parent: QObject | None = None,
    ) -> None:
        """Initialize processing worker.

        Args:
            model: Background removal model.
            input_path: Input video path.
            output_path: Output video path.
            output_config: Output configuration.
            preview_interval: Emit preview every N frames.
            parent: Parent QObject.
        """
        super().__init__(parent)

        self.model = model
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_config = output_config
        self.preview_interval = preview_interval

        self.signals = ProcessingSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of processing."""
        self._cancelled = True
        logger.info("Processing cancellation requested")

    def run(self) -> None:
        """Execute video processing in background thread."""
        from src.pipeline.output import OutputFormat, apply_alpha_processing
        from src.pipeline.reader import ReaderConfig, VideoReader
        from src.pipeline.writer import AdvancedVideoWriter

        start_time = time.time()
        frames_processed = 0
        reader = None
        writer = None

        try:
            self.signals.started.emit()
            self.signals.status_changed.emit("Loading model...")

            # Ensure model is loaded
            if not self.model.is_loaded:
                self.model.load()
                self.signals.model_loaded.emit()

            self.model.reset_state()
            self.signals.status_changed.emit("Opening video...")

            # Create reader
            reader_config = ReaderConfig(queue_size=32)
            reader = VideoReader(self.input_path, reader_config)
            reader.start()

            # Wait for metadata
            while reader.metadata is None and reader.is_running:
                time.sleep(0.01)

            if reader.metadata is None:
                raise RuntimeError("Failed to read video metadata")

            metadata = reader.metadata
            total_frames = metadata.frame_count

            self.signals.status_changed.emit(
                f"Processing {metadata.width}x{metadata.height} @ {metadata.fps:.1f}fps"
            )

            # Set audio source for output
            self.output_config.audio_source = self.input_path
            self.output_config.fps = metadata.fps

            # Get alpha processing settings
            alpha_threshold = self.output_config.alpha_threshold
            edge_softness = self.output_config.edge_softness

            # Create writer
            writer = AdvancedVideoWriter(
                self.output_path,
                metadata.width,
                metadata.height,
                self.output_config,
            )
            writer.start()

            # Process frames
            last_progress_time = time.time()

            for frame in reader:
                if self._cancelled:
                    self.signals.status_changed.emit("Cancelled")
                    break

                # Run inference
                foreground, alpha = self.model.inference(frame)
                frames_processed += 1

                # Apply alpha processing (threshold and softness)
                if alpha_threshold > 0 or edge_softness > 0:
                    alpha = apply_alpha_processing(
                        alpha,
                        threshold=alpha_threshold,
                        softness=edge_softness,
                    )

                # Write to output
                writer.write_with_alpha(foreground, alpha)

                # Emit preview frame
                if frames_processed % self.preview_interval == 0:
                    self.signals.frame_ready.emit(foreground.copy(), alpha.copy())

                # Emit progress
                now = time.time()
                if now - last_progress_time >= 0.1:  # Update every 100ms
                    elapsed = now - start_time
                    fps = frames_processed / elapsed if elapsed > 0 else 0
                    self.signals.progress.emit(frames_processed, total_frames, fps)
                    last_progress_time = now

            # Final progress update
            elapsed = time.time() - start_time
            fps = frames_processed / elapsed if elapsed > 0 else 0
            self.signals.progress.emit(frames_processed, total_frames, fps)

            # Create result
            result = ProcessingResult(
                success=not self._cancelled,
                input_path=self.input_path,
                output_path=writer.output_path,
                frames_processed=frames_processed,
                duration_seconds=elapsed,
            )

            if self._cancelled:
                result.error_message = "Processing was cancelled"

            self.signals.status_changed.emit(
                "Complete" if not self._cancelled else "Cancelled"
            )
            self.signals.finished.emit(result)

        except Exception as e:
            logger.exception("Processing error")
            elapsed = time.time() - start_time

            result = ProcessingResult(
                success=False,
                input_path=self.input_path,
                output_path=self.output_path,
                frames_processed=frames_processed,
                duration_seconds=elapsed,
                error_message=str(e),
            )

            self.signals.error.emit(str(e))
            self.signals.finished.emit(result)

        finally:
            if reader is not None:
                reader.stop()
            if writer is not None:
                writer.stop()


class ModelLoaderWorker(QThread):
    """Worker thread for loading model."""

    # Signals
    started = Signal()
    finished = Signal(bool, str)  # success, message
    progress = Signal(str)  # status message

    def __init__(
        self,
        model: BaseModel,
        parent: QObject | None = None,
    ) -> None:
        """Initialize model loader worker.

        Args:
            model: Model to load.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self.model = model

    def run(self) -> None:
        """Load model in background thread."""
        try:
            self.started.emit()
            self.progress.emit("Downloading model from TorchHub...")

            self.model.load()

            self.progress.emit("Model loaded successfully")
            self.finished.emit(True, "Model loaded successfully")

        except Exception as e:
            logger.exception("Model loading error")
            self.finished.emit(False, str(e))


class PreviewWorker(QThread):
    """Worker for generating preview frames."""

    frame_ready = Signal(object, object)  # foreground, alpha
    error = Signal(str)

    def __init__(
        self,
        model: BaseModel,
        video_path: str | Path,
        frame_number: int = 0,
        alpha_threshold: int = 0,
        edge_softness: int = 0,
        parent: QObject | None = None,
    ) -> None:
        """Initialize preview worker.

        Args:
            model: Background removal model.
            video_path: Video file path.
            frame_number: Frame number to preview.
            alpha_threshold: Alpha threshold (0-100%).
            edge_softness: Edge softness (0-20).
            parent: Parent QObject.
        """
        super().__init__(parent)
        self.model = model
        self.video_path = Path(video_path)
        self.frame_number = frame_number
        self.alpha_threshold = alpha_threshold
        self.edge_softness = edge_softness

    def run(self) -> None:
        """Generate preview frame."""
        import cv2

        from src.pipeline.output import apply_alpha_processing

        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError(f"Cannot read frame {self.frame_number}")

            # Ensure model is loaded
            if not self.model.is_loaded:
                self.model.load()

            # Reset state for single frame
            self.model.reset_state()

            # Run inference
            foreground, alpha = self.model.inference(frame)

            # Apply alpha processing (threshold and softness)
            if self.alpha_threshold > 0 or self.edge_softness > 0:
                alpha = apply_alpha_processing(
                    alpha,
                    threshold=self.alpha_threshold,
                    softness=self.edge_softness,
                )

            self.frame_ready.emit(foreground, alpha)

        except Exception as e:
            logger.exception("Preview error")
            self.error.emit(str(e))
