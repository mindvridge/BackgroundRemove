"""Video frame processor with batch inference support.

Implements the processing stage of the pipeline with
seq_chunk batching and torch.no_grad() optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy import ndarray

    from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Output type for processed frames."""

    FOREGROUND = "foreground"  # RGB foreground only
    ALPHA = "alpha"  # Alpha matte only
    COMPOSITE = "composite"  # Foreground with alpha channel (RGBA)
    GREEN_SCREEN = "green_screen"  # Foreground on green background


@dataclass
class ProcessorConfig:
    """Configuration for frame processor."""

    seq_chunk: int = 4  # Number of frames to process in batch
    output_type: OutputType = OutputType.COMPOSITE
    downsample_ratio: float | None = None  # None = auto-calculate
    background_color: tuple[int, int, int] = (0, 255, 0)  # For green screen

    # Post-processing options
    alpha_threshold: float = 0.0  # Threshold for binary alpha (0 = disabled)
    feather_amount: int = 0  # Edge feathering pixels (0 = disabled)
    output_resolution: tuple[int, int] | None = None  # (width, height), None = same


@dataclass
class ProcessedFrame:
    """Container for a processed frame result."""

    frame_index: int
    foreground: ndarray  # RGB foreground
    alpha: ndarray  # Alpha matte
    output: ndarray  # Final output based on output_type


class FrameProcessor:
    """Video frame processor with batch support.

    Processes frames using a background removal model with
    seq_chunk batching for improved throughput.

    Example:
        >>> processor = FrameProcessor(model, ProcessorConfig(seq_chunk=4))
        >>> for result in processor.process_frames(frame_iterator):
        ...     save_frame(result.output)
    """

    def __init__(
        self,
        model: BaseModel,
        config: ProcessorConfig | None = None,
    ) -> None:
        """Initialize frame processor.

        Args:
            model: Background removal model instance.
            config: Processor configuration.
        """
        self.model = model
        self.config = config or ProcessorConfig()
        self._frame_index = 0
        self._processed_count = 0

    @property
    def processed_count(self) -> int:
        """Number of frames processed."""
        return self._processed_count

    def reset(self) -> None:
        """Reset processor state for new video."""
        self._frame_index = 0
        self._processed_count = 0
        self.model.reset_state()
        logger.debug("Processor state reset")

    def _apply_alpha_threshold(self, alpha: ndarray) -> ndarray:
        """Apply threshold to create binary alpha mask.

        Args:
            alpha: Input alpha matte (0-255).

        Returns:
            Thresholded alpha matte.
        """
        if self.config.alpha_threshold <= 0:
            return alpha

        threshold = int(self.config.alpha_threshold * 255)
        return np.where(alpha > threshold, 255, 0).astype(np.uint8)

    def _apply_feathering(self, alpha: ndarray) -> ndarray:
        """Apply edge feathering to alpha matte.

        Args:
            alpha: Input alpha matte.

        Returns:
            Feathered alpha matte.
        """
        import cv2

        if self.config.feather_amount <= 0:
            return alpha

        kernel_size = self.config.feather_amount * 2 + 1
        return cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)

    def _resize_output(self, frame: ndarray) -> ndarray:
        """Resize frame to output resolution if specified.

        Args:
            frame: Input frame.

        Returns:
            Resized frame or original if no resize needed.
        """
        import cv2

        if self.config.output_resolution is None:
            return frame

        target_w, target_h = self.config.output_resolution
        if frame.shape[1] == target_w and frame.shape[0] == target_h:
            return frame

        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _create_output(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Create final output based on output_type configuration.

        Args:
            foreground: RGB foreground image.
            alpha: Alpha matte.

        Returns:
            Final output image.
        """
        output_type = self.config.output_type

        if output_type == OutputType.FOREGROUND:
            return foreground

        elif output_type == OutputType.ALPHA:
            return alpha

        elif output_type == OutputType.COMPOSITE:
            # RGBA composite
            if len(foreground.shape) == 2:
                foreground = np.stack([foreground] * 3, axis=-1)
            rgba = np.dstack([foreground, alpha])
            return rgba

        elif output_type == OutputType.GREEN_SCREEN:
            # Composite on colored background
            bg_color = np.array(self.config.background_color, dtype=np.uint8)
            alpha_norm = alpha.astype(np.float32) / 255.0

            if len(alpha_norm.shape) == 2:
                alpha_norm = alpha_norm[:, :, np.newaxis]

            background = np.full_like(foreground, bg_color)
            output = (foreground * alpha_norm + background * (1 - alpha_norm)).astype(
                np.uint8
            )
            return output

        else:
            return foreground

    @torch.inference_mode()
    def process_frame(self, frame: ndarray) -> ProcessedFrame:
        """Process a single frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format.

        Returns:
            ProcessedFrame with foreground, alpha, and output.
        """
        # Run model inference
        foreground, alpha = self.model.inference(
            frame,
            downsample_ratio=self.config.downsample_ratio,
        )

        # Apply post-processing
        alpha = self._apply_alpha_threshold(alpha)
        alpha = self._apply_feathering(alpha)

        # Create output
        output = self._create_output(foreground, alpha)
        output = self._resize_output(output)

        result = ProcessedFrame(
            frame_index=self._frame_index,
            foreground=foreground,
            alpha=alpha,
            output=output,
        )

        self._frame_index += 1
        self._processed_count += 1

        return result

    @torch.inference_mode()
    def process_batch(self, frames: list[ndarray]) -> list[ProcessedFrame]:
        """Process a batch of frames.

        Note: RVM processes frames sequentially to maintain recurrent state,
        but this method handles the batching logic for chunked processing.

        Args:
            frames: List of input frames.

        Returns:
            List of ProcessedFrame results.
        """
        results = []
        for frame in frames:
            result = self.process_frame(frame)
            results.append(result)
        return results

    def process_frames(
        self,
        frame_iterator: Iterator[ndarray],
        progress_callback: Callable[[int, int | None], None] | None = None,
        total_frames: int | None = None,
    ) -> Iterator[ProcessedFrame]:
        """Process frames from an iterator with seq_chunk batching.

        Args:
            frame_iterator: Iterator yielding video frames.
            progress_callback: Optional callback(processed, total) for progress.
            total_frames: Optional total frame count for progress reporting.

        Yields:
            ProcessedFrame for each processed frame.
        """
        chunk_size = self.config.seq_chunk
        buffer: list[ndarray] = []

        for frame in frame_iterator:
            buffer.append(frame)

            # Process when buffer reaches chunk size
            if len(buffer) >= chunk_size:
                results = self.process_batch(buffer)
                buffer.clear()

                for result in results:
                    if progress_callback is not None:
                        progress_callback(self._processed_count, total_frames)
                    yield result

        # Process remaining frames
        if buffer:
            results = self.process_batch(buffer)
            for result in results:
                if progress_callback is not None:
                    progress_callback(self._processed_count, total_frames)
                yield result

        logger.info(f"Processed {self._processed_count} frames")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FrameProcessor(seq_chunk={self.config.seq_chunk}, "
            f"output_type={self.config.output_type.value}, "
            f"processed={self._processed_count})"
        )
