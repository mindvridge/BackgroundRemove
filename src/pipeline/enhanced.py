"""Enhanced video pipeline with temporal consistency and edge refinement.

Provides high-quality video background removal with:
- Temporal consistency for flicker-free output
- Edge refinement for hair/fur detail preservation
- Quality presets for different use cases
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
import torch

from src.pipeline.edge_refine import EdgeConfig, EdgeMethod, EdgeRefiner
from src.pipeline.processor import OutputType, ProcessedFrame, ProcessorConfig
from src.pipeline.temporal import TemporalConfig, TemporalConsistency, TemporalMethod

if TYPE_CHECKING:
    from numpy import ndarray

    from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class QualityPreset(Enum):
    """Quality presets for enhanced pipeline."""

    FAST = "fast"  # Minimal processing, best for real-time
    BALANCED = "balanced"  # Good balance of speed and quality
    QUALITY = "quality"  # High quality, slower
    MAXIMUM = "maximum"  # Maximum quality, slowest


@dataclass
class EnhancedConfig:
    """Configuration for enhanced pipeline."""

    # Base processor settings
    seq_chunk: int = 4
    output_type: OutputType = OutputType.COMPOSITE
    downsample_ratio: float | None = None

    # Temporal consistency
    enable_temporal: bool = True
    temporal_method: TemporalMethod = TemporalMethod.OPTICAL_FLOW
    temporal_config: TemporalConfig | None = None

    # Edge refinement
    enable_edge_refine: bool = True
    edge_method: EdgeMethod = EdgeMethod.COMBINED
    edge_config: EdgeConfig | None = None

    # Quality preset (overrides individual settings)
    quality_preset: QualityPreset | None = None


def get_preset_config(preset: QualityPreset) -> EnhancedConfig:
    """Get configuration for a quality preset.

    Args:
        preset: Quality preset.

    Returns:
        Configured EnhancedConfig.
    """
    if preset == QualityPreset.FAST:
        return EnhancedConfig(
            seq_chunk=8,
            enable_temporal=True,
            temporal_method=TemporalMethod.EMA,
            temporal_config=TemporalConfig(
                method=TemporalMethod.EMA,
                ema_alpha=0.4,
            ),
            enable_edge_refine=True,
            edge_method=EdgeMethod.FAST,
            edge_config=EdgeConfig(
                method=EdgeMethod.FAST,
                guided_radius=4,
                feather_amount=1,
            ),
        )

    elif preset == QualityPreset.BALANCED:
        return EnhancedConfig(
            seq_chunk=4,
            enable_temporal=True,
            temporal_method=TemporalMethod.OPTICAL_FLOW,
            temporal_config=TemporalConfig(
                method=TemporalMethod.OPTICAL_FLOW,
                flow_quality="medium",
                ema_alpha=0.3,
            ),
            enable_edge_refine=True,
            edge_method=EdgeMethod.GUIDED_FILTER,
            edge_config=EdgeConfig(
                method=EdgeMethod.GUIDED_FILTER,
                guided_radius=8,
                guided_eps=0.01,
                feather_amount=2,
            ),
        )

    elif preset == QualityPreset.QUALITY:
        return EnhancedConfig(
            seq_chunk=2,
            enable_temporal=True,
            temporal_method=TemporalMethod.BIDIRECTIONAL,
            temporal_config=TemporalConfig(
                method=TemporalMethod.BIDIRECTIONAL,
                flow_quality="high",
                history_size=5,
            ),
            enable_edge_refine=True,
            edge_method=EdgeMethod.COMBINED,
            edge_config=EdgeConfig(
                method=EdgeMethod.COMBINED,
                guided_radius=12,
                guided_eps=0.005,
                matting_iterations=7,
                feather_amount=3,
                preserve_detail=True,
            ),
        )

    else:  # MAXIMUM
        return EnhancedConfig(
            seq_chunk=1,
            enable_temporal=True,
            temporal_method=TemporalMethod.ADAPTIVE,
            temporal_config=TemporalConfig(
                method=TemporalMethod.ADAPTIVE,
                flow_quality="high",
                flow_scale=1.0,
                history_size=7,
                blend_static_weight=0.8,
            ),
            enable_edge_refine=True,
            edge_method=EdgeMethod.COMBINED,
            edge_config=EdgeConfig(
                method=EdgeMethod.COMBINED,
                guided_radius=16,
                guided_eps=0.002,
                matting_iterations=10,
                erode_size=8,
                dilate_size=25,
                feather_amount=4,
                preserve_detail=True,
                detail_threshold=0.05,
            ),
        )


@dataclass
class EnhancedFrame:
    """Container for enhanced processed frame."""

    frame_index: int
    original_frame: ndarray
    foreground: ndarray
    alpha: ndarray
    refined_alpha: ndarray
    output: ndarray


class EnhancedProcessor:
    """Enhanced frame processor with temporal and edge refinement.

    Processes video frames with additional quality enhancements
    for professional-grade background removal.

    Example:
        >>> processor = EnhancedProcessor(model, EnhancedConfig(
        ...     quality_preset=QualityPreset.QUALITY
        ... ))
        >>> for result in processor.process_frames(frames):
        ...     save_frame(result.output)
    """

    def __init__(
        self,
        model: BaseModel,
        config: EnhancedConfig | None = None,
    ) -> None:
        """Initialize enhanced processor.

        Args:
            model: Background removal model.
            config: Enhanced configuration.
        """
        self.model = model

        # Apply preset if specified
        if config is not None and config.quality_preset is not None:
            self.config = get_preset_config(config.quality_preset)
        else:
            self.config = config or EnhancedConfig()

        # Initialize components
        self._init_components()

        self._frame_index = 0
        self._processed_count = 0

        logger.debug(
            f"Enhanced processor initialized: "
            f"temporal={self.config.enable_temporal}, "
            f"edge_refine={self.config.enable_edge_refine}"
        )

    def _init_components(self) -> None:
        """Initialize processing components."""
        # Temporal consistency
        if self.config.enable_temporal:
            temporal_config = self.config.temporal_config or TemporalConfig(
                method=self.config.temporal_method
            )
            self._temporal = TemporalConsistency(temporal_config)
        else:
            self._temporal = None

        # Edge refinement
        if self.config.enable_edge_refine:
            edge_config = self.config.edge_config or EdgeConfig(
                method=self.config.edge_method
            )
            self._edge_refiner = EdgeRefiner(edge_config)
        else:
            self._edge_refiner = None

    @property
    def processed_count(self) -> int:
        """Number of frames processed."""
        return self._processed_count

    def reset(self) -> None:
        """Reset processor state for new video."""
        self._frame_index = 0
        self._processed_count = 0
        self.model.reset_state()

        if self._temporal is not None:
            self._temporal.reset()

        logger.debug("Enhanced processor state reset")

    def _create_output(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Create final output based on output type.

        Args:
            foreground: RGB foreground.
            alpha: Alpha matte.

        Returns:
            Final output.
        """
        output_type = self.config.output_type

        if output_type == OutputType.FOREGROUND:
            return foreground
        elif output_type == OutputType.ALPHA:
            return alpha
        elif output_type == OutputType.COMPOSITE:
            if len(foreground.shape) == 2:
                foreground = np.stack([foreground] * 3, axis=-1)
            return np.dstack([foreground, alpha])
        elif output_type == OutputType.GREEN_SCREEN:
            bg_color = np.array([0, 255, 0], dtype=np.uint8)
            alpha_norm = alpha.astype(np.float32) / 255.0
            if len(alpha_norm.shape) == 2:
                alpha_norm = alpha_norm[:, :, np.newaxis]
            background = np.full_like(foreground, bg_color)
            return (foreground * alpha_norm + background * (1 - alpha_norm)).astype(np.uint8)
        else:
            return foreground

    @torch.inference_mode()
    def process_frame(self, frame: ndarray) -> EnhancedFrame:
        """Process a single frame with enhancements.

        Args:
            frame: Input frame (BGR format).

        Returns:
            EnhancedFrame with all processing results.
        """
        # Run model inference
        foreground, alpha = self.model.inference(
            frame,
            downsample_ratio=self.config.downsample_ratio,
        )

        refined_alpha = alpha.copy()

        # Apply edge refinement
        if self._edge_refiner is not None:
            refined_alpha = self._edge_refiner.refine(frame, refined_alpha)

        # Apply temporal consistency
        if self._temporal is not None:
            refined_alpha = self._temporal.process(frame, refined_alpha)

        # Create output
        output = self._create_output(foreground, refined_alpha)

        result = EnhancedFrame(
            frame_index=self._frame_index,
            original_frame=frame,
            foreground=foreground,
            alpha=alpha,
            refined_alpha=refined_alpha,
            output=output,
        )

        self._frame_index += 1
        self._processed_count += 1

        return result

    def process_frames(
        self,
        frame_iterator: Iterator[ndarray],
        progress_callback: Callable[[int, int | None], None] | None = None,
        total_frames: int | None = None,
    ) -> Iterator[EnhancedFrame]:
        """Process frames with enhanced pipeline.

        Args:
            frame_iterator: Iterator yielding video frames.
            progress_callback: Optional progress callback.
            total_frames: Optional total frame count.

        Yields:
            EnhancedFrame for each processed frame.
        """
        chunk_size = self.config.seq_chunk
        buffer: list[ndarray] = []

        for frame in frame_iterator:
            buffer.append(frame)

            if len(buffer) >= chunk_size:
                for f in buffer:
                    result = self.process_frame(f)
                    if progress_callback is not None:
                        progress_callback(self._processed_count, total_frames)
                    yield result
                buffer.clear()

        # Process remaining
        for f in buffer:
            result = self.process_frame(f)
            if progress_callback is not None:
                progress_callback(self._processed_count, total_frames)
            yield result

        logger.info(f"Enhanced processing complete: {self._processed_count} frames")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnhancedProcessor("
            f"temporal={self.config.temporal_method.value if self.config.enable_temporal else 'disabled'}, "
            f"edge={self.config.edge_method.value if self.config.enable_edge_refine else 'disabled'})"
        )


class EnhancedPipeline:
    """End-to-end enhanced video pipeline.

    Combines reader, enhanced processor, and writer for
    complete high-quality video processing.

    Example:
        >>> pipeline = EnhancedPipeline(model, QualityPreset.QUALITY)
        >>> result = pipeline.process("input.mp4", "output.mp4")
    """

    def __init__(
        self,
        model: BaseModel,
        preset: QualityPreset = QualityPreset.BALANCED,
        config: EnhancedConfig | None = None,
    ) -> None:
        """Initialize enhanced pipeline.

        Args:
            model: Background removal model.
            preset: Quality preset (ignored if config provided).
            config: Optional custom configuration.
        """
        self.model = model

        if config is not None:
            self.config = config
        else:
            self.config = get_preset_config(preset)

        self._processor = EnhancedProcessor(model, self.config)
        self._progress_callback: Callable[[int, int, float], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[int, int, float], None] | None,
    ) -> None:
        """Set progress callback.

        Args:
            callback: Function(current, total, fps) or None.
        """
        self._progress_callback = callback

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> dict:
        """Process a video file.

        Args:
            input_path: Input video path.
            output_path: Output video path.

        Returns:
            Processing statistics.
        """
        from src.pipeline.reader import ReaderConfig, VideoReader
        from src.pipeline.writer import VideoWriter, WriterConfig

        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Enhanced processing: {input_path} -> {output_path}")
        start_time = time.time()

        reader = None
        writer = None
        frames_processed = 0

        try:
            # Ensure model is loaded
            if not self.model.is_loaded:
                self.model.load()

            # Reset states
            self._processor.reset()

            # Create reader
            reader = VideoReader(input_path, ReaderConfig(queue_size=64))
            reader.start()

            # Wait for metadata
            while reader.metadata is None and reader.is_running:
                time.sleep(0.01)

            if reader.metadata is None:
                raise RuntimeError("Failed to read video metadata")

            metadata = reader.metadata

            # Determine output format
            has_alpha = self.config.output_type == OutputType.COMPOSITE

            # Create writer
            writer_config = WriterConfig(
                queue_size=64,
                fps=metadata.fps,
                codec="libx264" if not has_alpha else "png",
            )
            writer = VideoWriter(
                output_path,
                metadata.width,
                metadata.height,
                writer_config,
                has_alpha,
            )
            writer.start()

            # Process
            last_progress = time.time()

            def progress_cb(current: int, total: int | None) -> None:
                nonlocal last_progress, frames_processed
                frames_processed = current

                now = time.time()
                if now - last_progress >= 0.5:
                    elapsed = now - start_time
                    fps = current / elapsed if elapsed > 0 else 0

                    if self._progress_callback:
                        self._progress_callback(current, total or 0, fps)

                    last_progress = now

            for result in self._processor.process_frames(
                iter(reader),
                progress_callback=progress_cb,
                total_frames=metadata.frame_count,
            ):
                writer.write(result.output)

            frames_processed = self._processor.processed_count

        finally:
            if reader is not None:
                reader.stop()
            if writer is not None:
                writer.stop()

        elapsed = time.time() - start_time
        fps_achieved = frames_processed / elapsed if elapsed > 0 else 0

        logger.info(
            f"Enhanced processing complete: {frames_processed} frames "
            f"in {elapsed:.1f}s ({fps_achieved:.1f} fps)"
        )

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "frames_processed": frames_processed,
            "duration_seconds": elapsed,
            "fps_achieved": fps_achieved,
            "success": True,
        }


def process_video_enhanced(
    model: BaseModel,
    input_path: str | Path,
    output_path: str | Path,
    preset: QualityPreset = QualityPreset.BALANCED,
) -> dict:
    """Convenience function for enhanced video processing.

    Args:
        model: Background removal model.
        input_path: Input video path.
        output_path: Output video path.
        preset: Quality preset.

    Returns:
        Processing statistics.
    """
    pipeline = EnhancedPipeline(model, preset=preset)
    return pipeline.process(input_path, output_path)
