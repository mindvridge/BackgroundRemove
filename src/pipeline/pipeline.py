"""Video processing pipeline integrating reader, processor, and writer.

Provides a unified interface for end-to-end video background removal
using the Producer-Consumer pattern.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.pipeline.processor import FrameProcessor, OutputType, ProcessorConfig
from src.pipeline.reader import ReaderConfig, VideoReader
from src.pipeline.writer import VideoWriter, WriterConfig

if TYPE_CHECKING:
    from src.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""

    # Reader settings
    reader_queue_size: int = 64

    # Processor settings
    seq_chunk: int = 4
    output_type: OutputType = OutputType.COMPOSITE
    downsample_ratio: float | None = None

    # Writer settings
    writer_queue_size: int = 64
    output_codec: str = "libx264"
    output_crf: int = 23
    output_preset: str = "medium"
    output_bitrate: str | None = None
    copy_audio: bool = True

    # Pipeline settings
    overwrite_output: bool = True


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    input_path: Path
    output_path: Path
    frames_processed: int
    duration_seconds: float
    fps_achieved: float
    success: bool
    error: str | None = None


class VideoPipeline:
    """End-to-end video processing pipeline.

    Orchestrates VideoReader, FrameProcessor, and VideoWriter
    in a Producer-Consumer pattern for efficient video processing.

    Example:
        >>> pipeline = VideoPipeline(model)
        >>> result = pipeline.process("input.mp4", "output.mp4")
        >>> print(f"Processed {result.frames_processed} frames")
    """

    def __init__(
        self,
        model: BaseModel,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize video pipeline.

        Args:
            model: Background removal model instance.
            config: Pipeline configuration.
        """
        self.model = model
        self.config = config or PipelineConfig()
        self._progress_callback: Callable[[int, int, float], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[int, int, float], None] | None,
    ) -> None:
        """Set progress callback function.

        Args:
            callback: Function(current_frame, total_frames, fps) or None.
        """
        self._progress_callback = callback

    def _create_reader(self, input_path: Path) -> VideoReader:
        """Create configured video reader.

        Args:
            input_path: Input video path.

        Returns:
            Configured VideoReader instance.
        """
        reader_config = ReaderConfig(
            queue_size=self.config.reader_queue_size,
        )
        return VideoReader(input_path, reader_config)

    def _create_processor(self) -> FrameProcessor:
        """Create configured frame processor.

        Returns:
            Configured FrameProcessor instance.
        """
        processor_config = ProcessorConfig(
            seq_chunk=self.config.seq_chunk,
            output_type=self.config.output_type,
            downsample_ratio=self.config.downsample_ratio,
        )
        return FrameProcessor(self.model, processor_config)

    def _create_writer(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        audio_source: Path | None = None,
    ) -> VideoWriter:
        """Create configured video writer.

        Args:
            output_path: Output video path.
            width: Video width.
            height: Video height.
            fps: Frame rate.
            audio_source: Source video for audio copying.

        Returns:
            Configured VideoWriter instance.
        """
        # Determine if output has alpha channel
        has_alpha = self.config.output_type == OutputType.COMPOSITE

        writer_config = WriterConfig(
            queue_size=self.config.writer_queue_size,
            fps=fps,
            codec=self.config.output_codec,
            crf=self.config.output_crf,
            preset=self.config.output_preset,
            bitrate=self.config.output_bitrate,
            copy_audio=self.config.copy_audio,
            audio_source=audio_source if self.config.copy_audio else None,
            overwrite=self.config.overwrite_output,
        )

        return VideoWriter(output_path, width, height, writer_config, has_alpha)

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> PipelineResult:
        """Process a video file.

        Args:
            input_path: Path to input video.
            output_path: Path for output video.

        Returns:
            PipelineResult with processing statistics.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Processing: {input_path} -> {output_path}")
        start_time = time.time()

        reader = None
        writer = None
        frames_processed = 0
        error_msg = None

        try:
            # Ensure model is loaded
            if not self.model.is_loaded:
                self.model.load()

            # Reset model state for new video
            self.model.reset_state()

            # Create reader and start
            reader = self._create_reader(input_path)
            reader.start()

            # Wait for metadata to be available
            while reader.metadata is None and reader.is_running:
                time.sleep(0.01)

            if reader.metadata is None:
                raise RuntimeError("Failed to read video metadata")

            metadata = reader.metadata
            total_frames = metadata.frame_count

            logger.info(
                f"Video: {metadata.width}x{metadata.height} @ {metadata.fps:.2f}fps, "
                f"{total_frames} frames"
            )

            # Create processor
            processor = self._create_processor()

            # Create writer
            writer = self._create_writer(
                output_path,
                metadata.width,
                metadata.height,
                metadata.fps,
                input_path,
            )
            writer.start()

            # Processing loop with progress tracking
            last_progress_time = time.time()
            progress_interval = 0.5  # Update progress every 0.5 seconds

            def progress_callback(current: int, total: int | None) -> None:
                nonlocal last_progress_time, frames_processed
                frames_processed = current

                now = time.time()
                if now - last_progress_time >= progress_interval:
                    elapsed = now - start_time
                    fps = current / elapsed if elapsed > 0 else 0

                    if self._progress_callback is not None:
                        self._progress_callback(current, total or 0, fps)

                    last_progress_time = now

                    # Log progress
                    if total:
                        pct = (current / total) * 100
                        logger.debug(f"Progress: {current}/{total} ({pct:.1f}%) @ {fps:.1f} fps")

            # Process frames
            for result in processor.process_frames(
                iter(reader),
                progress_callback=progress_callback,
                total_frames=total_frames,
            ):
                # Write output frame
                if not writer.write(result.output):
                    logger.warning(f"Frame {result.frame_index} dropped (queue full)")

            frames_processed = processor.processed_count

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            error_msg = str(e)

        finally:
            # Cleanup
            if reader is not None:
                reader.stop()
            if writer is not None:
                writer.stop()

        # Calculate statistics
        elapsed = time.time() - start_time
        fps_achieved = frames_processed / elapsed if elapsed > 0 else 0

        result = PipelineResult(
            input_path=input_path,
            output_path=output_path,
            frames_processed=frames_processed,
            duration_seconds=elapsed,
            fps_achieved=fps_achieved,
            success=error_msg is None,
            error=error_msg,
        )

        if result.success:
            logger.info(
                f"Processing complete: {frames_processed} frames in {elapsed:.1f}s "
                f"({fps_achieved:.1f} fps)"
            )
        else:
            logger.error(f"Processing failed: {error_msg}")

        return result

    def process_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        output_suffix: str = "_removed",
    ) -> list[PipelineResult]:
        """Process multiple video files.

        Args:
            input_paths: List of input video paths.
            output_dir: Directory for output videos.
            output_suffix: Suffix to add to output filenames.

        Returns:
            List of PipelineResult for each video.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, input_path in enumerate(input_paths):
            input_path = Path(input_path)
            output_name = f"{input_path.stem}{output_suffix}{input_path.suffix}"
            output_path = output_dir / output_name

            logger.info(f"Processing video {i + 1}/{len(input_paths)}: {input_path.name}")

            result = self.process(input_path, output_path)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {successful}/{len(results)} videos processed")

        return results

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoPipeline(seq_chunk={self.config.seq_chunk}, "
            f"output_type={self.config.output_type.value})"
        )
