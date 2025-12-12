"""Image and video upscaling pipeline.

Provides unified interface for upscaling images and videos with:
- Automatic model selection based on content type
- Batch processing support
- Video frame upscaling with temporal consistency
- Progress tracking and callbacks
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import cv2
import numpy as np

from src.upscale.base import (
    BaseUpscaler,
    DegradationType,
    UpscaleConfig,
    UpscaleModel,
    UpscaleResult,
)

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


@dataclass
class VideoUpscaleResult:
    """Result of video upscaling."""

    input_path: Path
    output_path: Path
    frames_processed: int
    original_resolution: tuple[int, int]
    upscaled_resolution: tuple[int, int]
    scale_factor: float
    total_time: float
    average_fps: float
    model_used: str


def create_upscaler(config: UpscaleConfig) -> BaseUpscaler:
    """Factory function to create appropriate upscaler.

    Args:
        config: Upscaling configuration.

    Returns:
        Upscaler instance.
    """
    model = config.model

    if model in (
        UpscaleModel.REAL_ESRGAN_X4PLUS,
        UpscaleModel.REAL_ESRGAN_X4PLUS_ANIME,
        UpscaleModel.REAL_ESRGAN_X2PLUS,
    ):
        from src.upscale.real_esrgan import RealESRGANUpscaler
        return RealESRGANUpscaler(config)

    elif model in (
        UpscaleModel.SWINIR_REAL_SR_X4,
        UpscaleModel.SWINIR_CLASSICAL_SR_X4,
        UpscaleModel.SWINIR_LIGHTWEIGHT_X4,
    ):
        from src.upscale.swinir import SwinIRUpscaler
        return SwinIRUpscaler(config)

    else:
        raise ValueError(f"Unsupported model: {model}")


def auto_select_model(
    image: ndarray | None = None,
    content_type: str = "photo",
    quality_priority: bool = True,
) -> UpscaleModel:
    """Automatically select best model based on content.

    Args:
        image: Optional sample image for analysis.
        content_type: Content type hint ('photo', 'anime', 'art', 'architecture').
        quality_priority: Prioritize quality over speed.

    Returns:
        Recommended UpscaleModel.
    """
    if content_type == "anime":
        return UpscaleModel.REAL_ESRGAN_X4PLUS_ANIME

    if content_type == "architecture" or quality_priority:
        return UpscaleModel.SWINIR_REAL_SR_X4

    # Default to Real-ESRGAN for best speed/quality balance
    return UpscaleModel.REAL_ESRGAN_X4PLUS


class ImageUpscalePipeline:
    """Pipeline for upscaling single images or batches.

    Example:
        >>> pipeline = ImageUpscalePipeline(UpscaleConfig(scale=4))
        >>> result = pipeline.upscale_file("input.jpg", "output.png")
    """

    def __init__(self, config: UpscaleConfig | None = None) -> None:
        """Initialize pipeline.

        Args:
            config: Upscaling configuration.
        """
        self.config = config or UpscaleConfig()
        self._upscaler: BaseUpscaler | None = None

    def _ensure_upscaler(self) -> BaseUpscaler:
        """Ensure upscaler is loaded."""
        if self._upscaler is None:
            self._upscaler = create_upscaler(self.config)
            self._upscaler.load()
        return self._upscaler

    def upscale(self, image: ndarray) -> UpscaleResult:
        """Upscale a single image.

        Args:
            image: Input BGR image.

        Returns:
            UpscaleResult.
        """
        upscaler = self._ensure_upscaler()
        return upscaler.upscale(image)

    def upscale_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> UpscaleResult:
        """Upscale an image file.

        Args:
            input_path: Input image path.
            output_path: Output path (auto-generated if None).

        Returns:
            UpscaleResult with saved output path.
        """
        input_path = Path(input_path)

        if output_path is None:
            suffix = f"_x{self.config.scale}.{self.config.output_format}"
            output_path = input_path.with_stem(input_path.stem + suffix)
        else:
            output_path = Path(output_path)

        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Cannot load image: {input_path}")

        # Upscale
        result = self.upscale(image)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.output_format.lower() in ("jpg", "jpeg"):
            cv2.imwrite(
                str(output_path),
                result.image,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
            )
        elif self.config.output_format.lower() == "webp":
            cv2.imwrite(
                str(output_path),
                result.image,
                [cv2.IMWRITE_WEBP_QUALITY, self.config.jpeg_quality],
            )
        else:
            cv2.imwrite(str(output_path), result.image)

        logger.info(
            f"Upscaled {input_path.name}: {result.original_size} -> {result.upscaled_size} "
            f"in {result.processing_time:.2f}s"
        )

        return result

    def upscale_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[UpscaleResult]:
        """Upscale multiple images.

        Args:
            input_paths: List of input image paths.
            output_dir: Output directory.
            progress_callback: Optional callback(current, total).

        Returns:
            List of UpscaleResult.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(input_paths)

        for i, input_path in enumerate(input_paths):
            input_path = Path(input_path)
            output_path = output_dir / f"{input_path.stem}_x{self.config.scale}.{self.config.output_format}"

            try:
                result = self.upscale_file(input_path, output_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to upscale {input_path}: {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def unload(self) -> None:
        """Unload the upscaler."""
        if self._upscaler is not None:
            self._upscaler.unload()
            self._upscaler = None

    def __enter__(self) -> "ImageUpscalePipeline":
        self._ensure_upscaler()
        return self

    def __exit__(self, *args) -> None:
        self.unload()


class VideoUpscalePipeline:
    """Pipeline for upscaling videos frame by frame.

    Example:
        >>> pipeline = VideoUpscalePipeline(UpscaleConfig(scale=4))
        >>> result = pipeline.upscale("input.mp4", "output.mp4")
    """

    def __init__(
        self,
        config: UpscaleConfig | None = None,
        output_codec: str = "libx264",
        output_crf: int = 18,
    ) -> None:
        """Initialize video upscale pipeline.

        Args:
            config: Upscaling configuration.
            output_codec: FFmpeg output codec.
            output_crf: Output quality (CRF).
        """
        self.config = config or UpscaleConfig()
        self.output_codec = output_codec
        self.output_crf = output_crf
        self._upscaler: BaseUpscaler | None = None
        self._progress_callback: Callable[[int, int, float], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[int, int, float], None] | None,
    ) -> None:
        """Set progress callback.

        Args:
            callback: Function(current_frame, total_frames, fps).
        """
        self._progress_callback = callback

    def _ensure_upscaler(self) -> BaseUpscaler:
        """Ensure upscaler is loaded."""
        if self._upscaler is None:
            self._upscaler = create_upscaler(self.config)
            self._upscaler.load()
        return self._upscaler

    def upscale(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> VideoUpscaleResult:
        """Upscale a video file.

        Args:
            input_path: Input video path.
            output_path: Output video path.

        Returns:
            VideoUpscaleResult.
        """
        import subprocess

        input_path = Path(input_path)
        output_path = Path(output_path)

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate output size
        scale = self.config.scale
        out_width = width * scale
        out_height = height * scale

        logger.info(
            f"Upscaling video: {width}x{height} -> {out_width}x{out_height}, "
            f"{total_frames} frames @ {fps:.2f}fps"
        )

        # Ensure upscaler is ready
        upscaler = self._ensure_upscaler()

        # Setup FFmpeg output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}",
            "-r", str(fps),
            "-i", "-",
            "-i", str(input_path),  # For audio
            "-c:v", self.output_codec,
            "-crf", str(self.output_crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            str(output_path),
        ]

        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Process frames
        start_time = time.time()
        frames_processed = 0
        last_progress_time = start_time

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Upscale frame
                result = upscaler.upscale(frame)
                frames_processed += 1

                # Write to FFmpeg
                process.stdin.write(result.image.tobytes())

                # Progress callback
                now = time.time()
                if now - last_progress_time >= 0.5:
                    elapsed = now - start_time
                    current_fps = frames_processed / elapsed if elapsed > 0 else 0

                    if self._progress_callback:
                        self._progress_callback(frames_processed, total_frames, current_fps)

                    last_progress_time = now

        finally:
            cap.release()
            process.stdin.close()
            process.wait()

        total_time = time.time() - start_time
        avg_fps = frames_processed / total_time if total_time > 0 else 0

        logger.info(
            f"Video upscaling complete: {frames_processed} frames in {total_time:.1f}s "
            f"({avg_fps:.2f} fps)"
        )

        return VideoUpscaleResult(
            input_path=input_path,
            output_path=output_path,
            frames_processed=frames_processed,
            original_resolution=(width, height),
            upscaled_resolution=(out_width, out_height),
            scale_factor=scale,
            total_time=total_time,
            average_fps=avg_fps,
            model_used=upscaler.model_name,
        )

    def upscale_frames(
        self,
        frames: Iterator[ndarray],
        total_frames: int | None = None,
    ) -> Iterator[ndarray]:
        """Upscale frames from an iterator.

        Args:
            frames: Iterator of BGR frames.
            total_frames: Optional total frame count for progress.

        Yields:
            Upscaled frames.
        """
        upscaler = self._ensure_upscaler()
        processed = 0
        start_time = time.time()

        for frame in frames:
            result = upscaler.upscale(frame)
            processed += 1

            if self._progress_callback and total_frames:
                elapsed = time.time() - start_time
                fps = processed / elapsed if elapsed > 0 else 0
                self._progress_callback(processed, total_frames, fps)

            yield result.image

    def unload(self) -> None:
        """Unload the upscaler."""
        if self._upscaler is not None:
            self._upscaler.unload()
            self._upscaler = None


class ChainedUpscaler:
    """Chain multiple upscalers for enhanced quality.

    Applies upscalers sequentially for potentially better results:
    1. Real-ESRGAN for initial upscale
    2. SwinIR for detail refinement

    Note: This doubles processing time but can improve quality.
    """

    def __init__(
        self,
        first_config: UpscaleConfig | None = None,
        second_config: UpscaleConfig | None = None,
    ) -> None:
        """Initialize chained upscaler.

        Args:
            first_config: Config for first pass (default: Real-ESRGAN 2x).
            second_config: Config for second pass (default: SwinIR 2x).
        """
        self.first_config = first_config or UpscaleConfig(
            model=UpscaleModel.REAL_ESRGAN_X2PLUS,
            scale=2,
        )
        self.second_config = second_config or UpscaleConfig(
            model=UpscaleModel.SWINIR_REAL_SR_X4,
            scale=2,
        )

        self._first_upscaler: BaseUpscaler | None = None
        self._second_upscaler: BaseUpscaler | None = None

    def load(self) -> None:
        """Load both upscalers."""
        self._first_upscaler = create_upscaler(self.first_config)
        self._first_upscaler.load()

        self._second_upscaler = create_upscaler(self.second_config)
        self._second_upscaler.load()

    def upscale(self, image: ndarray) -> UpscaleResult:
        """Upscale with chained models.

        Args:
            image: Input BGR image.

        Returns:
            UpscaleResult from second pass.
        """
        if self._first_upscaler is None or self._second_upscaler is None:
            raise RuntimeError("Call load() first")

        start_time = time.time()
        original_h, original_w = image.shape[:2]

        # First pass
        result1 = self._first_upscaler.upscale(image)

        # Second pass
        result2 = self._second_upscaler.upscale(result1.image)

        total_time = time.time() - start_time

        return UpscaleResult(
            image=result2.image,
            original_size=(original_w, original_h),
            upscaled_size=result2.upscaled_size,
            scale_factor=result2.upscaled_size[0] / original_w,
            model_used=f"{self._first_upscaler.model_name}+{self._second_upscaler.model_name}",
            processing_time=total_time,
        )

    def unload(self) -> None:
        """Unload both upscalers."""
        if self._first_upscaler:
            self._first_upscaler.unload()
        if self._second_upscaler:
            self._second_upscaler.unload()

    def __enter__(self) -> "ChainedUpscaler":
        self.load()
        return self

    def __exit__(self, *args) -> None:
        self.unload()
