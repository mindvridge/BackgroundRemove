"""Video writer with thread-based FFmpeg encoding.

Implements consumer pattern for video frame writing using
a background thread and FFmpeg for encoding.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


@dataclass
class WriterConfig:
    """Configuration for VideoWriter."""

    # Queue settings
    queue_size: int = 64
    timeout: float = 1.0

    # Video settings
    fps: float = 30.0
    codec: str = "libx264"
    pixel_format: str = "yuv420p"  # For RGB output
    alpha_pixel_format: str = "yuva420p"  # For RGBA output

    # Encoding settings
    crf: int = 23  # Quality (0-51, lower = better)
    preset: str = "medium"  # ultrafast, fast, medium, slow, veryslow
    bitrate: str | None = None  # e.g., "5M" for 5 Mbps (overrides CRF)

    # Audio settings
    copy_audio: bool = True
    audio_source: str | Path | None = None  # Source video for audio

    # Output settings
    overwrite: bool = True


class VideoWriter:
    """Thread-based video writer using FFmpeg (Consumer).

    Writes video frames using FFmpeg in a background thread,
    consuming frames from a queue for non-blocking operation.

    Example:
        >>> writer = VideoWriter("output.mp4", width=1920, height=1080)
        >>> writer.start()
        >>> for frame in processed_frames:
        ...     writer.write(frame)
        >>> writer.stop()
    """

    # Sentinel value to signal end of stream
    END_OF_STREAM = None

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        config: WriterConfig | None = None,
        has_alpha: bool = False,
    ) -> None:
        """Initialize video writer.

        Args:
            output_path: Path for output video file.
            width: Output video width.
            height: Output video height.
            config: Writer configuration.
            has_alpha: Whether frames have alpha channel (RGBA).
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.config = config or WriterConfig()
        self.has_alpha = has_alpha

        self._queue: Queue[ndarray | None] = Queue(maxsize=self.config.queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None
        self._started = False
        self._frames_written = 0
        self._error: Exception | None = None

    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        return self._frames_written

    @property
    def is_running(self) -> bool:
        """Check if writer thread is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def queue_size(self) -> int:
        """Current number of frames in queue."""
        return self._queue.qsize()

    def _build_ffmpeg_command(self) -> list[str]:
        """Build FFmpeg command for video encoding.

        Returns:
            FFmpeg command as list of arguments.
        """
        config = self.config

        # Determine input pixel format
        if self.has_alpha:
            input_pix_fmt = "rgba"
            output_pix_fmt = config.alpha_pixel_format
            # Use codec that supports alpha
            codec = "png" if config.codec == "libx264" else config.codec
        else:
            input_pix_fmt = "rgb24"
            output_pix_fmt = config.pixel_format
            codec = config.codec

        cmd = [
            "ffmpeg",
            "-y" if config.overwrite else "-n",
            # Input settings
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", input_pix_fmt,
            "-s", f"{self.width}x{self.height}",
            "-r", str(config.fps),
            "-i", "-",  # Read from stdin
        ]

        # Add audio from source if requested
        if config.copy_audio and config.audio_source:
            audio_path = Path(config.audio_source)
            if audio_path.exists():
                cmd.extend(["-i", str(audio_path)])

        # Output codec settings
        cmd.extend(["-c:v", codec])

        # Quality settings
        if config.bitrate:
            cmd.extend(["-b:v", config.bitrate])
        elif codec == "libx264":
            cmd.extend(["-crf", str(config.crf)])
            cmd.extend(["-preset", config.preset])

        # Pixel format
        if codec in ("libx264", "libx265"):
            cmd.extend(["-pix_fmt", output_pix_fmt])

        # Audio settings
        if config.copy_audio and config.audio_source:
            cmd.extend(["-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0?"])

        # Output file
        cmd.append(str(self.output_path))

        return cmd

    def _writer_thread(self) -> None:
        """Background thread for writing video frames."""
        try:
            # Build and start FFmpeg process
            cmd = self._build_ffmpeg_command()
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            while not self._stop_event.is_set():
                try:
                    frame = self._queue.get(timeout=self.config.timeout)
                except Empty:
                    continue

                # Check for end of stream sentinel
                if frame is None:
                    break

                # Write frame to FFmpeg stdin
                try:
                    if self._process.stdin is not None:
                        self._process.stdin.write(frame.tobytes())
                        self._frames_written += 1
                except BrokenPipeError:
                    logger.error("FFmpeg pipe broken")
                    break

        except Exception as e:
            logger.error(f"Writer thread error: {e}")
            self._error = e
        finally:
            # Close FFmpeg process
            if self._process is not None:
                if self._process.stdin is not None:
                    try:
                        self._process.stdin.close()
                    except Exception:
                        pass

                # Wait for FFmpeg to finish
                try:
                    stdout, stderr = self._process.communicate(timeout=30)
                    if self._process.returncode != 0:
                        logger.error(f"FFmpeg error: {stderr.decode()}")
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    logger.warning("FFmpeg process killed due to timeout")

            logger.debug("Writer thread finished")

    def start(self) -> None:
        """Start the writer thread.

        Raises:
            RuntimeError: If writer is already started.
        """
        if self._started:
            raise RuntimeError("Writer already started")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()
        self._started = True

        logger.info(f"Writer started: {self.output_path}")

    def stop(self) -> None:
        """Stop the writer thread and finalize video."""
        if not self._started:
            return

        # Signal end of stream
        try:
            self._queue.put(self.END_OF_STREAM, timeout=self.config.timeout)
        except Full:
            pass

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=60.0)
            if self._thread.is_alive():
                logger.warning("Writer thread did not stop gracefully")

        self._started = False
        logger.info(f"Writer stopped. Frames written: {self._frames_written}")

    def write(self, frame: ndarray, timeout: float | None = None) -> bool:
        """Write a frame to the video.

        Args:
            frame: Frame as numpy array (H, W, C).
            timeout: Timeout in seconds. Uses config timeout if None.

        Returns:
            True if frame was queued successfully, False if queue is full.

        Raises:
            RuntimeError: If writer is not started or an error occurred.
        """
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")

        if self._error is not None:
            raise RuntimeError(f"Writer error: {self._error}")

        timeout = timeout if timeout is not None else self.config.timeout

        try:
            self._queue.put(frame, timeout=timeout)
            return True
        except Full:
            return False

    def __enter__(self) -> "VideoWriter":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        """String representation."""
        status = "running" if self.is_running else "stopped"
        return (
            f"VideoWriter(path={self.output_path!r}, "
            f"{self.width}x{self.height}, status={status})"
        )


class ImageSequenceWriter:
    """Writer for saving frames as image sequence.

    Alternative to video writer for saving individual frames.
    """

    def __init__(
        self,
        output_dir: str | Path,
        name_pattern: str = "frame_{:06d}.png",
        queue_size: int = 64,
    ) -> None:
        """Initialize image sequence writer.

        Args:
            output_dir: Output directory for images.
            name_pattern: Filename pattern with format placeholder.
            queue_size: Queue size for buffering.
        """
        self.output_dir = Path(output_dir)
        self.name_pattern = name_pattern
        self._queue: Queue[tuple[int, ndarray] | None] = Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._frames_written = 0

    def _writer_thread(self) -> None:
        """Background thread for saving images."""
        import cv2

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break

            idx, frame = item
            filename = self.name_pattern.format(idx)
            filepath = self.output_dir / filename

            cv2.imwrite(str(filepath), frame)
            self._frames_written += 1

    def start(self) -> None:
        """Start the writer thread."""
        if self._started:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop the writer thread."""
        if not self._started:
            return

        try:
            self._queue.put(None, timeout=1.0)
        except Full:
            pass

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

        self._started = False

    def write(self, frame: ndarray, index: int | None = None) -> bool:
        """Write a frame to the sequence.

        Args:
            frame: Frame as numpy array.
            index: Frame index. Auto-incremented if None.

        Returns:
            True if successful.
        """
        if index is None:
            index = self._frames_written

        try:
            self._queue.put((index, frame), timeout=1.0)
            return True
        except Full:
            return False

    def __enter__(self) -> "ImageSequenceWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class AdvancedVideoWriter:
    """Advanced video writer with multiple output format support.

    Supports various output formats including:
    - WebM with VP9 alpha channel
    - ProRes 4444 with alpha
    - Green screen compositing
    - Custom background replacement

    Example:
        >>> from src.pipeline.output import OutputConfig, OutputFormat
        >>> config = OutputConfig(format=OutputFormat.WEBM_VP9)
        >>> writer = AdvancedVideoWriter("output.webm", 1920, 1080, config)
        >>> writer.start()
        >>> writer.write_with_alpha(foreground, alpha)
        >>> writer.stop()
    """

    END_OF_STREAM = None

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        config: "OutputConfig | None" = None,
        queue_size: int = 64,
    ) -> None:
        """Initialize advanced video writer.

        Args:
            output_path: Path for output video.
            width: Video width.
            height: Video height.
            config: Output configuration.
            queue_size: Frame queue size.
        """
        from src.pipeline.output import OutputConfig, OutputFormat, CODEC_CONFIGS

        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.config = config or OutputConfig()
        self.queue_size = queue_size

        # Get codec config
        self._codec_config = CODEC_CONFIGS[self.config.format]

        # Ensure correct file extension
        expected_ext = f".{self._codec_config.container}"
        if self.output_path.suffix.lower() != expected_ext:
            self.output_path = self.output_path.with_suffix(expected_ext)

        self._queue: Queue[tuple[ndarray, ndarray] | None] = Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None
        self._compositor = None
        self._started = False
        self._frames_written = 0
        self._error: Exception | None = None

        # Setup compositor
        self._setup_compositor()

    def _setup_compositor(self) -> None:
        """Set up compositor based on output format."""
        from src.pipeline.output import (
            OutputFormat,
            GreenScreenCompositor,
            ImageBackgroundCompositor,
            VideoBackgroundCompositor,
        )

        fmt = self.config.format

        if fmt == OutputFormat.GREEN_SCREEN:
            self._compositor = GreenScreenCompositor(
                color=self.config.green_screen_color
            )
        elif fmt == OutputFormat.CUSTOM_BG and self.config.background_path:
            bg_path = Path(self.config.background_path)
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

            if bg_path.suffix.lower() in video_exts:
                self._compositor = VideoBackgroundCompositor(
                    bg_path,
                    blur_amount=self.config.background_blur,
                )
            else:
                self._compositor = ImageBackgroundCompositor(
                    bg_path,
                    blur_amount=self.config.background_blur,
                )

    @property
    def frames_written(self) -> int:
        """Number of frames written."""
        return self._frames_written

    @property
    def is_running(self) -> bool:
        """Check if writer is running."""
        return self._thread is not None and self._thread.is_alive()

    def _build_ffmpeg_command(self) -> list[str]:
        """Build FFmpeg command for encoding.

        Returns:
            FFmpeg command as list of arguments.
        """
        codec_config = self._codec_config

        # Determine input pixel format
        if codec_config.supports_alpha and self._compositor is None:
            input_pix_fmt = "rgba"
        else:
            input_pix_fmt = "rgb24"

        cmd = [
            "ffmpeg",
            "-y" if self.config.overwrite else "-n",
            # Input settings
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", input_pix_fmt,
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.config.fps),
            "-i", "-",  # stdin
        ]

        # Add audio source
        if self.config.copy_audio and self.config.audio_source:
            audio_path = Path(self.config.audio_source)
            if audio_path.exists():
                cmd.extend(["-i", str(audio_path)])

        # Output codec
        cmd.extend(["-c:v", codec_config.codec])

        # Pixel format
        cmd.extend(["-pix_fmt", codec_config.pixel_format])

        # Quality/encoding settings
        if self.config.bitrate:
            cmd.extend(["-b:v", self.config.bitrate])
        elif self.config.crf is not None:
            cmd.extend(["-crf", str(self.config.crf)])
        else:
            cmd.extend(codec_config.extra_args)

        # Audio settings
        if self.config.copy_audio and self.config.audio_source:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])

        cmd.append(str(self.output_path))

        return cmd

    def _process_frame(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Process frame for output.

        Args:
            foreground: RGB foreground (H, W, 3).
            alpha: Alpha matte (H, W).

        Returns:
            Processed frame ready for encoding.
        """
        import numpy as np

        if self._compositor is not None:
            # Composite with background
            return self._compositor.composite(foreground, alpha)
        elif self._codec_config.supports_alpha:
            # Create RGBA frame
            if len(alpha.shape) == 2:
                alpha = alpha[:, :, np.newaxis]
            return np.dstack([foreground, alpha])
        else:
            return foreground

    def _writer_thread(self) -> None:
        """Background thread for writing frames."""
        import numpy as np

        try:
            cmd = self._build_ffmpeg_command()
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=1.0)
                except Empty:
                    continue

                if item is None:
                    break

                foreground, alpha = item
                frame = self._process_frame(foreground, alpha)

                try:
                    if self._process.stdin is not None:
                        self._process.stdin.write(frame.tobytes())
                        self._frames_written += 1
                except BrokenPipeError:
                    logger.error("FFmpeg pipe broken")
                    break

        except Exception as e:
            logger.error(f"Writer thread error: {e}")
            self._error = e
        finally:
            if self._process is not None:
                if self._process.stdin is not None:
                    try:
                        self._process.stdin.close()
                    except Exception:
                        pass

                try:
                    stdout, stderr = self._process.communicate(timeout=60)
                    if self._process.returncode != 0:
                        logger.error(f"FFmpeg error: {stderr.decode()}")
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    logger.warning("FFmpeg killed due to timeout")

            # Release compositor resources
            from src.pipeline.output import VideoBackgroundCompositor
            if isinstance(self._compositor, VideoBackgroundCompositor):
                self._compositor.release()

            logger.debug("Writer thread finished")

    def start(self) -> None:
        """Start the writer thread."""
        if self._started:
            raise RuntimeError("Writer already started")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()
        self._started = True

        logger.info(f"Advanced writer started: {self.output_path}")
        logger.info(f"Format: {self.config.format.value}, Codec: {self._codec_config.codec}")

    def stop(self) -> None:
        """Stop the writer and finalize video."""
        if not self._started:
            return

        try:
            self._queue.put(self.END_OF_STREAM, timeout=5.0)
        except Full:
            pass

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=120.0)
            if self._thread.is_alive():
                logger.warning("Writer thread did not stop gracefully")

        self._started = False
        logger.info(f"Advanced writer stopped. Frames: {self._frames_written}")

    def write_with_alpha(
        self,
        foreground: ndarray,
        alpha: ndarray,
        timeout: float = 5.0,
    ) -> bool:
        """Write a frame with alpha channel.

        Args:
            foreground: RGB foreground image (H, W, 3).
            alpha: Alpha matte (H, W) with values 0-255.
            timeout: Queue timeout in seconds.

        Returns:
            True if frame was queued successfully.
        """
        if not self._started:
            raise RuntimeError("Writer not started. Call start() first.")

        if self._error is not None:
            raise RuntimeError(f"Writer error: {self._error}")

        try:
            self._queue.put((foreground, alpha), timeout=timeout)
            return True
        except Full:
            return False

    def __enter__(self) -> "AdvancedVideoWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return (
            f"AdvancedVideoWriter(path={self.output_path!r}, "
            f"format={self.config.format.value}, status={status})"
        )
