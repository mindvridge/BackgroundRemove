"""Video output formats and encoding options.

Provides various output format configurations including:
- WebM with VP9 alpha channel
- ProRes 4444 with alpha
- Green screen compositing
- Background replacement
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output video formats."""

    MP4_H264 = "mp4_h264"  # Standard H.264 (no alpha)
    WEBM_VP9 = "webm_vp9"  # WebM with VP9 alpha
    PRORES_4444 = "prores_4444"  # ProRes 4444 with alpha
    MOV_PNG = "mov_png"  # MOV with PNG codec (lossless alpha)
    GREEN_SCREEN = "green_screen"  # Green screen composite
    CUSTOM_BG = "custom_bg"  # Custom background composite


class CompressionPreset(Enum):
    """Compression quality presets for alpha videos."""

    LOSSLESS = "lossless"      # 최고 품질, 큰 용량
    HIGH = "high"              # 고품질, 중간 용량
    MEDIUM = "medium"          # 중간 품질, 작은 용량
    LOW = "low"                # 낮은 품질, 매우 작은 용량
    TINY = "tiny"              # 최소 용량 (품질 희생)


# Compression preset configurations for VP9 with alpha
# est_bitrate_mbps: 예상 비트레이트 (1080p 기준, Mbps)
COMPRESSION_PRESETS: dict[CompressionPreset, dict] = {
    CompressionPreset.LOSSLESS: {
        "crf": 0,
        "extra_args": ["-lossless", "1"],
        "scale": 1.0,
        "description": "무손실 (최대 용량)",
        "est_bitrate_mbps": 80.0,  # 1080p 기준
    },
    CompressionPreset.HIGH: {
        "crf": 20,
        "extra_args": ["-b:v", "0", "-deadline", "good", "-cpu-used", "1"],
        "scale": 1.0,
        "description": "고품질 (권장)",
        "est_bitrate_mbps": 8.0,
    },
    CompressionPreset.MEDIUM: {
        "crf": 32,
        "extra_args": ["-b:v", "0", "-deadline", "good", "-cpu-used", "2"],
        "scale": 1.0,
        "description": "중간 품질",
        "est_bitrate_mbps": 3.0,
    },
    CompressionPreset.LOW: {
        "crf": 40,
        "extra_args": ["-b:v", "0", "-deadline", "realtime", "-cpu-used", "4"],
        "scale": 0.75,
        "description": "낮은 품질 (작은 용량)",
        "est_bitrate_mbps": 1.2,
    },
    CompressionPreset.TINY: {
        "crf": 50,
        "extra_args": ["-b:v", "0", "-deadline", "realtime", "-cpu-used", "5"],
        "scale": 0.5,
        "description": "최소 용량",
        "est_bitrate_mbps": 0.4,
    },
}


@dataclass
class CodecConfig:
    """FFmpeg codec configuration."""

    codec: str
    pixel_format: str
    container: str
    extra_args: list[str] = field(default_factory=list)
    supports_alpha: bool = False


# Predefined codec configurations
CODEC_CONFIGS: dict[OutputFormat, CodecConfig] = {
    OutputFormat.MP4_H264: CodecConfig(
        codec="libx264",
        pixel_format="yuv420p",
        container="mp4",
        extra_args=["-crf", "23", "-preset", "medium"],
        supports_alpha=False,
    ),
    OutputFormat.WEBM_VP9: CodecConfig(
        codec="libvpx-vp9",
        pixel_format="yuva420p",
        container="webm",
        extra_args=[
            "-crf", "30",
            "-b:v", "0",
            "-deadline", "good",
            "-cpu-used", "2",
            "-auto-alt-ref", "0",
        ],
        supports_alpha=True,
    ),
    OutputFormat.PRORES_4444: CodecConfig(
        codec="prores_ks",
        pixel_format="yuva444p10le",
        container="mov",
        extra_args=["-profile:v", "4444", "-vendor", "apl0"],
        supports_alpha=True,
    ),
    OutputFormat.MOV_PNG: CodecConfig(
        codec="png",
        pixel_format="rgba",
        container="mov",
        extra_args=[],
        supports_alpha=True,
    ),
    OutputFormat.GREEN_SCREEN: CodecConfig(
        codec="libx264",
        pixel_format="yuv420p",
        container="mp4",
        extra_args=["-crf", "18", "-preset", "medium"],
        supports_alpha=False,
    ),
    OutputFormat.CUSTOM_BG: CodecConfig(
        codec="libx264",
        pixel_format="yuv420p",
        container="mp4",
        extra_args=["-crf", "18", "-preset", "medium"],
        supports_alpha=False,
    ),
}


@dataclass
class OutputConfig:
    """Configuration for video output."""

    format: OutputFormat = OutputFormat.MP4_H264
    fps: float = 30.0

    # Quality settings
    crf: int | None = None  # Override default CRF
    bitrate: str | None = None  # e.g., "10M" for 10 Mbps

    # Compression preset (for alpha formats like WebM VP9)
    compression_preset: CompressionPreset | None = None

    # Alpha/matting settings
    alpha_threshold: int = 0  # 0-100, percentage threshold for alpha
    edge_softness: int = 0  # 0-20, blur radius for edge softening

    # Green screen settings
    green_screen_color: tuple[int, int, int] = (0, 177, 64)

    # Background settings
    background_path: str | Path | None = None
    background_blur: int = 0  # Blur amount for background

    # Audio settings
    copy_audio: bool = True
    audio_source: str | Path | None = None

    # Output settings
    overwrite: bool = True


def apply_alpha_processing(
    alpha: "ndarray",
    threshold: int = 0,
    softness: int = 0,
) -> "ndarray":
    """Apply threshold and softness to alpha matte.

    Args:
        alpha: Alpha matte (H, W) with values 0-255.
        threshold: Threshold value (-100 to 100).
            Negative: boost/expand (keep more foreground)
            Positive: cut (remove more background)
        softness: Blur radius for edge softening (0-20).

    Returns:
        Processed alpha matte.
    """
    import cv2

    result = alpha.astype(np.float32)

    if threshold < 0:
        # Negative threshold: BOOST alpha (keep more foreground)
        # Apply power curve to boost semi-transparent pixels toward opaque
        boost_strength = abs(threshold) / 100.0  # 0.0 to 1.0
        # Normalize to 0-1
        normalized = result / 255.0
        # Apply power curve: lower power = more boost
        # power ranges from 1.0 (no boost) to 0.3 (strong boost)
        power = 1.0 - (boost_strength * 0.7)
        boosted = np.power(normalized, power)
        result = (boosted * 255.0).clip(0, 255)

    elif threshold > 0:
        # Positive threshold: CUT alpha (remove more background)
        # Convert percentage to 0-255 range
        thresh_value = int(threshold * 255 / 100)
        # Create mask for values below threshold
        mask = result < thresh_value
        result[mask] = 0
        # Rescale remaining values to use full range
        if thresh_value < 255:
            above_mask = result >= thresh_value
            result[above_mask] = np.clip(
                ((result[above_mask] - thresh_value) * 255 / (255 - thresh_value)),
                0, 255
            )

    result = result.astype(np.uint8)

    # Apply edge softness (Gaussian blur)
    if softness > 0:
        kernel_size = softness * 2 + 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

    return result


class Compositor(ABC):
    """Abstract base class for frame compositors."""

    @abstractmethod
    def composite(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Composite foreground with background.

        Args:
            foreground: RGB foreground image (H, W, 3).
            alpha: Alpha matte (H, W) with values 0-255.

        Returns:
            Composited RGB image (H, W, 3).
        """
        ...


class GreenScreenCompositor(Compositor):
    """Compositor for green screen output."""

    def __init__(
        self,
        color: tuple[int, int, int] = (0, 177, 64),
    ) -> None:
        """Initialize green screen compositor.

        Args:
            color: RGB color for green screen background.
        """
        self.color = color

    def composite(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Composite foreground onto green screen.

        Args:
            foreground: RGB foreground image (H, W, 3).
            alpha: Alpha matte (H, W) with values 0-255.

        Returns:
            Composited RGB image with green background.
        """
        # Normalize alpha to 0-1
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        # Create green background
        background = np.full_like(foreground, self.color, dtype=np.uint8)

        # Alpha blend
        output = (
            foreground.astype(np.float32) * alpha_norm +
            background.astype(np.float32) * (1 - alpha_norm)
        ).astype(np.uint8)

        return output


class ImageBackgroundCompositor(Compositor):
    """Compositor for static image background replacement."""

    def __init__(
        self,
        background_path: str | Path,
        blur_amount: int = 0,
    ) -> None:
        """Initialize image background compositor.

        Args:
            background_path: Path to background image.
            blur_amount: Gaussian blur kernel size (0 = no blur).
        """
        self.background_path = Path(background_path)
        self.blur_amount = blur_amount
        self._background: ndarray | None = None
        self._cached_size: tuple[int, int] | None = None

    def _load_and_resize(self, target_h: int, target_w: int) -> ndarray:
        """Load and resize background image to target size.

        Args:
            target_h: Target height.
            target_w: Target width.

        Returns:
            Resized background image.
        """
        if (
            self._background is not None and
            self._cached_size == (target_h, target_w)
        ):
            return self._background

        # Load image
        bg = cv2.imread(str(self.background_path))
        if bg is None:
            raise FileNotFoundError(f"Cannot load background: {self.background_path}")

        # Convert BGR to RGB
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        # Resize to target size (cover mode)
        bg_h, bg_w = bg.shape[:2]
        scale = max(target_h / bg_h, target_w / bg_w)
        new_h, new_w = int(bg_h * scale), int(bg_w * scale)
        bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop
        start_y = (new_h - target_h) // 2
        start_x = (new_w - target_w) // 2
        bg = bg[start_y:start_y + target_h, start_x:start_x + target_w]

        # Apply blur if specified
        if self.blur_amount > 0:
            kernel_size = self.blur_amount * 2 + 1
            bg = cv2.GaussianBlur(bg, (kernel_size, kernel_size), 0)

        self._background = bg
        self._cached_size = (target_h, target_w)

        return bg

    def composite(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Composite foreground onto image background.

        Args:
            foreground: RGB foreground image (H, W, 3).
            alpha: Alpha matte (H, W) with values 0-255.

        Returns:
            Composited RGB image.
        """
        h, w = foreground.shape[:2]
        background = self._load_and_resize(h, w)

        # Normalize alpha
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        # Alpha blend
        output = (
            foreground.astype(np.float32) * alpha_norm +
            background.astype(np.float32) * (1 - alpha_norm)
        ).astype(np.uint8)

        return output


class VideoBackgroundCompositor(Compositor):
    """Compositor for video background replacement."""

    def __init__(
        self,
        background_path: str | Path,
        blur_amount: int = 0,
        loop: bool = True,
    ) -> None:
        """Initialize video background compositor.

        Args:
            background_path: Path to background video.
            blur_amount: Gaussian blur kernel size (0 = no blur).
            loop: Loop video when it ends.
        """
        self.background_path = Path(background_path)
        self.blur_amount = blur_amount
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._cached_frame: ndarray | None = None
        self._target_size: tuple[int, int] | None = None

    def _ensure_open(self) -> None:
        """Ensure video capture is open."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.background_path))
            if not self._cap.isOpened():
                raise FileNotFoundError(
                    f"Cannot open background video: {self.background_path}"
                )

    def _read_next_frame(self, target_h: int, target_w: int) -> ndarray:
        """Read and process next frame from background video.

        Args:
            target_h: Target height.
            target_w: Target width.

        Returns:
            Processed background frame.
        """
        self._ensure_open()

        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if not ret:
                # Return cached frame or black
                if self._cached_frame is not None:
                    return self._cached_frame
                return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to target size
        frame_h, frame_w = frame.shape[:2]
        scale = max(target_h / frame_h, target_w / frame_w)
        new_h, new_w = int(frame_h * scale), int(frame_w * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop
        start_y = (new_h - target_h) // 2
        start_x = (new_w - target_w) // 2
        frame = frame[start_y:start_y + target_h, start_x:start_x + target_w]

        # Apply blur if specified
        if self.blur_amount > 0:
            kernel_size = self.blur_amount * 2 + 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        self._cached_frame = frame
        self._target_size = (target_h, target_w)

        return frame

    def composite(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Composite foreground onto video background.

        Args:
            foreground: RGB foreground image (H, W, 3).
            alpha: Alpha matte (H, W) with values 0-255.

        Returns:
            Composited RGB image.
        """
        h, w = foreground.shape[:2]
        background = self._read_next_frame(h, w)

        # Normalize alpha
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        # Alpha blend
        output = (
            foreground.astype(np.float32) * alpha_norm +
            background.astype(np.float32) * (1 - alpha_norm)
        ).astype(np.uint8)

        return output

    def reset(self) -> None:
        """Reset video to beginning."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        """Destructor to release resources."""
        self.release()


class FFmpegEncoder:
    """FFmpeg-based video encoder with various format support."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        config: OutputConfig,
    ) -> None:
        """Initialize FFmpeg encoder.

        Args:
            output_path: Path for output video.
            width: Video width.
            height: Video height.
            config: Output configuration.
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.config = config
        self._process: subprocess.Popen | None = None
        self._compositor: Compositor | None = None
        self._frames_written = 0

        # Initialize compositor if needed
        self._setup_compositor()

    @property
    def frames_written(self) -> int:
        """Number of frames written."""
        return self._frames_written

    def _setup_compositor(self) -> None:
        """Set up compositor based on output format."""
        fmt = self.config.format

        if fmt == OutputFormat.GREEN_SCREEN:
            self._compositor = GreenScreenCompositor(
                color=self.config.green_screen_color
            )
        elif fmt == OutputFormat.CUSTOM_BG and self.config.background_path:
            bg_path = Path(self.config.background_path)
            if bg_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
                self._compositor = VideoBackgroundCompositor(
                    bg_path,
                    blur_amount=self.config.background_blur,
                )
            else:
                self._compositor = ImageBackgroundCompositor(
                    bg_path,
                    blur_amount=self.config.background_blur,
                )

    def _get_codec_config(self) -> CodecConfig:
        """Get codec configuration for output format.

        Returns:
            CodecConfig for the selected format.
        """
        return CODEC_CONFIGS[self.config.format]

    def _build_ffmpeg_command(self) -> list[str]:
        """Build FFmpeg command for encoding.

        Returns:
            FFmpeg command as list of arguments.
        """
        codec_config = self._get_codec_config()

        # Determine input pixel format based on alpha support
        if codec_config.supports_alpha:
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

        # Add audio source if specified
        if self.config.copy_audio and self.config.audio_source:
            audio_path = Path(self.config.audio_source)
            if audio_path.exists():
                cmd.extend(["-i", str(audio_path)])

        # Output codec
        cmd.extend(["-c:v", codec_config.codec])

        # Pixel format
        cmd.extend(["-pix_fmt", codec_config.pixel_format])

        # Quality settings - override if specified in config
        if self.config.bitrate:
            cmd.extend(["-b:v", self.config.bitrate])
        elif self.config.crf is not None:
            cmd.extend(["-crf", str(self.config.crf)])
        elif self.config.compression_preset is not None and codec_config.supports_alpha:
            # Apply compression preset for alpha formats (WebM VP9)
            preset = COMPRESSION_PRESETS[self.config.compression_preset]
            cmd.extend(["-crf", str(preset["crf"])])
            cmd.extend(preset["extra_args"])
        else:
            # Use default extra args from codec config
            cmd.extend(codec_config.extra_args)

        # Audio settings
        if self.config.copy_audio and self.config.audio_source:
            # Use appropriate audio codec for container
            if codec_config.container == "webm":
                cmd.extend(["-c:a", "libopus", "-b:a", "128k"])
            else:
                cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])

        # Ensure correct extension
        output_path = self.output_path
        expected_ext = f".{codec_config.container}"
        if output_path.suffix.lower() != expected_ext:
            output_path = output_path.with_suffix(expected_ext)
            self.output_path = output_path

        cmd.append(str(output_path))

        return cmd

    def start(self) -> None:
        """Start the FFmpeg encoder process."""
        if self._process is not None:
            raise RuntimeError("Encoder already started")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_ffmpeg_command()
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info(f"Encoder started: {self.output_path}")

    def write_frame(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> None:
        """Write a frame to the encoder.

        Args:
            foreground: RGB foreground image (H, W, 3).
            alpha: Alpha matte (H, W) with values 0-255.

        Raises:
            RuntimeError: If encoder is not started.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Encoder not started. Call start() first.")

        codec_config = self._get_codec_config()

        # Composite if needed
        if self._compositor is not None:
            frame = self._compositor.composite(foreground, alpha)
        elif codec_config.supports_alpha:
            # Create RGBA frame
            if len(alpha.shape) == 2:
                alpha = alpha[:, :, np.newaxis]
            frame = np.dstack([foreground, alpha])
        else:
            frame = foreground

        # Write to FFmpeg stdin
        try:
            self._process.stdin.write(frame.tobytes())
            self._frames_written += 1
        except BrokenPipeError as e:
            raise RuntimeError("FFmpeg pipe broken") from e

    def stop(self) -> None:
        """Stop the encoder and finalize video."""
        if self._process is None:
            return

        # Close stdin
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass

        # Wait for FFmpeg to finish
        try:
            stdout, stderr = self._process.communicate(timeout=60)
            if self._process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
        except subprocess.TimeoutExpired:
            self._process.kill()
            logger.warning("FFmpeg process killed due to timeout")

        self._process = None

        # Release compositor resources
        if isinstance(self._compositor, VideoBackgroundCompositor):
            self._compositor.release()

        logger.info(f"Encoder stopped. Frames written: {self._frames_written}")

    def __enter__(self) -> "FFmpegEncoder":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def create_compositor(config: OutputConfig) -> Compositor | None:
    """Factory function to create appropriate compositor.

    Args:
        config: Output configuration.

    Returns:
        Compositor instance or None if not needed.
    """
    if config.format == OutputFormat.GREEN_SCREEN:
        return GreenScreenCompositor(config.green_screen_color)

    elif config.format == OutputFormat.CUSTOM_BG and config.background_path:
        bg_path = Path(config.background_path)

        # Check if background is video or image
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        if bg_path.suffix.lower() in video_extensions:
            return VideoBackgroundCompositor(
                bg_path,
                blur_amount=config.background_blur,
            )
        else:
            return ImageBackgroundCompositor(
                bg_path,
                blur_amount=config.background_blur,
            )

    return None
