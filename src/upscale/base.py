"""Image upscaling module.

Provides high-quality image upscaling using state-of-the-art AI models:
- Real-ESRGAN: Fast, high-quality (9.2/10), best for photographs
- SwinIR: Highest quality (9.7/10), transformer-based architecture

Based on research comparing AI upscalers:
- SwinIR uses Swin Transformer for superior long-range dependency modeling
- Real-ESRGAN offers best speed/quality balance
- Both support 2x, 4x upscaling with optional face enhancement
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class UpscaleModel(Enum):
    """Available upscaling models."""

    # Real-ESRGAN variants
    REAL_ESRGAN_X4PLUS = "realesrgan-x4plus"  # General purpose, 4x
    REAL_ESRGAN_X4PLUS_ANIME = "realesrgan-x4plus-anime"  # Anime optimized
    REAL_ESRGAN_X2PLUS = "realesrgan-x2plus"  # 2x upscale

    # SwinIR variants (highest quality)
    SWINIR_REAL_SR_X4 = "swinir-real-sr-x4"  # Real-world SR, 4x
    SWINIR_CLASSICAL_SR_X4 = "swinir-classical-sr-x4"  # Classical SR
    SWINIR_LIGHTWEIGHT_X4 = "swinir-lightweight-x4"  # Fast variant

    # Specialized
    GFPGAN = "gfpgan"  # Face enhancement
    CODEFORMER = "codeformer"  # Face restoration


class DegradationType(Enum):
    """Types of image degradation to handle."""

    NONE = "none"  # Clean upscaling only
    JPEG_ARTIFACT = "jpeg"  # JPEG compression artifacts
    BLUR = "blur"  # Gaussian/motion blur
    NOISE = "noise"  # Image noise
    MIXED = "mixed"  # Multiple degradations
    AUTO = "auto"  # Auto-detect


@dataclass
class UpscaleConfig:
    """Configuration for image upscaling."""

    # Model selection
    model: UpscaleModel = UpscaleModel.REAL_ESRGAN_X4PLUS
    scale: int = 4  # Upscaling factor (2 or 4)

    # Quality settings
    tile_size: int = 512  # Tile size for memory efficiency
    tile_overlap: int = 32  # Overlap between tiles
    denoise_strength: float = 0.5  # Denoising strength (0-1)
    degradation: DegradationType = DegradationType.AUTO

    # Face enhancement
    face_enhance: bool = False
    face_enhance_model: UpscaleModel = UpscaleModel.GFPGAN
    face_enhance_weight: float = 0.5  # Blend weight with upscaled bg

    # Performance
    use_fp16: bool = True
    use_gpu: bool = True
    device_id: int = 0
    batch_size: int = 1

    # Output
    output_format: str = "png"  # png, jpg, webp
    jpeg_quality: int = 95

    # Model paths (auto-downloaded if not specified)
    model_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "upscale_models"
    )


@dataclass
class UpscaleResult:
    """Result of image upscaling."""

    image: "ndarray"
    original_size: tuple[int, int]  # (width, height)
    upscaled_size: tuple[int, int]
    scale_factor: float
    model_used: str
    processing_time: float  # seconds


class BaseUpscaler(ABC):
    """Abstract base class for image upscalers.

    All upscaler implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: UpscaleConfig) -> None:
        """Initialize upscaler.

        Args:
            config: Upscaling configuration.
        """
        self.config = config
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the upscaling model."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        ...

    @abstractmethod
    def upscale(
        self,
        image: "ndarray",
        scale: int | None = None,
    ) -> UpscaleResult:
        """Upscale a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format.
            scale: Optional scale override.

        Returns:
            UpscaleResult with upscaled image and metadata.
        """
        ...

    def upscale_batch(
        self,
        images: list["ndarray"],
        scale: int | None = None,
    ) -> list[UpscaleResult]:
        """Upscale multiple images.

        Args:
            images: List of input images.
            scale: Optional scale override.

        Returns:
            List of UpscaleResult objects.
        """
        return [self.upscale(img, scale) for img in images]

    def __enter__(self) -> "BaseUpscaler":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload()


# Model download URLs
MODEL_URLS = {
    UpscaleModel.REAL_ESRGAN_X4PLUS: {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
    },
    UpscaleModel.REAL_ESRGAN_X4PLUS_ANIME: {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
    },
    UpscaleModel.REAL_ESRGAN_X2PLUS: {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
    },
    UpscaleModel.GFPGAN: {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
    },
}


def get_model_path(model: UpscaleModel, model_dir: Path) -> Path:
    """Get path to model file, downloading if necessary.

    Args:
        model: Model to get path for.
        model_dir: Directory to store models.

    Returns:
        Path to model file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    if model not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model}")

    info = MODEL_URLS[model]
    model_path = model_dir / info["filename"]

    if not model_path.exists():
        logger.info(f"Downloading {model.value} model...")
        _download_model(info["url"], model_path)

    return model_path


def _download_model(url: str, path: Path) -> None:
    """Download model file from URL.

    Args:
        url: Download URL.
        path: Destination path.
    """
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, path)
        logger.info(f"Downloaded model to {path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise
