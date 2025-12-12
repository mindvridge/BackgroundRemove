"""Image upscaling module.

Provides high-quality AI-powered image and video upscaling using
state-of-the-art deep learning models.

Models available:
- Real-ESRGAN: Fast, high-quality (9.2/10), best for photographs
- SwinIR: Highest quality (9.7/10), transformer-based, best for details

Example:
    >>> from src.upscale import ImageUpscalePipeline, UpscaleConfig, UpscaleModel
    >>>
    >>> # Quick upscale with default settings
    >>> pipeline = ImageUpscalePipeline()
    >>> result = pipeline.upscale_file("input.jpg", "output.png")
    >>>
    >>> # High-quality upscale with SwinIR
    >>> config = UpscaleConfig(
    ...     model=UpscaleModel.SWINIR_REAL_SR_X4,
    ...     scale=4,
    ...     use_fp16=True,
    ... )
    >>> pipeline = ImageUpscalePipeline(config)
    >>> result = pipeline.upscale(image)
"""

from src.upscale.base import (
    BaseUpscaler,
    DegradationType,
    UpscaleConfig,
    UpscaleModel,
    UpscaleResult,
    get_model_path,
)
from src.upscale.pipeline import (
    ChainedUpscaler,
    ImageUpscalePipeline,
    VideoUpscalePipeline,
    VideoUpscaleResult,
    auto_select_model,
    create_upscaler,
)
from src.upscale.real_esrgan import RealESRGANUpscaler
from src.upscale.swinir import SwinIRUpscaler

__all__ = [
    # Base classes
    "BaseUpscaler",
    "UpscaleConfig",
    "UpscaleModel",
    "UpscaleResult",
    "DegradationType",
    # Upscalers
    "RealESRGANUpscaler",
    "SwinIRUpscaler",
    # Pipelines
    "ImageUpscalePipeline",
    "VideoUpscalePipeline",
    "VideoUpscaleResult",
    "ChainedUpscaler",
    # Utilities
    "create_upscaler",
    "auto_select_model",
    "get_model_path",
]
