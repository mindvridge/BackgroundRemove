"""Image upscaling module.

Provides high-quality AI-powered image and video upscaling using
state-of-the-art deep learning models.

Models available:
- Real-ESRGAN: Fast, high-quality (9.2/10), best for photographs
- SwinIR: Highest quality (9.7/10), transformer-based, best for details
- HAT: State-of-the-art (9.8/10), Hybrid Attention Transformer

Quality Presets:
- Fast: Real-ESRGAN only (~2s/image)
- Balanced: Real-ESRGAN + light processing (~5s/image)
- Quality: HAT + full processing (~15s/image)
- Ultra: Multi-pass HAT + SwinIR (~30s/image)

Example:
    >>> from src.upscale import AdvancedUpscalePipeline, QualityPreset
    >>>
    >>> # Quick quality upscale
    >>> pipeline = AdvancedUpscalePipeline(QualityPreset.QUALITY)
    >>> result = pipeline.upscale_file("input.jpg", "output.png")
    >>>
    >>> # Maximum quality
    >>> pipeline = AdvancedUpscalePipeline(QualityPreset.ULTRA)
    >>> result = pipeline.upscale(image)
    >>> print(f"Upscaled in {result.processing_time:.2f}s")

Simple API:
    >>> from src.upscale import upscale_quality, upscale_fast
    >>>
    >>> # Fast upscale
    >>> result = upscale_fast(image, scale=4)
    >>>
    >>> # High-quality upscale
    >>> result = upscale_quality(image, scale=4)
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
from src.upscale.hat import HATUpscaler
from src.upscale.preprocessing import (
    DenoiseMethod,
    ImagePreprocessor,
    PreprocessConfig,
    SharpenMethod,
    preprocess_for_upscale,
    quick_denoise,
    quick_sharpen,
)
from src.upscale.postprocessing import (
    ArtifactType,
    ColorGradingPreset,
    ImagePostprocessor,
    PostprocessConfig,
    apply_color_grade,
    postprocess_upscaled,
)
from src.upscale.advanced_pipeline import (
    AdvancedPipelineConfig,
    AdvancedUpscalePipeline,
    AdvancedUpscaleResult,
    QualityPreset,
    UpscalerType,
    compare_quality_presets,
    upscale_fast,
    upscale_quality,
    upscale_ultra,
)

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
    "HATUpscaler",
    # Basic Pipelines
    "ImageUpscalePipeline",
    "VideoUpscalePipeline",
    "VideoUpscaleResult",
    "ChainedUpscaler",
    # Advanced Pipeline
    "AdvancedUpscalePipeline",
    "AdvancedPipelineConfig",
    "AdvancedUpscaleResult",
    "QualityPreset",
    "UpscalerType",
    # Preprocessing
    "ImagePreprocessor",
    "PreprocessConfig",
    "DenoiseMethod",
    "SharpenMethod",
    # Postprocessing
    "ImagePostprocessor",
    "PostprocessConfig",
    "ArtifactType",
    "ColorGradingPreset",
    # Convenience functions
    "create_upscaler",
    "auto_select_model",
    "get_model_path",
    "upscale_fast",
    "upscale_quality",
    "upscale_ultra",
    "quick_denoise",
    "quick_sharpen",
    "preprocess_for_upscale",
    "postprocess_upscaled",
    "apply_color_grade",
    "compare_quality_presets",
]
