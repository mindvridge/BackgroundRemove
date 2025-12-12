"""Image upscaling module.

Provides high-quality AI-powered image and video upscaling using
state-of-the-art deep learning models.

Models available:
- Real-ESRGAN: Fast, high-quality (9.2/10), best for photographs
- SwinIR: High quality (9.7/10), transformer-based, best for details
- HAT: State-of-the-art (9.8/10), Hybrid Attention Transformer
- SD x4 Upscaler: Diffusion-based (10/10), best for creative upscaling
- SUPIR: Restoration + upscaling (10/10), best for degraded images

Quality Presets:
- Fast: Real-ESRGAN only (~2s/image)
- Balanced: Real-ESRGAN + light processing (~5s/image)
- Quality: HAT + full processing (~15s/image)
- Ultra: Multi-pass HAT + SwinIR (~30s/image)
- Diffusion: SD x4 upscaler (~20s/image, 8GB+ VRAM)
- Maximum: SUPIR restoration (~45s/image, 12GB+ VRAM)

Example:
    >>> from src.upscale import AdvancedUpscalePipeline, QualityPreset
    >>>
    >>> # Quick quality upscale
    >>> pipeline = AdvancedUpscalePipeline(QualityPreset.QUALITY)
    >>> result = pipeline.upscale_file("input.jpg", "output.png")
    >>>
    >>> # Maximum quality with SUPIR
    >>> pipeline = AdvancedUpscalePipeline(QualityPreset.MAXIMUM)
    >>> result = pipeline.upscale(degraded_image)
    >>> print(f"Restored in {result.processing_time:.2f}s")

Simple API:
    >>> from src.upscale import upscale_quality, sd_upscale, supir_restore
    >>>
    >>> # High-quality upscale
    >>> result = upscale_quality(image, scale=4)
    >>>
    >>> # Diffusion-based upscale
    >>> result = sd_upscale(image, prompt="detailed photo")
    >>>
    >>> # Restore degraded image
    >>> result = supir_restore(old_photo, scale=2)
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

# Lazy imports for optional diffusion-based upscalers
# These require additional dependencies (diffusers, transformers)


def _get_sd_upscaler():
    """Lazy import for SD upscaler."""
    from src.upscale.sd_upscaler import (
        SDUpscaleConfig,
        SDUpscaleModel,
        SDXLRefinerUpscaler,
        StableDiffusionUpscaler,
        sd_upscale,
        sd_upscale_with_refiner,
    )
    return {
        "StableDiffusionUpscaler": StableDiffusionUpscaler,
        "SDXLRefinerUpscaler": SDXLRefinerUpscaler,
        "SDUpscaleConfig": SDUpscaleConfig,
        "SDUpscaleModel": SDUpscaleModel,
        "sd_upscale": sd_upscale,
        "sd_upscale_with_refiner": sd_upscale_with_refiner,
    }


def _get_supir_upscaler():
    """Lazy import for SUPIR upscaler."""
    from src.upscale.supir_upscaler import (
        DegradationLevel,
        SUPIRConfig,
        SUPIRModel,
        SUPIRUpscaler,
        restore_old_photo,
        supir_restore,
    )
    return {
        "SUPIRUpscaler": SUPIRUpscaler,
        "SUPIRConfig": SUPIRConfig,
        "SUPIRModel": SUPIRModel,
        "DegradationLevel": DegradationLevel,
        "supir_restore": supir_restore,
        "restore_old_photo": restore_old_photo,
    }


# Dynamic attribute access for lazy loading
def __getattr__(name):
    """Lazy load diffusion-based upscalers."""
    sd_components = {
        "StableDiffusionUpscaler",
        "SDXLRefinerUpscaler",
        "SDUpscaleConfig",
        "SDUpscaleModel",
        "sd_upscale",
        "sd_upscale_with_refiner",
    }
    supir_components = {
        "SUPIRUpscaler",
        "SUPIRConfig",
        "SUPIRModel",
        "DegradationLevel",
        "supir_restore",
        "restore_old_photo",
    }

    if name in sd_components:
        return _get_sd_upscaler()[name]
    elif name in supir_components:
        return _get_supir_upscaler()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "BaseUpscaler",
    "UpscaleConfig",
    "UpscaleModel",
    "UpscaleResult",
    "DegradationType",
    # Upscalers (always available)
    "RealESRGANUpscaler",
    "SwinIRUpscaler",
    "HATUpscaler",
    # Diffusion upscalers (lazy loaded)
    "StableDiffusionUpscaler",
    "SDXLRefinerUpscaler",
    "SDUpscaleConfig",
    "SDUpscaleModel",
    "SUPIRUpscaler",
    "SUPIRConfig",
    "SUPIRModel",
    "DegradationLevel",
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
    "sd_upscale",
    "sd_upscale_with_refiner",
    "supir_restore",
    "restore_old_photo",
    "quick_denoise",
    "quick_sharpen",
    "preprocess_for_upscale",
    "postprocess_upscaled",
    "apply_color_grade",
    "compare_quality_presets",
]
