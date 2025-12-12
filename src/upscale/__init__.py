"""Image upscaling module.

Provides high-quality AI-powered image and video upscaling using
state-of-the-art deep learning models.

Models available:
- Real-ESRGAN: Fast, high-quality (9.2/10), best for photographs
- SwinIR: High quality (9.7/10), transformer-based, best for details
- HAT: State-of-the-art (9.8/10), Hybrid Attention Transformer
- SD x4 Upscaler: Diffusion-based (10/10), best for creative upscaling
- SUPIR: Restoration + upscaling (10/10), best for degraded images
- CodeFormer: Face restoration (10/10), best for faces
- Ultimate Pipeline: All combined (10.7/10), maximum quality

Quality Presets:
- Fast: Real-ESRGAN only (~2s/image)
- Balanced: Real-ESRGAN + light processing (~5s/image)
- Quality: HAT + full processing (~15s/image)
- Ultra: Multi-pass HAT + SwinIR (~30s/image)
- Diffusion: SD x4 upscaler (~20s/image, 8GB+ VRAM)
- Maximum: SUPIR restoration (~45s/image, 12GB+ VRAM)
- Ultimate: All techniques combined (~120s/image, 16GB+ VRAM)

Example:
    >>> from src.upscale import UltimatePipeline, UltimatePreset
    >>>
    >>> # Ultimate quality upscale (10.7/10)
    >>> pipeline = UltimatePipeline(UltimatePreset.MAXIMUM)
    >>> result = pipeline.upscale(image)
    >>>
    >>> # Quick quality upscale
    >>> from src.upscale import AdvancedUpscalePipeline, QualityPreset
    >>> pipeline = AdvancedUpscalePipeline(QualityPreset.QUALITY)
    >>> result = pipeline.upscale_file("input.jpg", "output.png")

Simple API:
    >>> from src.upscale import ultimate_upscale, restore_faces
    >>>
    >>> # Ultimate quality
    >>> result = ultimate_upscale(image, scale=4)
    >>>
    >>> # Face restoration
    >>> result = restore_faces(image, fidelity=0.5)
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


def _get_codeformer():
    """Lazy import for CodeFormer."""
    from src.upscale.codeformer import (
        CodeFormerConfig,
        CodeFormerRestorer,
        enhance_portrait,
        restore_faces,
    )
    return {
        "CodeFormerRestorer": CodeFormerRestorer,
        "CodeFormerConfig": CodeFormerConfig,
        "restore_faces": restore_faces,
        "enhance_portrait": enhance_portrait,
    }


def _get_region_segmenter():
    """Lazy import for region segmentation."""
    from src.upscale.region_segmenter import (
        Region,
        RegionSegmenter,
        RegionType,
        SegmenterConfig,
    )
    return {
        "RegionSegmenter": RegionSegmenter,
        "SegmenterConfig": SegmenterConfig,
        "RegionType": RegionType,
        "Region": Region,
    }


def _get_ensemble():
    """Lazy import for ensemble."""
    from src.upscale.ensemble import (
        BlendingMethod,
        EnsembleConfig,
        EnsembleUpscaler,
        ensemble_upscale,
    )
    return {
        "EnsembleUpscaler": EnsembleUpscaler,
        "EnsembleConfig": EnsembleConfig,
        "BlendingMethod": BlendingMethod,
        "ensemble_upscale": ensemble_upscale,
    }


def _get_ultimate():
    """Lazy import for ultimate pipeline."""
    from src.upscale.ultimate_pipeline import (
        UltimateConfig,
        UltimatePipeline,
        UltimatePreset,
        UltimateResult,
        ultimate_upscale,
        upscale_document,
        upscale_portrait,
    )
    return {
        "UltimatePipeline": UltimatePipeline,
        "UltimateConfig": UltimateConfig,
        "UltimatePreset": UltimatePreset,
        "UltimateResult": UltimateResult,
        "ultimate_upscale": ultimate_upscale,
        "upscale_portrait": upscale_portrait,
        "upscale_document": upscale_document,
    }


# Dynamic attribute access for lazy loading
def __getattr__(name):
    """Lazy load optional components."""
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
    codeformer_components = {
        "CodeFormerRestorer",
        "CodeFormerConfig",
        "restore_faces",
        "enhance_portrait",
    }
    segmenter_components = {
        "RegionSegmenter",
        "SegmenterConfig",
        "RegionType",
        "Region",
    }
    ensemble_components = {
        "EnsembleUpscaler",
        "EnsembleConfig",
        "BlendingMethod",
        "ensemble_upscale",
    }
    ultimate_components = {
        "UltimatePipeline",
        "UltimateConfig",
        "UltimatePreset",
        "UltimateResult",
        "ultimate_upscale",
        "upscale_portrait",
        "upscale_document",
    }

    if name in sd_components:
        return _get_sd_upscaler()[name]
    elif name in supir_components:
        return _get_supir_upscaler()[name]
    elif name in codeformer_components:
        return _get_codeformer()[name]
    elif name in segmenter_components:
        return _get_region_segmenter()[name]
    elif name in ensemble_components:
        return _get_ensemble()[name]
    elif name in ultimate_components:
        return _get_ultimate()[name]

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
    # Face restoration (lazy loaded)
    "CodeFormerRestorer",
    "CodeFormerConfig",
    "restore_faces",
    "enhance_portrait",
    # Region segmentation (lazy loaded)
    "RegionSegmenter",
    "SegmenterConfig",
    "RegionType",
    "Region",
    # Ensemble (lazy loaded)
    "EnsembleUpscaler",
    "EnsembleConfig",
    "BlendingMethod",
    "ensemble_upscale",
    # Ultimate Pipeline (lazy loaded)
    "UltimatePipeline",
    "UltimateConfig",
    "UltimatePreset",
    "UltimateResult",
    "ultimate_upscale",
    "upscale_portrait",
    "upscale_document",
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
