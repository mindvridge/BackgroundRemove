"""Advanced upscaling pipeline with pre/post-processing.

Provides high-quality upscaling through a complete pipeline:
1. Preprocessing: Denoise, sharpen, remove artifacts
2. Upscaling: HAT, SwinIR, Real-ESRGAN (single or chained)
3. Postprocessing: Color correction, detail enhancement, artifact removal

Quality tiers:
- Fast: Real-ESRGAN only, minimal processing (~2s/image)
- Balanced: Real-ESRGAN + light pre/post (~5s/image)
- Quality: HAT + full processing (~15s/image)
- Ultra: Multi-pass HAT + SwinIR + full processing (~30s/image)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.upscale.base import BaseUpscaler, UpscaleConfig, UpscaleModel, UpscaleResult
from src.upscale.hat import HATUpscaler
from src.upscale.postprocessing import (
    ColorGradingPreset,
    ImagePostprocessor,
    PostprocessConfig,
)
from src.upscale.preprocessing import (
    DenoiseMethod,
    ImagePreprocessor,
    PreprocessConfig,
    SharpenMethod,
)
from src.upscale.real_esrgan import RealESRGANUpscaler
from src.upscale.swinir import SwinIRUpscaler

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class QualityPreset(Enum):
    """Quality presets for the advanced pipeline."""

    FAST = "fast"  # Speed priority
    BALANCED = "balanced"  # Good balance
    QUALITY = "quality"  # Quality priority
    ULTRA = "ultra"  # Maximum quality


class UpscalerType(Enum):
    """Available upscaler types."""

    REAL_ESRGAN = "real_esrgan"
    SWINIR = "swinir"
    HAT = "hat"
    HAT_L = "hat_l"  # Large variant


@dataclass
class AdvancedPipelineConfig:
    """Configuration for advanced upscaling pipeline."""

    # Quality preset (overrides individual settings if set)
    preset: QualityPreset | None = None

    # Upscaler selection
    primary_upscaler: UpscalerType = UpscalerType.HAT
    secondary_upscaler: UpscalerType | None = None  # For chained upscaling
    scale: int = 4

    # Preprocessing
    enable_preprocessing: bool = True
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)

    # Postprocessing
    enable_postprocessing: bool = True
    postprocess_config: PostprocessConfig = field(default_factory=PostprocessConfig)

    # Multi-pass upscaling
    multi_pass: bool = False  # Use 2x+2x instead of 4x
    passes: int = 2  # Number of 2x passes for multi-pass mode

    # Performance
    use_fp16: bool = True
    use_gpu: bool = True
    tile_size: int = 512
    tile_overlap: int = 32

    # Output
    output_format: str = "png"
    jpeg_quality: int = 95


# Preset configurations
PRESET_CONFIGS = {
    QualityPreset.FAST: {
        "primary_upscaler": UpscalerType.REAL_ESRGAN,
        "secondary_upscaler": None,
        "enable_preprocessing": False,
        "enable_postprocessing": False,
        "multi_pass": False,
        "tile_size": 512,
    },
    QualityPreset.BALANCED: {
        "primary_upscaler": UpscalerType.REAL_ESRGAN,
        "secondary_upscaler": None,
        "enable_preprocessing": True,
        "preprocess_config": PreprocessConfig(
            denoise_method=DenoiseMethod.FAST_NLM,
            denoise_strength=0.3,
            sharpen_method=SharpenMethod.NONE,
        ),
        "enable_postprocessing": True,
        "postprocess_config": PostprocessConfig(
            remove_halos=True,
            enhance_details=False,
            final_sharpen=True,
            sharpen_amount=0.15,
        ),
        "multi_pass": False,
        "tile_size": 512,
    },
    QualityPreset.QUALITY: {
        "primary_upscaler": UpscalerType.HAT,
        "secondary_upscaler": None,
        "enable_preprocessing": True,
        "preprocess_config": PreprocessConfig(
            denoise_method=DenoiseMethod.FAST_NLM,
            denoise_strength=0.4,
            sharpen_method=SharpenMethod.UNSHARP_MASK,
            sharpen_amount=0.2,
            remove_jpeg_artifacts=True,
        ),
        "enable_postprocessing": True,
        "postprocess_config": PostprocessConfig(
            remove_halos=True,
            remove_ringing=True,
            enhance_details=True,
            detail_strength=0.25,
            final_sharpen=True,
            sharpen_amount=0.15,
        ),
        "multi_pass": False,
        "tile_size": 384,
    },
    QualityPreset.ULTRA: {
        "primary_upscaler": UpscalerType.HAT_L,
        "secondary_upscaler": UpscalerType.SWINIR,
        "enable_preprocessing": True,
        "preprocess_config": PreprocessConfig(
            denoise_method=DenoiseMethod.NLM,
            denoise_strength=0.5,
            sharpen_method=SharpenMethod.ADAPTIVE,
            sharpen_amount=0.25,
            remove_jpeg_artifacts=True,
            clahe_enabled=True,
            clahe_clip_limit=1.5,
        ),
        "enable_postprocessing": True,
        "postprocess_config": PostprocessConfig(
            remove_halos=True,
            remove_ringing=True,
            remove_checkerboard=True,
            enhance_details=True,
            detail_strength=0.3,
            texture_enhancement=True,
            texture_strength=0.2,
            final_sharpen=True,
            sharpen_amount=0.2,
            reduce_banding=True,
        ),
        "multi_pass": True,
        "passes": 2,
        "tile_size": 256,
    },
}


@dataclass
class AdvancedUpscaleResult(UpscaleResult):
    """Extended result with pipeline information."""

    preprocessing_time: float = 0.0
    upscaling_time: float = 0.0
    postprocessing_time: float = 0.0
    upscalers_used: list[str] = field(default_factory=list)
    passes_completed: int = 1
    quality_preset: str = "custom"


class AdvancedUpscalePipeline:
    """Advanced image upscaling pipeline with full processing chain.

    This pipeline provides the highest quality upscaling by combining:
    1. Intelligent preprocessing (denoising, artifact removal)
    2. State-of-the-art AI upscaling (HAT, SwinIR, Real-ESRGAN)
    3. Quality postprocessing (detail enhancement, color correction)

    Example:
        >>> # Quick quality upscale
        >>> pipeline = AdvancedUpscalePipeline(QualityPreset.QUALITY)
        >>> result = pipeline.upscale_file("input.jpg", "output.png")

        >>> # Custom configuration
        >>> config = AdvancedPipelineConfig(
        ...     primary_upscaler=UpscalerType.HAT,
        ...     scale=4,
        ...     enable_preprocessing=True,
        ... )
        >>> pipeline = AdvancedUpscalePipeline(config=config)
        >>> result = pipeline.upscale(image)
    """

    def __init__(
        self,
        preset: QualityPreset | None = None,
        config: AdvancedPipelineConfig | None = None,
    ) -> None:
        """Initialize advanced pipeline.

        Args:
            preset: Quality preset to use.
            config: Custom configuration (overrides preset).
        """
        if config is not None:
            self.config = config
            if config.preset is not None:
                self._apply_preset(config.preset)
        elif preset is not None:
            self.config = AdvancedPipelineConfig()
            self._apply_preset(preset)
        else:
            self.config = AdvancedPipelineConfig()

        self._upscalers: dict[UpscalerType, BaseUpscaler] = {}
        self._preprocessor: ImagePreprocessor | None = None
        self._postprocessor: ImagePostprocessor | None = None

    def _apply_preset(self, preset: QualityPreset) -> None:
        """Apply preset configuration."""
        if preset not in PRESET_CONFIGS:
            return

        preset_config = PRESET_CONFIGS[preset]
        for key, value in preset_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.config.preset = preset

    def _get_upscaler(self, upscaler_type: UpscalerType) -> BaseUpscaler:
        """Get or create upscaler instance."""
        if upscaler_type in self._upscalers:
            return self._upscalers[upscaler_type]

        # Create base config
        base_config = UpscaleConfig(
            scale=2 if self.config.multi_pass else self.config.scale,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            use_fp16=self.config.use_fp16,
            use_gpu=self.config.use_gpu,
        )

        # Create upscaler based on type
        if upscaler_type == UpscalerType.REAL_ESRGAN:
            base_config.model = UpscaleModel.REAL_ESRGAN_X4PLUS
            upscaler = RealESRGANUpscaler(base_config)
        elif upscaler_type == UpscalerType.SWINIR:
            base_config.model = UpscaleModel.SWINIR_REAL_SR_X4
            upscaler = SwinIRUpscaler(base_config)
        elif upscaler_type == UpscalerType.HAT:
            upscaler = HATUpscaler(base_config, variant="HAT")
        elif upscaler_type == UpscalerType.HAT_L:
            upscaler = HATUpscaler(base_config, variant="HAT-L")
        else:
            raise ValueError(f"Unknown upscaler type: {upscaler_type}")

        self._upscalers[upscaler_type] = upscaler
        return upscaler

    def _get_preprocessor(self) -> ImagePreprocessor:
        """Get or create preprocessor."""
        if self._preprocessor is None:
            self._preprocessor = ImagePreprocessor(self.config.preprocess_config)
        return self._preprocessor

    def _get_postprocessor(self) -> ImagePostprocessor:
        """Get or create postprocessor."""
        if self._postprocessor is None:
            self._postprocessor = ImagePostprocessor(self.config.postprocess_config)
        return self._postprocessor

    def upscale(self, image: "ndarray") -> AdvancedUpscaleResult:
        """Upscale image with full processing pipeline.

        Args:
            image: Input BGR image.

        Returns:
            AdvancedUpscaleResult with upscaled image and metrics.
        """
        total_start = time.time()
        h, w = image.shape[:2]
        original = image.copy()
        current = image

        preprocessing_time = 0.0
        upscaling_time = 0.0
        postprocessing_time = 0.0
        upscalers_used = []
        passes_completed = 0

        # Step 1: Preprocessing
        if self.config.enable_preprocessing:
            pre_start = time.time()
            preprocessor = self._get_preprocessor()
            current = preprocessor.process(current)
            preprocessing_time = time.time() - pre_start
            logger.info(f"Preprocessing completed in {preprocessing_time:.2f}s")

        # Step 2: Upscaling
        up_start = time.time()

        if self.config.multi_pass:
            # Multi-pass upscaling (2x + 2x = 4x)
            for i in range(self.config.passes):
                upscaler = self._get_upscaler(self.config.primary_upscaler)
                if not upscaler.is_loaded:
                    upscaler.load()

                result = upscaler.upscale(current, scale=2)
                current = result.image
                upscalers_used.append(upscaler.model_name)
                passes_completed += 1
                logger.info(f"Pass {i + 1}/{self.config.passes} completed")

                # Use secondary upscaler for later passes if available
                if self.config.secondary_upscaler and i < self.config.passes - 1:
                    secondary = self._get_upscaler(self.config.secondary_upscaler)
                    if not secondary.is_loaded:
                        secondary.load()
                    # Apply refinement
                    refined = secondary.upscale(current, scale=1)
                    current = refined.image
        else:
            # Single-pass upscaling
            upscaler = self._get_upscaler(self.config.primary_upscaler)
            if not upscaler.is_loaded:
                upscaler.load()

            result = upscaler.upscale(current, scale=self.config.scale)
            current = result.image
            upscalers_used.append(upscaler.model_name)
            passes_completed = 1

            # Optional secondary upscaler for refinement
            if self.config.secondary_upscaler:
                secondary = self._get_upscaler(self.config.secondary_upscaler)
                if not secondary.is_loaded:
                    secondary.load()

                # Downscale slightly and re-upscale for detail refinement
                h_up, w_up = current.shape[:2]
                downscaled = cv2.resize(
                    current,
                    (int(w_up * 0.9), int(h_up * 0.9)),
                    interpolation=cv2.INTER_AREA,
                )
                refined = secondary.upscale(downscaled)
                # Resize back and blend
                refined_resized = cv2.resize(
                    refined.image, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4
                )
                current = cv2.addWeighted(current, 0.7, refined_resized, 0.3, 0)
                upscalers_used.append(secondary.model_name)

        upscaling_time = time.time() - up_start
        logger.info(f"Upscaling completed in {upscaling_time:.2f}s")

        # Step 3: Postprocessing
        if self.config.enable_postprocessing:
            post_start = time.time()
            postprocessor = self._get_postprocessor()
            current = postprocessor.process(current, original, self.config.scale)
            postprocessing_time = time.time() - post_start
            logger.info(f"Postprocessing completed in {postprocessing_time:.2f}s")

        total_time = time.time() - total_start

        return AdvancedUpscaleResult(
            image=current,
            original_size=(w, h),
            upscaled_size=(current.shape[1], current.shape[0]),
            scale_factor=current.shape[1] / w,
            model_used=", ".join(upscalers_used),
            processing_time=total_time,
            preprocessing_time=preprocessing_time,
            upscaling_time=upscaling_time,
            postprocessing_time=postprocessing_time,
            upscalers_used=upscalers_used,
            passes_completed=passes_completed,
            quality_preset=self.config.preset.value if self.config.preset else "custom",
        )

    def upscale_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> AdvancedUpscaleResult:
        """Upscale image file.

        Args:
            input_path: Path to input image.
            output_path: Path to save output (optional).

        Returns:
            AdvancedUpscaleResult.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Read image
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        # Process
        result = self.upscale(image)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            ext = output_path.suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                cv2.imwrite(
                    str(output_path),
                    result.image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                )
            elif ext == ".webp":
                cv2.imwrite(
                    str(output_path),
                    result.image,
                    [cv2.IMWRITE_WEBP_QUALITY, self.config.jpeg_quality],
                )
            else:
                cv2.imwrite(str(output_path), result.image)

            logger.info(f"Saved upscaled image to {output_path}")

        return result

    def unload(self) -> None:
        """Unload all models and free memory."""
        for upscaler in self._upscalers.values():
            upscaler.unload()
        self._upscalers.clear()

    def __enter__(self) -> "AdvancedUpscalePipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()


# =============================================================================
# Convenience functions
# =============================================================================


def upscale_fast(image: "ndarray", scale: int = 4) -> "ndarray":
    """Fast upscaling with minimal processing.

    Args:
        image: Input BGR image.
        scale: Upscaling factor.

    Returns:
        Upscaled image.
    """
    with AdvancedUpscalePipeline(QualityPreset.FAST) as pipeline:
        pipeline.config.scale = scale
        result = pipeline.upscale(image)
        return result.image


def upscale_quality(image: "ndarray", scale: int = 4) -> "ndarray":
    """High-quality upscaling with HAT.

    Args:
        image: Input BGR image.
        scale: Upscaling factor.

    Returns:
        Upscaled image.
    """
    with AdvancedUpscalePipeline(QualityPreset.QUALITY) as pipeline:
        pipeline.config.scale = scale
        result = pipeline.upscale(image)
        return result.image


def upscale_ultra(image: "ndarray", scale: int = 4) -> "ndarray":
    """Maximum quality upscaling with full pipeline.

    Args:
        image: Input BGR image.
        scale: Upscaling factor.

    Returns:
        Upscaled image.
    """
    with AdvancedUpscalePipeline(QualityPreset.ULTRA) as pipeline:
        pipeline.config.scale = scale
        result = pipeline.upscale(image)
        return result.image


def compare_quality_presets(
    image: "ndarray",
    output_dir: str | Path,
    scale: int = 4,
) -> dict[str, AdvancedUpscaleResult]:
    """Compare all quality presets on an image.

    Args:
        image: Input BGR image.
        output_dir: Directory to save comparison images.
        scale: Upscaling factor.

    Returns:
        Dictionary of preset name to result.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for preset in QualityPreset:
        logger.info(f"Processing with {preset.value} preset...")
        with AdvancedUpscalePipeline(preset) as pipeline:
            pipeline.config.scale = scale
            result = pipeline.upscale(image)
            results[preset.value] = result

            # Save result
            output_path = output_dir / f"upscaled_{preset.value}.png"
            cv2.imwrite(str(output_path), result.image)
            logger.info(
                f"{preset.value}: {result.processing_time:.2f}s, "
                f"size: {result.upscaled_size}"
            )

    return results
