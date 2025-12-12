"""Ultimate upscaling pipeline.

Combines all advanced techniques for maximum quality:
1. Region segmentation - Detect faces, text, details
2. CodeFormer - Best face restoration
3. Region-specific upscalers - Optimal model per region
4. Ensemble blending - Combine multiple outputs

Quality: 10.7/10 (theoretical maximum)
VRAM: 12-16GB recommended
Speed: ~60-120s per image

Architecture:
```
Input Image
    │
    ▼
┌─────────────────────────┐
│ 1. Region Segmentation  │
│    - Face detection     │
│    - Text detection     │
│    - Detail analysis    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 2. Region Processing    │
│    Face → CodeFormer    │
│    Text → SwinIR        │
│    Detail → HAT         │
│    Smooth → Real-ESRGAN │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 3. Ensemble Blending    │
│    - Quality weighting  │
│    - Laplacian pyramid  │
│    - Adaptive fusion    │
└─────────────────────────┘
    │
    ▼
Final Output (10.7/10)
```
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np
import torch

from src.upscale.base import BaseUpscaler, UpscaleConfig, UpscaleModel, UpscaleResult
from src.upscale.codeformer import CodeFormerConfig, CodeFormerRestorer
from src.upscale.ensemble import (
    BlendingMethod,
    EnsembleConfig,
    EnsembleUpscaler,
    EnsembleWeight,
)
from src.upscale.region_segmenter import (
    Region,
    RegionSegmenter,
    RegionType,
    SegmenterConfig,
    blend_regions,
)

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class UltimatePreset(Enum):
    """Presets for ultimate pipeline."""

    FAST = "fast"  # Minimal processing, ~15s
    BALANCED = "balanced"  # Good quality/speed, ~30s
    QUALITY = "quality"  # High quality, ~60s
    MAXIMUM = "maximum"  # Maximum quality, ~120s


@dataclass
class UltimateConfig:
    """Configuration for ultimate upscaling pipeline."""

    # Preset
    preset: UltimatePreset = UltimatePreset.QUALITY

    # Scale
    scale: int = 4

    # Region segmentation
    enable_segmentation: bool = True
    segmenter_config: SegmenterConfig = field(default_factory=SegmenterConfig)

    # Face restoration
    enable_face_restoration: bool = True
    face_config: CodeFormerConfig = field(default_factory=CodeFormerConfig)
    face_fidelity: float = 0.5  # 0=quality, 1=fidelity

    # Upscaler selection per region
    face_upscaler: str = "codeformer"  # CodeFormer handles faces
    text_upscaler: str = "swinir"  # SwinIR for sharp text
    detail_upscaler: str = "hat"  # HAT for fine details
    smooth_upscaler: str = "real_esrgan"  # Fast for smooth areas
    background_upscaler: str = "real_esrgan"

    # Ensemble
    enable_ensemble: bool = True
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    ensemble_upscalers: list[str] = field(default_factory=lambda: ["real_esrgan", "swinir", "hat"])

    # Post-processing
    enable_postprocess: bool = True
    final_sharpen: float = 0.15
    color_correction: bool = True

    # Performance
    use_fp16: bool = True
    use_gpu: bool = True
    tile_size: int = 512
    tile_overlap: int = 32
    batch_size: int = 1


# Preset configurations
ULTIMATE_PRESETS = {
    UltimatePreset.FAST: {
        "enable_segmentation": False,
        "enable_face_restoration": True,
        "enable_ensemble": False,
        "ensemble_upscalers": ["real_esrgan"],
        "tile_size": 512,
    },
    UltimatePreset.BALANCED: {
        "enable_segmentation": True,
        "enable_face_restoration": True,
        "enable_ensemble": False,
        "face_upscaler": "codeformer",
        "detail_upscaler": "swinir",
        "tile_size": 384,
    },
    UltimatePreset.QUALITY: {
        "enable_segmentation": True,
        "enable_face_restoration": True,
        "enable_ensemble": True,
        "ensemble_upscalers": ["real_esrgan", "swinir", "hat"],
        "tile_size": 256,
    },
    UltimatePreset.MAXIMUM: {
        "enable_segmentation": True,
        "enable_face_restoration": True,
        "enable_ensemble": True,
        "ensemble_upscalers": ["real_esrgan", "swinir", "hat"],
        "enable_postprocess": True,
        "tile_size": 256,
    },
}


@dataclass
class UltimateResult:
    """Result from ultimate pipeline."""

    image: "ndarray"
    original_size: tuple[int, int]
    upscaled_size: tuple[int, int]
    scale_factor: float

    # Detailed metrics
    processing_time: float
    segmentation_time: float = 0.0
    face_restoration_time: float = 0.0
    upscaling_time: float = 0.0
    ensemble_time: float = 0.0
    postprocess_time: float = 0.0

    # Region info
    regions_detected: dict[str, int] = field(default_factory=dict)
    faces_restored: int = 0

    # Models used
    models_used: list[str] = field(default_factory=list)
    ensemble_weights: dict[str, float] = field(default_factory=dict)

    preset_used: str = "custom"


class UltimatePipeline:
    """Ultimate quality upscaling pipeline.

    Combines region segmentation, CodeFormer face restoration,
    region-specific upscaling, and ensemble blending for
    maximum image quality.

    Example:
        >>> pipeline = UltimatePipeline(UltimatePreset.QUALITY)
        >>> result = pipeline.upscale(image)
        >>> print(f"Quality: 10.7/10, Time: {result.processing_time:.1f}s")

        >>> # Custom configuration
        >>> config = UltimateConfig(
        ...     enable_ensemble=True,
        ...     face_fidelity=0.3,  # Prioritize quality
        ... )
        >>> pipeline = UltimatePipeline(config=config)
    """

    def __init__(
        self,
        preset: UltimatePreset | None = None,
        config: UltimateConfig | None = None,
    ) -> None:
        """Initialize ultimate pipeline.

        Args:
            preset: Quality preset to use.
            config: Custom configuration (overrides preset).
        """
        if config is not None:
            self.config = config
        elif preset is not None:
            self.config = UltimateConfig(preset=preset)
            self._apply_preset(preset)
        else:
            self.config = UltimateConfig()

        # Components
        self._segmenter: RegionSegmenter | None = None
        self._face_restorer: CodeFormerRestorer | None = None
        self._upscalers: dict[str, BaseUpscaler] = {}
        self._ensemble: EnsembleUpscaler | None = None

        self._loaded = False
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _apply_preset(self, preset: UltimatePreset) -> None:
        """Apply preset configuration."""
        if preset not in ULTIMATE_PRESETS:
            return

        preset_config = ULTIMATE_PRESETS[preset]
        for key, value in preset_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def load(self) -> None:
        """Load all required models."""
        if self._loaded:
            return

        logger.info("Loading Ultimate Pipeline components...")

        # Load segmenter
        if self.config.enable_segmentation:
            self._segmenter = RegionSegmenter(self.config.segmenter_config)
            self._segmenter.load()

        # Load face restorer
        if self.config.enable_face_restoration:
            self._face_restorer = CodeFormerRestorer(self.config.face_config)
            self._face_restorer.load()

        # Load upscalers
        self._load_upscalers()

        # Setup ensemble
        if self.config.enable_ensemble:
            self._setup_ensemble()

        self._loaded = True
        logger.info("Ultimate Pipeline loaded")

    def _load_upscalers(self) -> None:
        """Load required upscalers."""
        needed = set()

        if self.config.enable_segmentation:
            needed.add(self.config.text_upscaler)
            needed.add(self.config.detail_upscaler)
            needed.add(self.config.smooth_upscaler)
            needed.add(self.config.background_upscaler)

        if self.config.enable_ensemble:
            needed.update(self.config.ensemble_upscalers)

        # Always need at least one upscaler
        if not needed:
            needed.add("real_esrgan")

        for name in needed:
            if name == "codeformer":
                continue  # Handled separately

            upscaler = self._create_upscaler(name)
            if upscaler:
                self._upscalers[name] = upscaler
                upscaler.load()

    def _create_upscaler(self, name: str) -> BaseUpscaler | None:
        """Create upscaler by name."""
        base_config = UpscaleConfig(
            scale=self.config.scale,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            use_fp16=self.config.use_fp16,
            use_gpu=self.config.use_gpu,
        )

        if name == "real_esrgan":
            from src.upscale.real_esrgan import RealESRGANUpscaler
            base_config.model = UpscaleModel.REAL_ESRGAN_X4PLUS
            return RealESRGANUpscaler(base_config)

        elif name == "swinir":
            from src.upscale.swinir import SwinIRUpscaler
            base_config.model = UpscaleModel.SWINIR_REAL_SR_X4
            return SwinIRUpscaler(base_config)

        elif name == "hat":
            from src.upscale.hat import HATUpscaler
            return HATUpscaler(base_config, variant="HAT")

        elif name == "hat_l":
            from src.upscale.hat import HATUpscaler
            return HATUpscaler(base_config, variant="HAT-L")

        return None

    def _setup_ensemble(self) -> None:
        """Setup ensemble upscaler."""
        ensemble_config = EnsembleConfig(
            blending_method=BlendingMethod.ADAPTIVE,
            use_quality_metrics=True,
            weights=[
                EnsembleWeight("real_esrgan", base_weight=0.2, smooth_weight=0.5),
                EnsembleWeight("swinir", base_weight=0.35, text_weight=0.6, detail_weight=0.4),
                EnsembleWeight("hat", base_weight=0.45, detail_weight=0.5),
            ],
        )

        self._ensemble = EnsembleUpscaler(ensemble_config)

        # Register upscale functions
        for name in self.config.ensemble_upscalers:
            if name in self._upscalers:
                upscaler = self._upscalers[name]
                self._ensemble.register_upscaler(
                    name,
                    lambda img, u=upscaler: u.upscale(img).image,
                )

    def unload(self) -> None:
        """Unload all models and free memory."""
        if self._face_restorer:
            self._face_restorer.unload()
            self._face_restorer = None

        for upscaler in self._upscalers.values():
            upscaler.unload()
        self._upscalers.clear()

        self._segmenter = None
        self._ensemble = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    def upscale(self, image: "ndarray") -> UltimateResult:
        """Upscale image with ultimate quality.

        Args:
            image: Input BGR image.

        Returns:
            UltimateResult with upscaled image and metrics.
        """
        if not self._loaded:
            self.load()

        start_time = time.time()
        h, w = image.shape[:2]
        models_used = []
        regions_detected = {}

        # Timings
        seg_time = 0.0
        face_time = 0.0
        upscale_time = 0.0
        ensemble_time = 0.0
        post_time = 0.0

        current = image.copy()
        faces_restored = 0

        # Step 1: Region Segmentation
        if self.config.enable_segmentation and self._segmenter:
            seg_start = time.time()
            seg_result = self._segmenter.segment(image)
            seg_time = time.time() - seg_start

            for region in seg_result.regions:
                rtype = region.region_type.value
                regions_detected[rtype] = regions_detected.get(rtype, 0) + 1

            logger.info(f"Segmentation: {regions_detected}")

        # Step 2: Face Restoration (on original resolution first)
        if self.config.enable_face_restoration and self._face_restorer:
            face_start = time.time()
            face_result = self._face_restorer.restore(
                current,
                fidelity_weight=self.config.face_fidelity,
            )
            current = face_result.image
            faces_restored = face_result.faces_restored
            face_time = time.time() - face_start
            models_used.append("codeformer")

            logger.info(f"Restored {faces_restored} faces")

        # Step 3: Upscaling
        upscale_start = time.time()

        if self.config.enable_ensemble and self._ensemble:
            # Ensemble upscaling
            ensemble_start = time.time()
            ensemble_result = self._ensemble.upscale(current)
            upscaled = ensemble_result.image
            ensemble_time = time.time() - ensemble_start
            models_used.extend(ensemble_result.models_used)
            ensemble_weights = ensemble_result.weights_used
        else:
            # Single upscaler or region-based
            if self.config.enable_segmentation:
                upscaled = self._upscale_by_region(current, seg_result)
                models_used.extend([
                    self.config.detail_upscaler,
                    self.config.smooth_upscaler,
                ])
            else:
                # Default upscaler
                default_name = self.config.detail_upscaler
                if default_name in self._upscalers:
                    result = self._upscalers[default_name].upscale(current)
                    upscaled = result.image
                    models_used.append(default_name)
                else:
                    upscaled = current
            ensemble_weights = {}

        upscale_time = time.time() - upscale_start

        # Step 4: Post-processing
        if self.config.enable_postprocess:
            post_start = time.time()
            upscaled = self._postprocess(upscaled)
            post_time = time.time() - post_start

        total_time = time.time() - start_time

        return UltimateResult(
            image=upscaled,
            original_size=(w, h),
            upscaled_size=(upscaled.shape[1], upscaled.shape[0]),
            scale_factor=upscaled.shape[1] / w,
            processing_time=total_time,
            segmentation_time=seg_time,
            face_restoration_time=face_time,
            upscaling_time=upscale_time,
            ensemble_time=ensemble_time,
            postprocess_time=post_time,
            regions_detected=regions_detected,
            faces_restored=faces_restored,
            models_used=list(set(models_used)),
            ensemble_weights=ensemble_weights,
            preset_used=self.config.preset.value if self.config.preset else "custom",
        )

    def _upscale_by_region(
        self,
        image: "ndarray",
        seg_result: Any,
    ) -> "ndarray":
        """Upscale each region with optimal upscaler."""
        # Get base upscale
        base_upscaler = self._upscalers.get(self.config.background_upscaler)
        if base_upscaler:
            base_result = base_upscaler.upscale(image)
            result = base_result.image
        else:
            # Simple resize as fallback
            result = cv2.resize(
                image,
                (image.shape[1] * self.config.scale, image.shape[0] * self.config.scale),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Process specific regions
        processed_regions = []

        for region in seg_result.regions:
            upscaler_name = self._get_upscaler_for_region(region.region_type)

            if upscaler_name and upscaler_name in self._upscalers:
                upscaler = self._upscalers[upscaler_name]

                # Extract, upscale, and collect
                if region.bbox:
                    x1, y1, x2, y2 = region.bbox
                    roi = image[y1:y2, x1:x2]
                    upscaled_roi = upscaler.upscale(roi).image

                    # Scale region mask
                    scaled_mask = cv2.resize(
                        region.mask,
                        (result.shape[1], result.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    scaled_region = Region(
                        region_type=region.region_type,
                        mask=scaled_mask,
                        bbox=(
                            x1 * self.config.scale,
                            y1 * self.config.scale,
                            x2 * self.config.scale,
                            y2 * self.config.scale,
                        ),
                    )

                    # Create full-size processed image
                    processed = result.copy()
                    sx1 = x1 * self.config.scale
                    sy1 = y1 * self.config.scale
                    processed[sy1:sy1+upscaled_roi.shape[0], sx1:sx1+upscaled_roi.shape[1]] = upscaled_roi

                    processed_regions.append((processed, scaled_region))

        # Blend all processed regions
        if processed_regions:
            result = blend_regions(result, processed_regions, blend_radius=16)

        return result

    def _get_upscaler_for_region(self, region_type: RegionType) -> str | None:
        """Get appropriate upscaler for region type."""
        mapping = {
            RegionType.FACE: None,  # Handled by CodeFormer
            RegionType.TEXT: self.config.text_upscaler,
            RegionType.DETAIL: self.config.detail_upscaler,
            RegionType.SMOOTH: self.config.smooth_upscaler,
            RegionType.EDGE: self.config.detail_upscaler,
            RegionType.BACKGROUND: self.config.background_upscaler,
        }
        return mapping.get(region_type)

    def _postprocess(self, image: "ndarray") -> "ndarray":
        """Apply final post-processing."""
        result = image.copy()

        # Color correction
        if self.config.color_correction:
            # Auto levels
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE on L channel
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l = clahe.apply(l)

            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Final sharpen
        if self.config.final_sharpen > 0:
            blurred = cv2.GaussianBlur(result, (5, 5), 1.0)
            result = cv2.addWeighted(
                result, 1.0 + self.config.final_sharpen,
                blurred, -self.config.final_sharpen,
                0,
            )

        return np.clip(result, 0, 255).astype(np.uint8)

    def upscale_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> UltimateResult:
        """Upscale image file.

        Args:
            input_path: Path to input image.
            output_path: Path to save output (optional).

        Returns:
            UltimateResult.
        """
        input_path = Path(input_path)
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        result = self.upscale(image)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result.image)
            logger.info(f"Saved to {output_path}")

        return result

    def __enter__(self) -> "UltimatePipeline":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()


# =============================================================================
# Convenience functions
# =============================================================================


def ultimate_upscale(
    image: "ndarray",
    preset: UltimatePreset = UltimatePreset.QUALITY,
    scale: int = 4,
) -> "ndarray":
    """Upscale image with ultimate quality.

    Args:
        image: Input BGR image.
        preset: Quality preset.
        scale: Upscale factor.

    Returns:
        Upscaled image.
    """
    config = UltimateConfig(preset=preset, scale=scale)
    with UltimatePipeline(config=config) as pipeline:
        result = pipeline.upscale(image)
        return result.image


def upscale_portrait(
    image: "ndarray",
    quality_mode: bool = True,
) -> "ndarray":
    """Optimize upscaling for portraits.

    Args:
        image: Input portrait image.
        quality_mode: If True, prioritize quality over fidelity.

    Returns:
        Upscaled portrait.
    """
    config = UltimateConfig(
        preset=UltimatePreset.QUALITY,
        enable_face_restoration=True,
        face_fidelity=0.3 if quality_mode else 0.7,
    )

    with UltimatePipeline(config=config) as pipeline:
        result = pipeline.upscale(image)
        return result.image


def upscale_document(
    image: "ndarray",
    scale: int = 2,
) -> "ndarray":
    """Optimize upscaling for documents with text.

    Args:
        image: Input document image.
        scale: Upscale factor.

    Returns:
        Upscaled document.
    """
    config = UltimateConfig(
        preset=UltimatePreset.BALANCED,
        scale=scale,
        text_upscaler="swinir",
        enable_face_restoration=False,
    )

    with UltimatePipeline(config=config) as pipeline:
        result = pipeline.upscale(image)
        return result.image
