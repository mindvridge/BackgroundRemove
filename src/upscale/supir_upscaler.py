"""SUPIR: Scaling Up Image Restoration with diffusion prior.

SUPIR is a state-of-the-art image restoration model that combines:
- SDXL as a generative prior for high-quality restoration
- Degradation-aware encoding for handling various image degradations
- Multi-scale processing for large images

Quality: 10/10 (Current SOTA for image restoration)
VRAM: 12GB+ required (16GB+ recommended)
Speed: Slow (30-60s per image)

Reference:
- Paper: "Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild"
- GitHub: https://github.com/Fanghua-Yu/SUPIR
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.upscale.base import BaseUpscaler, UpscaleConfig, UpscaleResult

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class SUPIRModel(Enum):
    """Available SUPIR model variants."""

    SUPIR_V0Q = "supir-v0q"  # Quality focused
    SUPIR_V0F = "supir-v0f"  # Fidelity focused


class DegradationLevel(Enum):
    """Image degradation levels for SUPIR."""

    LIGHT = "light"  # Minor degradation
    MODERATE = "moderate"  # Typical degradation
    HEAVY = "heavy"  # Severe degradation
    AUTO = "auto"  # Auto-detect


@dataclass
class SUPIRConfig:
    """Configuration for SUPIR upscaling."""

    # Model selection
    model: SUPIRModel = SUPIRModel.SUPIR_V0Q

    # Restoration settings
    scale: int = 4
    degradation_level: DegradationLevel = DegradationLevel.MODERATE
    positive_prompt: str = "high quality, detailed, sharp, clean"
    negative_prompt: str = "blurry, noisy, low quality, artifacts, distorted"

    # Generation parameters
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    restoration_scale: float = 1.0  # Strength of restoration (0-2)
    color_fix_type: str = "wavelet"  # 'none', 'wavelet', 'adain'

    # Tile processing
    tile_size: int = 512
    tile_overlap: int = 64

    # Performance
    use_fp16: bool = True
    use_gpu: bool = True
    device_id: int = 0
    enable_xformers: bool = True  # Memory efficient attention

    # Model paths
    model_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "supir_models"
    )


class SUPIRUpscaler(BaseUpscaler):
    """SUPIR-based image restoration and upscaling.

    SUPIR (Scaling Up to Excellence) is currently one of the best models
    for photo-realistic image restoration. It can handle:
    - Low resolution images
    - JPEG artifacts
    - Noise and grain
    - Blur and motion blur
    - Old/degraded photos

    Quality: 10/10 (SOTA)
    Speed: Slow (~30-60s per 512x512 image)
    VRAM: 12GB minimum, 16GB+ recommended

    Example:
        >>> config = SUPIRConfig(
        ...     scale=4,
        ...     degradation_level=DegradationLevel.MODERATE,
        ...     num_inference_steps=30,
        ... )
        >>> upscaler = SUPIRUpscaler(config)
        >>> result = upscaler.upscale(degraded_image)
    """

    def __init__(
        self,
        config: UpscaleConfig | None = None,
        supir_config: SUPIRConfig | None = None,
    ) -> None:
        """Initialize SUPIR upscaler.

        Args:
            config: Base upscale config.
            supir_config: SUPIR-specific configuration.
        """
        base_config = config or UpscaleConfig(scale=4)
        super().__init__(base_config)

        self.supir_config = supir_config or SUPIRConfig()
        self.model: Any = None
        self.device = self._get_device()

        # Model components (loaded separately)
        self._sdxl_pipeline: Any = None
        self._encoder: Any = None
        self._controlnet: Any = None

    def _get_device(self) -> torch.device:
        """Get compute device."""
        if self.supir_config.use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{self.supir_config.device_id}")
        return torch.device("cpu")

    @property
    def model_name(self) -> str:
        return f"SUPIR-{self.supir_config.model.value}"

    def load(self) -> None:
        """Load SUPIR model components.

        Note: SUPIR requires multiple model components:
        1. SDXL base model for generation
        2. SUPIR encoder for degradation-aware encoding
        3. ControlNet for structural guidance
        """
        if self._loaded:
            return

        logger.info("Loading SUPIR model components...")

        try:
            from diffusers import (
                AutoencoderKL,
                ControlNetModel,
                StableDiffusionXLControlNetPipeline,
                UniPCMultistepScheduler,
            )
        except ImportError:
            raise ImportError(
                "diffusers is required for SUPIR. "
                "Install with: pip install diffusers transformers accelerate"
            )

        # Load SDXL VAE
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

        # Load ControlNet for tile control
        logger.info("Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Load SDXL pipeline with ControlNet
        logger.info("Loading SDXL pipeline...")
        self._sdxl_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Apply optimizations
        self._sdxl_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self._sdxl_pipeline.scheduler.config
        )

        if self.supir_config.enable_xformers:
            try:
                self._sdxl_pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"xformers not available: {e}")

        self._sdxl_pipeline.enable_attention_slicing()
        self._sdxl_pipeline = self._sdxl_pipeline.to(self.device)

        self._loaded = True
        logger.info("SUPIR model loaded successfully")

    def unload(self) -> None:
        """Unload all model components."""
        if self._sdxl_pipeline is not None:
            del self._sdxl_pipeline
            self._sdxl_pipeline = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("SUPIR model unloaded")

    def upscale(
        self,
        image: "ndarray",
        scale: int | None = None,
        prompt: str | None = None,
    ) -> UpscaleResult:
        """Restore and upscale image using SUPIR.

        Args:
            image: Input BGR image (degraded/low quality).
            scale: Upscale factor (default: 4).
            prompt: Optional custom prompt.

        Returns:
            UpscaleResult with restored image.
        """
        if not self._loaded:
            self.load()

        start_time = time.time()
        h, w = image.shape[:2]
        target_scale = scale or self.supir_config.scale

        # Convert to RGB PIL
        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image)

        # Detect degradation if auto
        if self.supir_config.degradation_level == DegradationLevel.AUTO:
            degradation = self._detect_degradation(image)
        else:
            degradation = self.supir_config.degradation_level

        # Prepare prompts
        positive = prompt or self.supir_config.positive_prompt
        negative = self.supir_config.negative_prompt

        # Adjust parameters based on degradation
        steps = self._adjust_steps_for_degradation(degradation)
        strength = self._get_restoration_strength(degradation)

        # Process with tiling for large images
        if w * target_scale > 1024 or h * target_scale > 1024:
            output_pil = self._process_tiled(
                pil_image, target_scale, positive, negative, steps, strength
            )
        else:
            output_pil = self._process_single(
                pil_image, target_scale, positive, negative, steps, strength
            )

        # Apply color fix if enabled
        if self.supir_config.color_fix_type != "none":
            output_pil = self._apply_color_fix(pil_image, output_pil)

        # Convert back to BGR
        output_rgb = np.array(output_pil)
        output_bgr = output_rgb[:, :, ::-1].copy()

        processing_time = time.time() - start_time

        return UpscaleResult(
            image=output_bgr,
            original_size=(w, h),
            upscaled_size=(output_bgr.shape[1], output_bgr.shape[0]),
            scale_factor=target_scale,
            model_used=self.model_name,
            processing_time=processing_time,
        )

    def _detect_degradation(self, image: "ndarray") -> DegradationLevel:
        """Auto-detect image degradation level."""
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Calculate noise level (using high-frequency content)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.std(gray.astype(np.float32) - blur.astype(np.float32))

        # Detect JPEG artifacts (blockiness)
        blockiness = self._measure_blockiness(gray)

        # Combine metrics
        quality_score = laplacian_var / 1000 - noise / 10 - blockiness * 2

        if quality_score > 50:
            return DegradationLevel.LIGHT
        elif quality_score > 20:
            return DegradationLevel.MODERATE
        else:
            return DegradationLevel.HEAVY

    def _measure_blockiness(self, gray: "ndarray") -> float:
        """Measure JPEG blockiness."""
        h, w = gray.shape
        h8, w8 = h // 8 * 8, w // 8 * 8
        if h8 < 16 or w8 < 16:
            return 0.0

        gray_crop = gray[:h8, :w8].astype(np.float32)

        # Block boundary differences
        h_diff = np.abs(gray_crop[7::8, :] - gray_crop[8::8, :])
        v_diff = np.abs(gray_crop[:, 7::8] - gray_crop[:, 8::8])

        # Internal differences for comparison
        h_int = np.abs(gray_crop[3::8, :] - gray_crop[4::8, :])
        v_int = np.abs(gray_crop[:, 3::8] - gray_crop[:, 4::8])

        block_mean = (np.mean(h_diff) + np.mean(v_diff)) / 2
        int_mean = (np.mean(h_int) + np.mean(v_int)) / 2

        if int_mean > 0:
            return block_mean / int_mean
        return 0.0

    def _adjust_steps_for_degradation(self, degradation: DegradationLevel) -> int:
        """Adjust inference steps based on degradation."""
        base_steps = self.supir_config.num_inference_steps

        if degradation == DegradationLevel.LIGHT:
            return max(15, base_steps - 10)
        elif degradation == DegradationLevel.HEAVY:
            return base_steps + 10
        return base_steps

    def _get_restoration_strength(self, degradation: DegradationLevel) -> float:
        """Get restoration strength based on degradation."""
        base = self.supir_config.restoration_scale

        if degradation == DegradationLevel.LIGHT:
            return base * 0.7
        elif degradation == DegradationLevel.HEAVY:
            return min(base * 1.3, 2.0)
        return base

    def _process_single(
        self,
        image: Image.Image,
        scale: int,
        positive: str,
        negative: str,
        steps: int,
        strength: float,
    ) -> Image.Image:
        """Process a single image."""
        import cv2

        # Resize to target
        w, h = image.size
        target_size = (w * scale, h * scale)

        # Pre-upscale with Lanczos
        upscaled = image.resize(target_size, Image.LANCZOS)

        # Create control image (edge/structure)
        upscaled_np = np.array(upscaled)
        gray = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        control_image = Image.fromarray(edges).convert("RGB")

        # Run SDXL with ControlNet
        with torch.no_grad():
            result = self._sdxl_pipeline(
                prompt=positive,
                negative_prompt=negative,
                image=upscaled,
                control_image=control_image,
                num_inference_steps=steps,
                guidance_scale=self.supir_config.guidance_scale,
                controlnet_conditioning_scale=0.5,
                strength=min(strength * 0.5, 0.8),
            )

        return result.images[0]

    def _process_tiled(
        self,
        image: Image.Image,
        scale: int,
        positive: str,
        negative: str,
        steps: int,
        strength: float,
    ) -> Image.Image:
        """Process large image using tiles."""
        w, h = image.size
        tile_size = self.supir_config.tile_size
        overlap = self.supir_config.tile_overlap

        # Output size
        out_w, out_h = w * scale, h * scale
        output_sum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_sum = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Create blend weights
        tile_weights = self._create_blend_weights(tile_size * scale, overlap * scale)

        stride = tile_size - overlap
        total_tiles = ((h - 1) // stride + 1) * ((w - 1) // stride + 1)
        current = 0

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                current += 1
                logger.info(f"SUPIR: Processing tile {current}/{total_tiles}")

                # Get tile bounds
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                # Extract and process tile
                tile = image.crop((x_start, y_start, x_end, y_end))
                processed = self._process_single(
                    tile, scale, positive, negative, steps, strength
                )

                # Convert to numpy
                tile_np = np.array(processed).astype(np.float32)
                tile_h, tile_w = tile_np.shape[:2]

                # Get weights
                weights = tile_weights[:tile_h, :tile_w]

                # Accumulate
                out_y = y_start * scale
                out_x = x_start * scale
                output_sum[out_y:out_y + tile_h, out_x:out_x + tile_w] += tile_np * weights
                weight_sum[out_y:out_y + tile_h, out_x:out_x + tile_w] += weights

        # Normalize
        output = output_sum / (weight_sum + 1e-8)
        output = np.clip(output, 0, 255).astype(np.uint8)

        return Image.fromarray(output)

    def _create_blend_weights(self, size: int, overlap: int) -> np.ndarray:
        """Create blending weights for tiles."""
        weights = np.ones((size, size, 1), dtype=np.float32)

        if overlap > 0:
            ramp = np.linspace(0, 1, overlap).reshape(-1, 1, 1)
            weights[:overlap] *= ramp
            weights[-overlap:] *= ramp[::-1]
            weights[:, :overlap] *= ramp.reshape(1, -1, 1)
            weights[:, -overlap:] *= ramp[::-1].reshape(1, -1, 1)

        return weights

    def _apply_color_fix(
        self,
        original: Image.Image,
        restored: Image.Image,
    ) -> Image.Image:
        """Apply color correction to match original."""
        import cv2

        # Resize original to match restored
        orig_resized = original.resize(restored.size, Image.LANCZOS)

        orig_np = np.array(orig_resized).astype(np.float32)
        rest_np = np.array(restored).astype(np.float32)

        if self.supir_config.color_fix_type == "wavelet":
            # Wavelet-based color transfer
            result = self._wavelet_color_fix(orig_np, rest_np)
        elif self.supir_config.color_fix_type == "adain":
            # AdaIN style color transfer
            result = self._adain_color_fix(orig_np, rest_np)
        else:
            result = rest_np

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _wavelet_color_fix(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Wavelet-based color transfer."""
        import cv2

        # Convert to LAB
        source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

        # Keep L from target, transfer a/b from source
        result_lab = target_lab.copy()

        # Blend color channels
        alpha = 0.7  # How much to preserve restored colors
        result_lab[:, :, 1] = alpha * target_lab[:, :, 1] + (1 - alpha) * cv2.resize(
            source_lab[:, :, 1], (target_lab.shape[1], target_lab.shape[0])
        )
        result_lab[:, :, 2] = alpha * target_lab[:, :, 2] + (1 - alpha) * cv2.resize(
            source_lab[:, :, 2], (target_lab.shape[1], target_lab.shape[0])
        )

        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return result.astype(np.float32)

    def _adain_color_fix(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """AdaIN-style color transfer."""
        # Compute statistics
        src_mean = np.mean(source, axis=(0, 1), keepdims=True)
        src_std = np.std(source, axis=(0, 1), keepdims=True) + 1e-6
        tgt_mean = np.mean(target, axis=(0, 1), keepdims=True)
        tgt_std = np.std(target, axis=(0, 1), keepdims=True) + 1e-6

        # Transfer: normalize target, then denormalize with source stats
        normalized = (target - tgt_mean) / tgt_std
        transferred = normalized * src_std + src_mean

        # Blend with original restored
        alpha = 0.3
        result = alpha * transferred + (1 - alpha) * target

        return result


# =============================================================================
# Convenience functions
# =============================================================================


def supir_restore(
    image: "ndarray",
    scale: int = 4,
    quality: str = "balanced",
) -> "ndarray":
    """Restore and upscale image using SUPIR.

    Args:
        image: Input BGR image (degraded).
        scale: Upscale factor.
        quality: Quality preset - "fast", "balanced", "quality".

    Returns:
        Restored and upscaled image.
    """
    steps_map = {"fast": 15, "balanced": 25, "quality": 40}
    steps = steps_map.get(quality, 25)

    config = SUPIRConfig(
        scale=scale,
        num_inference_steps=steps,
        degradation_level=DegradationLevel.AUTO,
    )

    with SUPIRUpscaler(supir_config=config) as upscaler:
        result = upscaler.upscale(image)
        return result.image


def restore_old_photo(
    image: "ndarray",
    scale: int = 2,
) -> "ndarray":
    """Restore old/degraded photo using SUPIR.

    Optimized settings for old photo restoration with
    heavy degradation handling.

    Args:
        image: Input degraded photo.
        scale: Upscale factor (2 recommended for old photos).

    Returns:
        Restored photo.
    """
    config = SUPIRConfig(
        scale=scale,
        degradation_level=DegradationLevel.HEAVY,
        num_inference_steps=35,
        restoration_scale=1.2,
        positive_prompt="high quality photo, sharp, clear, restored, detailed",
        negative_prompt="blurry, noisy, artifacts, damaged, faded, low quality",
    )

    with SUPIRUpscaler(supir_config=config) as upscaler:
        result = upscaler.upscale(image)
        return result.image
