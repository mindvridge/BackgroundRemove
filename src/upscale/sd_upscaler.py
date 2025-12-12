"""Stable Diffusion based image upscaler.

Provides the highest quality upscaling using diffusion models:
- SD x4 Upscaler: Official Stability AI 4x upscaler (10/10 quality)
- SDXL Refiner: Additional detail enhancement
- Tiled processing for large images

Requirements:
- diffusers>=0.25.0
- transformers>=4.35.0
- accelerate>=0.25.0
- 8GB+ VRAM recommended

Reference:
- https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
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
from PIL import Image

from src.upscale.base import BaseUpscaler, UpscaleConfig, UpscaleResult

if TYPE_CHECKING:
    from diffusers import StableDiffusionUpscalePipeline
    from numpy import ndarray

logger = logging.getLogger(__name__)


class SDUpscaleModel(Enum):
    """Available Stable Diffusion upscale models."""

    SD_X4_UPSCALER = "stabilityai/stable-diffusion-x4-upscaler"
    SD_X2_LATENT = "stabilityai/sd-x2-latent-upscaler"


@dataclass
class SDUpscaleConfig:
    """Configuration for Stable Diffusion upscaling."""

    # Model selection
    model: SDUpscaleModel = SDUpscaleModel.SD_X4_UPSCALER

    # Generation settings
    prompt: str = ""  # Optional prompt for guided upscaling
    negative_prompt: str = "blurry, low quality, artifacts, noise, jpeg artifacts"
    num_inference_steps: int = 20  # More steps = better quality, slower
    guidance_scale: float = 7.5  # How closely to follow prompt
    noise_level: int = 20  # Noise added to input (0-100), helps with details

    # Tile processing for large images
    tile_size: int = 512  # Input tile size (output will be 4x)
    tile_overlap: int = 64  # Overlap for seamless blending
    tile_batch_size: int = 1  # Tiles to process at once

    # Performance
    use_fp16: bool = True
    use_gpu: bool = True
    device_id: int = 0
    enable_attention_slicing: bool = True  # Reduce VRAM usage
    enable_vae_tiling: bool = True  # Further VRAM reduction
    enable_cpu_offload: bool = False  # Offload to CPU when not in use

    # Output
    output_format: str = "png"

    # Model cache
    model_cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "huggingface"
    )


class StableDiffusionUpscaler(BaseUpscaler):
    """Stable Diffusion x4 Upscaler.

    Uses the official Stability AI upscaler model for highest quality
    image upscaling. This model was specifically trained for the task
    of 4x image super-resolution.

    Quality: 10/10 (highest available)
    Speed: Slow (~10-30s per image depending on size)
    VRAM: 6-8GB minimum, 12GB+ recommended

    Example:
        >>> config = SDUpscaleConfig(
        ...     num_inference_steps=25,
        ...     noise_level=20,
        ... )
        >>> upscaler = StableDiffusionUpscaler(config)
        >>> result = upscaler.upscale(image)
    """

    def __init__(
        self,
        config: UpscaleConfig | None = None,
        sd_config: SDUpscaleConfig | None = None,
    ) -> None:
        """Initialize Stable Diffusion upscaler.

        Args:
            config: Base upscale config (for compatibility).
            sd_config: SD-specific configuration.
        """
        base_config = config or UpscaleConfig(scale=4)
        super().__init__(base_config)

        self.sd_config = sd_config or SDUpscaleConfig()
        self.pipeline: "StableDiffusionUpscalePipeline | None" = None
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Get compute device."""
        if self.sd_config.use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{self.sd_config.device_id}")
        return torch.device("cpu")

    @property
    def model_name(self) -> str:
        return "SD-x4-Upscaler"

    def load(self) -> None:
        """Load the Stable Diffusion upscaler pipeline."""
        if self._loaded:
            return

        logger.info("Loading Stable Diffusion x4 Upscaler...")

        try:
            from diffusers import StableDiffusionUpscalePipeline
        except ImportError:
            raise ImportError(
                "diffusers is required for SD upscaling. "
                "Install with: pip install diffusers transformers accelerate"
            )

        # Determine dtype
        dtype = torch.float16 if self.sd_config.use_fp16 else torch.float32

        # Load pipeline
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.sd_config.model.value,
            torch_dtype=dtype,
            cache_dir=self.sd_config.model_cache_dir,
        )

        # Apply optimizations
        if self.sd_config.enable_attention_slicing:
            self.pipeline.enable_attention_slicing()

        if self.sd_config.enable_vae_tiling:
            self.pipeline.enable_vae_tiling()

        if self.sd_config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)

        # Disable safety checker for speed (upscaling doesn't generate unsafe content)
        self.pipeline.safety_checker = None

        self._loaded = True
        logger.info(f"SD Upscaler loaded on {self.device}")

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("SD Upscaler unloaded")

    def upscale(
        self,
        image: "ndarray",
        scale: int | None = None,
        prompt: str | None = None,
    ) -> UpscaleResult:
        """Upscale image using Stable Diffusion.

        Args:
            image: Input BGR image (numpy array).
            scale: Scale factor (only 4 supported for this model).
            prompt: Optional prompt for guided upscaling.

        Returns:
            UpscaleResult with upscaled image.
        """
        if not self._loaded:
            self.load()

        start_time = time.time()
        h, w = image.shape[:2]

        # SD upscaler only supports 4x
        actual_scale = 4

        # Convert BGR to RGB PIL Image
        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image)

        # Process with tiling if image is large
        max_input_size = self.sd_config.tile_size
        if w > max_input_size or h > max_input_size:
            output_pil = self._process_tiled(pil_image, prompt)
        else:
            output_pil = self._process_single(pil_image, prompt)

        # Convert back to BGR numpy
        output_rgb = np.array(output_pil)
        output_bgr = output_rgb[:, :, ::-1].copy()

        processing_time = time.time() - start_time

        return UpscaleResult(
            image=output_bgr,
            original_size=(w, h),
            upscaled_size=(output_bgr.shape[1], output_bgr.shape[0]),
            scale_factor=actual_scale,
            model_used=self.model_name,
            processing_time=processing_time,
        )

    def _process_single(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> Image.Image:
        """Process a single image (no tiling)."""
        prompt = prompt or self.sd_config.prompt or ""
        negative_prompt = self.sd_config.negative_prompt

        # Run the pipeline
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=self.sd_config.num_inference_steps,
                guidance_scale=self.sd_config.guidance_scale,
                noise_level=self.sd_config.noise_level,
            )

        return result.images[0]

    def _process_tiled(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> Image.Image:
        """Process large image using tiles."""
        w, h = image.size
        tile_size = self.sd_config.tile_size
        overlap = self.sd_config.tile_overlap
        scale = 4

        # Output size
        out_w, out_h = w * scale, h * scale
        output = Image.new("RGB", (out_w, out_h))

        # Create weight mask for blending
        tile_weights = self._create_tile_weights(tile_size * scale, overlap * scale)

        # Accumulator for weighted blending
        weight_sum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        output_sum = np.zeros((out_h, out_w, 3), dtype=np.float32)

        # Process tiles
        stride = tile_size - overlap
        total_tiles = ((h - 1) // stride + 1) * ((w - 1) // stride + 1)
        current_tile = 0

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                current_tile += 1
                logger.info(f"Processing tile {current_tile}/{total_tiles}")

                # Calculate tile bounds
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                # Extract tile
                tile = image.crop((x_start, y_start, x_end, y_end))

                # Pad if needed
                if tile.size != (tile_size, tile_size):
                    padded = Image.new("RGB", (tile_size, tile_size))
                    padded.paste(tile, (0, 0))
                    tile = padded

                # Process tile
                upscaled_tile = self._process_single(tile, prompt)

                # Convert to numpy for blending
                tile_np = np.array(upscaled_tile).astype(np.float32)

                # Calculate output position
                out_x = x_start * scale
                out_y = y_start * scale
                tile_h, tile_w = tile_np.shape[:2]

                # Get weights for this tile
                weights = tile_weights[:tile_h, :tile_w, :]

                # Accumulate
                output_sum[out_y:out_y + tile_h, out_x:out_x + tile_w] += tile_np * weights
                weight_sum[out_y:out_y + tile_h, out_x:out_x + tile_w] += weights

        # Normalize
        output_np = output_sum / (weight_sum + 1e-8)
        output_np = np.clip(output_np, 0, 255).astype(np.uint8)

        return Image.fromarray(output_np)

    def _create_tile_weights(self, tile_size: int, overlap: int) -> np.ndarray:
        """Create weight mask for tile blending."""
        weights = np.ones((tile_size, tile_size, 3), dtype=np.float32)

        if overlap > 0:
            # Create linear ramps for overlap regions
            ramp = np.linspace(0, 1, overlap)

            # Apply ramps to edges
            for i in range(overlap):
                weights[i, :, :] *= ramp[i]
                weights[-(i + 1), :, :] *= ramp[i]
                weights[:, i, :] *= ramp[i]
                weights[:, -(i + 1), :] *= ramp[i]

        return weights


class SDXLRefinerUpscaler(BaseUpscaler):
    """SDXL Refiner for detail enhancement.

    Uses SDXL's refiner model to add fine details to upscaled images.
    Best used after initial upscaling with another model.

    Quality: 9.5/10 for detail enhancement
    Speed: Moderate (~5-15s per image)
    VRAM: 8GB+ required
    """

    def __init__(
        self,
        config: UpscaleConfig | None = None,
        refiner_strength: float = 0.3,
        num_inference_steps: int = 20,
    ) -> None:
        """Initialize SDXL Refiner.

        Args:
            config: Base upscale config.
            refiner_strength: How much to refine (0-1).
            num_inference_steps: Number of denoising steps.
        """
        base_config = config or UpscaleConfig(scale=1)
        super().__init__(base_config)

        self.refiner_strength = refiner_strength
        self.num_inference_steps = num_inference_steps
        self.pipeline: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model_name(self) -> str:
        return "SDXL-Refiner"

    def load(self) -> None:
        """Load SDXL Refiner pipeline."""
        if self._loaded:
            return

        logger.info("Loading SDXL Refiner...")

        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
        except ImportError:
            raise ImportError(
                "diffusers is required. "
                "Install with: pip install diffusers transformers accelerate"
            )

        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        self.pipeline.enable_attention_slicing()
        self.pipeline = self.pipeline.to(self.device)

        self._loaded = True
        logger.info("SDXL Refiner loaded")

    def unload(self) -> None:
        """Unload model."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    def upscale(
        self,
        image: "ndarray",
        scale: int | None = None,
        prompt: str = "high quality, detailed, sharp",
    ) -> UpscaleResult:
        """Refine image details using SDXL.

        Note: This doesn't actually upscale, it enhances details.
        Use after upscaling with another model.
        """
        if not self._loaded:
            self.load()

        start_time = time.time()
        h, w = image.shape[:2]

        # Convert to PIL
        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image)

        # Run refiner
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt="blurry, low quality, artifacts",
                image=pil_image,
                strength=self.refiner_strength,
                num_inference_steps=self.num_inference_steps,
            )

        # Convert back
        output_rgb = np.array(result.images[0])
        output_bgr = output_rgb[:, :, ::-1].copy()

        processing_time = time.time() - start_time

        return UpscaleResult(
            image=output_bgr,
            original_size=(w, h),
            upscaled_size=(output_bgr.shape[1], output_bgr.shape[0]),
            scale_factor=1.0,
            model_used=self.model_name,
            processing_time=processing_time,
        )


# =============================================================================
# Convenience functions
# =============================================================================


def sd_upscale(
    image: "ndarray",
    prompt: str = "",
    steps: int = 20,
    noise_level: int = 20,
) -> "ndarray":
    """Upscale image using Stable Diffusion x4.

    Args:
        image: Input BGR image.
        prompt: Optional guidance prompt.
        steps: Number of inference steps (more = better, slower).
        noise_level: Noise level (0-100).

    Returns:
        Upscaled image (4x).
    """
    config = SDUpscaleConfig(
        num_inference_steps=steps,
        noise_level=noise_level,
    )

    with StableDiffusionUpscaler(sd_config=config) as upscaler:
        result = upscaler.upscale(image, prompt=prompt)
        return result.image


def sd_upscale_with_refiner(
    image: "ndarray",
    prompt: str = "",
    steps: int = 25,
    refiner_strength: float = 0.3,
) -> "ndarray":
    """Upscale with SD x4 then enhance with SDXL Refiner.

    Args:
        image: Input BGR image.
        prompt: Guidance prompt.
        steps: Inference steps for upscaler.
        refiner_strength: Refiner strength (0-1).

    Returns:
        Upscaled and refined image.
    """
    # First upscale
    upscaler_config = SDUpscaleConfig(num_inference_steps=steps)
    with StableDiffusionUpscaler(sd_config=upscaler_config) as upscaler:
        upscaled = upscaler.upscale(image, prompt=prompt)

    # Then refine
    with SDXLRefinerUpscaler(refiner_strength=refiner_strength) as refiner:
        refined = refiner.upscale(upscaled.image, prompt=prompt)

    return refined.image
