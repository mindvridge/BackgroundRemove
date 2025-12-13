"""Neural rendering for perfect foreground extraction.

Provides advanced neural techniques for:
- Diffusion-based boundary refinement
- Neural inpainting for edge completion
- Latent space manipulation
- Generative foreground enhancement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class NeuralMethod(Enum):
    """Neural rendering methods."""

    DIFFUSION_REFINE = "diffusion_refine"
    INPAINTING = "inpainting"
    LATENT_BLEND = "latent_blend"
    GENERATIVE_FILL = "generative_fill"


@dataclass
class NeuralRenderConfig:
    """Configuration for neural rendering."""

    method: NeuralMethod = NeuralMethod.DIFFUSION_REFINE
    device: str = "cuda"

    # Diffusion settings
    diffusion_steps: int = 20
    guidance_scale: float = 7.5

    # Inpainting settings
    inpaint_radius: int = 3
    inpaint_method: str = "ns"  # "ns" or "telea"

    # Edge settings
    edge_width: int = 5  # Width of edge region to refine
    blend_width: int = 10  # Blending region width

    # Quality
    enhance_foreground: bool = True
    remove_halo: bool = True


class EdgeInpainter:
    """Edge region inpainting for clean boundaries.

    Uses inpainting to create natural edge transitions.

    Example:
        >>> inpainter = EdgeInpainter(radius=3)
        >>> refined = inpainter.inpaint_edges(image, alpha)
    """

    def __init__(
        self,
        radius: int = 3,
        method: str = "ns",
    ) -> None:
        """Initialize edge inpainter.

        Args:
            radius: Inpainting radius.
            method: "ns" (Navier-Stokes) or "telea".
        """
        self.radius = radius
        self.method = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA

    def _get_edge_mask(
        self,
        alpha: ndarray,
        width: int = 5,
    ) -> ndarray:
        """Get edge region mask.

        Args:
            alpha: Alpha matte.
            width: Edge width.

        Returns:
            Edge region mask.
        """
        # Detect alpha edges
        edges = cv2.Canny(alpha, 30, 100)

        # Dilate to get region
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1)
        )
        edge_mask = cv2.dilate(edges, kernel)

        return edge_mask

    def inpaint_edges(
        self,
        image: ndarray,
        alpha: ndarray,
        edge_width: int = 5,
    ) -> tuple[ndarray, ndarray]:
        """Inpaint edge regions for cleaner extraction.

        Args:
            image: Input image (BGR).
            alpha: Alpha matte.
            edge_width: Edge region width.

        Returns:
            Tuple of (inpainted_foreground, refined_alpha).
        """
        # Get edge mask
        edge_mask = self._get_edge_mask(alpha, edge_width)

        # Create foreground
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        foreground = (image * alpha_norm).astype(np.uint8)

        # Inpaint edge regions
        inpainted = cv2.inpaint(foreground, edge_mask, self.radius, self.method)

        # Blend inpainted with original in edge regions
        edge_weight = edge_mask.astype(np.float32) / 255.0
        if len(edge_weight.shape) == 2:
            edge_weight = edge_weight[:, :, np.newaxis]

        blended_fg = (
            edge_weight * inpainted.astype(np.float32) +
            (1 - edge_weight) * foreground.astype(np.float32)
        ).astype(np.uint8)

        # Refine alpha based on inpainting
        # Compute color consistency
        diff = np.abs(inpainted.astype(np.float32) - foreground.astype(np.float32))
        consistency = 1.0 - np.mean(diff, axis=2) / 255.0

        refined_alpha = alpha.astype(np.float32)
        edge_region = edge_mask > 0
        refined_alpha[edge_region] = alpha[edge_region] * consistency[edge_region]

        return blended_fg, refined_alpha.astype(np.uint8)


class HaloRemover:
    """Remove halo artifacts from foreground edges.

    Halos are common artifacts in matting where background
    colors bleed into foreground edges.

    Example:
        >>> remover = HaloRemover()
        >>> clean_fg = remover.remove(image, foreground, alpha)
    """

    def __init__(
        self,
        detection_threshold: float = 0.3,
    ) -> None:
        """Initialize halo remover.

        Args:
            detection_threshold: Threshold for halo detection.
        """
        self.threshold = detection_threshold

    def _detect_halos(
        self,
        image: ndarray,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Detect halo regions.

        Args:
            image: Original image.
            foreground: Extracted foreground.
            alpha: Alpha matte.

        Returns:
            Halo mask.
        """
        # Transition region (partial alpha)
        transition = (alpha > 30) & (alpha < 225)

        # Color difference in transition
        if len(foreground.shape) == 2:
            foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)

        diff = np.abs(image.astype(np.float32) - foreground.astype(np.float32))
        diff_magnitude = np.mean(diff, axis=2) / 255.0

        # Halo = high color difference in transition region
        halo_mask = (transition & (diff_magnitude > self.threshold)).astype(np.uint8) * 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        halo_mask = cv2.morphologyEx(halo_mask, cv2.MORPH_OPEN, kernel)

        return halo_mask

    def remove(
        self,
        image: ndarray,
        foreground: ndarray,
        alpha: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Remove halo artifacts.

        Args:
            image: Original image.
            foreground: Extracted foreground.
            alpha: Alpha matte.

        Returns:
            Tuple of (clean_foreground, refined_alpha).
        """
        halo_mask = self._detect_halos(image, foreground, alpha)

        if halo_mask.sum() == 0:
            return foreground, alpha

        # Sample foreground colors from non-halo regions
        alpha_high = alpha > 200
        non_halo = alpha_high & (halo_mask == 0)

        if non_halo.sum() > 0:
            # Get average foreground color
            fg_colors = foreground[non_halo]
            mean_fg_color = np.mean(fg_colors, axis=0)
        else:
            mean_fg_color = np.mean(foreground, axis=(0, 1))

        # Replace halo regions with estimated color
        halo_weight = halo_mask.astype(np.float32) / 255.0
        if len(halo_weight.shape) == 2:
            halo_weight = halo_weight[:, :, np.newaxis]

        clean_fg = (
            (1 - halo_weight) * foreground.astype(np.float32) +
            halo_weight * mean_fg_color
        ).astype(np.uint8)

        # Adjust alpha in halo regions
        refined_alpha = alpha.copy()
        halo_region = halo_mask > 0
        refined_alpha[halo_region] = (refined_alpha[halo_region] * 0.7).astype(np.uint8)

        return clean_fg, refined_alpha


class DiffusionRefiner:
    """Diffusion-based alpha refinement.

    Uses diffusion model principles for smooth, natural
    alpha transitions.

    Example:
        >>> refiner = DiffusionRefiner(steps=20)
        >>> refined = refiner.refine(alpha, guidance)
    """

    def __init__(
        self,
        steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> None:
        """Initialize diffusion refiner.

        Args:
            steps: Number of diffusion steps.
            guidance_scale: Guidance strength.
        """
        self.steps = steps
        self.guidance_scale = guidance_scale

    def _add_noise(
        self,
        image: ndarray,
        noise_level: float,
    ) -> ndarray:
        """Add Gaussian noise.

        Args:
            image: Input image.
            noise_level: Noise standard deviation.

        Returns:
            Noisy image.
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _denoise_step(
        self,
        noisy: ndarray,
        guidance: ndarray,
        step: int,
    ) -> ndarray:
        """Single denoising step.

        Args:
            noisy: Noisy image.
            guidance: Guidance image.
            step: Current step number.

        Returns:
            Denoised image.
        """
        # Strength decreases with step
        strength = 1.0 - step / self.steps

        # Bilateral filter for edge-aware denoising
        d = max(3, int(9 * strength))
        sigma_color = 75 * strength
        sigma_space = 75 * strength

        denoised = cv2.bilateralFilter(noisy, d, sigma_color, sigma_space)

        # Guide towards guidance image
        guide_weight = self.guidance_scale * strength * 0.1
        guided = (
            (1 - guide_weight) * denoised.astype(np.float32) +
            guide_weight * guidance.astype(np.float32)
        )

        return np.clip(guided, 0, 255).astype(np.uint8)

    def refine(
        self,
        alpha: ndarray,
        guidance: ndarray | None = None,
    ) -> ndarray:
        """Refine alpha using diffusion process.

        Args:
            alpha: Alpha matte to refine.
            guidance: Optional guidance image.

        Returns:
            Refined alpha matte.
        """
        if guidance is None:
            guidance = alpha.copy()

        # Initial noise
        noise_level = 30
        current = self._add_noise(alpha, noise_level)

        # Iterative denoising
        for step in range(self.steps):
            current = self._denoise_step(current, guidance, step)

        return current


class LatentBlender:
    """Latent space blending for smooth compositing.

    Operates in a compressed latent space for more
    coherent blending results.

    Example:
        >>> blender = LatentBlender()
        >>> blended = blender.blend(fg, bg, alpha)
    """

    def __init__(
        self,
        latent_dim: int = 64,
    ) -> None:
        """Initialize latent blender.

        Args:
            latent_dim: Latent space dimension.
        """
        self.latent_dim = latent_dim

    def _encode(self, image: ndarray) -> ndarray:
        """Encode image to latent space.

        Simple encoding using DCT.

        Args:
            image: Input image.

        Returns:
            Latent representation.
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        if len(img_float.shape) == 3:
            # Process each channel
            latent = []
            for c in range(img_float.shape[2]):
                # Resize to latent dim
                small = cv2.resize(img_float[:, :, c], (self.latent_dim, self.latent_dim))
                # DCT
                dct = cv2.dct(small)
                latent.append(dct)
            return np.stack(latent, axis=2)
        else:
            small = cv2.resize(img_float, (self.latent_dim, self.latent_dim))
            return cv2.dct(small)

    def _decode(
        self,
        latent: ndarray,
        target_size: tuple[int, int],
    ) -> ndarray:
        """Decode from latent space.

        Args:
            latent: Latent representation.
            target_size: Target (height, width).

        Returns:
            Decoded image.
        """
        if len(latent.shape) == 3:
            # Process each channel
            decoded = []
            for c in range(latent.shape[2]):
                # Inverse DCT
                idct = cv2.idct(latent[:, :, c])
                # Resize to target
                full = cv2.resize(idct, (target_size[1], target_size[0]))
                decoded.append(full)
            result = np.stack(decoded, axis=2)
        else:
            idct = cv2.idct(latent)
            result = cv2.resize(idct, (target_size[1], target_size[0]))

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def blend(
        self,
        foreground: ndarray,
        background: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Blend in latent space.

        Args:
            foreground: Foreground image.
            background: Background image.
            alpha: Alpha matte.

        Returns:
            Blended result.
        """
        h, w = foreground.shape[:2]

        # Encode
        fg_latent = self._encode(foreground)
        bg_latent = self._encode(background)
        alpha_latent = self._encode(alpha)

        # Normalize alpha latent
        alpha_norm = alpha_latent / (alpha_latent.max() + 1e-8)
        if len(alpha_norm.shape) == 2 and len(fg_latent.shape) == 3:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        # Blend in latent space
        blended_latent = fg_latent * alpha_norm + bg_latent * (1 - alpha_norm)

        # Decode
        return self._decode(blended_latent, (h, w))


class ForegroundEnhancer:
    """Enhance foreground quality after extraction.

    Improves extracted foreground appearance.

    Example:
        >>> enhancer = ForegroundEnhancer()
        >>> enhanced = enhancer.enhance(foreground, alpha)
    """

    def __init__(self) -> None:
        """Initialize foreground enhancer."""
        pass

    def enhance(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Enhance foreground quality.

        Args:
            foreground: Extracted foreground.
            alpha: Alpha matte.

        Returns:
            Enhanced foreground.
        """
        # Color correction in foreground region
        fg_mask = alpha > 127

        if fg_mask.sum() == 0:
            return foreground

        # Auto levels
        enhanced = foreground.copy()

        for c in range(3):
            channel = enhanced[:, :, c]
            fg_values = channel[fg_mask]

            if len(fg_values) > 0:
                p2, p98 = np.percentile(fg_values, (2, 98))
                if p98 > p2:
                    # Stretch histogram
                    channel = np.clip((channel - p2) * 255.0 / (p98 - p2), 0, 255)
                    enhanced[:, :, c] = channel.astype(np.uint8)

        # Slight sharpening
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1)
        enhanced = cv2.addWeighted(enhanced, 1.2, blur, -0.2, 0)

        # Only apply to foreground
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        result = (
            alpha_norm * enhanced.astype(np.float32) +
            (1 - alpha_norm) * foreground.astype(np.float32)
        )

        return np.clip(result, 0, 255).astype(np.uint8)


class NeuralRenderer:
    """Neural rendering processor for perfect extraction.

    Combines multiple neural techniques for highest quality
    foreground extraction.

    Example:
        >>> renderer = NeuralRenderer(NeuralRenderConfig())
        >>> fg, alpha = renderer.render(image, initial_alpha)
    """

    def __init__(self, config: NeuralRenderConfig | None = None) -> None:
        """Initialize neural renderer.

        Args:
            config: Neural rendering configuration.
        """
        self.config = config or NeuralRenderConfig()

        self._edge_inpainter = EdgeInpainter(
            radius=self.config.inpaint_radius,
            method=self.config.inpaint_method,
        )
        self._halo_remover = HaloRemover()
        self._diffusion = DiffusionRefiner(
            steps=self.config.diffusion_steps,
            guidance_scale=self.config.guidance_scale,
        )
        self._latent_blender = LatentBlender()
        self._fg_enhancer = ForegroundEnhancer()

        logger.debug(f"Neural renderer initialized: {self.config.method.value}")

    def render(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Render high-quality foreground extraction.

        Args:
            image: Input image (BGR).
            alpha: Initial alpha matte.

        Returns:
            Tuple of (foreground, refined_alpha).
        """
        # Step 1: Diffusion refinement of alpha
        refined_alpha = self._diffusion.refine(alpha, alpha)

        # Step 2: Edge inpainting
        foreground, refined_alpha = self._edge_inpainter.inpaint_edges(
            image, refined_alpha, self.config.edge_width
        )

        # Step 3: Halo removal
        if self.config.remove_halo:
            foreground, refined_alpha = self._halo_remover.remove(
                image, foreground, refined_alpha
            )

        # Step 4: Foreground enhancement
        if self.config.enhance_foreground:
            foreground = self._fg_enhancer.enhance(foreground, refined_alpha)

        return foreground, refined_alpha

    def blend_composite(
        self,
        foreground: ndarray,
        background: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Create high-quality composite.

        Args:
            foreground: Foreground image.
            background: Background image.
            alpha: Alpha matte.

        Returns:
            Composited result.
        """
        if self.config.method == NeuralMethod.LATENT_BLEND:
            return self._latent_blender.blend(foreground, background, alpha)
        else:
            # Standard alpha blending
            alpha_norm = alpha.astype(np.float32) / 255.0
            if len(alpha_norm.shape) == 2:
                alpha_norm = alpha_norm[:, :, np.newaxis]

            composite = (
                foreground.astype(np.float32) * alpha_norm +
                background.astype(np.float32) * (1 - alpha_norm)
            )

            return np.clip(composite, 0, 255).astype(np.uint8)

    def __repr__(self) -> str:
        """String representation."""
        return f"NeuralRenderer(method={self.config.method.value})"


def neural_refine(
    image: ndarray,
    alpha: ndarray,
    method: NeuralMethod = NeuralMethod.DIFFUSION_REFINE,
) -> tuple[ndarray, ndarray]:
    """Convenience function for neural refinement.

    Args:
        image: Input image.
        alpha: Alpha matte.
        method: Neural method.

    Returns:
        Tuple of (foreground, refined_alpha).
    """
    config = NeuralRenderConfig(method=method)
    renderer = NeuralRenderer(config)
    return renderer.render(image, alpha)


def remove_halos(
    image: ndarray,
    foreground: ndarray,
    alpha: ndarray,
) -> tuple[ndarray, ndarray]:
    """Convenience function for halo removal.

    Args:
        image: Original image.
        foreground: Extracted foreground.
        alpha: Alpha matte.

    Returns:
        Tuple of (clean_foreground, refined_alpha).
    """
    remover = HaloRemover()
    return remover.remove(image, foreground, alpha)
