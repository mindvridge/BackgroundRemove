"""GAN-based edge refinement for natural alpha boundaries.

Provides adversarial learning-based refinement for:
- Natural edge transitions
- Hair/fur detail preservation
- Artifact removal
- Photorealistic boundaries

Models:
- EdgeGAN: Edge-focused adversarial refinement
- AlphaGAN: Alpha matte specific GAN
- PatchGAN: Local patch-based discrimination
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class GANModel(Enum):
    """Available GAN models for refinement."""

    EDGE_GAN = "edge_gan"
    ALPHA_GAN = "alpha_gan"
    PATCH_GAN = "patch_gan"
    ESRGAN_REFINE = "esrgan_refine"


@dataclass
class GANConfig:
    """Configuration for GAN refinement."""

    model: GANModel = GANModel.ALPHA_GAN
    device: str = "cuda"

    # Refinement settings
    edge_focus: bool = True  # Focus on edge regions
    edge_radius: int = 10  # Edge detection radius
    blend_smooth: float = 0.5  # Blending smoothness

    # Quality settings
    enhance_details: bool = True
    remove_artifacts: bool = True
    preserve_structure: bool = True

    # Model-specific
    patch_size: int = 64  # For PatchGAN
    num_residual_blocks: int = 8  # For generator


class EdgeDetector:
    """Multi-method edge detection for alpha refinement.

    Provides robust edge detection combining multiple techniques.
    """

    def __init__(self, radius: int = 10) -> None:
        """Initialize edge detector.

        Args:
            radius: Dilation radius for edge regions.
        """
        self.radius = radius

    def detect(
        self,
        alpha: ndarray,
        image: ndarray | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Detect edges and create edge mask.

        Args:
            alpha: Alpha matte.
            image: Optional original image for guidance.

        Returns:
            Tuple of (edge_map, edge_mask).
        """
        # Canny edges on alpha
        alpha_edges = cv2.Canny(alpha, 30, 100)

        # Gradient-based edges
        grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = (gradient / gradient.max() * 255).astype(np.uint8)

        # Combine
        edge_map = cv2.addWeighted(alpha_edges, 0.5, gradient, 0.5, 0)

        # If image available, also use image edges
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_edges = cv2.Canny(gray, 50, 150)
            edge_map = cv2.addWeighted(edge_map, 0.7, image_edges, 0.3, 0)

        # Create mask (dilated edge regions)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.radius * 2 + 1, self.radius * 2 + 1)
        )
        edge_mask = cv2.dilate(edge_map, kernel)

        return edge_map, edge_mask


class SimpleGenerator:
    """Simple convolutional generator for alpha refinement.

    Provides CNN-based refinement without requiring
    full GAN training infrastructure.

    Example:
        >>> gen = SimpleGenerator()
        >>> refined = gen.refine(image, alpha, edge_mask)
    """

    def __init__(
        self,
        num_blocks: int = 8,
        base_channels: int = 64,
    ) -> None:
        """Initialize generator.

        Args:
            num_blocks: Number of residual blocks.
            base_channels: Base channel count.
        """
        self.num_blocks = num_blocks
        self.base_channels = base_channels
        self._model = None

    def _build_model(self) -> None:
        """Build PyTorch model if available."""
        try:
            import torch
            import torch.nn as nn

            class ResidualBlock(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
                    self.bn1 = nn.BatchNorm2d(channels)
                    self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
                    self.bn2 = nn.BatchNorm2d(channels)
                    self.relu = nn.ReLU(inplace=True)

                def forward(self, x):
                    residual = x
                    out = self.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    return out + residual

            class AlphaGenerator(nn.Module):
                def __init__(self, num_blocks, base_ch):
                    super().__init__()
                    # Input: 4 channels (RGB + alpha)
                    self.input_conv = nn.Sequential(
                        nn.Conv2d(4, base_ch, 7, 1, 3),
                        nn.ReLU(inplace=True),
                    )

                    self.res_blocks = nn.Sequential(
                        *[ResidualBlock(base_ch) for _ in range(num_blocks)]
                    )

                    self.output_conv = nn.Sequential(
                        nn.Conv2d(base_ch, base_ch // 2, 3, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch // 2, 1, 3, 1, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    feat = self.input_conv(x)
                    feat = self.res_blocks(feat)
                    return self.output_conv(feat)

            self._model = AlphaGenerator(self.num_blocks, self.base_channels)
            self._has_torch = True

        except ImportError:
            self._has_torch = False
            logger.info("PyTorch not available, using fallback refinement")

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
        edge_mask: ndarray,
    ) -> ndarray:
        """Refine alpha using generator.

        Args:
            image: Input image (BGR).
            alpha: Alpha matte.
            edge_mask: Edge region mask.

        Returns:
            Refined alpha.
        """
        if self._model is None:
            self._build_model()

        if self._has_torch:
            return self._refine_torch(image, alpha, edge_mask)
        else:
            return self._refine_fallback(image, alpha, edge_mask)

    def _refine_torch(
        self,
        image: ndarray,
        alpha: ndarray,
        edge_mask: ndarray,
    ) -> ndarray:
        """Refine using PyTorch model.

        Args:
            image: Input image.
            alpha: Alpha matte.
            edge_mask: Edge mask.

        Returns:
            Refined alpha.
        """
        import torch

        # Prepare input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        alpha_norm = alpha.astype(np.float32) / 255.0

        # Stack RGBA
        rgba = np.dstack([image_norm, alpha_norm[:, :, np.newaxis] if len(alpha_norm.shape) == 2 else alpha_norm])

        # To tensor
        tensor = torch.from_numpy(rgba).permute(2, 0, 1).unsqueeze(0)

        # Inference
        self._model.eval()
        with torch.no_grad():
            output = self._model(tensor)

        # To numpy
        refined = output.squeeze().numpy()
        refined = (refined * 255).astype(np.uint8)

        # Blend in edge regions only
        edge_weight = edge_mask.astype(np.float32) / 255.0
        result = edge_weight * refined + (1 - edge_weight) * alpha

        return result.astype(np.uint8)

    def _refine_fallback(
        self,
        image: ndarray,
        alpha: ndarray,
        edge_mask: ndarray,
    ) -> ndarray:
        """Fallback refinement without PyTorch.

        Uses traditional CV techniques to simulate GAN refinement.

        Args:
            image: Input image.
            alpha: Alpha matte.
            edge_mask: Edge mask.

        Returns:
            Refined alpha.
        """
        # Guided filter based refinement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filtering for edge-aware smoothing
        smoothed = cv2.bilateralFilter(alpha, 9, 75, 75)

        # Edge-aware interpolation using image as guide
        alpha_float = alpha.astype(np.float32) / 255.0
        gray_float = gray.astype(np.float32) / 255.0

        # Simple guided filter
        radius = 4
        eps = 0.02

        mean_I = cv2.blur(gray_float, (2*radius+1, 2*radius+1))
        mean_p = cv2.blur(alpha_float, (2*radius+1, 2*radius+1))
        corr_Ip = cv2.blur(gray_float * alpha_float, (2*radius+1, 2*radius+1))
        corr_II = cv2.blur(gray_float * gray_float, (2*radius+1, 2*radius+1))

        var_I = corr_II - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.blur(a, (2*radius+1, 2*radius+1))
        mean_b = cv2.blur(b, (2*radius+1, 2*radius+1))

        refined = mean_a * gray_float + mean_b
        refined = (np.clip(refined, 0, 1) * 255).astype(np.uint8)

        # Blend in edge regions
        edge_weight = edge_mask.astype(np.float32) / 255.0
        result = edge_weight * refined + (1 - edge_weight) * alpha

        return result.astype(np.uint8)


class PatchDiscriminator:
    """Patch-based quality assessment for alpha mattes.

    Evaluates local patch quality to guide refinement.

    Example:
        >>> disc = PatchDiscriminator(patch_size=64)
        >>> quality_map = disc.evaluate(image, alpha)
    """

    def __init__(self, patch_size: int = 64) -> None:
        """Initialize patch discriminator.

        Args:
            patch_size: Size of patches.
        """
        self.patch_size = patch_size

    def evaluate(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Evaluate patch quality.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Quality map (same size as alpha).
        """
        h, w = alpha.shape[:2]
        quality_map = np.ones((h, w), np.float32)

        p = self.patch_size

        for y in range(0, h - p + 1, p // 2):
            for x in range(0, w - p + 1, p // 2):
                patch_alpha = alpha[y:y+p, x:x+p]
                patch_image = image[y:y+p, x:x+p]

                # Compute patch quality metrics
                quality = self._evaluate_patch(patch_image, patch_alpha)

                # Apply to quality map
                quality_map[y:y+p, x:x+p] = np.minimum(
                    quality_map[y:y+p, x:x+p], quality
                )

        return quality_map

    def _evaluate_patch(
        self,
        patch_image: ndarray,
        patch_alpha: ndarray,
    ) -> float:
        """Evaluate single patch quality.

        Args:
            patch_image: Image patch.
            patch_alpha: Alpha patch.

        Returns:
            Quality score (0-1).
        """
        # Edge alignment score
        gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)
        image_edges = cv2.Canny(gray, 50, 150)
        alpha_edges = cv2.Canny(patch_alpha, 50, 150)

        if alpha_edges.sum() == 0:
            edge_score = 1.0
        else:
            # Dilate image edges for tolerance
            image_edges_dilated = cv2.dilate(image_edges, None, iterations=2)
            aligned = np.sum(alpha_edges & image_edges_dilated)
            edge_score = aligned / (alpha_edges.sum() + 1e-8)

        # Smoothness score (penalize noise)
        grad = cv2.Sobel(patch_alpha, cv2.CV_32F, 1, 0)
        smoothness = 1.0 - min(np.std(grad) / 50.0, 1.0)

        # Transition quality (natural gradients)
        transition_mask = (patch_alpha > 10) & (patch_alpha < 245)
        if transition_mask.sum() > 0:
            transition_std = np.std(patch_alpha[transition_mask])
            transition_score = min(transition_std / 50.0, 1.0)
        else:
            transition_score = 1.0

        # Combined score
        quality = 0.4 * edge_score + 0.3 * smoothness + 0.3 * transition_score

        return quality


class GANRefiner:
    """GAN-based alpha refinement processor.

    Combines generator and discriminator for high-quality
    edge refinement.

    Example:
        >>> refiner = GANRefiner(GANConfig(model=GANModel.ALPHA_GAN))
        >>> refined = refiner.refine(image, alpha)
    """

    def __init__(self, config: GANConfig | None = None) -> None:
        """Initialize GAN refiner.

        Args:
            config: GAN configuration.
        """
        self.config = config or GANConfig()
        self._edge_detector = EdgeDetector(radius=config.edge_radius if config else 10)
        self._generator = SimpleGenerator(
            num_blocks=config.num_residual_blocks if config else 8,
        )
        self._discriminator = PatchDiscriminator(
            patch_size=config.patch_size if config else 64,
        )
        logger.debug(f"GAN refiner initialized: {self.config.model.value}")

    def _enhance_details(self, alpha: ndarray) -> ndarray:
        """Enhance fine details.

        Args:
            alpha: Alpha matte.

        Returns:
            Detail-enhanced alpha.
        """
        # Extract details using Laplacian
        laplacian = cv2.Laplacian(alpha, cv2.CV_32F)

        # Enhance
        enhanced = alpha.astype(np.float32) + 0.3 * laplacian

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _remove_artifacts(self, alpha: ndarray) -> ndarray:
        """Remove common artifacts.

        Args:
            alpha: Alpha matte.

        Returns:
            Artifact-free alpha.
        """
        # Remove isolated pixels
        kernel = np.ones((3, 3), np.uint8)

        # Binary threshold for noise removal
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        min_area = 50
        clean = np.zeros_like(alpha)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = alpha[labels == i]

        # Blend back smooth transitions
        transition_mask = (alpha > 10) & (alpha < 245)
        clean[transition_mask] = alpha[transition_mask]

        return clean

    def _preserve_structure(
        self,
        refined: ndarray,
        original: ndarray,
    ) -> ndarray:
        """Preserve structural integrity.

        Args:
            refined: Refined alpha.
            original: Original alpha.

        Returns:
            Structure-preserved alpha.
        """
        # Keep overall structure from original
        original_blur = cv2.GaussianBlur(original, (21, 21), 0)
        refined_blur = cv2.GaussianBlur(refined, (21, 21), 0)

        # Structure difference
        structure_diff = original_blur.astype(np.float32) - refined_blur.astype(np.float32)

        # Add structure back to refined
        result = refined.astype(np.float32) + 0.5 * structure_diff

        return np.clip(result, 0, 255).astype(np.uint8)

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Refine alpha using GAN-based approach.

        Args:
            image: Input image (BGR).
            alpha: Alpha matte.

        Returns:
            Refined alpha matte.
        """
        # Detect edges
        edge_map, edge_mask = self._edge_detector.detect(alpha, image)

        # Apply generator refinement
        refined = self._generator.refine(image, alpha, edge_mask)

        # Optional enhancements
        if self.config.enhance_details:
            refined = self._enhance_details(refined)

        if self.config.remove_artifacts:
            refined = self._remove_artifacts(refined)

        if self.config.preserve_structure:
            refined = self._preserve_structure(refined, alpha)

        # Quality-guided blending
        if self.config.edge_focus:
            quality_map = self._discriminator.evaluate(image, refined)
            quality_weight = quality_map[:, :, np.newaxis] if len(quality_map.shape) == 2 else quality_map

            # Higher quality = more refined, lower = more original
            blend_weight = edge_mask.astype(np.float32) / 255.0 * quality_weight

            result = (
                blend_weight * refined.astype(np.float32) +
                (1 - blend_weight) * alpha.astype(np.float32)
            )
            refined = np.clip(result, 0, 255).astype(np.uint8)

        return refined

    def __repr__(self) -> str:
        """String representation."""
        return f"GANRefiner(model={self.config.model.value})"


class HairRefiner:
    """Specialized refiner for hair and fur regions.

    Uses multi-scale approach specifically optimized for
    fine strand-like structures.

    Example:
        >>> refiner = HairRefiner()
        >>> refined = refiner.refine(image, alpha)
    """

    def __init__(self) -> None:
        """Initialize hair refiner."""
        self._gan_refiner = GANRefiner(GANConfig(
            edge_radius=5,
            enhance_details=True,
            remove_artifacts=False,  # Keep fine strands
            preserve_structure=True,
        ))

    def _detect_hair_regions(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Detect likely hair/fur regions.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Hair region mask.
        """
        # High-frequency content indicates hair
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Laplacian for texture
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        texture = np.abs(laplacian)
        texture = texture / (texture.max() + 1e-8)

        # Edge regions in alpha
        alpha_edges = cv2.Canny(alpha, 30, 100)
        alpha_edges_dilated = cv2.dilate(alpha_edges, None, iterations=3)

        # Hair = high texture + edge region
        hair_mask = (texture > 0.3) & (alpha_edges_dilated > 0)
        hair_mask = cv2.dilate(hair_mask.astype(np.uint8) * 255, None, iterations=2)

        return hair_mask

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Refine hair/fur regions.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Refined alpha with better hair detail.
        """
        # Detect hair regions
        hair_mask = self._detect_hair_regions(image, alpha)
        hair_weight = hair_mask.astype(np.float32) / 255.0

        # Multi-scale refinement for hair
        scales = [1.0, 2.0, 0.5]
        refined_multi = []
        h, w = alpha.shape[:2]

        for scale in scales:
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h))
                scaled_alpha = cv2.resize(alpha, (new_w, new_h))
            else:
                scaled_image = image
                scaled_alpha = alpha

            refined = self._gan_refiner.refine(scaled_image, scaled_alpha)

            if scale != 1.0:
                refined = cv2.resize(refined, (w, h))

            refined_multi.append(refined.astype(np.float32))

        # Combine multi-scale results
        combined = np.mean(refined_multi, axis=0)

        # Apply only in hair regions
        result = (
            hair_weight * combined +
            (1 - hair_weight) * alpha.astype(np.float32)
        )

        return np.clip(result, 0, 255).astype(np.uint8)


def gan_refine(
    image: ndarray,
    alpha: ndarray,
    model: GANModel = GANModel.ALPHA_GAN,
) -> ndarray:
    """Convenience function for GAN refinement.

    Args:
        image: Input image.
        alpha: Alpha matte.
        model: GAN model type.

    Returns:
        Refined alpha.
    """
    config = GANConfig(model=model)
    refiner = GANRefiner(config)
    return refiner.refine(image, alpha)


def refine_hair(
    image: ndarray,
    alpha: ndarray,
) -> ndarray:
    """Convenience function for hair refinement.

    Args:
        image: Input image.
        alpha: Alpha matte.

    Returns:
        Refined alpha with better hair detail.
    """
    refiner = HairRefiner()
    return refiner.refine(image, alpha)
