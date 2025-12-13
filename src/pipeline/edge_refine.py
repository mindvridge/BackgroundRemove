"""Edge refinement module for video background removal.

Provides advanced edge handling for hair, fur, and fine details
using guided filtering, alpha matting, and morphological operations.

Techniques implemented:
- Guided filter for edge-aware smoothing
- KNN-based alpha matting
- Trimap generation and refinement
- Morphological edge enhancement
- Detail-preserving feathering
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


class EdgeMethod(Enum):
    """Edge refinement method."""

    NONE = "none"  # No refinement
    GUIDED_FILTER = "guided_filter"  # Guided filter only
    ALPHA_MATTING = "alpha_matting"  # KNN matting
    MORPHOLOGICAL = "morphological"  # Morphological operations
    COMBINED = "combined"  # All techniques combined
    FAST = "fast"  # Fast mode for real-time


@dataclass
class EdgeConfig:
    """Configuration for edge refinement."""

    method: EdgeMethod = EdgeMethod.COMBINED

    # Guided filter parameters
    guided_radius: int = 8  # Filter radius
    guided_eps: float = 0.01  # Regularization (lower = sharper edges)

    # Alpha matting parameters
    matting_kernel_size: int = 3  # KNN kernel size
    matting_iterations: int = 5  # Number of iterations
    erode_size: int = 10  # Erode size for trimap foreground
    dilate_size: int = 20  # Dilate size for trimap unknown

    # Morphological parameters
    morph_kernel_size: int = 3  # Morphological kernel size
    edge_detect_low: int = 50  # Canny low threshold
    edge_detect_high: int = 150  # Canny high threshold

    # Feathering parameters
    feather_amount: int = 3  # Feathering radius
    feather_blend: float = 0.5  # Blend with original (0-1)

    # Detail preservation
    preserve_detail: bool = True  # Enable detail preservation
    detail_threshold: float = 0.1  # Detail sensitivity


class GuidedFilter:
    """Fast guided filter implementation.

    Edge-aware filter that preserves edges while smoothing.
    Based on "Guided Image Filtering" by He et al.

    Example:
        >>> gf = GuidedFilter(radius=8, eps=0.01)
        >>> filtered = gf.filter(image, guide)
    """

    def __init__(
        self,
        radius: int = 8,
        eps: float = 0.01,
    ) -> None:
        """Initialize guided filter.

        Args:
            radius: Filter radius.
            eps: Regularization parameter.
        """
        self.radius = radius
        self.eps = eps

    def _box_filter(self, image: ndarray, radius: int) -> ndarray:
        """Apply box filter (mean filter).

        Args:
            image: Input image.
            radius: Filter radius.

        Returns:
            Filtered image.
        """
        ksize = 2 * radius + 1
        return cv2.blur(image, (ksize, ksize))

    def filter(
        self,
        image: ndarray,
        guide: ndarray | None = None,
    ) -> ndarray:
        """Apply guided filter.

        Args:
            image: Input image to filter (H, W) or (H, W, C).
            guide: Guide image (defaults to input image).

        Returns:
            Filtered image.
        """
        if guide is None:
            guide = image

        # Ensure float32
        image = image.astype(np.float32)
        guide = guide.astype(np.float32)

        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0
        if guide.max() > 1.0:
            guide = guide / 255.0

        # Handle grayscale guide
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)

        r = self.radius
        eps = self.eps

        # Mean of I, p, I*I, I*p
        mean_I = self._box_filter(guide, r)
        mean_p = self._box_filter(image, r)
        mean_Ip = self._box_filter(guide * image, r)
        mean_II = self._box_filter(guide * guide, r)

        # Variance and covariance
        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p

        # Linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Mean of a and b
        mean_a = self._box_filter(a, r)
        mean_b = self._box_filter(b, r)

        # Output
        output = mean_a * guide + mean_b

        # Restore range
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        return output


class TrimapGenerator:
    """Generate trimap from alpha matte.

    Creates foreground, background, and unknown regions
    for alpha matting algorithms.

    Example:
        >>> generator = TrimapGenerator(erode=10, dilate=20)
        >>> trimap = generator.generate(alpha)
    """

    def __init__(
        self,
        erode_size: int = 10,
        dilate_size: int = 20,
    ) -> None:
        """Initialize trimap generator.

        Args:
            erode_size: Erosion size for foreground.
            dilate_size: Dilation size for unknown region.
        """
        self.erode_size = erode_size
        self.dilate_size = dilate_size

    def generate(
        self,
        alpha: ndarray,
        threshold: float = 0.5,
    ) -> ndarray:
        """Generate trimap from alpha matte.

        Args:
            alpha: Alpha matte (0-255 or 0-1).
            threshold: Threshold for foreground/background.

        Returns:
            Trimap (0=background, 128=unknown, 255=foreground).
        """
        # Normalize
        if alpha.max() <= 1.0:
            alpha = (alpha * 255).astype(np.uint8)
        else:
            alpha = alpha.astype(np.uint8)

        # Binary mask
        threshold_value = int(threshold * 255)
        _, binary = cv2.threshold(alpha, threshold_value, 255, cv2.THRESH_BINARY)

        # Create kernels
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.erode_size, self.erode_size),
        )
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.dilate_size, self.dilate_size),
        )

        # Erode for definite foreground
        foreground = cv2.erode(binary, erode_kernel)

        # Dilate for possible foreground (includes unknown)
        possible_fg = cv2.dilate(binary, dilate_kernel)

        # Create trimap
        trimap = np.zeros_like(alpha)
        trimap[possible_fg > 0] = 128  # Unknown
        trimap[foreground > 0] = 255  # Foreground

        return trimap


class KNNMatting:
    """KNN-based alpha matting.

    Refines alpha matte using K-nearest neighbors
    color sampling from known regions.

    Example:
        >>> matting = KNNMatting(kernel_size=3)
        >>> refined = matting.refine(image, trimap)
    """

    def __init__(
        self,
        kernel_size: int = 3,
        iterations: int = 5,
    ) -> None:
        """Initialize KNN matting.

        Args:
            kernel_size: Kernel size for neighbor sampling.
            iterations: Number of refinement iterations.
        """
        self.kernel_size = kernel_size
        self.iterations = iterations

    def refine(
        self,
        image: ndarray,
        trimap: ndarray,
    ) -> ndarray:
        """Refine alpha using KNN matting.

        Args:
            image: Input image (BGR).
            trimap: Trimap (0=bg, 128=unknown, 255=fg).

        Returns:
            Refined alpha matte.
        """
        # Convert to float
        image = image.astype(np.float32) / 255.0
        trimap = trimap.astype(np.float32)

        # Initialize alpha
        alpha = np.zeros_like(trimap)
        alpha[trimap == 255] = 1.0
        alpha[trimap == 0] = 0.0
        alpha[trimap == 128] = 0.5

        # Unknown region mask
        unknown = (trimap == 128)

        if not unknown.any():
            return (alpha * 255).astype(np.uint8)

        # Get foreground and background colors
        fg_mask = (trimap == 255)
        bg_mask = (trimap == 0)

        # Sample colors
        fg_colors = image[fg_mask] if fg_mask.any() else np.array([[1, 1, 1]])
        bg_colors = image[bg_mask] if bg_mask.any() else np.array([[0, 0, 0]])

        # Mean colors
        fg_mean = fg_colors.mean(axis=0) if len(fg_colors) > 0 else np.array([1, 1, 1])
        bg_mean = bg_colors.mean(axis=0) if len(bg_colors) > 0 else np.array([0, 0, 0])

        # Iterative refinement
        for _ in range(self.iterations):
            unknown_coords = np.where(unknown)

            for y, x in zip(*unknown_coords):
                # Get local patch
                y1 = max(0, y - self.kernel_size)
                y2 = min(image.shape[0], y + self.kernel_size + 1)
                x1 = max(0, x - self.kernel_size)
                x2 = min(image.shape[1], x + self.kernel_size + 1)

                patch = image[y1:y2, x1:x2]
                patch_alpha = alpha[y1:y2, x1:x2]

                # Current pixel color
                pixel_color = image[y, x]

                # Distance to foreground and background
                fg_dist = np.linalg.norm(pixel_color - fg_mean)
                bg_dist = np.linalg.norm(pixel_color - bg_mean)

                # Compute alpha based on color distance
                total_dist = fg_dist + bg_dist + 1e-8
                local_alpha = bg_dist / total_dist

                # Blend with neighbor alpha
                neighbor_alpha = patch_alpha.mean()
                alpha[y, x] = 0.7 * local_alpha + 0.3 * neighbor_alpha

        return (np.clip(alpha, 0, 1) * 255).astype(np.uint8)


class EdgeRefiner:
    """Edge refinement processor for video matting.

    Provides multiple techniques for improving edge quality
    in background removal results.

    Example:
        >>> refiner = EdgeRefiner(EdgeConfig(method=EdgeMethod.COMBINED))
        >>> refined_alpha = refiner.refine(frame, alpha)
    """

    def __init__(self, config: EdgeConfig | None = None) -> None:
        """Initialize edge refiner.

        Args:
            config: Edge refinement configuration.
        """
        self.config = config or EdgeConfig()
        self._guided_filter = GuidedFilter(
            radius=self.config.guided_radius,
            eps=self.config.guided_eps,
        )
        self._trimap_gen = TrimapGenerator(
            erode_size=self.config.erode_size,
            dilate_size=self.config.dilate_size,
        )
        self._knn_matting = KNNMatting(
            kernel_size=self.config.matting_kernel_size,
            iterations=self.config.matting_iterations,
        )
        logger.debug(f"Edge refiner initialized: {self.config.method.value}")

    def _apply_guided_filter(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply guided filter refinement.

        Args:
            frame: Input frame (BGR).
            alpha: Alpha matte.

        Returns:
            Refined alpha.
        """
        # Use frame as guide for edge-aware smoothing
        gray_guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._guided_filter.filter(alpha, gray_guide)

    def _apply_alpha_matting(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply alpha matting refinement.

        Args:
            frame: Input frame (BGR).
            alpha: Alpha matte.

        Returns:
            Refined alpha.
        """
        # Generate trimap
        trimap = self._trimap_gen.generate(alpha)

        # Apply KNN matting
        return self._knn_matting.refine(frame, trimap)

    def _apply_morphological(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply morphological edge enhancement.

        Args:
            frame: Input frame (BGR).
            alpha: Alpha matte.

        Returns:
            Refined alpha.
        """
        ksize = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (ksize, ksize),
        )

        # Detect edges in original image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            self.config.edge_detect_low,
            self.config.edge_detect_high,
        )

        # Dilate edges to create edge region
        edge_region = cv2.dilate(edges, kernel, iterations=2)

        # Opening to remove noise
        opened = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

        # Closing to fill holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # In edge regions, preserve original details
        edge_mask = edge_region.astype(np.float32) / 255.0
        edge_mask = edge_mask[:, :, np.newaxis] if len(edge_mask.shape) == 2 else edge_mask

        alpha_float = alpha.astype(np.float32)
        closed_float = closed.astype(np.float32)

        # Blend: edge regions keep original, non-edge regions use morphological
        result = edge_mask * alpha_float + (1 - edge_mask) * closed_float

        return result.astype(np.uint8)

    def _apply_combined(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply combined edge refinement.

        Args:
            frame: Input frame (BGR).
            alpha: Alpha matte.

        Returns:
            Refined alpha.
        """
        # Step 1: Guided filter for initial smoothing
        guided = self._apply_guided_filter(frame, alpha)

        # Step 2: Morphological for structure
        morphed = self._apply_morphological(frame, guided)

        # Step 3: Alpha matting for fine details
        # Only apply in edge regions to save computation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edge_region = cv2.dilate(edges, None, iterations=3)

        if edge_region.any():
            # Generate trimap only in edge regions
            trimap = self._trimap_gen.generate(morphed)

            # Only refine unknown regions near edges
            edge_mask = (edge_region > 0) & (trimap == 128)
            if edge_mask.any():
                matted = self._knn_matting.refine(frame, trimap)

                # Blend matted result in edge regions
                edge_mask_float = edge_mask.astype(np.float32)
                morphed_float = morphed.astype(np.float32)
                matted_float = matted.astype(np.float32)

                result = (
                    edge_mask_float * matted_float +
                    (1 - edge_mask_float) * morphed_float
                )
                morphed = result.astype(np.uint8)

        return morphed

    def _apply_fast(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply fast edge refinement for real-time.

        Args:
            frame: Input frame (BGR).
            alpha: Alpha matte.

        Returns:
            Refined alpha.
        """
        # Fast guided filter with small radius
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downscale for speed
        h, w = alpha.shape[:2]
        scale = 0.5
        small_alpha = cv2.resize(alpha, (int(w * scale), int(h * scale)))
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        # Quick guided filter
        gf = GuidedFilter(radius=4, eps=0.02)
        filtered = gf.filter(small_alpha, small_gray)

        # Upscale
        filtered = cv2.resize(filtered, (w, h))

        # Quick morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        return filtered

    def _apply_feathering(
        self,
        alpha: ndarray,
        original_alpha: ndarray,
    ) -> ndarray:
        """Apply edge feathering.

        Args:
            alpha: Refined alpha.
            original_alpha: Original alpha for blending.

        Returns:
            Feathered alpha.
        """
        amount = self.config.feather_amount
        if amount <= 0:
            return alpha

        # Detect edge regions
        edges = cv2.Canny(alpha, 50, 150)
        edge_region = cv2.dilate(edges, None, iterations=amount)
        edge_mask = edge_region.astype(np.float32) / 255.0

        # Apply Gaussian blur to edges
        ksize = amount * 2 + 1
        blurred = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

        # Blend
        alpha_float = alpha.astype(np.float32)
        blurred_float = blurred.astype(np.float32)

        blend = self.config.feather_blend
        feathered = (
            edge_mask * (blend * blurred_float + (1 - blend) * alpha_float) +
            (1 - edge_mask) * alpha_float
        )

        return feathered.astype(np.uint8)

    def _preserve_details(
        self,
        frame: ndarray,
        refined: ndarray,
        original: ndarray,
    ) -> ndarray:
        """Preserve fine details from original alpha.

        Args:
            frame: Input frame.
            refined: Refined alpha.
            original: Original alpha.

        Returns:
            Alpha with preserved details.
        """
        if not self.config.preserve_detail:
            return refined

        # Detect high-frequency details in original
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Laplacian for detail detection
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        detail_mask = np.abs(laplacian)
        detail_mask = detail_mask / (detail_mask.max() + 1e-8)

        # Threshold
        threshold = self.config.detail_threshold
        detail_mask = np.clip((detail_mask - threshold) / (1 - threshold), 0, 1)

        # In detail regions, blend original back
        refined_float = refined.astype(np.float32)
        original_float = original.astype(np.float32)

        result = (
            detail_mask * (0.5 * original_float + 0.5 * refined_float) +
            (1 - detail_mask) * refined_float
        )

        return result.astype(np.uint8)

    def refine(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Refine alpha matte edges.

        Args:
            frame: Input frame (BGR format).
            alpha: Input alpha matte (0-255).

        Returns:
            Refined alpha matte.
        """
        method = self.config.method
        original = alpha.copy()

        if method == EdgeMethod.NONE:
            result = alpha
        elif method == EdgeMethod.GUIDED_FILTER:
            result = self._apply_guided_filter(frame, alpha)
        elif method == EdgeMethod.ALPHA_MATTING:
            result = self._apply_alpha_matting(frame, alpha)
        elif method == EdgeMethod.MORPHOLOGICAL:
            result = self._apply_morphological(frame, alpha)
        elif method == EdgeMethod.COMBINED:
            result = self._apply_combined(frame, alpha)
        elif method == EdgeMethod.FAST:
            result = self._apply_fast(frame, alpha)
        else:
            result = alpha

        # Apply feathering
        result = self._apply_feathering(result, original)

        # Preserve details
        result = self._preserve_details(frame, result, original)

        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"EdgeRefiner(method={self.config.method.value})"


def refine_edges(
    frame: ndarray,
    alpha: ndarray,
    method: EdgeMethod = EdgeMethod.COMBINED,
    **kwargs,
) -> ndarray:
    """Convenience function for edge refinement.

    Args:
        frame: Input frame (BGR).
        alpha: Alpha matte.
        method: Edge refinement method.
        **kwargs: Additional config parameters.

    Returns:
        Refined alpha matte.
    """
    config = EdgeConfig(method=method, **kwargs)
    refiner = EdgeRefiner(config)
    return refiner.refine(frame, alpha)


def guided_filter(
    image: ndarray,
    guide: ndarray | None = None,
    radius: int = 8,
    eps: float = 0.01,
) -> ndarray:
    """Apply guided filter to image.

    Convenience wrapper for GuidedFilter class.

    Args:
        image: Image to filter.
        guide: Guide image (defaults to image).
        radius: Filter radius.
        eps: Regularization.

    Returns:
        Filtered image.
    """
    gf = GuidedFilter(radius=radius, eps=eps)
    return gf.filter(image, guide)


def generate_trimap(
    alpha: ndarray,
    erode_size: int = 10,
    dilate_size: int = 20,
) -> ndarray:
    """Generate trimap from alpha matte.

    Args:
        alpha: Alpha matte.
        erode_size: Erosion for foreground.
        dilate_size: Dilation for unknown.

    Returns:
        Trimap.
    """
    generator = TrimapGenerator(erode_size, dilate_size)
    return generator.generate(alpha)
