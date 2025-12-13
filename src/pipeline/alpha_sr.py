"""Super-resolution for alpha mattes.

Provides high-quality upscaling specifically optimized for
alpha masks, preserving edge sharpness and fine details.

Techniques:
- Edge-guided upscaling
- Detail-preserving interpolation
- Neural network-based SR
- Guided super-resolution
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


class AlphaSRMethod(Enum):
    """Alpha super-resolution methods."""

    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    EDGE_GUIDED = "edge_guided"
    DETAIL_PRESERVING = "detail_preserving"
    NEURAL = "neural"
    GUIDED = "guided"


@dataclass
class AlphaSRConfig:
    """Configuration for alpha super-resolution."""

    method: AlphaSRMethod = AlphaSRMethod.GUIDED
    scale: int = 2  # Upscaling factor

    # Edge-guided settings
    edge_threshold: float = 0.1
    edge_sharpness: float = 1.5

    # Detail preservation
    detail_weight: float = 0.7
    smoothness_weight: float = 0.3

    # Guided SR settings
    guide_radius: int = 4
    guide_eps: float = 0.01

    # Neural SR settings
    neural_model: str = "esrgan"  # "esrgan", "swinir", "hat"


class AlphaSuperResolution:
    """Super-resolution processor for alpha mattes.

    Provides multiple methods for high-quality alpha upscaling
    while preserving edge sharpness and fine details.

    Example:
        >>> sr = AlphaSuperResolution(AlphaSRConfig(scale=2))
        >>> upscaled = sr.upscale(alpha, guide_image)
    """

    def __init__(self, config: AlphaSRConfig | None = None) -> None:
        """Initialize alpha super-resolution.

        Args:
            config: SR configuration.
        """
        self.config = config or AlphaSRConfig()
        self._neural_model = None
        logger.debug(f"Alpha SR initialized: {self.config.method.value}")

    def _bicubic_upscale(
        self,
        alpha: ndarray,
        target_size: tuple[int, int],
    ) -> ndarray:
        """Simple bicubic upscaling.

        Args:
            alpha: Input alpha matte.
            target_size: Target (width, height).

        Returns:
            Upscaled alpha.
        """
        return cv2.resize(alpha, target_size, interpolation=cv2.INTER_CUBIC)

    def _lanczos_upscale(
        self,
        alpha: ndarray,
        target_size: tuple[int, int],
    ) -> ndarray:
        """Lanczos upscaling.

        Args:
            alpha: Input alpha matte.
            target_size: Target (width, height).

        Returns:
            Upscaled alpha.
        """
        return cv2.resize(alpha, target_size, interpolation=cv2.INTER_LANCZOS4)

    def _edge_guided_upscale(
        self,
        alpha: ndarray,
        target_size: tuple[int, int],
        guide: ndarray | None = None,
    ) -> ndarray:
        """Edge-guided upscaling.

        Args:
            alpha: Input alpha matte.
            target_size: Target (width, height).
            guide: Optional guide image.

        Returns:
            Upscaled alpha with sharp edges.
        """
        # Initial upscale
        upscaled = cv2.resize(alpha, target_size, interpolation=cv2.INTER_CUBIC)

        # Detect edges in original
        edges = cv2.Canny(alpha, 50, 150)
        edges_up = cv2.resize(edges, target_size, interpolation=cv2.INTER_NEAREST)

        # Create edge mask
        edge_mask = cv2.dilate(edges_up, None, iterations=2)
        edge_mask = edge_mask.astype(np.float32) / 255.0

        # Sharpen edges using unsharp mask
        blur = cv2.GaussianBlur(upscaled, (0, 0), 3)
        sharpened = cv2.addWeighted(
            upscaled, 1 + self.config.edge_sharpness,
            blur, -self.config.edge_sharpness,
            0
        )

        # Apply sharpening only to edge regions
        result = (
            edge_mask * sharpened +
            (1 - edge_mask) * upscaled
        ).astype(np.uint8)

        # Use guide image if available
        if guide is not None:
            result = self._apply_guide(result, guide, target_size)

        return result

    def _detail_preserving_upscale(
        self,
        alpha: ndarray,
        target_size: tuple[int, int],
    ) -> ndarray:
        """Detail-preserving upscaling.

        Args:
            alpha: Input alpha matte.
            target_size: Target (width, height).

        Returns:
            Upscaled alpha with preserved details.
        """
        # Extract detail layer
        blur = cv2.GaussianBlur(alpha, (5, 5), 0)
        detail = alpha.astype(np.float32) - blur.astype(np.float32)

        # Upscale base and detail separately
        base_up = cv2.resize(blur, target_size, interpolation=cv2.INTER_CUBIC)
        detail_up = cv2.resize(detail, target_size, interpolation=cv2.INTER_CUBIC)

        # Enhance detail
        detail_up = detail_up * self.config.edge_sharpness

        # Combine
        result = base_up.astype(np.float32) + detail_up
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _guided_upscale(
        self,
        alpha: ndarray,
        target_size: tuple[int, int],
        guide: ndarray | None = None,
    ) -> ndarray:
        """Guided super-resolution.

        Args:
            alpha: Input alpha matte.
            target_size: Target (width, height).
            guide: Guide image (optional).

        Returns:
            Upscaled alpha.
        """
        # Initial upscale
        upscaled = cv2.resize(alpha, target_size, interpolation=cv2.INTER_CUBIC)

        if guide is None:
            return upscaled

        # Resize guide if needed
        if guide.shape[:2][::-1] != target_size:
            guide = cv2.resize(guide, target_size)

        # Convert guide to grayscale if needed
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)

        # Apply guided filter
        result = self._guided_filter(
            upscaled,
            guide,
            self.config.guide_radius,
            self.config.guide_eps,
        )

        return result

    def _guided_filter(
        self,
        src: ndarray,
        guide: ndarray,
        radius: int,
        eps: float,
    ) -> ndarray:
        """Apply guided filter.

        Args:
            src: Source image.
            guide: Guide image.
            radius: Filter radius.
            eps: Regularization.

        Returns:
            Filtered image.
        """
        src = src.astype(np.float32) / 255.0
        guide = guide.astype(np.float32) / 255.0

        # Box filter
        def box_filter(img, r):
            ksize = 2 * r + 1
            return cv2.blur(img, (ksize, ksize))

        mean_I = box_filter(guide, radius)
        mean_p = box_filter(src, radius)
        mean_Ip = box_filter(guide * src, radius)
        mean_II = box_filter(guide * guide, radius)

        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)

        result = mean_a * guide + mean_b

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def _neural_upscale(
        self,
        alpha: ndarray,
        target_size: tuple[int, int],
    ) -> ndarray:
        """Neural network-based upscaling.

        Args:
            alpha: Input alpha matte.
            target_size: Target (width, height).

        Returns:
            Upscaled alpha.
        """
        # Try to use Real-ESRGAN for alpha
        try:
            if self._neural_model is None:
                self._load_neural_model()

            if self._neural_model is not None:
                # Convert to 3-channel for model
                alpha_3ch = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)

                # Upscale
                upscaled_3ch = self._neural_model.upscale(alpha_3ch)

                # Convert back to grayscale
                upscaled = cv2.cvtColor(upscaled_3ch, cv2.COLOR_BGR2GRAY)

                # Resize to exact target if needed
                if upscaled.shape[:2][::-1] != target_size:
                    upscaled = cv2.resize(upscaled, target_size)

                return upscaled

        except Exception as e:
            logger.warning(f"Neural upscale failed: {e}, using fallback")

        # Fallback to detail-preserving
        return self._detail_preserving_upscale(alpha, target_size)

    def _load_neural_model(self) -> None:
        """Load neural upscaling model."""
        try:
            from src.upscale.real_esrgan import RealESRGAN

            self._neural_model = RealESRGAN(scale=self.config.scale)
            self._neural_model.load()
            logger.info("Neural SR model loaded")

        except Exception as e:
            logger.warning(f"Could not load neural model: {e}")
            self._neural_model = None

    def _apply_guide(
        self,
        alpha: ndarray,
        guide: ndarray,
        target_size: tuple[int, int],
    ) -> ndarray:
        """Apply guide image for edge refinement.

        Args:
            alpha: Upscaled alpha.
            guide: Guide image.
            target_size: Target size.

        Returns:
            Refined alpha.
        """
        # Resize guide
        if guide.shape[:2][::-1] != target_size:
            guide = cv2.resize(guide, target_size)

        # Get edges from guide
        if len(guide.shape) == 3:
            gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        else:
            gray = guide

        edges = cv2.Canny(gray, 50, 150)
        edge_region = cv2.dilate(edges, None, iterations=2)
        edge_mask = edge_region.astype(np.float32) / 255.0

        # Apply guided filter in edge regions
        filtered = self._guided_filter(
            alpha, gray, self.config.guide_radius, self.config.guide_eps
        )

        # Blend
        result = (
            edge_mask * filtered.astype(np.float32) +
            (1 - edge_mask) * alpha.astype(np.float32)
        ).astype(np.uint8)

        return result

    def upscale(
        self,
        alpha: ndarray,
        guide: ndarray | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> ndarray:
        """Upscale alpha matte.

        Args:
            alpha: Input alpha matte.
            guide: Optional guide image for edge guidance.
            target_size: Target (width, height), or None for scale factor.

        Returns:
            Upscaled alpha matte.
        """
        # Calculate target size
        if target_size is None:
            h, w = alpha.shape[:2]
            target_size = (w * self.config.scale, h * self.config.scale)

        method = self.config.method

        if method == AlphaSRMethod.BICUBIC:
            result = self._bicubic_upscale(alpha, target_size)
        elif method == AlphaSRMethod.LANCZOS:
            result = self._lanczos_upscale(alpha, target_size)
        elif method == AlphaSRMethod.EDGE_GUIDED:
            result = self._edge_guided_upscale(alpha, target_size, guide)
        elif method == AlphaSRMethod.DETAIL_PRESERVING:
            result = self._detail_preserving_upscale(alpha, target_size)
        elif method == AlphaSRMethod.NEURAL:
            result = self._neural_upscale(alpha, target_size)
        elif method == AlphaSRMethod.GUIDED:
            result = self._guided_upscale(alpha, target_size, guide)
        else:
            result = self._bicubic_upscale(alpha, target_size)

        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"AlphaSuperResolution(method={self.config.method.value}, scale={self.config.scale})"


class LowResProcessor:
    """Process at low resolution for speed, then upscale.

    Useful for real-time processing where full resolution
    is too slow.

    Example:
        >>> processor = LowResProcessor(process_scale=0.5)
        >>> fast_alpha = processor.process(frame, matting_model)
    """

    def __init__(
        self,
        process_scale: float = 0.5,
        sr_config: AlphaSRConfig | None = None,
    ) -> None:
        """Initialize low-res processor.

        Args:
            process_scale: Scale for processing (0-1).
            sr_config: Super-resolution config for upscaling.
        """
        self.process_scale = process_scale
        self.sr = AlphaSuperResolution(sr_config or AlphaSRConfig(
            method=AlphaSRMethod.GUIDED,
        ))

    def process(
        self,
        frame: ndarray,
        model,
        upscale_guide: bool = True,
    ) -> tuple[ndarray, ndarray]:
        """Process frame at low resolution and upscale.

        Args:
            frame: Input frame.
            model: Matting model with inference(frame) -> (fg, alpha).
            upscale_guide: Use frame as guide for upscaling.

        Returns:
            Tuple of (foreground, alpha) at original resolution.
        """
        h, w = frame.shape[:2]
        target_size = (w, h)

        # Downscale for processing
        small_w = int(w * self.process_scale)
        small_h = int(h * self.process_scale)
        small_frame = cv2.resize(frame, (small_w, small_h))

        # Process at low resolution
        fg_small, alpha_small = model.inference(small_frame)

        # Upscale alpha
        guide = frame if upscale_guide else None
        alpha = self.sr.upscale(alpha_small, guide, target_size)

        # Upscale foreground using alpha as guide
        fg = cv2.resize(fg_small, target_size, interpolation=cv2.INTER_CUBIC)

        return fg, alpha


def upscale_alpha(
    alpha: ndarray,
    scale: int = 2,
    method: AlphaSRMethod = AlphaSRMethod.GUIDED,
    guide: ndarray | None = None,
) -> ndarray:
    """Convenience function for alpha upscaling.

    Args:
        alpha: Input alpha matte.
        scale: Upscaling factor.
        method: SR method.
        guide: Optional guide image.

    Returns:
        Upscaled alpha.
    """
    config = AlphaSRConfig(method=method, scale=scale)
    sr = AlphaSuperResolution(config)
    return sr.upscale(alpha, guide)
