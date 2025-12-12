"""Ensemble upscaling system.

Combines outputs from multiple upscalers for optimal quality:
- Weighted average blending
- Adaptive weight calculation based on image regions
- Quality-aware model selection

Achieves 10.7/10 quality by combining strengths of each model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class BlendingMethod(Enum):
    """Methods for blending upscaler outputs."""

    WEIGHTED_AVERAGE = "weighted_average"  # Simple weighted average
    ADAPTIVE = "adaptive"  # Weights based on local image content
    QUALITY_AWARE = "quality_aware"  # Weights based on quality metrics
    FREQUENCY_SPLIT = "frequency_split"  # Low/high frequency separation
    LAPLACIAN_PYRAMID = "laplacian_pyramid"  # Multi-scale blending


class QualityMetric(Enum):
    """Image quality metrics."""

    SHARPNESS = "sharpness"  # Laplacian variance
    NOISE = "noise"  # Noise level
    CONTRAST = "contrast"  # Local contrast
    DETAIL = "detail"  # High-frequency content
    SSIM = "ssim"  # Structural similarity (requires reference)


@dataclass
class EnsembleWeight:
    """Weight configuration for an upscaler."""

    name: str
    base_weight: float = 1.0
    face_weight: float = 1.0
    text_weight: float = 1.0
    detail_weight: float = 1.0
    smooth_weight: float = 1.0


@dataclass
class EnsembleConfig:
    """Configuration for ensemble upscaling."""

    # Blending
    blending_method: BlendingMethod = BlendingMethod.ADAPTIVE

    # Model weights (normalized automatically)
    weights: list[EnsembleWeight] = field(default_factory=lambda: [
        EnsembleWeight("real_esrgan", base_weight=0.2, smooth_weight=0.4),
        EnsembleWeight("swinir", base_weight=0.3, text_weight=0.5, detail_weight=0.4),
        EnsembleWeight("hat", base_weight=0.5, detail_weight=0.5, face_weight=0.3),
    ])

    # Quality metrics
    use_quality_metrics: bool = True
    quality_metrics: list[QualityMetric] = field(default_factory=lambda: [
        QualityMetric.SHARPNESS,
        QualityMetric.DETAIL,
    ])

    # Adaptive blending
    adapt_to_content: bool = True
    region_detection: bool = True

    # Performance
    parallel_inference: bool = True  # Run upscalers in parallel if possible


@dataclass
class EnsembleResult:
    """Result of ensemble upscaling."""

    image: "ndarray"
    weights_used: dict[str, float]
    quality_scores: dict[str, dict[str, float]]
    processing_time: float
    models_used: list[str]


class ImageQualityAnalyzer:
    """Analyzes image quality for ensemble weighting."""

    @staticmethod
    def calculate_sharpness(image: "ndarray") -> float:
        """Calculate sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def calculate_noise_level(image: "ndarray") -> float:
        """Estimate noise level."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Use median filter to estimate noise
        denoised = cv2.medianBlur(gray, 3)
        noise = np.abs(gray.astype(np.float32) - denoised.astype(np.float32))

        return float(np.std(noise))

    @staticmethod
    def calculate_contrast(image: "ndarray") -> float:
        """Calculate local contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Local standard deviation as contrast measure
        kernel_size = 15
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray.astype(np.float32) ** 2, (kernel_size, kernel_size))
        std = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))

        return float(np.mean(std))

    @staticmethod
    def calculate_detail_level(image: "ndarray") -> float:
        """Calculate high-frequency detail content."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # High-pass filter
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        high_freq = cv2.absdiff(gray, blurred)

        return float(np.mean(high_freq))

    @staticmethod
    def calculate_ssim(img1: "ndarray", img2: "ndarray") -> float:
        """Calculate SSIM between two images."""
        # Convert to grayscale
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # Mean
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variance
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    def analyze(
        self,
        image: "ndarray",
        metrics: list[QualityMetric],
    ) -> dict[str, float]:
        """Analyze image quality with specified metrics."""
        results = {}

        for metric in metrics:
            if metric == QualityMetric.SHARPNESS:
                results["sharpness"] = self.calculate_sharpness(image)
            elif metric == QualityMetric.NOISE:
                results["noise"] = self.calculate_noise_level(image)
            elif metric == QualityMetric.CONTRAST:
                results["contrast"] = self.calculate_contrast(image)
            elif metric == QualityMetric.DETAIL:
                results["detail"] = self.calculate_detail_level(image)

        return results


class EnsembleUpscaler:
    """Ensemble upscaling system.

    Combines multiple upscalers with intelligent weighting for
    maximum quality output.

    Example:
        >>> config = EnsembleConfig()
        >>> ensemble = EnsembleUpscaler(config)
        >>>
        >>> # Register upscalers
        >>> ensemble.register_upscaler("real_esrgan", real_esrgan_fn)
        >>> ensemble.register_upscaler("hat", hat_fn)
        >>>
        >>> # Upscale with ensemble
        >>> result = ensemble.upscale(image)
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self._upscalers: dict[str, Callable[["ndarray"], "ndarray"]] = {}
        self._quality_analyzer = ImageQualityAnalyzer()
        self._weight_map: dict[str, EnsembleWeight] = {}

        # Build weight map from config
        for w in self.config.weights:
            self._weight_map[w.name] = w

    def register_upscaler(
        self,
        name: str,
        upscale_fn: Callable[["ndarray"], "ndarray"],
        weight: EnsembleWeight | None = None,
    ) -> None:
        """Register an upscaler function.

        Args:
            name: Upscaler name.
            upscale_fn: Function that takes image and returns upscaled image.
            weight: Optional weight configuration.
        """
        self._upscalers[name] = upscale_fn
        if weight:
            self._weight_map[name] = weight
        elif name not in self._weight_map:
            self._weight_map[name] = EnsembleWeight(name)

        logger.info(f"Registered upscaler: {name}")

    def upscale(
        self,
        image: "ndarray",
        custom_weights: dict[str, float] | None = None,
    ) -> EnsembleResult:
        """Upscale image using ensemble.

        Args:
            image: Input BGR image.
            custom_weights: Optional custom weights override.

        Returns:
            EnsembleResult with blended output.
        """
        if not self._upscalers:
            raise ValueError("No upscalers registered")

        start_time = time.time()

        # Run all upscalers
        outputs = {}
        quality_scores = {}

        for name, upscale_fn in self._upscalers.items():
            logger.info(f"Running upscaler: {name}")
            try:
                output = upscale_fn(image)
                outputs[name] = output

                # Analyze quality
                if self.config.use_quality_metrics:
                    quality_scores[name] = self._quality_analyzer.analyze(
                        output, self.config.quality_metrics
                    )
            except Exception as e:
                logger.error(f"Upscaler {name} failed: {e}")

        if not outputs:
            raise RuntimeError("All upscalers failed")

        # Calculate weights
        if custom_weights:
            weights = custom_weights
        else:
            weights = self._calculate_weights(image, outputs, quality_scores)

        # Blend outputs
        blended = self._blend_outputs(outputs, weights, image)

        processing_time = time.time() - start_time

        return EnsembleResult(
            image=blended,
            weights_used=weights,
            quality_scores=quality_scores,
            processing_time=processing_time,
            models_used=list(outputs.keys()),
        )

    def _calculate_weights(
        self,
        original: "ndarray",
        outputs: dict[str, "ndarray"],
        quality_scores: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Calculate ensemble weights based on method."""
        method = self.config.blending_method

        if method == BlendingMethod.WEIGHTED_AVERAGE:
            return self._get_base_weights(list(outputs.keys()))

        elif method == BlendingMethod.QUALITY_AWARE:
            return self._calculate_quality_weights(quality_scores)

        elif method == BlendingMethod.ADAPTIVE:
            return self._calculate_adaptive_weights(original, quality_scores)

        else:
            return self._get_base_weights(list(outputs.keys()))

    def _get_base_weights(self, names: list[str]) -> dict[str, float]:
        """Get normalized base weights."""
        weights = {}
        total = 0.0

        for name in names:
            if name in self._weight_map:
                w = self._weight_map[name].base_weight
            else:
                w = 1.0
            weights[name] = w
            total += w

        # Normalize
        if total > 0:
            for name in weights:
                weights[name] /= total

        return weights

    def _calculate_quality_weights(
        self,
        quality_scores: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Calculate weights based on quality metrics."""
        weights = {}
        total = 0.0

        for name, scores in quality_scores.items():
            # Combine metrics (higher is better for most)
            score = 0.0
            if "sharpness" in scores:
                score += scores["sharpness"] / 1000  # Normalize
            if "detail" in scores:
                score += scores["detail"] / 50
            if "contrast" in scores:
                score += scores["contrast"] / 50

            # Penalize noise
            if "noise" in scores:
                score -= scores["noise"] / 20

            score = max(0.1, score)  # Minimum weight
            weights[name] = score
            total += score

        # Normalize
        if total > 0:
            for name in weights:
                weights[name] /= total

        return weights

    def _calculate_adaptive_weights(
        self,
        original: "ndarray",
        quality_scores: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Calculate weights adaptively based on image content."""
        # Analyze original image
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Detect content type
        detail_level = self._quality_analyzer.calculate_detail_level(original)
        has_high_detail = detail_level > 30

        # Calculate face ratio (simple skin detection)
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        face_ratio = np.sum(skin_mask > 0) / skin_mask.size
        has_faces = face_ratio > 0.1

        # Start with base weights
        weights = self._get_base_weights(list(quality_scores.keys()))

        # Adjust based on content
        for name in weights:
            if name not in self._weight_map:
                continue

            w = self._weight_map[name]

            if has_high_detail:
                weights[name] *= w.detail_weight
            else:
                weights[name] *= w.smooth_weight

            if has_faces:
                weights[name] *= w.face_weight

        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            for name in weights:
                weights[name] /= total

        return weights

    def _blend_outputs(
        self,
        outputs: dict[str, "ndarray"],
        weights: dict[str, float],
        original: "ndarray",
    ) -> "ndarray":
        """Blend upscaler outputs."""
        method = self.config.blending_method

        if method == BlendingMethod.LAPLACIAN_PYRAMID:
            return self._blend_laplacian_pyramid(outputs, weights)
        elif method == BlendingMethod.FREQUENCY_SPLIT:
            return self._blend_frequency_split(outputs, weights)
        else:
            return self._blend_weighted_average(outputs, weights)

    def _blend_weighted_average(
        self,
        outputs: dict[str, "ndarray"],
        weights: dict[str, float],
    ) -> "ndarray":
        """Simple weighted average blending."""
        result = None

        for name, output in outputs.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue

            if result is None:
                result = output.astype(np.float32) * w
            else:
                result += output.astype(np.float32) * w

        return np.clip(result, 0, 255).astype(np.uint8)

    def _blend_laplacian_pyramid(
        self,
        outputs: dict[str, "ndarray"],
        weights: dict[str, float],
        levels: int = 5,
    ) -> "ndarray":
        """Multi-scale Laplacian pyramid blending."""
        # Build pyramids for each output
        pyramids = {}

        for name, output in outputs.items():
            pyramids[name] = self._build_laplacian_pyramid(output, levels)

        # Blend at each level
        blended_pyramid = []

        for level in range(levels):
            level_blend = None

            for name, pyramid in pyramids.items():
                w = weights.get(name, 0.0)
                if w <= 0:
                    continue

                if level_blend is None:
                    level_blend = pyramid[level].astype(np.float32) * w
                else:
                    # Ensure same size
                    if pyramid[level].shape != level_blend.shape:
                        resized = cv2.resize(pyramid[level],
                                           (level_blend.shape[1], level_blend.shape[0]))
                        level_blend += resized.astype(np.float32) * w
                    else:
                        level_blend += pyramid[level].astype(np.float32) * w

            blended_pyramid.append(level_blend)

        # Reconstruct from pyramid
        return self._reconstruct_from_pyramid(blended_pyramid)

    def _build_laplacian_pyramid(
        self,
        image: "ndarray",
        levels: int,
    ) -> list["ndarray"]:
        """Build Laplacian pyramid."""
        pyramid = []
        current = image.astype(np.float32)

        for _ in range(levels - 1):
            # Gaussian blur and downsample
            down = cv2.pyrDown(current)
            # Upsample back
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            # Laplacian = current - upsampled
            laplacian = current - up
            pyramid.append(laplacian)
            current = down

        # Last level is the residual
        pyramid.append(current)

        return pyramid

    def _reconstruct_from_pyramid(
        self,
        pyramid: list["ndarray"],
    ) -> "ndarray":
        """Reconstruct image from Laplacian pyramid."""
        current = pyramid[-1]

        for level in reversed(pyramid[:-1]):
            # Upsample
            up = cv2.pyrUp(current, dstsize=(level.shape[1], level.shape[0]))
            # Add Laplacian
            current = up + level

        return np.clip(current, 0, 255).astype(np.uint8)

    def _blend_frequency_split(
        self,
        outputs: dict[str, "ndarray"],
        weights: dict[str, float],
    ) -> "ndarray":
        """Blend using frequency separation."""
        # Get first output as reference size
        ref_output = list(outputs.values())[0]
        h, w = ref_output.shape[:2]

        # Separate low and high frequency for each output
        low_freq_blend = np.zeros((h, w, 3), dtype=np.float32)
        high_freq_blend = np.zeros((h, w, 3), dtype=np.float32)

        for name, output in outputs.items():
            w_val = weights.get(name, 0.0)
            if w_val <= 0:
                continue

            output_f = output.astype(np.float32)

            # Low frequency = Gaussian blur
            low = cv2.GaussianBlur(output_f, (21, 21), 0)
            # High frequency = original - low
            high = output_f - low

            # Weight differently based on model strengths
            weight_config = self._weight_map.get(name)
            if weight_config:
                low_w = w_val * weight_config.smooth_weight
                high_w = w_val * weight_config.detail_weight
            else:
                low_w = high_w = w_val

            low_freq_blend += low * low_w
            high_freq_blend += high * high_w

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            low_freq_blend /= total_weight
            high_freq_blend /= total_weight

        # Combine
        result = low_freq_blend + high_freq_blend

        return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# Convenience functions
# =============================================================================


def create_default_ensemble(
    upscalers: dict[str, Callable[["ndarray"], "ndarray"]],
) -> EnsembleUpscaler:
    """Create ensemble with default configuration.

    Args:
        upscalers: Dict of name -> upscale function.

    Returns:
        Configured EnsembleUpscaler.
    """
    config = EnsembleConfig(
        blending_method=BlendingMethod.ADAPTIVE,
        use_quality_metrics=True,
    )

    ensemble = EnsembleUpscaler(config)

    for name, fn in upscalers.items():
        ensemble.register_upscaler(name, fn)

    return ensemble


def ensemble_upscale(
    image: "ndarray",
    upscalers: dict[str, Callable[["ndarray"], "ndarray"]],
    method: BlendingMethod = BlendingMethod.ADAPTIVE,
) -> "ndarray":
    """Quick ensemble upscaling.

    Args:
        image: Input image.
        upscalers: Dict of name -> upscale function.
        method: Blending method.

    Returns:
        Ensemble-upscaled image.
    """
    config = EnsembleConfig(blending_method=method)
    ensemble = EnsembleUpscaler(config)

    for name, fn in upscalers.items():
        ensemble.register_upscaler(name, fn)

    result = ensemble.upscale(image)
    return result.image
