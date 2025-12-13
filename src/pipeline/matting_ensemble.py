"""Multi-model ensemble for video matting.

Combines multiple matting models to achieve better quality
than any single model alone.

Ensemble methods:
- Weighted average blending
- Confidence-based selection
- Uncertainty-aware fusion
- Quality-guided combination
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

    from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Ensemble combination methods."""

    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    UNCERTAINTY_FUSION = "uncertainty_fusion"
    QUALITY_GUIDED = "quality_guided"
    LAPLACIAN_PYRAMID = "laplacian_pyramid"


@dataclass
class EnsembleConfig:
    """Configuration for matting ensemble."""

    method: EnsembleMethod = EnsembleMethod.QUALITY_GUIDED

    # Model weights (normalized automatically)
    model_weights: dict[str, float] = field(default_factory=lambda: {
        "rvm": 0.4,
        "modnet": 0.3,
        "vitmatte": 0.3,
    })

    # Confidence thresholds
    confidence_threshold: float = 0.8
    uncertainty_threshold: float = 0.2

    # Quality assessment
    edge_weight: float = 0.4  # Weight for edge quality
    smoothness_weight: float = 0.3  # Weight for smoothness
    consistency_weight: float = 0.3  # Weight for temporal consistency

    # Pyramid levels for Laplacian
    pyramid_levels: int = 4


class MattingResult:
    """Container for matting result with metadata."""

    def __init__(
        self,
        foreground: ndarray,
        alpha: ndarray,
        model_name: str,
        confidence: ndarray | None = None,
    ) -> None:
        """Initialize matting result.

        Args:
            foreground: Foreground image.
            alpha: Alpha matte.
            model_name: Name of the model that produced this.
            confidence: Optional confidence map.
        """
        self.foreground = foreground
        self.alpha = alpha
        self.model_name = model_name
        self.confidence = confidence

    @property
    def alpha_float(self) -> ndarray:
        """Get alpha as float (0-1)."""
        return self.alpha.astype(np.float32) / 255.0


class QualityAssessor:
    """Assess quality of alpha mattes.

    Provides metrics for evaluating matting quality
    without ground truth.

    Example:
        >>> assessor = QualityAssessor()
        >>> score = assessor.assess(image, alpha)
    """

    def __init__(
        self,
        edge_weight: float = 0.4,
        smoothness_weight: float = 0.3,
        consistency_weight: float = 0.3,
    ) -> None:
        """Initialize quality assessor.

        Args:
            edge_weight: Weight for edge quality.
            smoothness_weight: Weight for smoothness.
            consistency_weight: Weight for consistency.
        """
        self.edge_weight = edge_weight
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight
        self._prev_alpha = None

    def assess(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> float:
        """Assess quality of alpha matte.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Quality score (0-1).
        """
        edge_score = self._assess_edges(image, alpha)
        smoothness_score = self._assess_smoothness(alpha)
        consistency_score = self._assess_consistency(alpha)

        score = (
            self.edge_weight * edge_score +
            self.smoothness_weight * smoothness_score +
            self.consistency_weight * consistency_score
        )

        # Update history
        self._prev_alpha = alpha.copy()

        return score

    def _assess_edges(self, image: ndarray, alpha: ndarray) -> float:
        """Assess edge alignment quality.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Edge quality score.
        """
        # Get edges from image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_edges = cv2.Canny(gray, 50, 150)

        # Get edges from alpha
        alpha_edges = cv2.Canny(alpha, 50, 150)

        # Dilate for tolerance
        kernel = np.ones((3, 3), np.uint8)
        image_edges_dilated = cv2.dilate(image_edges, kernel, iterations=2)

        # Compute alignment
        if alpha_edges.sum() == 0:
            return 0.5

        aligned = np.sum(alpha_edges & image_edges_dilated)
        total = np.sum(alpha_edges)

        return aligned / total if total > 0 else 0.5

    def _assess_smoothness(self, alpha: ndarray) -> float:
        """Assess smoothness of alpha transitions.

        Args:
            alpha: Alpha matte.

        Returns:
            Smoothness score.
        """
        # Compute gradient
        grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Find transition regions
        transition = (alpha > 10) & (alpha < 245)

        if transition.sum() == 0:
            return 1.0

        # Measure smoothness in transitions
        avg_gradient = gradient_mag[transition].mean()

        # Normalize (lower gradient = smoother)
        smoothness = 1.0 - min(avg_gradient / 100, 1.0)

        return smoothness

    def _assess_consistency(self, alpha: ndarray) -> float:
        """Assess temporal consistency.

        Args:
            alpha: Current alpha matte.

        Returns:
            Consistency score.
        """
        if self._prev_alpha is None:
            return 1.0

        # Compute difference
        diff = np.abs(alpha.astype(np.float32) - self._prev_alpha.astype(np.float32))
        avg_diff = diff.mean() / 255.0

        # Lower difference = better consistency
        consistency = 1.0 - min(avg_diff * 5, 1.0)

        return consistency

    def assess_batch(
        self,
        image: ndarray,
        alphas: list[ndarray],
    ) -> list[float]:
        """Assess quality of multiple alpha mattes.

        Args:
            image: Input image.
            alphas: List of alpha mattes.

        Returns:
            List of quality scores.
        """
        scores = []
        for alpha in alphas:
            score = self.assess(image, alpha)
            scores.append(score)
            # Reset prev_alpha to avoid cross-contamination
        self._prev_alpha = None
        return scores


class MattingEnsemble:
    """Ensemble of matting models.

    Combines multiple matting models using various fusion
    methods for improved quality.

    Example:
        >>> ensemble = MattingEnsemble(config)
        >>> ensemble.add_model("rvm", rvm_model)
        >>> ensemble.add_model("modnet", modnet_model)
        >>> fg, alpha = ensemble.predict(image)
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        """Initialize matting ensemble.

        Args:
            config: Ensemble configuration.
        """
        self.config = config or EnsembleConfig()
        self._models: dict[str, Callable] = {}
        self._quality_assessor = QualityAssessor(
            edge_weight=self.config.edge_weight,
            smoothness_weight=self.config.smoothness_weight,
            consistency_weight=self.config.consistency_weight,
        )
        logger.debug(f"Matting ensemble initialized: {self.config.method.value}")

    def add_model(
        self,
        name: str,
        model: BaseModel | Callable,
        weight: float | None = None,
    ) -> None:
        """Add a model to the ensemble.

        Args:
            name: Model identifier.
            model: Model instance or callable(image) -> (fg, alpha).
            weight: Optional weight override.
        """
        self._models[name] = model

        if weight is not None:
            self.config.model_weights[name] = weight

        # Normalize weights
        total = sum(self.config.model_weights.get(n, 0.1) for n in self._models)
        for n in self._models:
            if n not in self.config.model_weights:
                self.config.model_weights[n] = 0.1 / len(self._models)

        logger.debug(f"Added model '{name}' to ensemble")

    def _run_models(
        self,
        image: ndarray,
    ) -> list[MattingResult]:
        """Run all models on image.

        Args:
            image: Input image.

        Returns:
            List of matting results.
        """
        results = []

        for name, model in self._models.items():
            try:
                if hasattr(model, 'inference'):
                    fg, alpha = model.inference(image)
                elif hasattr(model, 'predict'):
                    fg, alpha = model.predict(image)
                else:
                    fg, alpha = model(image)

                results.append(MattingResult(
                    foreground=fg,
                    alpha=alpha,
                    model_name=name,
                ))

            except Exception as e:
                logger.warning(f"Model '{name}' failed: {e}")
                continue

        return results

    def _weighted_average(
        self,
        results: list[MattingResult],
    ) -> ndarray:
        """Combine using weighted average.

        Args:
            results: List of matting results.

        Returns:
            Combined alpha matte.
        """
        if not results:
            raise ValueError("No results to combine")

        # Get weights
        weights = []
        alphas = []

        for result in results:
            weight = self.config.model_weights.get(result.model_name, 0.1)
            weights.append(weight)
            alphas.append(result.alpha_float)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted sum
        combined = np.zeros_like(alphas[0])
        for alpha, weight in zip(alphas, weights):
            combined += alpha * weight

        return (combined * 255).astype(np.uint8)

    def _confidence_based(
        self,
        image: ndarray,
        results: list[MattingResult],
    ) -> ndarray:
        """Combine using confidence-based selection.

        Args:
            image: Input image.
            results: List of matting results.

        Returns:
            Combined alpha matte.
        """
        if len(results) == 1:
            return results[0].alpha

        # Compute confidence for each result
        for result in results:
            # Use alpha values near 0 or 255 as high confidence
            alpha_float = result.alpha_float
            confidence = np.abs(alpha_float - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
            result.confidence = confidence

        # Select based on confidence
        combined = np.zeros_like(results[0].alpha_float)
        total_conf = np.zeros_like(combined)

        for result in results:
            weight = self.config.model_weights.get(result.model_name, 0.1)
            conf = result.confidence * weight
            combined += result.alpha_float * conf
            total_conf += conf

        # Normalize
        combined = combined / (total_conf + 1e-8)

        return (combined * 255).astype(np.uint8)

    def _uncertainty_fusion(
        self,
        image: ndarray,
        results: list[MattingResult],
    ) -> ndarray:
        """Combine using uncertainty-aware fusion.

        Uses disagreement between models as uncertainty signal.

        Args:
            image: Input image.
            results: List of matting results.

        Returns:
            Combined alpha matte.
        """
        if len(results) == 1:
            return results[0].alpha

        alphas = [r.alpha_float for r in results]

        # Compute mean and variance
        mean_alpha = np.mean(alphas, axis=0)
        var_alpha = np.var(alphas, axis=0)

        # High variance = high uncertainty
        uncertainty = np.sqrt(var_alpha)
        certainty = 1 - np.clip(uncertainty / self.config.uncertainty_threshold, 0, 1)

        # In certain regions, use weighted average
        # In uncertain regions, use quality-weighted selection
        weighted_avg = self._weighted_average(results)
        weighted_avg_float = weighted_avg.astype(np.float32) / 255.0

        # Quality assessment for uncertain regions
        scores = self._quality_assessor.assess_batch(image, [r.alpha for r in results])
        best_idx = np.argmax(scores)
        best_alpha = results[best_idx].alpha_float

        # Blend based on certainty
        combined = certainty * weighted_avg_float + (1 - certainty) * best_alpha

        return (combined * 255).astype(np.uint8)

    def _quality_guided(
        self,
        image: ndarray,
        results: list[MattingResult],
    ) -> ndarray:
        """Combine using quality-guided selection.

        Args:
            image: Input image.
            results: List of matting results.

        Returns:
            Combined alpha matte.
        """
        if len(results) == 1:
            return results[0].alpha

        # Assess quality of each result
        scores = self._quality_assessor.assess_batch(image, [r.alpha for r in results])

        # Convert scores to weights
        scores = np.array(scores)
        quality_weights = np.exp(scores * 2)  # Exponential to favor higher quality
        quality_weights = quality_weights / quality_weights.sum()

        # Combine with model weights
        final_weights = []
        for i, result in enumerate(results):
            model_weight = self.config.model_weights.get(result.model_name, 0.1)
            final_weight = quality_weights[i] * model_weight
            final_weights.append(final_weight)

        # Normalize
        final_weights = np.array(final_weights)
        final_weights = final_weights / final_weights.sum()

        # Weighted combination
        combined = np.zeros_like(results[0].alpha_float)
        for result, weight in zip(results, final_weights):
            combined += result.alpha_float * weight

        return (combined * 255).astype(np.uint8)

    def _laplacian_pyramid(
        self,
        image: ndarray,
        results: list[MattingResult],
    ) -> ndarray:
        """Combine using Laplacian pyramid fusion.

        Args:
            image: Input image.
            results: List of matting results.

        Returns:
            Combined alpha matte.
        """
        if len(results) == 1:
            return results[0].alpha

        levels = self.config.pyramid_levels
        alphas = [r.alpha_float for r in results]

        # Build Laplacian pyramids
        pyramids = []
        for alpha in alphas:
            pyramid = self._build_laplacian_pyramid(alpha, levels)
            pyramids.append(pyramid)

        # Get quality scores for weighting
        scores = self._quality_assessor.assess_batch(image, [r.alpha for r in results])
        weights = np.exp(np.array(scores))
        weights = weights / weights.sum()

        # Blend at each level
        blended_pyramid = []
        for level in range(len(pyramids[0])):
            blended_level = np.zeros_like(pyramids[0][level])
            for pyramid, weight in zip(pyramids, weights):
                blended_level += pyramid[level] * weight
            blended_pyramid.append(blended_level)

        # Reconstruct from blended pyramid
        combined = self._reconstruct_from_pyramid(blended_pyramid)

        return (np.clip(combined, 0, 1) * 255).astype(np.uint8)

    def _build_laplacian_pyramid(
        self,
        image: ndarray,
        levels: int,
    ) -> list[ndarray]:
        """Build Laplacian pyramid.

        Args:
            image: Input image.
            levels: Number of levels.

        Returns:
            List of pyramid levels.
        """
        gaussian = [image.astype(np.float32)]

        for _ in range(levels - 1):
            down = cv2.pyrDown(gaussian[-1])
            gaussian.append(down)

        laplacian = []
        for i in range(levels - 1):
            up = cv2.pyrUp(gaussian[i + 1], dstsize=gaussian[i].shape[:2][::-1])
            lap = gaussian[i] - up
            laplacian.append(lap)

        laplacian.append(gaussian[-1])

        return laplacian

    def _reconstruct_from_pyramid(
        self,
        pyramid: list[ndarray],
    ) -> ndarray:
        """Reconstruct image from Laplacian pyramid.

        Args:
            pyramid: Laplacian pyramid levels.

        Returns:
            Reconstructed image.
        """
        image = pyramid[-1]

        for i in range(len(pyramid) - 2, -1, -1):
            up = cv2.pyrUp(image, dstsize=pyramid[i].shape[:2][::-1])
            image = up + pyramid[i]

        return image

    def predict(
        self,
        image: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Predict using ensemble.

        Args:
            image: Input image (BGR).

        Returns:
            Tuple of (foreground, alpha).
        """
        # Run all models
        results = self._run_models(image)

        if not results:
            raise RuntimeError("All models failed")

        # Combine results
        method = self.config.method

        if method == EnsembleMethod.WEIGHTED_AVERAGE:
            alpha = self._weighted_average(results)
        elif method == EnsembleMethod.CONFIDENCE_BASED:
            alpha = self._confidence_based(image, results)
        elif method == EnsembleMethod.UNCERTAINTY_FUSION:
            alpha = self._uncertainty_fusion(image, results)
        elif method == EnsembleMethod.QUALITY_GUIDED:
            alpha = self._quality_guided(image, results)
        elif method == EnsembleMethod.LAPLACIAN_PYRAMID:
            alpha = self._laplacian_pyramid(image, results)
        else:
            alpha = self._weighted_average(results)

        # Generate foreground
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]
        foreground = (image * alpha_norm).astype(np.uint8)

        return foreground, alpha

    def __repr__(self) -> str:
        """String representation."""
        models = list(self._models.keys())
        return f"MattingEnsemble(method={self.config.method.value}, models={models})"


def create_ensemble_from_models(
    models: dict[str, BaseModel | Callable],
    method: EnsembleMethod = EnsembleMethod.QUALITY_GUIDED,
    weights: dict[str, float] | None = None,
) -> MattingEnsemble:
    """Create ensemble from model dictionary.

    Args:
        models: Dictionary of name -> model.
        method: Ensemble method.
        weights: Optional model weights.

    Returns:
        Configured MattingEnsemble.
    """
    config = EnsembleConfig(method=method)
    if weights:
        config.model_weights = weights

    ensemble = MattingEnsemble(config)
    for name, model in models.items():
        ensemble.add_model(name, model)

    return ensemble
