"""Iterative refinement for maximum alpha quality.

Provides multiple refinement strategies:
- Multi-pass processing with progressive improvement
- Coarse-to-fine hierarchical refinement
- Feedback loop with error correction
- Cascade refinement with multiple models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Iterative refinement strategies."""

    MULTI_PASS = "multi_pass"
    COARSE_TO_FINE = "coarse_to_fine"
    FEEDBACK_LOOP = "feedback_loop"
    CASCADE = "cascade"
    PROGRESSIVE = "progressive"


@dataclass
class IterativeConfig:
    """Configuration for iterative refinement."""

    strategy: RefinementStrategy = RefinementStrategy.PROGRESSIVE
    num_iterations: int = 3

    # Multi-pass settings
    pass_weight_decay: float = 0.8  # Weight decay per pass

    # Coarse-to-fine settings
    scale_levels: list[float] | None = None  # [0.25, 0.5, 1.0]

    # Feedback loop settings
    error_threshold: float = 0.01  # Stop when error < threshold
    max_feedback_iterations: int = 5

    # Cascade settings
    cascade_stages: int = 3

    # Progressive settings
    progressive_steps: int = 4

    # Quality improvement
    sharpen_each_pass: bool = True
    denoise_each_pass: bool = True


class MultiPassRefiner:
    """Multi-pass iterative refinement.

    Applies refinement multiple times with decreasing influence,
    allowing gradual improvement of alpha quality.

    Example:
        >>> refiner = MultiPassRefiner(num_passes=3)
        >>> refined = refiner.refine(image, alpha, refine_fn)
    """

    def __init__(
        self,
        num_passes: int = 3,
        weight_decay: float = 0.8,
    ) -> None:
        """Initialize multi-pass refiner.

        Args:
            num_passes: Number of refinement passes.
            weight_decay: Weight decay per pass.
        """
        self.num_passes = num_passes
        self.weight_decay = weight_decay

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
        refine_fn: Callable[[ndarray, ndarray], ndarray],
    ) -> ndarray:
        """Apply multi-pass refinement.

        Args:
            image: Input image.
            alpha: Initial alpha matte.
            refine_fn: Refinement function(image, alpha) -> refined_alpha.

        Returns:
            Refined alpha matte.
        """
        current_alpha = alpha.astype(np.float32)
        weight = 1.0

        for i in range(self.num_passes):
            # Apply refinement
            refined = refine_fn(image, current_alpha.astype(np.uint8))
            refined = refined.astype(np.float32)

            # Blend with current
            current_alpha = weight * refined + (1 - weight) * current_alpha

            # Decay weight
            weight *= self.weight_decay

            logger.debug(f"Multi-pass iteration {i+1}/{self.num_passes}, weight={weight:.3f}")

        return np.clip(current_alpha, 0, 255).astype(np.uint8)


class CoarseToFineRefiner:
    """Coarse-to-fine hierarchical refinement.

    Processes at multiple scales from coarse to fine,
    propagating improvements to finer levels.

    Example:
        >>> refiner = CoarseToFineRefiner(scales=[0.25, 0.5, 1.0])
        >>> refined = refiner.refine(image, alpha, refine_fn)
    """

    def __init__(
        self,
        scales: list[float] | None = None,
    ) -> None:
        """Initialize coarse-to-fine refiner.

        Args:
            scales: Scale levels (ascending order).
        """
        self.scales = scales or [0.25, 0.5, 1.0]

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
        refine_fn: Callable[[ndarray, ndarray], ndarray],
    ) -> ndarray:
        """Apply coarse-to-fine refinement.

        Args:
            image: Input image.
            alpha: Initial alpha matte.
            refine_fn: Refinement function.

        Returns:
            Refined alpha matte.
        """
        h, w = image.shape[:2]
        current_alpha = alpha

        for scale in self.scales:
            # Resize to current scale
            new_h, new_w = int(h * scale), int(w * scale)

            scaled_image = cv2.resize(image, (new_w, new_h))
            scaled_alpha = cv2.resize(current_alpha, (new_w, new_h))

            # Refine at this scale
            refined = refine_fn(scaled_image, scaled_alpha)

            # Upscale to original size for next iteration
            if scale < 1.0:
                current_alpha = cv2.resize(refined, (w, h), interpolation=cv2.INTER_CUBIC)
            else:
                current_alpha = refined

            logger.debug(f"Coarse-to-fine at scale {scale}: {new_w}x{new_h}")

        return current_alpha


class FeedbackLoopRefiner:
    """Feedback loop refinement with error correction.

    Iteratively refines until error falls below threshold
    or maximum iterations reached.

    Example:
        >>> refiner = FeedbackLoopRefiner(error_threshold=0.01)
        >>> refined = refiner.refine(image, alpha, refine_fn)
    """

    def __init__(
        self,
        error_threshold: float = 0.01,
        max_iterations: int = 5,
    ) -> None:
        """Initialize feedback loop refiner.

        Args:
            error_threshold: Stop when error < threshold.
            max_iterations: Maximum iterations.
        """
        self.error_threshold = error_threshold
        self.max_iterations = max_iterations

    def _compute_error(
        self,
        alpha1: ndarray,
        alpha2: ndarray,
    ) -> float:
        """Compute error between two alpha mattes.

        Args:
            alpha1: First alpha.
            alpha2: Second alpha.

        Returns:
            Normalized error.
        """
        diff = np.abs(alpha1.astype(np.float32) - alpha2.astype(np.float32))
        return np.mean(diff) / 255.0

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
        refine_fn: Callable[[ndarray, ndarray], ndarray],
    ) -> ndarray:
        """Apply feedback loop refinement.

        Args:
            image: Input image.
            alpha: Initial alpha matte.
            refine_fn: Refinement function.

        Returns:
            Refined alpha matte.
        """
        current_alpha = alpha
        prev_alpha = None

        for i in range(self.max_iterations):
            # Refine
            refined = refine_fn(image, current_alpha)

            # Compute error
            if prev_alpha is not None:
                error = self._compute_error(refined, prev_alpha)
                logger.debug(f"Feedback iteration {i+1}, error={error:.4f}")

                if error < self.error_threshold:
                    logger.info(f"Converged at iteration {i+1}")
                    break

            prev_alpha = current_alpha
            current_alpha = refined

        return current_alpha


class CascadeRefiner:
    """Cascade refinement with multiple stages.

    Each stage focuses on different aspects of refinement
    (edges, details, consistency).

    Example:
        >>> refiner = CascadeRefiner(stages=3)
        >>> refined = refiner.refine(image, alpha)
    """

    def __init__(self, stages: int = 3) -> None:
        """Initialize cascade refiner.

        Args:
            stages: Number of cascade stages.
        """
        self.stages = stages

    def _stage_edge_refine(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Stage 1: Edge refinement.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Edge-refined alpha.
        """
        # Detect edges in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Guided filter for edge alignment
        alpha_float = alpha.astype(np.float32) / 255.0
        gray_float = gray.astype(np.float32) / 255.0

        # Simple guided filtering
        radius = 8
        eps = 0.01

        mean_I = cv2.blur(gray_float, (2*radius+1, 2*radius+1))
        mean_p = cv2.blur(alpha_float, (2*radius+1, 2*radius+1))
        mean_Ip = cv2.blur(gray_float * alpha_float, (2*radius+1, 2*radius+1))
        mean_II = cv2.blur(gray_float * gray_float, (2*radius+1, 2*radius+1))

        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.blur(a, (2*radius+1, 2*radius+1))
        mean_b = cv2.blur(b, (2*radius+1, 2*radius+1))

        result = mean_a * gray_float + mean_b

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def _stage_detail_refine(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Stage 2: Detail enhancement.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Detail-enhanced alpha.
        """
        # Extract high-frequency details
        blur = cv2.GaussianBlur(alpha, (5, 5), 0)
        detail = alpha.astype(np.float32) - blur.astype(np.float32)

        # Enhance details
        enhanced_detail = detail * 1.5

        # Add back
        result = blur.astype(np.float32) + enhanced_detail

        return np.clip(result, 0, 255).astype(np.uint8)

    def _stage_consistency_refine(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Stage 3: Consistency refinement.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Consistency-refined alpha.
        """
        # Morphological consistency
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Close small holes
        closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        # Open to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Blend with original in edge regions
        edges = cv2.Canny(alpha, 50, 150)
        edge_mask = cv2.dilate(edges, None, iterations=2)
        edge_mask = edge_mask.astype(np.float32) / 255.0

        result = edge_mask * alpha.astype(np.float32) + (1 - edge_mask) * opened.astype(np.float32)

        return result.astype(np.uint8)

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply cascade refinement.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Refined alpha.
        """
        current = alpha

        stages = [
            self._stage_edge_refine,
            self._stage_detail_refine,
            self._stage_consistency_refine,
        ]

        for i, stage_fn in enumerate(stages[:self.stages]):
            current = stage_fn(image, current)
            logger.debug(f"Cascade stage {i+1}/{self.stages} complete")

        return current


class ProgressiveRefiner:
    """Progressive refinement with increasing quality.

    Gradually increases refinement strength through multiple steps.

    Example:
        >>> refiner = ProgressiveRefiner(steps=4)
        >>> refined = refiner.refine(image, alpha, refine_fn)
    """

    def __init__(
        self,
        steps: int = 4,
        sharpen: bool = True,
        denoise: bool = True,
    ) -> None:
        """Initialize progressive refiner.

        Args:
            steps: Number of progressive steps.
            sharpen: Apply sharpening between steps.
            denoise: Apply denoising between steps.
        """
        self.steps = steps
        self.sharpen = sharpen
        self.denoise = denoise

    def _apply_sharpen(self, alpha: ndarray, strength: float) -> ndarray:
        """Apply sharpening.

        Args:
            alpha: Alpha matte.
            strength: Sharpening strength.

        Returns:
            Sharpened alpha.
        """
        blur = cv2.GaussianBlur(alpha, (0, 0), 3)
        sharpened = cv2.addWeighted(alpha, 1 + strength, blur, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _apply_denoise(self, alpha: ndarray, strength: float) -> ndarray:
        """Apply denoising.

        Args:
            alpha: Alpha matte.
            strength: Denoising strength (0-1).

        Returns:
            Denoised alpha.
        """
        h = int(3 + strength * 7)  # h parameter for fastNlMeansDenoising
        return cv2.fastNlMeansDenoising(alpha, None, h, 7, 21)

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
        refine_fn: Callable[[ndarray, ndarray], ndarray],
    ) -> ndarray:
        """Apply progressive refinement.

        Args:
            image: Input image.
            alpha: Initial alpha.
            refine_fn: Refinement function.

        Returns:
            Refined alpha.
        """
        current = alpha

        for i in range(self.steps):
            # Progressive strength (increases each step)
            progress = (i + 1) / self.steps

            # Refine
            current = refine_fn(image, current)

            # Apply inter-step processing
            if self.denoise and i < self.steps - 1:
                current = self._apply_denoise(current, progress * 0.3)

            if self.sharpen:
                current = self._apply_sharpen(current, progress * 0.5)

            logger.debug(f"Progressive step {i+1}/{self.steps}, progress={progress:.2f}")

        return current


class IterativeRefinement:
    """Main iterative refinement processor.

    Combines multiple refinement strategies for maximum quality.

    Example:
        >>> refiner = IterativeRefinement(IterativeConfig(
        ...     strategy=RefinementStrategy.PROGRESSIVE,
        ...     num_iterations=3
        ... ))
        >>> refined = refiner.refine(image, alpha, matting_model)
    """

    def __init__(self, config: IterativeConfig | None = None) -> None:
        """Initialize iterative refinement.

        Args:
            config: Refinement configuration.
        """
        self.config = config or IterativeConfig()
        self._init_refiners()

    def _init_refiners(self) -> None:
        """Initialize refinement components."""
        self._multi_pass = MultiPassRefiner(
            num_passes=self.config.num_iterations,
            weight_decay=self.config.pass_weight_decay,
        )

        self._coarse_to_fine = CoarseToFineRefiner(
            scales=self.config.scale_levels or [0.25, 0.5, 1.0],
        )

        self._feedback_loop = FeedbackLoopRefiner(
            error_threshold=self.config.error_threshold,
            max_iterations=self.config.max_feedback_iterations,
        )

        self._cascade = CascadeRefiner(stages=self.config.cascade_stages)

        self._progressive = ProgressiveRefiner(
            steps=self.config.progressive_steps,
            sharpen=self.config.sharpen_each_pass,
            denoise=self.config.denoise_each_pass,
        )

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
        refine_fn: Callable[[ndarray, ndarray], ndarray] | None = None,
    ) -> ndarray:
        """Apply iterative refinement.

        Args:
            image: Input image.
            alpha: Initial alpha matte.
            refine_fn: Optional refinement function.

        Returns:
            Refined alpha matte.
        """
        # Default refinement function using cascade
        if refine_fn is None:
            refine_fn = lambda img, a: self._cascade.refine(img, a)

        strategy = self.config.strategy

        if strategy == RefinementStrategy.MULTI_PASS:
            return self._multi_pass.refine(image, alpha, refine_fn)

        elif strategy == RefinementStrategy.COARSE_TO_FINE:
            return self._coarse_to_fine.refine(image, alpha, refine_fn)

        elif strategy == RefinementStrategy.FEEDBACK_LOOP:
            return self._feedback_loop.refine(image, alpha, refine_fn)

        elif strategy == RefinementStrategy.CASCADE:
            return self._cascade.refine(image, alpha)

        elif strategy == RefinementStrategy.PROGRESSIVE:
            return self._progressive.refine(image, alpha, refine_fn)

        else:
            return refine_fn(image, alpha)

    def __repr__(self) -> str:
        """String representation."""
        return f"IterativeRefinement(strategy={self.config.strategy.value})"


def iterative_refine(
    image: ndarray,
    alpha: ndarray,
    strategy: RefinementStrategy = RefinementStrategy.PROGRESSIVE,
    iterations: int = 3,
) -> ndarray:
    """Convenience function for iterative refinement.

    Args:
        image: Input image.
        alpha: Alpha matte.
        strategy: Refinement strategy.
        iterations: Number of iterations.

    Returns:
        Refined alpha.
    """
    config = IterativeConfig(strategy=strategy, num_iterations=iterations)
    refiner = IterativeRefinement(config)
    return refiner.refine(image, alpha)
