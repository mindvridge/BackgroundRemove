"""Temporal consistency module for video background removal.

Provides frame-to-frame consistency to eliminate flickering
and temporal artifacts in video matting results.

Techniques implemented:
- Optical flow-based mask propagation
- Temporal smoothing with exponential moving average
- Bidirectional consistency checking
- Motion-adaptive blending
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class TemporalMethod(Enum):
    """Temporal consistency method."""

    NONE = "none"  # No temporal processing
    EMA = "ema"  # Exponential moving average
    OPTICAL_FLOW = "optical_flow"  # Optical flow-based propagation
    BIDIRECTIONAL = "bidirectional"  # Bidirectional consistency
    ADAPTIVE = "adaptive"  # Motion-adaptive blending


@dataclass
class TemporalConfig:
    """Configuration for temporal consistency."""

    method: TemporalMethod = TemporalMethod.OPTICAL_FLOW

    # EMA parameters
    ema_alpha: float = 0.3  # Smoothing factor (0-1, lower = smoother)

    # Optical flow parameters
    flow_quality: str = "medium"  # "fast", "medium", "high"
    flow_scale: float = 0.5  # Scale for flow computation (lower = faster)

    # Consistency parameters
    consistency_threshold: float = 0.1  # Threshold for consistency check
    motion_threshold: float = 0.05  # Threshold for motion detection

    # Blending parameters
    blend_static_weight: float = 0.7  # Weight for static regions
    blend_motion_weight: float = 0.3  # Weight for motion regions

    # History
    history_size: int = 3  # Number of frames to keep in history


@dataclass
class TemporalState:
    """State for temporal processing."""

    prev_frame: ndarray | None = None
    prev_alpha: ndarray | None = None
    prev_flow: ndarray | None = None
    alpha_history: list[ndarray] = field(default_factory=list)
    frame_history: list[ndarray] = field(default_factory=list)
    frame_count: int = 0


class TemporalConsistency:
    """Temporal consistency processor for video matting.

    Reduces flickering and temporal artifacts by ensuring
    smooth transitions between frames.

    Example:
        >>> temporal = TemporalConsistency(TemporalConfig(method=TemporalMethod.OPTICAL_FLOW))
        >>> for frame, alpha in video_frames:
        ...     smoothed_alpha = temporal.process(frame, alpha)
    """

    def __init__(self, config: TemporalConfig | None = None) -> None:
        """Initialize temporal consistency processor.

        Args:
            config: Temporal consistency configuration.
        """
        self.config = config or TemporalConfig()
        self.state = TemporalState()
        self._flow_params = self._get_flow_params()
        logger.debug(f"Temporal consistency initialized: {self.config.method.value}")

    def _get_flow_params(self) -> dict:
        """Get optical flow parameters based on quality setting."""
        quality = self.config.flow_quality

        if quality == "fast":
            return {
                "pyr_scale": 0.5,
                "levels": 2,
                "winsize": 11,
                "iterations": 2,
                "poly_n": 5,
                "poly_sigma": 1.1,
                "flags": 0,
            }
        elif quality == "high":
            return {
                "pyr_scale": 0.5,
                "levels": 5,
                "winsize": 21,
                "iterations": 5,
                "poly_n": 7,
                "poly_sigma": 1.5,
                "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            }
        else:  # medium
            return {
                "pyr_scale": 0.5,
                "levels": 3,
                "winsize": 15,
                "iterations": 3,
                "poly_n": 5,
                "poly_sigma": 1.2,
                "flags": 0,
            }

    def reset(self) -> None:
        """Reset temporal state for new video."""
        self.state = TemporalState()
        logger.debug("Temporal state reset")

    def _compute_optical_flow(
        self,
        prev_gray: ndarray,
        curr_gray: ndarray,
    ) -> ndarray:
        """Compute optical flow between two frames.

        Args:
            prev_gray: Previous frame in grayscale.
            curr_gray: Current frame in grayscale.

        Returns:
            Optical flow field (H, W, 2).
        """
        scale = self.config.flow_scale

        if scale < 1.0:
            h, w = prev_gray.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            prev_small = cv2.resize(prev_gray, (new_w, new_h))
            curr_small = cv2.resize(curr_gray, (new_w, new_h))
        else:
            prev_small = prev_gray
            curr_small = curr_gray

        flow = cv2.calcOpticalFlowFarneback(
            prev_small,
            curr_small,
            None,
            **self._flow_params,
        )

        if scale < 1.0:
            flow = cv2.resize(flow, (w, h))
            flow *= (1.0 / scale)

        return flow

    def _warp_with_flow(
        self,
        image: ndarray,
        flow: ndarray,
    ) -> ndarray:
        """Warp an image using optical flow.

        Args:
            image: Image to warp.
            flow: Optical flow field.

        Returns:
            Warped image.
        """
        h, w = flow.shape[:2]

        # Create mesh grid
        flow_map = np.column_stack([
            np.tile(np.arange(w), h),
            np.repeat(np.arange(h), w),
        ]).reshape(h, w, 2).astype(np.float32)

        # Add flow to grid
        flow_map += flow

        # Remap
        warped = cv2.remap(
            image,
            flow_map[:, :, 0],
            flow_map[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return warped

    def _compute_motion_mask(
        self,
        flow: ndarray,
    ) -> ndarray:
        """Compute motion mask from optical flow.

        Args:
            flow: Optical flow field.

        Returns:
            Motion mask (0-1).
        """
        # Compute flow magnitude
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        # Normalize
        max_mag = magnitude.max()
        if max_mag > 0:
            magnitude = magnitude / max_mag

        # Threshold
        motion_mask = np.clip(
            (magnitude - self.config.motion_threshold) /
            (1.0 - self.config.motion_threshold),
            0, 1,
        ).astype(np.float32)

        return motion_mask

    def _ema_smooth(
        self,
        current_alpha: ndarray,
    ) -> ndarray:
        """Apply exponential moving average smoothing.

        Args:
            current_alpha: Current alpha matte.

        Returns:
            Smoothed alpha matte.
        """
        if self.state.prev_alpha is None:
            return current_alpha

        alpha = self.config.ema_alpha
        smoothed = (
            alpha * current_alpha.astype(np.float32) +
            (1 - alpha) * self.state.prev_alpha.astype(np.float32)
        )

        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def _optical_flow_smooth(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply optical flow-based smoothing.

        Args:
            frame: Current frame (BGR).
            alpha: Current alpha matte.

        Returns:
            Smoothed alpha matte.
        """
        if self.state.prev_frame is None or self.state.prev_alpha is None:
            return alpha

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.state.prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = self._compute_optical_flow(prev_gray, curr_gray)

        # Warp previous alpha to current frame
        warped_alpha = self._warp_with_flow(
            self.state.prev_alpha.astype(np.float32),
            flow,
        )

        # Compute motion mask
        motion_mask = self._compute_motion_mask(flow)
        motion_mask = motion_mask[:, :, np.newaxis] if len(motion_mask.shape) == 2 else motion_mask

        # Blend based on motion
        alpha_float = alpha.astype(np.float32)

        # Static regions: prefer warped (temporally consistent)
        # Motion regions: prefer current (more accurate)
        static_weight = self.config.blend_static_weight
        motion_weight = self.config.blend_motion_weight

        blended = (
            (1 - motion_mask) * (static_weight * warped_alpha + (1 - static_weight) * alpha_float) +
            motion_mask * (motion_weight * warped_alpha + (1 - motion_weight) * alpha_float)
        )

        # Store flow for bidirectional
        self.state.prev_flow = flow

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _bidirectional_smooth(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply bidirectional consistency smoothing.

        Uses forward and backward flow for improved consistency.

        Args:
            frame: Current frame (BGR).
            alpha: Current alpha matte.

        Returns:
            Smoothed alpha matte.
        """
        if len(self.state.alpha_history) < 2:
            return self._optical_flow_smooth(frame, alpha)

        # Get previous frames
        prev_frame = self.state.frame_history[-1]
        prev_prev_frame = self.state.frame_history[-2]

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_prev_gray = cv2.cvtColor(prev_prev_frame, cv2.COLOR_BGR2GRAY)

        # Forward flow (prev -> curr)
        flow_forward = self._compute_optical_flow(prev_gray, curr_gray)

        # Backward flow (curr -> prev)
        flow_backward = self._compute_optical_flow(curr_gray, prev_gray)

        # Consistency check
        warped_backward_flow = self._warp_with_flow(flow_backward, flow_forward)
        consistency = np.sqrt(
            (flow_forward[:, :, 0] + warped_backward_flow[:, :, 0]) ** 2 +
            (flow_forward[:, :, 1] + warped_backward_flow[:, :, 1]) ** 2
        )

        # Consistency mask (1 = consistent, 0 = inconsistent)
        consistency_mask = np.exp(-consistency / self.config.consistency_threshold)
        consistency_mask = consistency_mask[:, :, np.newaxis]

        # Warp previous alpha
        prev_alpha = self.state.alpha_history[-1]
        warped_alpha = self._warp_with_flow(prev_alpha.astype(np.float32), flow_forward)

        # Blend based on consistency
        alpha_float = alpha.astype(np.float32)
        blended = consistency_mask * warped_alpha + (1 - consistency_mask) * alpha_float

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _adaptive_smooth(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Apply motion-adaptive smoothing.

        Combines multiple techniques based on scene analysis.

        Args:
            frame: Current frame (BGR).
            alpha: Current alpha matte.

        Returns:
            Smoothed alpha matte.
        """
        if self.state.prev_frame is None:
            return alpha

        # Compute frame difference
        frame_diff = cv2.absdiff(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.state.prev_frame, cv2.COLOR_BGR2GRAY),
        )

        # Compute global motion level
        motion_level = np.mean(frame_diff) / 255.0

        if motion_level < 0.02:
            # Very static: heavy temporal smoothing
            return self._ema_smooth(alpha)
        elif motion_level < 0.1:
            # Moderate motion: optical flow
            return self._optical_flow_smooth(frame, alpha)
        elif motion_level < 0.3:
            # High motion: bidirectional if available
            if len(self.state.alpha_history) >= 2:
                return self._bidirectional_smooth(frame, alpha)
            return self._optical_flow_smooth(frame, alpha)
        else:
            # Very high motion: minimal smoothing
            if self.state.prev_alpha is not None:
                alpha_float = alpha.astype(np.float32)
                prev_float = self.state.prev_alpha.astype(np.float32)
                return np.clip(0.9 * alpha_float + 0.1 * prev_float, 0, 255).astype(np.uint8)
            return alpha

    def process(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Process alpha matte for temporal consistency.

        Args:
            frame: Current frame (BGR format).
            alpha: Current alpha matte (0-255).

        Returns:
            Temporally smoothed alpha matte.
        """
        method = self.config.method

        if method == TemporalMethod.NONE:
            result = alpha
        elif method == TemporalMethod.EMA:
            result = self._ema_smooth(alpha)
        elif method == TemporalMethod.OPTICAL_FLOW:
            result = self._optical_flow_smooth(frame, alpha)
        elif method == TemporalMethod.BIDIRECTIONAL:
            result = self._bidirectional_smooth(frame, alpha)
        elif method == TemporalMethod.ADAPTIVE:
            result = self._adaptive_smooth(frame, alpha)
        else:
            result = alpha

        # Update state
        self._update_state(frame, result)

        return result

    def _update_state(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> None:
        """Update temporal state.

        Args:
            frame: Current frame.
            alpha: Processed alpha matte.
        """
        self.state.prev_frame = frame.copy()
        self.state.prev_alpha = alpha.copy()
        self.state.frame_count += 1

        # Update history
        self.state.alpha_history.append(alpha.copy())
        self.state.frame_history.append(frame.copy())

        # Trim history
        max_history = self.config.history_size
        if len(self.state.alpha_history) > max_history:
            self.state.alpha_history = self.state.alpha_history[-max_history:]
            self.state.frame_history = self.state.frame_history[-max_history:]

    def __repr__(self) -> str:
        """String representation."""
        return f"TemporalConsistency(method={self.config.method.value})"


class TemporalFilter:
    """Simple temporal filter using Kalman filtering.

    Provides lightweight temporal smoothing suitable for
    real-time applications.

    Example:
        >>> filter = TemporalFilter(smoothness=0.5)
        >>> smoothed = filter.filter(alpha)
    """

    def __init__(
        self,
        smoothness: float = 0.5,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        """Initialize temporal filter.

        Args:
            smoothness: Smoothness factor (0-1, higher = smoother).
            process_noise: Process noise covariance.
            measurement_noise: Measurement noise covariance.
        """
        self.smoothness = smoothness
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self._estimate: ndarray | None = None
        self._error_cov: ndarray | None = None
        self._initialized = False

    def reset(self) -> None:
        """Reset filter state."""
        self._estimate = None
        self._error_cov = None
        self._initialized = False

    def filter(self, measurement: ndarray) -> ndarray:
        """Apply Kalman filter to alpha matte.

        Args:
            measurement: Current alpha measurement.

        Returns:
            Filtered alpha matte.
        """
        measurement = measurement.astype(np.float32)

        if not self._initialized:
            self._estimate = measurement.copy()
            self._error_cov = np.ones_like(measurement) * self.measurement_noise
            self._initialized = True
            return measurement.astype(np.uint8)

        # Predict
        predicted = self._estimate
        predicted_cov = self._error_cov + self.process_noise

        # Update
        kalman_gain = predicted_cov / (predicted_cov + self.measurement_noise)

        # Adjust gain based on smoothness
        kalman_gain *= (1 - self.smoothness)

        self._estimate = predicted + kalman_gain * (measurement - predicted)
        self._error_cov = (1 - kalman_gain) * predicted_cov

        return np.clip(self._estimate, 0, 255).astype(np.uint8)


def apply_temporal_consistency(
    frames: list[tuple[ndarray, ndarray]],
    method: TemporalMethod = TemporalMethod.OPTICAL_FLOW,
    **kwargs,
) -> list[ndarray]:
    """Apply temporal consistency to a sequence of frames.

    Convenience function for batch processing.

    Args:
        frames: List of (frame, alpha) tuples.
        method: Temporal consistency method.
        **kwargs: Additional config parameters.

    Returns:
        List of temporally consistent alpha mattes.
    """
    config = TemporalConfig(method=method, **kwargs)
    processor = TemporalConsistency(config)

    results = []
    for frame, alpha in frames:
        smoothed = processor.process(frame, alpha)
        results.append(smoothed)

    return results


def temporal_smooth(
    alpha_sequence: list[ndarray],
    smoothness: float = 0.5,
) -> list[ndarray]:
    """Apply simple temporal smoothing to alpha sequence.

    Args:
        alpha_sequence: List of alpha mattes.
        smoothness: Smoothness factor (0-1).

    Returns:
        Smoothed alpha sequence.
    """
    filter = TemporalFilter(smoothness=smoothness)
    return [filter.filter(alpha) for alpha in alpha_sequence]
