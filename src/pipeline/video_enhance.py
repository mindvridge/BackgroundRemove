"""Video-specific enhancement for temporal matting.

Provides specialized techniques for video matting:
- Memory network for long-range temporal consistency
- 3D convolution for spatio-temporal features
- Attention-based frame aggregation
- Motion-compensated refinement
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class VideoMethod(Enum):
    """Video enhancement methods."""

    MEMORY_NETWORK = "memory_network"
    TEMPORAL_ATTENTION = "temporal_attention"
    MOTION_COMPENSATED = "motion_compensated"
    MULTI_FRAME_FUSION = "multi_frame_fusion"
    RECURRENT = "recurrent"


@dataclass
class VideoEnhanceConfig:
    """Configuration for video enhancement."""

    method: VideoMethod = VideoMethod.MEMORY_NETWORK

    # Memory settings
    memory_size: int = 10  # Number of frames to remember
    memory_decay: float = 0.9  # Decay factor for older frames

    # Attention settings
    attention_heads: int = 4
    attention_window: int = 5  # Frames to attend to

    # Motion settings
    motion_threshold: float = 0.1
    motion_compensation: bool = True

    # Fusion settings
    fusion_weights: list[float] | None = None  # [current, prev, prev_prev, ...]

    # Quality settings
    consistency_weight: float = 0.5
    sharpness_weight: float = 0.3


class FrameMemory:
    """Memory buffer for video frames and alpha mattes.

    Stores recent frames with associated features for
    temporal reasoning.

    Example:
        >>> memory = FrameMemory(size=10)
        >>> memory.add(frame, alpha, features)
        >>> similar = memory.query(current_features)
    """

    def __init__(
        self,
        size: int = 10,
        decay: float = 0.9,
    ) -> None:
        """Initialize frame memory.

        Args:
            size: Maximum memory size.
            decay: Weight decay for older frames.
        """
        self.size = size
        self.decay = decay

        self._frames: deque[ndarray] = deque(maxlen=size)
        self._alphas: deque[ndarray] = deque(maxlen=size)
        self._features: deque[ndarray] = deque(maxlen=size)
        self._weights: deque[float] = deque(maxlen=size)

    def add(
        self,
        frame: ndarray,
        alpha: ndarray,
        features: ndarray | None = None,
    ) -> None:
        """Add frame to memory.

        Args:
            frame: Video frame.
            alpha: Alpha matte.
            features: Optional extracted features.
        """
        self._frames.append(frame.copy())
        self._alphas.append(alpha.copy())
        self._features.append(features if features is not None else self._extract_features(frame, alpha))

        # Update weights (decay older)
        new_weights = deque(maxlen=self.size)
        for w in self._weights:
            new_weights.append(w * self.decay)
        new_weights.append(1.0)
        self._weights = new_weights

    def _extract_features(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Extract features from frame.

        Args:
            frame: Video frame.
            alpha: Alpha matte.

        Returns:
            Feature vector.
        """
        # Simple feature extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram features
        hist_frame = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        hist_alpha = cv2.calcHist([alpha], [0], None, [64], [0, 256]).flatten()

        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / edges.size

        # Combine
        features = np.concatenate([
            hist_frame / hist_frame.sum(),
            hist_alpha / (hist_alpha.sum() + 1e-8),
            [edge_density],
        ])

        return features

    def query(
        self,
        current_features: ndarray,
        top_k: int = 3,
    ) -> list[tuple[ndarray, ndarray, float]]:
        """Query similar frames from memory.

        Args:
            current_features: Current frame features.
            top_k: Number of similar frames to return.

        Returns:
            List of (frame, alpha, similarity) tuples.
        """
        if not self._features:
            return []

        # Compute similarities
        similarities = []
        for i, feat in enumerate(self._features):
            # Cosine similarity
            sim = np.dot(current_features, feat) / (
                np.linalg.norm(current_features) * np.linalg.norm(feat) + 1e-8
            )
            # Weight by recency
            sim *= self._weights[i]
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for idx, sim in similarities[:top_k]:
            results.append((
                self._frames[idx],
                self._alphas[idx],
                sim,
            ))

        return results

    def get_recent(self, n: int = 3) -> list[tuple[ndarray, ndarray]]:
        """Get n most recent frames.

        Args:
            n: Number of frames.

        Returns:
            List of (frame, alpha) tuples.
        """
        results = []
        for i in range(min(n, len(self._frames))):
            idx = len(self._frames) - 1 - i
            results.append((self._frames[idx], self._alphas[idx]))
        return results

    def clear(self) -> None:
        """Clear memory."""
        self._frames.clear()
        self._alphas.clear()
        self._features.clear()
        self._weights.clear()

    def __len__(self) -> int:
        """Memory size."""
        return len(self._frames)


class TemporalAttention:
    """Attention-based temporal aggregation.

    Aggregates information from multiple frames using
    attention mechanism.

    Example:
        >>> attention = TemporalAttention(window=5)
        >>> aggregated = attention.aggregate(frames, alphas)
    """

    def __init__(
        self,
        window: int = 5,
        num_heads: int = 4,
    ) -> None:
        """Initialize temporal attention.

        Args:
            window: Attention window size.
            num_heads: Number of attention heads.
        """
        self.window = window
        self.num_heads = num_heads

    def _compute_attention_weights(
        self,
        query: ndarray,
        keys: list[ndarray],
    ) -> ndarray:
        """Compute attention weights.

        Args:
            query: Query features (current frame).
            keys: Key features (other frames).

        Returns:
            Attention weights.
        """
        weights = []
        for key in keys:
            # Simple dot-product attention
            score = np.sum(query * key) / (np.sqrt(query.size) + 1e-8)
            weights.append(score)

        # Softmax
        weights = np.array(weights)
        weights = np.exp(weights - weights.max())
        weights = weights / (weights.sum() + 1e-8)

        return weights

    def aggregate(
        self,
        current_frame: ndarray,
        current_alpha: ndarray,
        history_frames: list[ndarray],
        history_alphas: list[ndarray],
    ) -> ndarray:
        """Aggregate alpha from multiple frames.

        Args:
            current_frame: Current frame.
            current_alpha: Current alpha estimate.
            history_frames: Previous frames.
            history_alphas: Previous alpha mattes.

        Returns:
            Aggregated alpha matte.
        """
        if not history_frames:
            return current_alpha

        # Extract features for attention
        def extract_features(frame: ndarray) -> ndarray:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downsample for efficiency
            small = cv2.resize(gray, (64, 64))
            return small.flatten().astype(np.float32) / 255.0

        query = extract_features(current_frame)
        keys = [extract_features(f) for f in history_frames[-self.window:]]
        values = history_alphas[-self.window:]

        # Compute attention
        weights = self._compute_attention_weights(query, keys)

        # Weighted aggregation
        h, w = current_alpha.shape[:2]
        aggregated = np.zeros((h, w), dtype=np.float32)

        for alpha, weight in zip(values, weights):
            resized = cv2.resize(alpha, (w, h)).astype(np.float32)
            aggregated += weight * resized

        # Blend with current
        result = 0.6 * current_alpha.astype(np.float32) + 0.4 * aggregated

        return np.clip(result, 0, 255).astype(np.uint8)


class MotionCompensator:
    """Motion-compensated alpha refinement.

    Aligns previous frames to current using optical flow
    for better temporal consistency.

    Example:
        >>> compensator = MotionCompensator()
        >>> aligned = compensator.compensate(prev_alpha, flow)
    """

    def __init__(
        self,
        flow_quality: str = "medium",
    ) -> None:
        """Initialize motion compensator.

        Args:
            flow_quality: Flow computation quality.
        """
        self.flow_quality = flow_quality
        self._flow_params = self._get_flow_params()

    def _get_flow_params(self) -> dict:
        """Get optical flow parameters."""
        if self.flow_quality == "fast":
            return {
                "pyr_scale": 0.5,
                "levels": 2,
                "winsize": 11,
                "iterations": 2,
                "poly_n": 5,
                "poly_sigma": 1.1,
                "flags": 0,
            }
        elif self.flow_quality == "high":
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

    def compute_flow(
        self,
        prev_frame: ndarray,
        curr_frame: ndarray,
    ) -> ndarray:
        """Compute optical flow.

        Args:
            prev_frame: Previous frame.
            curr_frame: Current frame.

        Returns:
            Optical flow field.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self._flow_params
        )

        return flow

    def warp(
        self,
        image: ndarray,
        flow: ndarray,
    ) -> ndarray:
        """Warp image using flow.

        Args:
            image: Image to warp.
            flow: Optical flow.

        Returns:
            Warped image.
        """
        h, w = flow.shape[:2]

        # Create remap coordinates
        flow_map = np.column_stack([
            np.tile(np.arange(w), h),
            np.repeat(np.arange(h), w),
        ]).reshape(h, w, 2).astype(np.float32)

        flow_map += flow

        warped = cv2.remap(
            image,
            flow_map[:, :, 0],
            flow_map[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return warped

    def compensate(
        self,
        prev_frame: ndarray,
        curr_frame: ndarray,
        prev_alpha: ndarray,
    ) -> ndarray:
        """Compensate previous alpha for current frame.

        Args:
            prev_frame: Previous frame.
            curr_frame: Current frame.
            prev_alpha: Previous alpha matte.

        Returns:
            Motion-compensated alpha.
        """
        flow = self.compute_flow(prev_frame, curr_frame)
        warped = self.warp(prev_alpha, flow)
        return warped


class MultiFrameFusion:
    """Multi-frame fusion for robust matting.

    Combines multiple frame observations for improved
    alpha quality.

    Example:
        >>> fusion = MultiFrameFusion(window=5)
        >>> fused = fusion.fuse(alphas, weights)
    """

    def __init__(
        self,
        window: int = 5,
        weights: list[float] | None = None,
    ) -> None:
        """Initialize multi-frame fusion.

        Args:
            window: Fusion window size.
            weights: Frame weights (default: temporal decay).
        """
        self.window = window
        self.weights = weights or self._default_weights()

    def _default_weights(self) -> list[float]:
        """Generate default temporal weights."""
        weights = []
        for i in range(self.window):
            weights.append(0.9 ** i)
        # Normalize
        total = sum(weights)
        return [w / total for w in weights]

    def fuse(
        self,
        alphas: list[ndarray],
        custom_weights: list[float] | None = None,
    ) -> ndarray:
        """Fuse multiple alpha mattes.

        Args:
            alphas: List of alpha mattes (newest first).
            custom_weights: Optional custom weights.

        Returns:
            Fused alpha matte.
        """
        if not alphas:
            raise ValueError("No alphas to fuse")

        weights = custom_weights or self.weights[:len(alphas)]

        # Ensure same size
        h, w = alphas[0].shape[:2]
        fused = np.zeros((h, w), dtype=np.float32)
        total_weight = 0

        for alpha, weight in zip(alphas, weights):
            resized = cv2.resize(alpha, (w, h)) if alpha.shape[:2] != (h, w) else alpha
            fused += weight * resized.astype(np.float32)
            total_weight += weight

        fused /= (total_weight + 1e-8)

        return np.clip(fused, 0, 255).astype(np.uint8)


class VideoEnhancer:
    """Video-specific alpha enhancement processor.

    Combines multiple video-aware techniques for
    optimal temporal matting quality.

    Example:
        >>> enhancer = VideoEnhancer(VideoEnhanceConfig(
        ...     method=VideoMethod.MEMORY_NETWORK
        ... ))
        >>> for frame, alpha in video:
        ...     enhanced = enhancer.enhance(frame, alpha)
    """

    def __init__(self, config: VideoEnhanceConfig | None = None) -> None:
        """Initialize video enhancer.

        Args:
            config: Video enhancement configuration.
        """
        self.config = config or VideoEnhanceConfig()

        # Initialize components
        self._memory = FrameMemory(
            size=self.config.memory_size,
            decay=self.config.memory_decay,
        )
        self._attention = TemporalAttention(
            window=self.config.attention_window,
            num_heads=self.config.attention_heads,
        )
        self._motion = MotionCompensator()
        self._fusion = MultiFrameFusion(
            window=self.config.attention_window,
            weights=self.config.fusion_weights,
        )

        self._prev_frame = None
        self._prev_alpha = None
        self._frame_count = 0

        logger.debug(f"Video enhancer initialized: {self.config.method.value}")

    def reset(self) -> None:
        """Reset enhancer state."""
        self._memory.clear()
        self._prev_frame = None
        self._prev_alpha = None
        self._frame_count = 0

    def _memory_network_enhance(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Memory network based enhancement.

        Args:
            frame: Current frame.
            alpha: Current alpha.

        Returns:
            Enhanced alpha.
        """
        # Query similar frames from memory
        features = self._memory._extract_features(frame, alpha)
        similar = self._memory.query(features, top_k=3)

        if not similar:
            self._memory.add(frame, alpha, features)
            return alpha

        # Aggregate from similar frames
        aggregated = np.zeros_like(alpha, dtype=np.float32)
        total_weight = 0

        for mem_frame, mem_alpha, similarity in similar:
            # Motion compensate if needed
            if self.config.motion_compensation and self._prev_frame is not None:
                try:
                    warped = self._motion.compensate(mem_frame, frame, mem_alpha)
                except cv2.error:
                    warped = mem_alpha
            else:
                warped = mem_alpha

            # Resize if needed
            if warped.shape != alpha.shape:
                warped = cv2.resize(warped, (alpha.shape[1], alpha.shape[0]))

            aggregated += similarity * warped.astype(np.float32)
            total_weight += similarity

        if total_weight > 0:
            aggregated /= total_weight

        # Blend with current
        consistency = self.config.consistency_weight
        result = consistency * aggregated + (1 - consistency) * alpha.astype(np.float32)

        # Update memory
        self._memory.add(frame, alpha, features)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _temporal_attention_enhance(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Attention-based temporal enhancement.

        Args:
            frame: Current frame.
            alpha: Current alpha.

        Returns:
            Enhanced alpha.
        """
        recent = self._memory.get_recent(self.config.attention_window)

        if not recent:
            self._memory.add(frame, alpha)
            return alpha

        history_frames = [f for f, _ in recent]
        history_alphas = [a for _, a in recent]

        enhanced = self._attention.aggregate(
            frame, alpha, history_frames, history_alphas
        )

        self._memory.add(frame, alpha)

        return enhanced

    def _motion_compensated_enhance(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Motion-compensated enhancement.

        Args:
            frame: Current frame.
            alpha: Current alpha.

        Returns:
            Enhanced alpha.
        """
        if self._prev_frame is None or self._prev_alpha is None:
            self._prev_frame = frame
            self._prev_alpha = alpha
            return alpha

        # Warp previous alpha
        try:
            warped = self._motion.compensate(self._prev_frame, frame, self._prev_alpha)
        except cv2.error:
            warped = self._prev_alpha

        # Compute motion mask
        flow = self._motion.compute_flow(self._prev_frame, frame)
        motion_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        motion_mask = (motion_mag > self.config.motion_threshold).astype(np.float32)

        # In motion regions, prefer current; in static, prefer temporal
        result = (
            motion_mask * alpha.astype(np.float32) +
            (1 - motion_mask) * (
                self.config.consistency_weight * warped.astype(np.float32) +
                (1 - self.config.consistency_weight) * alpha.astype(np.float32)
            )
        )

        self._prev_frame = frame
        self._prev_alpha = result.astype(np.uint8)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _multi_frame_fusion_enhance(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Multi-frame fusion enhancement.

        Args:
            frame: Current frame.
            alpha: Current alpha.

        Returns:
            Enhanced alpha.
        """
        recent = self._memory.get_recent(self.config.attention_window)

        alphas = [alpha]
        for _, a in recent:
            alphas.append(a)

        fused = self._fusion.fuse(alphas)
        self._memory.add(frame, alpha)

        return fused

    def enhance(
        self,
        frame: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Enhance alpha using video-aware methods.

        Args:
            frame: Current video frame.
            alpha: Current alpha matte.

        Returns:
            Enhanced alpha matte.
        """
        method = self.config.method

        if method == VideoMethod.MEMORY_NETWORK:
            result = self._memory_network_enhance(frame, alpha)
        elif method == VideoMethod.TEMPORAL_ATTENTION:
            result = self._temporal_attention_enhance(frame, alpha)
        elif method == VideoMethod.MOTION_COMPENSATED:
            result = self._motion_compensated_enhance(frame, alpha)
        elif method == VideoMethod.MULTI_FRAME_FUSION:
            result = self._multi_frame_fusion_enhance(frame, alpha)
        else:
            result = alpha

        self._frame_count += 1

        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"VideoEnhancer(method={self.config.method.value})"


def video_enhance(
    frames: list[ndarray],
    alphas: list[ndarray],
    method: VideoMethod = VideoMethod.MEMORY_NETWORK,
) -> list[ndarray]:
    """Convenience function for video enhancement.

    Args:
        frames: List of video frames.
        alphas: List of alpha mattes.
        method: Enhancement method.

    Returns:
        List of enhanced alpha mattes.
    """
    config = VideoEnhanceConfig(method=method)
    enhancer = VideoEnhancer(config)

    enhanced = []
    for frame, alpha in zip(frames, alphas):
        result = enhancer.enhance(frame, alpha)
        enhanced.append(result)

    return enhanced
