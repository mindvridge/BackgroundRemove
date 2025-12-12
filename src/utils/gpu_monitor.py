"""GPU memory monitoring and management utilities.

Provides tools for monitoring GPU memory usage and automatically
adjusting processing parameters for optimal performance.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryInfo:
    """GPU memory information."""

    device_id: int
    device_name: str
    total_memory: int  # bytes
    used_memory: int  # bytes
    free_memory: int  # bytes

    @property
    def usage_percent(self) -> float:
        """Memory usage percentage."""
        if self.total_memory == 0:
            return 0.0
        return (self.used_memory / self.total_memory) * 100

    @property
    def free_percent(self) -> float:
        """Free memory percentage."""
        return 100.0 - self.usage_percent

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        """Used memory in GB."""
        return self.used_memory / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        """Free memory in GB."""
        return self.free_memory / (1024 ** 3)


class GPUMonitor:
    """GPU memory monitor with automatic adjustment support.

    Monitors GPU memory usage and provides recommendations for
    downsample ratio and batch size adjustments.

    Example:
        >>> monitor = GPUMonitor()
        >>> info = monitor.get_memory_info()
        >>> ratio = monitor.recommend_downsample_ratio(1920, 1080)
    """

    # Memory thresholds
    HIGH_MEMORY_THRESHOLD = 0.85  # 85% usage
    CRITICAL_MEMORY_THRESHOLD = 0.95  # 95% usage

    # Base memory requirements (approximate, in GB)
    BASE_MODEL_MEMORY = 0.5  # Model weights
    PER_PIXEL_MEMORY = 12e-9  # ~12 bytes per pixel for inference

    def __init__(self, device_id: int = 0) -> None:
        """Initialize GPU monitor.

        Args:
            device_id: GPU device ID to monitor.
        """
        self.device_id = device_id
        self._cuda_available = self._check_cuda()
        self._lock = threading.Lock()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA is available.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._cuda_available

    def get_memory_info(self) -> GPUMemoryInfo | None:
        """Get current GPU memory information.

        Returns:
            GPUMemoryInfo or None if not available.
        """
        if not self._cuda_available:
            return None

        try:
            import torch

            with self._lock:
                device = torch.device(f"cuda:{self.device_id}")
                torch.cuda.synchronize(device)

                total = torch.cuda.get_device_properties(self.device_id).total_memory
                reserved = torch.cuda.memory_reserved(self.device_id)
                allocated = torch.cuda.memory_allocated(self.device_id)

                # Use allocated as "used" for more accurate picture
                used = allocated
                free = total - reserved

                device_name = torch.cuda.get_device_name(self.device_id)

                return GPUMemoryInfo(
                    device_id=self.device_id,
                    device_name=device_name,
                    total_memory=total,
                    used_memory=used,
                    free_memory=free,
                )

        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None

    def get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage.

        Returns:
            Memory usage as percentage (0-100), or 0 if not available.
        """
        info = self.get_memory_info()
        return info.usage_percent if info else 0.0

    def estimate_inference_memory(
        self,
        width: int,
        height: int,
        batch_size: int = 1,
        fp16: bool = False,
    ) -> float:
        """Estimate memory required for inference.

        Args:
            width: Frame width.
            height: Frame height.
            batch_size: Batch size.
            fp16: Whether using FP16.

        Returns:
            Estimated memory in GB.
        """
        pixels = width * height * batch_size
        bytes_per_pixel = self.PER_PIXEL_MEMORY

        if fp16:
            bytes_per_pixel *= 0.5

        # Account for intermediate tensors (roughly 4x input)
        memory_gb = (pixels * bytes_per_pixel * 4) / (1024 ** 3)

        # Add base model memory
        memory_gb += self.BASE_MODEL_MEMORY

        return memory_gb

    def recommend_downsample_ratio(
        self,
        width: int,
        height: int,
        target_memory_usage: float = 0.7,
        min_ratio: float = 0.25,
        max_ratio: float = 1.0,
    ) -> float:
        """Recommend downsample ratio based on available memory.

        Args:
            width: Input frame width.
            height: Input frame height.
            target_memory_usage: Target memory usage (0-1).
            min_ratio: Minimum allowed ratio.
            max_ratio: Maximum allowed ratio.

        Returns:
            Recommended downsample ratio.
        """
        info = self.get_memory_info()

        if info is None:
            # Default to conservative ratio if no GPU info
            return min(0.5, max_ratio)

        # Calculate available memory for inference
        target_free = info.total_memory * (1 - target_memory_usage)
        available = info.free_memory - target_free

        if available <= 0:
            return min_ratio

        # Estimate current memory need
        estimated = self.estimate_inference_memory(width, height)
        available_gb = available / (1024 ** 3)

        if estimated <= available_gb:
            return max_ratio

        # Calculate ratio to fit in available memory
        ratio = (available_gb / estimated) ** 0.5
        ratio = max(min_ratio, min(max_ratio, ratio))

        logger.debug(
            f"Recommended ratio: {ratio:.3f} "
            f"(available: {available_gb:.2f}GB, estimated: {estimated:.2f}GB)"
        )

        return ratio

    def should_reduce_quality(self) -> bool:
        """Check if quality should be reduced due to memory pressure.

        Returns:
            True if memory usage is high.
        """
        usage = self.get_memory_usage_percent()
        return usage > self.HIGH_MEMORY_THRESHOLD * 100

    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical.

        Returns:
            True if memory usage is critical.
        """
        usage = self.get_memory_usage_percent()
        return usage > self.CRITICAL_MEMORY_THRESHOLD * 100

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if not self._cuda_available:
            return

        try:
            import torch
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")


class AdaptiveDownsampler:
    """Adaptive downsampling based on GPU memory and frame characteristics.

    Dynamically adjusts downsample ratio during processing to maintain
    optimal performance and memory usage.
    """

    def __init__(
        self,
        gpu_monitor: GPUMonitor | None = None,
        max_dimension: int = 512,
        target_memory_usage: float = 0.7,
        adjustment_interval: int = 30,  # frames
    ) -> None:
        """Initialize adaptive downsampler.

        Args:
            gpu_monitor: GPU monitor instance.
            max_dimension: Maximum dimension for downsampling.
            target_memory_usage: Target GPU memory usage.
            adjustment_interval: Frames between ratio adjustments.
        """
        self.gpu_monitor = gpu_monitor or GPUMonitor()
        self.max_dimension = max_dimension
        self.target_memory_usage = target_memory_usage
        self.adjustment_interval = adjustment_interval

        self._current_ratio = 1.0
        self._frame_count = 0
        self._last_adjustment_time = 0.0
        self._ratio_history: list[float] = []

    def calculate_base_ratio(self, height: int, width: int) -> float:
        """Calculate base downsample ratio from dimensions.

        Args:
            height: Frame height.
            width: Frame width.

        Returns:
            Base downsample ratio.
        """
        smaller_dim = min(height, width)

        if smaller_dim <= self.max_dimension:
            return 1.0

        return self.max_dimension / smaller_dim

    def get_ratio(
        self,
        height: int,
        width: int,
        force_recalculate: bool = False,
    ) -> float:
        """Get current downsample ratio, adjusting if needed.

        Args:
            height: Frame height.
            width: Frame width.
            force_recalculate: Force ratio recalculation.

        Returns:
            Downsample ratio to use.
        """
        self._frame_count += 1

        # Check if adjustment is needed
        should_adjust = (
            force_recalculate or
            self._frame_count % self.adjustment_interval == 0
        )

        if should_adjust:
            self._adjust_ratio(height, width)

        return self._current_ratio

    def _adjust_ratio(self, height: int, width: int) -> None:
        """Adjust ratio based on current conditions.

        Args:
            height: Frame height.
            width: Frame width.
        """
        base_ratio = self.calculate_base_ratio(height, width)

        if self.gpu_monitor.is_available:
            memory_ratio = self.gpu_monitor.recommend_downsample_ratio(
                width, height,
                target_memory_usage=self.target_memory_usage,
            )
            # Use the more conservative ratio
            new_ratio = min(base_ratio, memory_ratio)
        else:
            new_ratio = base_ratio

        # Smooth adjustment to avoid sudden changes
        if self._ratio_history:
            avg_history = sum(self._ratio_history[-5:]) / len(self._ratio_history[-5:])
            new_ratio = 0.7 * new_ratio + 0.3 * avg_history

        self._ratio_history.append(new_ratio)
        if len(self._ratio_history) > 100:
            self._ratio_history = self._ratio_history[-50:]

        if abs(new_ratio - self._current_ratio) > 0.05:
            logger.debug(f"Adjusting ratio: {self._current_ratio:.3f} -> {new_ratio:.3f}")
            self._current_ratio = new_ratio

    def reset(self) -> None:
        """Reset adaptive state for new video."""
        self._current_ratio = 1.0
        self._frame_count = 0
        self._ratio_history.clear()


class PerformanceMonitor:
    """Monitor processing performance metrics."""

    def __init__(self, window_size: int = 30) -> None:
        """Initialize performance monitor.

        Args:
            window_size: Number of samples for moving average.
        """
        self.window_size = window_size
        self._frame_times: list[float] = []
        self._last_time: float | None = None
        self._total_frames = 0
        self._start_time: float | None = None

    def start(self) -> None:
        """Start monitoring."""
        self._start_time = time.time()
        self._last_time = self._start_time

    def tick(self) -> None:
        """Record a frame completion."""
        now = time.time()

        if self._last_time is not None:
            frame_time = now - self._last_time
            self._frame_times.append(frame_time)

            if len(self._frame_times) > self.window_size:
                self._frame_times.pop(0)

        self._last_time = now
        self._total_frames += 1

    @property
    def current_fps(self) -> float:
        """Get current FPS (moving average)."""
        if not self._frame_times:
            return 0.0

        avg_time = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    @property
    def average_fps(self) -> float:
        """Get average FPS since start."""
        if self._start_time is None or self._total_frames == 0:
            return 0.0

        elapsed = time.time() - self._start_time
        return self._total_frames / elapsed if elapsed > 0 else 0.0

    @property
    def total_frames(self) -> int:
        """Total frames processed."""
        return self._total_frames

    @property
    def elapsed_time(self) -> float:
        """Elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def reset(self) -> None:
        """Reset monitor state."""
        self._frame_times.clear()
        self._last_time = None
        self._total_frames = 0
        self._start_time = None
