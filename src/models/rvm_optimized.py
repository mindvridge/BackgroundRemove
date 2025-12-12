"""Optimized RVM model with TensorRT and CUDA execution providers.

Provides high-performance inference using ONNX Runtime with:
- TensorRT Execution Provider (if available)
- CUDA Execution Provider
- FP16 inference support
- Session caching and reuse
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


@dataclass
class OptimizedONNXConfig:
    """Configuration for optimized ONNX inference."""

    model_path: str | Path
    max_downsample_size: int = 512

    # Execution provider settings
    use_tensorrt: bool = True
    use_cuda: bool = True
    use_fp16: bool = True
    device_id: int = 0

    # TensorRT settings
    trt_max_workspace_size: int = 2 << 30  # 2GB
    trt_fp16_enable: bool = True
    trt_int8_enable: bool = False
    trt_engine_cache_enable: bool = True
    trt_engine_cache_path: str = ".cache/tensorrt"

    # CUDA settings
    cuda_arena_extend_strategy: str = "kSameAsRequested"
    cuda_cudnn_conv_algo_search: str = "DEFAULT"
    cuda_do_copy_in_default_stream: bool = True

    # Session settings
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    intra_op_num_threads: int = 0  # 0 = auto
    inter_op_num_threads: int = 0  # 0 = auto
    enable_mem_pattern: bool = True
    enable_cpu_mem_arena: bool = True


@dataclass
class OptimizedRecurrentState:
    """Optimized recurrent state with pre-allocated buffers."""

    r1: ndarray | None = None
    r2: ndarray | None = None
    r3: ndarray | None = None
    r4: ndarray | None = None
    _initialized_shape: tuple[int, int] | None = field(default=None, repr=False)

    def initialize(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize state buffers with pre-allocation.

        Args:
            batch_size: Batch size.
            height: Input height.
            width: Input width.
            dtype: Data type.
        """
        shape = (height, width)

        if self._initialized_shape == shape and self.r1 is not None:
            # Already initialized for this shape, just zero out
            self.r1.fill(0)
            self.r2.fill(0)
            self.r3.fill(0)
            self.r4.fill(0)
            return

        self.r1 = np.zeros((batch_size, 16, height // 2, width // 2), dtype=dtype)
        self.r2 = np.zeros((batch_size, 20, height // 4, width // 4), dtype=dtype)
        self.r3 = np.zeros((batch_size, 40, height // 8, width // 8), dtype=dtype)
        self.r4 = np.zeros((batch_size, 64, height // 16, width // 16), dtype=dtype)
        self._initialized_shape = shape

        logger.debug(f"Initialized recurrent state for {width}x{height}")

    def reset(self) -> None:
        """Reset state values to zero (keep buffers)."""
        if self.r1 is not None:
            self.r1.fill(0)
        if self.r2 is not None:
            self.r2.fill(0)
        if self.r3 is not None:
            self.r3.fill(0)
        if self.r4 is not None:
            self.r4.fill(0)

    def is_initialized(self) -> bool:
        """Check if states are initialized."""
        return all(s is not None for s in [self.r1, self.r2, self.r3, self.r4])


class RVMOptimizedModel:
    """Optimized RVM model with TensorRT/CUDA acceleration.

    Features:
    - Automatic provider selection (TensorRT > CUDA > CPU)
    - FP16 inference support
    - Session caching and reuse
    - Pre-allocated buffers for zero-copy inference

    Example:
        >>> config = OptimizedONNXConfig(model_path="rvm.onnx", use_fp16=True)
        >>> model = RVMOptimizedModel(config)
        >>> model.load()
        >>> fgr, pha = model.inference(frame)
    """

    # Class-level session cache
    _session_cache: dict[str, Any] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: OptimizedONNXConfig) -> None:
        """Initialize optimized model.

        Args:
            config: Model configuration.
        """
        self.config = config
        self.session = None
        self.recurrent_state = OptimizedRecurrentState()
        self._loaded = False
        self._provider: str = ""
        self._dtype = np.float16 if config.use_fp16 else np.float32

        # Pre-allocated buffers
        self._input_buffer: ndarray | None = None
        self._dsr_buffer: ndarray | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def provider(self) -> str:
        """Get current execution provider."""
        return self._provider

    def _get_available_providers(self) -> list[str]:
        """Get list of available ONNX Runtime providers.

        Returns:
            List of available provider names.
        """
        import onnxruntime as ort
        return ort.get_available_providers()

    def _build_provider_options(self) -> list[tuple[str, dict] | str]:
        """Build execution provider list with options.

        Returns:
            List of providers with their options.
        """
        providers = []
        available = self._get_available_providers()

        logger.debug(f"Available providers: {available}")

        # TensorRT Provider
        if self.config.use_tensorrt and "TensorrtExecutionProvider" in available:
            trt_options = {
                "device_id": self.config.device_id,
                "trt_max_workspace_size": self.config.trt_max_workspace_size,
                "trt_fp16_enable": self.config.trt_fp16_enable,
                "trt_int8_enable": self.config.trt_int8_enable,
            }

            if self.config.trt_engine_cache_enable:
                cache_path = Path(self.config.trt_engine_cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)
                trt_options["trt_engine_cache_enable"] = True
                trt_options["trt_engine_cache_path"] = str(cache_path)

            providers.append(("TensorrtExecutionProvider", trt_options))
            logger.info("TensorRT provider configured")

        # CUDA Provider
        if self.config.use_cuda and "CUDAExecutionProvider" in available:
            cuda_options = {
                "device_id": self.config.device_id,
                "arena_extend_strategy": self.config.cuda_arena_extend_strategy,
                "cudnn_conv_algo_search": self.config.cuda_cudnn_conv_algo_search,
                "do_copy_in_default_stream": self.config.cuda_do_copy_in_default_stream,
            }
            providers.append(("CUDAExecutionProvider", cuda_options))
            logger.info("CUDA provider configured")

        # CPU Provider (fallback)
        providers.append("CPUExecutionProvider")

        return providers

    def _create_session_options(self):
        """Create ONNX Runtime session options.

        Returns:
            SessionOptions instance.
        """
        import onnxruntime as ort

        options = ort.SessionOptions()

        # Graph optimization
        opt_levels = {
            "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        options.graph_optimization_level = opt_levels.get(
            self.config.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )

        # Thread settings
        if self.config.intra_op_num_threads > 0:
            options.intra_op_num_threads = self.config.intra_op_num_threads
        if self.config.inter_op_num_threads > 0:
            options.inter_op_num_threads = self.config.inter_op_num_threads

        # Memory settings
        options.enable_mem_pattern = self.config.enable_mem_pattern
        options.enable_cpu_mem_arena = self.config.enable_cpu_mem_arena

        return options

    def _get_cache_key(self) -> str:
        """Generate cache key for session caching.

        Returns:
            Cache key string.
        """
        model_path = Path(self.config.model_path)
        return f"{model_path.name}_{self.config.use_fp16}_{self.config.device_id}"

    def load(self) -> None:
        """Load the ONNX model with optimized providers.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If loading fails.
        """
        import onnxruntime as ort

        if self._loaded:
            logger.warning("Model already loaded")
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        cache_key = self._get_cache_key()

        # Check session cache
        with self._cache_lock:
            if cache_key in self._session_cache:
                logger.info("Reusing cached ONNX session")
                self.session = self._session_cache[cache_key]
                self._provider = self.session.get_providers()[0]
                self._loaded = True
                return

        try:
            logger.info(f"Loading optimized ONNX model: {model_path}")

            providers = self._build_provider_options()
            options = self._create_session_options()

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=providers,
            )

            self._provider = self.session.get_providers()[0]
            logger.info(f"Model loaded with provider: {self._provider}")

            # Cache the session
            with self._cache_lock:
                self._session_cache[cache_key] = self.session

            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

    def _calculate_downsample_ratio(self, height: int, width: int) -> float:
        """Calculate optimal downsample ratio.

        Args:
            height: Input height.
            width: Input width.

        Returns:
            Downsample ratio.
        """
        max_size = self.config.max_downsample_size
        smaller_dim = min(height, width)

        if smaller_dim <= max_size:
            return 1.0

        return max(0.1, min(1.0, max_size / smaller_dim))

    def preprocess(self, frame: ndarray) -> ndarray:
        """Preprocess frame for inference.

        Args:
            frame: Input BGR frame (H, W, C).

        Returns:
            Preprocessed array (1, C, H, W).
        """
        import cv2

        # BGR to RGB
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # HWC to NCHW, normalize
        h, w = frame.shape[:2]

        # Reuse input buffer if possible
        if (
            self._input_buffer is None or
            self._input_buffer.shape[2:] != (h, w)
        ):
            self._input_buffer = np.empty((1, 3, h, w), dtype=self._dtype)

        np.copyto(
            self._input_buffer,
            frame.transpose(2, 0, 1)[np.newaxis].astype(self._dtype) / 255.0,
        )

        return self._input_buffer

    def postprocess(self, fgr: ndarray, pha: ndarray) -> tuple[ndarray, ndarray]:
        """Postprocess model outputs.

        Args:
            fgr: Foreground array (1, C, H, W).
            pha: Alpha array (1, 1, H, W).

        Returns:
            Tuple of (foreground, alpha) as (H, W, C) and (H, W).
        """
        fgr_np = (fgr[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        pha_np = (pha[0, 0] * 255).astype(np.uint8)
        return fgr_np, pha_np

    def reset_state(self) -> None:
        """Reset recurrent state for new video."""
        self.recurrent_state.reset()

    def inference(
        self,
        frame: ndarray,
        downsample_ratio: float | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Run inference on a frame.

        Args:
            frame: Input BGR frame (H, W, C).
            downsample_ratio: Optional downsample ratio override.

        Returns:
            Tuple of (foreground, alpha) arrays.
        """
        if not self._loaded or self.session is None:
            raise RuntimeError("Model not loaded")

        h, w = frame.shape[:2]

        # Initialize recurrent state if needed
        if not self.recurrent_state.is_initialized():
            self.recurrent_state.initialize(1, h, w, self._dtype)

        # Preprocess
        src = self.preprocess(frame)

        # Calculate downsample ratio
        if downsample_ratio is None:
            downsample_ratio = self._calculate_downsample_ratio(h, w)

        # Reuse dsr buffer
        if self._dsr_buffer is None:
            self._dsr_buffer = np.array([downsample_ratio], dtype=self._dtype)
        else:
            self._dsr_buffer[0] = downsample_ratio

        # Build inputs
        inputs = {
            "src": src,
            "r1i": self.recurrent_state.r1,
            "r2i": self.recurrent_state.r2,
            "r3i": self.recurrent_state.r3,
            "r4i": self.recurrent_state.r4,
            "downsample_ratio": self._dsr_buffer,
        }

        # Run inference
        outputs = self.session.run(None, inputs)
        fgr, pha, r1o, r2o, r3o, r4o = outputs

        # Update recurrent state (in-place copy)
        np.copyto(self.recurrent_state.r1, r1o)
        np.copyto(self.recurrent_state.r2, r2o)
        np.copyto(self.recurrent_state.r3, r3o)
        np.copyto(self.recurrent_state.r4, r4o)

        return self.postprocess(fgr, pha)

    def warmup(self, height: int = 480, width: int = 640, iterations: int = 3) -> None:
        """Warm up the model with dummy inference.

        Args:
            height: Warmup frame height.
            width: Warmup frame width.
            iterations: Number of warmup iterations.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        logger.info(f"Warming up model ({iterations} iterations)...")

        dummy_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        for i in range(iterations):
            self.inference(dummy_frame)
            self.reset_state()

        logger.info("Warmup complete")

    def unload(self) -> None:
        """Unload model (keeps session in cache for reuse)."""
        self.recurrent_state.reset()
        self._input_buffer = None
        self._dsr_buffer = None
        self._loaded = False

        logger.info("Model unloaded (session cached)")

    @classmethod
    def clear_session_cache(cls) -> None:
        """Clear all cached sessions."""
        with cls._cache_lock:
            cls._session_cache.clear()
        logger.info("Session cache cleared")

    def __enter__(self) -> "RVMOptimizedModel":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"RVMOptimizedModel(provider={self._provider!r}, "
            f"fp16={self.config.use_fp16}, status={status})"
        )
