"""Model session manager for efficient model loading and reuse.

Provides centralized model management with:
- Session caching and reuse
- Automatic GPU memory management
- FP16 inference with torch.autocast
- Model warm-up and optimization
"""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from numpy import ndarray

    from src.models.base import BaseModel
    from src.utils.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Available model backends."""

    PYTORCH = "pytorch"
    ONNX_CPU = "onnx_cpu"
    ONNX_CUDA = "onnx_cuda"
    ONNX_TENSORRT = "onnx_tensorrt"


@dataclass
class SessionConfig:
    """Configuration for model session."""

    model_variant: str = "mobilenetv3"
    backend: ModelBackend = ModelBackend.PYTORCH

    # Performance settings
    use_fp16: bool = True
    use_autocast: bool = True
    use_cudnn_benchmark: bool = True
    use_channels_last: bool = True

    # Memory settings
    max_cached_sessions: int = 2
    auto_clear_cache: bool = True
    memory_fraction: float = 0.8

    # ONNX settings
    onnx_model_path: str | Path | None = None

    # Optimization settings
    warmup_iterations: int = 3
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)


class FP16InferenceContext:
    """Context manager for FP16 inference with autocast.

    Provides automatic mixed precision inference using torch.autocast
    for improved performance on supported GPUs.

    Example:
        >>> with FP16InferenceContext(enabled=True):
        ...     output = model(input)
    """

    def __init__(
        self,
        enabled: bool = True,
        device_type: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize FP16 context.

        Args:
            enabled: Whether to enable autocast.
            device_type: Device type for autocast.
            dtype: Target dtype for autocast.
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.device_type = device_type
        self.dtype = dtype
        self._autocast_context = None

    def __enter__(self):
        """Enter context."""
        if self.enabled:
            self._autocast_context = torch.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
            )
            self._autocast_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self._autocast_context is not None:
            self._autocast_context.__exit__(exc_type, exc_val, exc_tb)


class OptimizedPyTorchModel:
    """Optimized PyTorch model wrapper with FP16 and autocast support.

    Features:
    - Automatic mixed precision with torch.autocast
    - cuDNN benchmark mode
    - Channels-last memory format
    - Optional torch.compile optimization
    """

    def __init__(
        self,
        config: SessionConfig,
        gpu_monitor: "GPUMonitor | None" = None,
    ) -> None:
        """Initialize optimized model.

        Args:
            config: Session configuration.
            gpu_monitor: Optional GPU monitor.
        """
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.model: nn.Module | None = None
        self.device: torch.device | None = None
        self._loaded = False
        self._recurrent_state: tuple | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _setup_cuda_optimizations(self) -> None:
        """Apply CUDA optimizations."""
        if not torch.cuda.is_available():
            return

        # Enable cuDNN benchmark for consistent input sizes
        if self.config.use_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.debug("cuDNN benchmark enabled")

        # Set memory fraction
        if self.config.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                self.config.memory_fraction
            )

    def load(self) -> None:
        """Load and optimize the model."""
        if self._loaded:
            return

        logger.info(f"Loading PyTorch model ({self.config.model_variant})...")

        # Setup CUDA
        self._setup_cuda_optimizations()

        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load from TorchHub
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            self.config.model_variant,
            trust_repo=True,
        )

        # Move to device
        self.model = self.model.to(self.device)

        # Apply optimizations
        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            logger.debug("FP16 weights enabled")

        if self.config.use_channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.debug("Channels-last memory format enabled")

        # Compile model (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        self.model.eval()
        self._loaded = True

        # Warmup
        if self.config.warmup_iterations > 0:
            self._warmup()

        logger.info(f"Model loaded on {self.device}")

    def _warmup(self) -> None:
        """Warm up model with dummy inference."""
        logger.debug(f"Warming up ({self.config.warmup_iterations} iterations)...")

        dummy = torch.randn(1, 3, 480, 640, device=self.device)
        if self.config.use_fp16:
            dummy = dummy.half()
        if self.config.use_channels_last:
            dummy = dummy.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            for _ in range(self.config.warmup_iterations):
                with FP16InferenceContext(self.config.use_autocast):
                    rec = [None] * 4
                    dsr = torch.tensor([0.25], device=self.device)
                    if self.config.use_fp16:
                        dsr = dsr.half()
                    self.model(dummy, *rec, dsr)

        self._recurrent_state = None
        logger.debug("Warmup complete")

    def reset_state(self) -> None:
        """Reset recurrent state."""
        self._recurrent_state = None

    def preprocess(self, frame: ndarray) -> torch.Tensor:
        """Preprocess frame for inference.

        Args:
            frame: BGR frame (H, W, C).

        Returns:
            Preprocessed tensor.
        """
        import cv2

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To tensor
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device).float() / 255.0

        if self.config.use_fp16:
            tensor = tensor.half()
        if self.config.use_channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)

        return tensor

    def postprocess(
        self,
        fgr: torch.Tensor,
        pha: torch.Tensor,
    ) -> tuple[ndarray, ndarray]:
        """Postprocess outputs.

        Args:
            fgr: Foreground tensor.
            pha: Alpha tensor.

        Returns:
            Numpy arrays (foreground, alpha).
        """
        import numpy as np

        fgr_np = (fgr[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pha_np = (pha[0, 0].cpu().float().numpy() * 255).astype(np.uint8)

        return fgr_np, pha_np

    def _calculate_downsample_ratio(self, h: int, w: int, max_size: int = 512) -> float:
        """Calculate downsample ratio."""
        smaller = min(h, w)
        if smaller <= max_size:
            return 1.0
        return max_size / smaller

    @torch.inference_mode()
    def inference(
        self,
        frame: ndarray,
        downsample_ratio: float | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Run inference with autocast optimization.

        Args:
            frame: BGR frame (H, W, C).
            downsample_ratio: Optional downsample ratio.

        Returns:
            Tuple of (foreground, alpha).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        h, w = frame.shape[:2]

        # Preprocess
        src = self.preprocess(frame)

        # Downsample ratio
        if downsample_ratio is None:
            downsample_ratio = self._calculate_downsample_ratio(h, w)

        dsr = torch.tensor([downsample_ratio], device=self.device)
        if self.config.use_fp16:
            dsr = dsr.half()

        # Get recurrent state
        if self._recurrent_state is None:
            rec = [None] * 4
        else:
            rec = list(self._recurrent_state)

        # Inference with autocast
        with FP16InferenceContext(self.config.use_autocast):
            fgr, pha, *new_rec = self.model(src, *rec, dsr)

        # Update state
        self._recurrent_state = tuple(new_rec)

        return self.postprocess(fgr, pha)

    def unload(self) -> None:
        """Unload model."""
        if self.model is not None:
            del self.model
            self.model = None

        self._recurrent_state = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()


class ModelSessionManager:
    """Centralized manager for model sessions.

    Provides:
    - Session caching and reuse
    - Automatic backend selection
    - Memory management
    - Thread-safe session access
    """

    _instance: "ModelSessionManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelSessionManager":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize manager."""
        if self._initialized:
            return

        self._sessions: dict[str, Any] = {}
        self._session_refs: dict[str, weakref.ref] = {}
        self._default_config = SessionConfig()
        self._gpu_monitor: GPUMonitor | None = None
        self._initialized = True

        logger.debug("ModelSessionManager initialized")

    def set_gpu_monitor(self, monitor: "GPUMonitor") -> None:
        """Set GPU monitor for memory management.

        Args:
            monitor: GPU monitor instance.
        """
        self._gpu_monitor = monitor

    def _get_session_key(self, config: SessionConfig) -> str:
        """Generate session key.

        Args:
            config: Session configuration.

        Returns:
            Session key string.
        """
        return f"{config.backend.value}_{config.model_variant}_{config.use_fp16}"

    def _auto_select_backend(self) -> ModelBackend:
        """Automatically select best available backend.

        Returns:
            Best available backend.
        """
        # Check for TensorRT
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()

            if "TensorrtExecutionProvider" in providers:
                return ModelBackend.ONNX_TENSORRT
            elif "CUDAExecutionProvider" in providers:
                return ModelBackend.ONNX_CUDA
        except ImportError:
            pass

        # Check CUDA for PyTorch
        if torch.cuda.is_available():
            return ModelBackend.PYTORCH

        return ModelBackend.ONNX_CPU

    def get_session(
        self,
        config: SessionConfig | None = None,
        auto_select_backend: bool = True,
    ) -> Any:
        """Get or create a model session.

        Args:
            config: Session configuration.
            auto_select_backend: Auto-select best backend.

        Returns:
            Model session instance.
        """
        if config is None:
            config = self._default_config

        if auto_select_backend and config.backend == ModelBackend.PYTORCH:
            config.backend = self._auto_select_backend()

        key = self._get_session_key(config)

        with self._lock:
            # Check existing session
            if key in self._sessions:
                logger.debug(f"Reusing cached session: {key}")
                return self._sessions[key]

            # Clear old sessions if needed
            if len(self._sessions) >= config.max_cached_sessions:
                self._clear_oldest_session()

            # Create new session
            session = self._create_session(config)
            self._sessions[key] = session

            return session

    def _create_session(self, config: SessionConfig) -> Any:
        """Create a new model session.

        Args:
            config: Session configuration.

        Returns:
            New model session.
        """
        logger.info(f"Creating session: backend={config.backend.value}")

        if config.backend == ModelBackend.PYTORCH:
            model = OptimizedPyTorchModel(config, self._gpu_monitor)
            model.load()
            return model

        elif config.backend in (
            ModelBackend.ONNX_CUDA,
            ModelBackend.ONNX_TENSORRT,
            ModelBackend.ONNX_CPU,
        ):
            if config.onnx_model_path is None:
                raise ValueError("ONNX model path required for ONNX backend")

            from src.models.rvm_optimized import OptimizedONNXConfig, RVMOptimizedModel

            onnx_config = OptimizedONNXConfig(
                model_path=config.onnx_model_path,
                use_tensorrt=config.backend == ModelBackend.ONNX_TENSORRT,
                use_cuda=config.backend in (
                    ModelBackend.ONNX_CUDA,
                    ModelBackend.ONNX_TENSORRT,
                ),
                use_fp16=config.use_fp16,
            )
            model = RVMOptimizedModel(onnx_config)
            model.load()
            return model

        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

    def _clear_oldest_session(self) -> None:
        """Clear the oldest cached session."""
        if not self._sessions:
            return

        # Remove first (oldest) session
        key = next(iter(self._sessions))
        session = self._sessions.pop(key)

        if hasattr(session, "unload"):
            session.unload()

        logger.debug(f"Cleared session: {key}")

    def clear_all_sessions(self) -> None:
        """Clear all cached sessions."""
        with self._lock:
            for key, session in list(self._sessions.items()):
                if hasattr(session, "unload"):
                    session.unload()
            self._sessions.clear()

        logger.info("All sessions cleared")

    def get_memory_stats(self) -> dict:
        """Get memory statistics.

        Returns:
            Memory statistics dictionary.
        """
        stats = {
            "cached_sessions": len(self._sessions),
            "gpu_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            stats["gpu_memory_reserved"] = torch.cuda.memory_reserved()

        return stats


# Global session manager instance
session_manager = ModelSessionManager()


def get_optimized_model(
    use_fp16: bool = True,
    backend: ModelBackend | None = None,
    onnx_path: str | Path | None = None,
) -> Any:
    """Convenience function to get an optimized model.

    Args:
        use_fp16: Enable FP16 inference.
        backend: Specific backend to use (auto-select if None).
        onnx_path: Path to ONNX model.

    Returns:
        Optimized model instance.
    """
    config = SessionConfig(
        use_fp16=use_fp16,
        backend=backend or ModelBackend.PYTORCH,
        onnx_model_path=onnx_path,
    )

    return session_manager.get_session(config, auto_select_backend=backend is None)
