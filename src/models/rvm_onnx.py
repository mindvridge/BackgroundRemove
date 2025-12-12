"""ONNX Runtime inference wrapper for RVM model.

This module provides ONNX Runtime-based inference for the RVM model,
supporting both CPU and GPU execution providers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


@dataclass
class ONNXConfig:
    """Configuration for ONNX Runtime inference."""

    model_path: str | Path
    max_downsample_size: int = 512
    fp16: bool = False
    use_gpu: bool | None = None  # Auto-detect if None
    gpu_device_id: int = 0

    # Session options
    intra_op_num_threads: int = 0  # 0 = auto
    inter_op_num_threads: int = 0  # 0 = auto


@dataclass
class ONNXRecurrentState:
    """Container for RVM ONNX recurrent states."""

    r1: ndarray | None = None
    r2: ndarray | None = None
    r3: ndarray | None = None
    r4: ndarray | None = None

    def reset(self) -> None:
        """Reset all recurrent states."""
        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.r4 = None

    def is_initialized(self) -> bool:
        """Check if states are initialized."""
        return all(s is not None for s in [self.r1, self.r2, self.r3, self.r4])


class RVMOnnxModel:
    """ONNX Runtime wrapper for RVM model.

    Provides efficient inference using ONNX Runtime with automatic
    GPU/CPU provider selection.

    Example:
        >>> model = RVMOnnxModel(ONNXConfig(model_path="rvm.onnx"))
        >>> model.load()
        >>> fgr, pha = model.inference(frame)
    """

    def __init__(self, config: ONNXConfig) -> None:
        """Initialize ONNX model wrapper.

        Args:
            config: ONNX inference configuration.
        """
        self.config = config
        self.session = None
        self.recurrent_state = ONNXRecurrentState()
        self._loaded = False
        self._provider: str = ""
        self._dtype = np.float32

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def provider(self) -> str:
        """Get current execution provider."""
        return self._provider

    def _detect_providers(self) -> list[str]:
        """Detect available execution providers.

        Returns:
            List of execution providers to use.
        """
        import onnxruntime as ort

        available = ort.get_available_providers()
        logger.debug(f"Available ONNX providers: {available}")

        providers = []

        # Check GPU preference
        use_gpu = self.config.use_gpu
        if use_gpu is None:
            # Auto-detect: prefer GPU if available
            use_gpu = "CUDAExecutionProvider" in available

        if use_gpu and "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self.config.gpu_device_id,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    },
                )
            )
            logger.info("Using CUDA execution provider")
        elif use_gpu and "CUDAExecutionProvider" not in available:
            logger.warning("GPU requested but CUDA provider not available")

        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")

        return providers

    def _create_session_options(self):
        """Create ONNX Runtime session options.

        Returns:
            SessionOptions instance.
        """
        import onnxruntime as ort

        options = ort.SessionOptions()

        # Thread settings
        if self.config.intra_op_num_threads > 0:
            options.intra_op_num_threads = self.config.intra_op_num_threads
        if self.config.inter_op_num_threads > 0:
            options.inter_op_num_threads = self.config.inter_op_num_threads

        # Optimization
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return options

    def load(self) -> None:
        """Load the ONNX model.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        import onnxruntime as ort

        if self._loaded:
            logger.warning("Model already loaded. Skipping.")
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        try:
            logger.info(f"Loading ONNX model: {model_path}")

            providers = self._detect_providers()
            options = self._create_session_options()

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=providers,
            )

            # Get actual provider being used
            self._provider = self.session.get_providers()[0]
            logger.info(f"ONNX session created with provider: {self._provider}")

            # Set dtype based on FP16 setting
            if self.config.fp16:
                self._dtype = np.float16
                logger.info("Using FP16 inference")
            else:
                self._dtype = np.float32

            self._loaded = True
            logger.info("ONNX model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def _calculate_downsample_ratio(
        self,
        height: int,
        width: int,
    ) -> float:
        """Calculate optimal downsample ratio.

        Args:
            height: Input frame height.
            width: Input frame width.

        Returns:
            Downsample ratio between 0 and 1.
        """
        max_size = self.config.max_downsample_size
        smaller_dim = min(height, width)

        if smaller_dim <= max_size:
            ratio = 1.0
        else:
            ratio = max_size / smaller_dim

        return max(0.1, min(1.0, ratio))

    def _initialize_recurrent_state(
        self,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        """Initialize recurrent states for given input dimensions.

        Args:
            batch_size: Batch size.
            height: Input height.
            width: Input width.
        """
        dtype = self._dtype

        self.recurrent_state.r1 = np.zeros(
            (batch_size, 16, height // 2, width // 2), dtype=dtype
        )
        self.recurrent_state.r2 = np.zeros(
            (batch_size, 20, height // 4, width // 4), dtype=dtype
        )
        self.recurrent_state.r3 = np.zeros(
            (batch_size, 40, height // 8, width // 8), dtype=dtype
        )
        self.recurrent_state.r4 = np.zeros(
            (batch_size, 64, height // 16, width // 16), dtype=dtype
        )

        logger.debug(f"Initialized recurrent states for {width}x{height}")

    def reset_state(self) -> None:
        """Reset recurrent states for new video sequence."""
        self.recurrent_state.reset()
        logger.debug("Recurrent states reset")

    def preprocess(self, frame: ndarray) -> ndarray:
        """Preprocess numpy frame for model input.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format.

        Returns:
            Preprocessed array (1, C, H, W) normalized to [0, 1].
        """
        import cv2

        # BGR to RGB
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # HWC to NCHW and normalize
        arr = frame.transpose(2, 0, 1)[np.newaxis, ...].astype(self._dtype) / 255.0

        return arr

    def postprocess(
        self,
        fgr: ndarray,
        pha: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Postprocess model output to numpy arrays.

        Args:
            fgr: Foreground array (1, C, H, W).
            pha: Alpha matte array (1, 1, H, W).

        Returns:
            Tuple of (foreground, alpha) as numpy arrays (H, W, C) and (H, W).
        """
        # NCHW to HWC
        fgr_np = (fgr[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        pha_np = (pha[0, 0] * 255).astype(np.uint8)

        return fgr_np, pha_np

    def inference(
        self,
        frame: ndarray,
        downsample_ratio: float | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Run inference on a single frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format.
            downsample_ratio: Manual downsample ratio. Auto-calculated if None.

        Returns:
            Tuple of (foreground, alpha_matte) as numpy arrays.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded or self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h, w = frame.shape[:2]

        # Initialize recurrent state if needed
        if not self.recurrent_state.is_initialized():
            self._initialize_recurrent_state(1, h, w)

        # Preprocess
        src = self.preprocess(frame)

        # Calculate downsample ratio
        if downsample_ratio is None:
            downsample_ratio = self._calculate_downsample_ratio(h, w)

        dsr = np.array([downsample_ratio], dtype=self._dtype)

        # Prepare inputs
        inputs = {
            "src": src,
            "r1i": self.recurrent_state.r1,
            "r2i": self.recurrent_state.r2,
            "r3i": self.recurrent_state.r3,
            "r4i": self.recurrent_state.r4,
            "downsample_ratio": dsr,
        }

        # Run inference
        outputs = self.session.run(None, inputs)
        fgr, pha, r1o, r2o, r3o, r4o = outputs

        # Update recurrent states
        self.recurrent_state.r1 = r1o
        self.recurrent_state.r2 = r2o
        self.recurrent_state.r3 = r3o
        self.recurrent_state.r4 = r4o

        return self.postprocess(fgr, pha)

    def inference_batch(
        self,
        frames: list[ndarray],
        downsample_ratio: float | None = None,
    ) -> list[tuple[ndarray, ndarray]]:
        """Run inference on multiple frames sequentially.

        Args:
            frames: List of input frames.
            downsample_ratio: Manual downsample ratio for all frames.

        Returns:
            List of (foreground, alpha_matte) tuples.
        """
        results = []
        for frame in frames:
            result = self.inference(frame, downsample_ratio)
            results.append(result)
        return results

    def get_input_info(self) -> dict:
        """Get information about model inputs.

        Returns:
            Dictionary with input names and shapes.
        """
        if not self._loaded or self.session is None:
            raise RuntimeError("Model not loaded")

        info = {}
        for inp in self.session.get_inputs():
            info[inp.name] = {
                "shape": inp.shape,
                "type": inp.type,
            }
        return info

    def get_output_info(self) -> dict:
        """Get information about model outputs.

        Returns:
            Dictionary with output names and shapes.
        """
        if not self._loaded or self.session is None:
            raise RuntimeError("Model not loaded")

        info = {}
        for out in self.session.get_outputs():
            info[out.name] = {
                "shape": out.shape,
                "type": out.type,
            }
        return info

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.session is not None:
            del self.session
            self.session = None

        self.recurrent_state.reset()
        self._loaded = False
        self._provider = ""

        logger.info("ONNX model unloaded")

    def __enter__(self) -> "RVMOnnxModel":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload()

    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self._loaded else "not loaded"
        provider = self._provider if self._provider else "N/A"
        return (
            f"RVMOnnxModel(model_path={self.config.model_path!r}, "
            f"status={status}, provider={provider})"
        )
