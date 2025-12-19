"""RVM (Robust Video Matting) model wrapper.

This module provides a wrapper for the RVM model with support for:
- TorchHub model loading (MobileNetV3 backbone)
- Automatic GPU/CPU detection
- Automatic downsample ratio calculation
- FP16 inference support
- ONNX export functionality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


@dataclass
class RVMConfig:
    """Configuration for RVM model."""

    variant: str = "mobilenetv3"
    max_downsample_size: int = 512
    fp16: bool = False
    device: str | None = None  # Auto-detect if None
    hub_repo: str = "PeterL1n/RobustVideoMatting"

    # Recurrent state settings
    use_recurrent: bool = True

    # ONNX export settings
    onnx_opset_version: int = 14
    onnx_dynamic_axes: bool = True


@dataclass
class RecurrentState:
    """Container for RVM recurrent states."""

    r1: torch.Tensor | None = None
    r2: torch.Tensor | None = None
    r3: torch.Tensor | None = None
    r4: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset all recurrent states."""
        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.r4 = None

    def as_tuple(self) -> tuple[torch.Tensor | None, ...]:
        """Return states as tuple for model input."""
        return (self.r1, self.r2, self.r3, self.r4)

    def update(
        self,
        r1: torch.Tensor,
        r2: torch.Tensor,
        r3: torch.Tensor,
        r4: torch.Tensor,
    ) -> None:
        """Update recurrent states."""
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4


class RVMModel:
    """Wrapper for Robust Video Matting model.

    Provides easy-to-use interface for video matting with automatic
    device detection, downsample ratio calculation, and FP16 support.

    Example:
        >>> model = RVMModel()
        >>> model.load()
        >>> fgr, pha = model.inference(frame)
    """

    def __init__(self, config: RVMConfig | None = None) -> None:
        """Initialize RVM model wrapper.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or RVMConfig()
        self.model: nn.Module | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype = torch.float32
        self.recurrent_state = RecurrentState()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _detect_device(self) -> torch.device:
        """Auto-detect the best available device.

        Returns:
            torch.device: CUDA if available, else CPU.
        """
        if self.config.device is not None:
            device = torch.device(self.config.device)
            logger.info(f"Using specified device: {device}")
            return device

        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available. Using GPU: {gpu_name}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU")

        return device

    def _calculate_downsample_ratio(
        self,
        height: int,
        width: int,
    ) -> float:
        """Calculate optimal downsample ratio based on input size.

        The ratio is calculated to ensure the smaller dimension
        does not exceed max_downsample_size (default 512).

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

        # Clamp ratio to reasonable bounds
        ratio = max(0.1, min(1.0, ratio))

        logger.debug(
            f"Input: {width}x{height}, "
            f"max_size: {max_size}, "
            f"downsample_ratio: {ratio:.4f}"
        )
        return ratio

    def _patch_model_for_pytorch2(self) -> None:
        """Patch model for PyTorch 2.x compatibility.

        PyTorch 2.x doesn't accept tensors as scale_factor in F.interpolate.
        This patches the model's _interpolate method to convert tensors to floats.
        """
        import types
        import torch.nn.functional as F

        original_interpolate = self.model._interpolate

        def patched_interpolate(self, x, scale_factor):
            # Convert tensor to float if needed
            if isinstance(scale_factor, torch.Tensor):
                scale_factor = scale_factor.item()
            return F.interpolate(
                x,
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False,
                recompute_scale_factor=False,
            )

        # Bind the patched method to the model
        self.model._interpolate = types.MethodType(patched_interpolate, self.model)
        logger.debug("Patched model for PyTorch 2.x compatibility")

    def load(self) -> None:
        """Load the RVM model from TorchHub.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._loaded:
            logger.warning("Model already loaded. Skipping.")
            return

        try:
            logger.info(f"Loading RVM model ({self.config.variant}) from TorchHub...")

            # Detect device
            self.device = self._detect_device()

            # Set dtype based on FP16 setting and device
            if self.config.fp16 and self.device.type == "cuda":
                self.dtype = torch.float16
                logger.info("FP16 inference enabled")
            else:
                self.dtype = torch.float32
                if self.config.fp16 and self.device.type != "cuda":
                    logger.warning("FP16 requested but not on CUDA. Using FP32.")

            # Load model from TorchHub
            self.model = torch.hub.load(
                self.config.hub_repo,
                self.config.variant,
                trust_repo=True,
            )

            # Patch _interpolate for PyTorch 2.x compatibility
            self._patch_model_for_pytorch2()

            # Move to device and set dtype
            self.model = self.model.to(device=self.device, dtype=self.dtype)
            self.model.eval()

            self._loaded = True
            logger.info("RVM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load RVM model: {e}")
            raise RuntimeError(f"Failed to load RVM model: {e}") from e

    def reset_state(self) -> None:
        """Reset recurrent states for new video sequence."""
        self.recurrent_state.reset()
        logger.debug("Recurrent states reset")

    def preprocess(self, frame: ndarray) -> torch.Tensor:
        """Preprocess numpy frame for model input.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format.

        Returns:
            Preprocessed tensor (1, C, H, W) normalized to [0, 1].
        """
        import cv2
        import numpy as np

        # BGR to RGB
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # HWC to CHW and add batch dimension
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)

        # Normalize to [0, 1] and convert dtype
        tensor = tensor.to(device=self.device, dtype=self.dtype) / 255.0

        return tensor

    def postprocess(
        self,
        fgr: torch.Tensor,
        pha: torch.Tensor,
    ) -> tuple[ndarray, ndarray]:
        """Postprocess model output to numpy arrays.

        Args:
            fgr: Foreground tensor (1, C, H, W).
            pha: Alpha matte tensor (1, 1, H, W).

        Returns:
            Tuple of (foreground, alpha) as numpy arrays.
        """
        import numpy as np

        # Convert to numpy
        fgr_np = (fgr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pha_np = (pha[0, 0].cpu().numpy() * 255).astype(np.uint8)

        return fgr_np, pha_np

    @torch.inference_mode()
    def inference(
        self,
        frame: ndarray,
        downsample_ratio: float | None = None,
        return_tensor: bool = False,
    ) -> tuple[ndarray | torch.Tensor, ndarray | torch.Tensor]:
        """Run inference on a single frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format.
            downsample_ratio: Manual downsample ratio. Auto-calculated if None.
            return_tensor: If True, return tensors instead of numpy arrays.

        Returns:
            Tuple of (foreground, alpha_matte).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Preprocess
        src = self.preprocess(frame)
        h, w = frame.shape[:2]

        # Calculate downsample ratio if not provided
        if downsample_ratio is None:
            downsample_ratio = self._calculate_downsample_ratio(h, w)

        # Prepare downsample ratio tensor
        dsr = torch.tensor([downsample_ratio], device=self.device, dtype=self.dtype)

        # Get recurrent states
        rec = self.recurrent_state.as_tuple()

        # Run inference
        fgr, pha, *new_rec = self.model(src, *rec, dsr)

        # Update recurrent states if using recurrent mode
        if self.config.use_recurrent and len(new_rec) == 4:
            self.recurrent_state.update(*new_rec)

        if return_tensor:
            return fgr, pha

        return self.postprocess(fgr, pha)

    @torch.inference_mode()
    def inference_batch(
        self,
        frames: list[ndarray],
        downsample_ratio: float | None = None,
    ) -> list[tuple[ndarray, ndarray]]:
        """Run inference on multiple frames.

        Note: This processes frames sequentially to maintain recurrent state.

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

    def export_onnx(
        self,
        output_path: str | Path,
        input_shape: tuple[int, int, int, int] = (1, 3, 1080, 1920),
    ) -> Path:
        """Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model.
            input_shape: Input tensor shape (B, C, H, W).

        Returns:
            Path to exported ONNX model.

        Raises:
            RuntimeError: If model is not loaded or export fails.
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting RVM model to ONNX: {output_path}")

        try:
            # Create dummy inputs
            b, c, h, w = input_shape
            dummy_src = torch.randn(b, c, h, w, device=self.device, dtype=self.dtype)
            dummy_r1 = torch.zeros(b, 16, h // 2, w // 2, device=self.device, dtype=self.dtype)
            dummy_r2 = torch.zeros(b, 20, h // 4, w // 4, device=self.device, dtype=self.dtype)
            dummy_r3 = torch.zeros(b, 40, h // 8, w // 8, device=self.device, dtype=self.dtype)
            dummy_r4 = torch.zeros(b, 64, h // 16, w // 16, device=self.device, dtype=self.dtype)
            dummy_dsr = torch.tensor([0.25], device=self.device, dtype=self.dtype)

            # Input/output names
            input_names = ["src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"]
            output_names = ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"]

            # Dynamic axes for variable input size
            dynamic_axes = None
            if self.config.onnx_dynamic_axes:
                dynamic_axes = {
                    "src": {0: "batch", 2: "height", 3: "width"},
                    "fgr": {0: "batch", 2: "height", 3: "width"},
                    "pha": {0: "batch", 2: "height", 3: "width"},
                    "r1i": {0: "batch", 2: "h2", 3: "w2"},
                    "r2i": {0: "batch", 2: "h4", 3: "w4"},
                    "r3i": {0: "batch", 2: "h8", 3: "w8"},
                    "r4i": {0: "batch", 2: "h16", 3: "w16"},
                    "r1o": {0: "batch", 2: "h2", 3: "w2"},
                    "r2o": {0: "batch", 2: "h4", 3: "w4"},
                    "r3o": {0: "batch", 2: "h8", 3: "w8"},
                    "r4o": {0: "batch", 2: "h16", 3: "w16"},
                }

            # Export to ONNX
            torch.onnx.export(
                self.model,
                (dummy_src, dummy_r1, dummy_r2, dummy_r3, dummy_r4, dummy_dsr),
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=self.config.onnx_opset_version,
                dynamic_axes=dynamic_axes,
            )

            logger.info(f"ONNX export successful: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise RuntimeError(f"ONNX export failed: {e}") from e

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        self.recurrent_state.reset()
        self._loaded = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")

    def __enter__(self) -> "RVMModel":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload()

    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self._loaded else "not loaded"
        device = self.device if self.device else "N/A"
        dtype = self.dtype if self._loaded else "N/A"
        return (
            f"RVMModel(variant={self.config.variant!r}, "
            f"status={status}, device={device}, dtype={dtype})"
        )
