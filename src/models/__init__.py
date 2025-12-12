"""Background removal models module.

This module provides model implementations for video background removal:
- RVMModel: PyTorch-based RVM (Robust Video Matting) model
- RVMOnnxModel: ONNX Runtime-based RVM inference
- RVMOptimizedModel: TensorRT/CUDA optimized ONNX inference
- OptimizedPyTorchModel: FP16 + autocast optimized PyTorch model
- ModelSessionManager: Centralized session management
"""

from src.models.base import BaseModel
from src.models.rvm import RecurrentState, RVMConfig, RVMModel
from src.models.rvm_onnx import ONNXConfig, ONNXRecurrentState, RVMOnnxModel
from src.models.rvm_optimized import (
    OptimizedONNXConfig,
    OptimizedRecurrentState,
    RVMOptimizedModel,
)
from src.models.session_manager import (
    FP16InferenceContext,
    ModelBackend,
    ModelSessionManager,
    OptimizedPyTorchModel,
    SessionConfig,
    get_optimized_model,
    session_manager,
)

__all__ = [
    # Base
    "BaseModel",
    # PyTorch RVM
    "RVMModel",
    "RVMConfig",
    "RecurrentState",
    # ONNX RVM
    "RVMOnnxModel",
    "ONNXConfig",
    "ONNXRecurrentState",
    # Optimized ONNX
    "RVMOptimizedModel",
    "OptimizedONNXConfig",
    "OptimizedRecurrentState",
    # Optimized PyTorch
    "OptimizedPyTorchModel",
    "FP16InferenceContext",
    # Session management
    "ModelSessionManager",
    "SessionConfig",
    "ModelBackend",
    "session_manager",
    "get_optimized_model",
]
