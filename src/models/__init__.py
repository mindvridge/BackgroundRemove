"""Background removal models module.

This module provides model implementations for video background removal:
- RVMModel: PyTorch-based RVM (Robust Video Matting) model
- RVMOnnxModel: ONNX Runtime-based RVM inference
"""

from src.models.base import BaseModel
from src.models.rvm import RecurrentState, RVMConfig, RVMModel
from src.models.rvm_onnx import ONNXConfig, ONNXRecurrentState, RVMOnnxModel

__all__ = [
    "BaseModel",
    "RVMModel",
    "RVMConfig",
    "RecurrentState",
    "RVMOnnxModel",
    "ONNXConfig",
    "ONNXRecurrentState",
]
