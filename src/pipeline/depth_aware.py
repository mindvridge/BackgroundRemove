"""Depth-aware processing for video background removal.

Uses depth estimation to improve foreground/background separation
and handle complex scenes with multiple depth layers.

Models supported:
- MiDaS: Robust depth estimation
- ZoeDepth: Metric depth estimation
- Depth Anything: Latest depth foundation model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class DepthModel(Enum):
    """Available depth estimation models."""

    MIDAS = "midas"
    MIDAS_SMALL = "midas_small"
    ZOEDEPTH = "zoedepth"
    DEPTH_ANYTHING = "depth_anything"


@dataclass
class DepthConfig:
    """Configuration for depth estimation."""

    model: DepthModel = DepthModel.MIDAS
    device: str = "cuda"

    # Processing settings
    process_scale: float = 1.0  # Scale for depth processing

    # Depth-based separation
    foreground_depth_threshold: float = 0.3  # Relative depth threshold
    depth_blur_size: int = 5  # Blur kernel for depth smoothing

    # Edge refinement
    use_depth_edges: bool = True  # Use depth edges for refinement
    edge_weight: float = 0.5  # Weight for depth edge contribution


class DepthEstimator:
    """Depth estimation using various models.

    Provides monocular depth estimation for single images
    using pre-trained deep learning models.

    Example:
        >>> estimator = DepthEstimator(DepthConfig(model=DepthModel.MIDAS))
        >>> estimator.load()
        >>> depth = estimator.estimate(image)
    """

    def __init__(self, config: DepthConfig | None = None) -> None:
        """Initialize depth estimator.

        Args:
            config: Depth estimation configuration.
        """
        self.config = config or DepthConfig()
        self._model = None
        self._transform = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load depth estimation model."""
        if self._is_loaded:
            return

        model_type = self.config.model

        try:
            if model_type in [DepthModel.MIDAS, DepthModel.MIDAS_SMALL]:
                self._load_midas()
            elif model_type == DepthModel.ZOEDEPTH:
                self._load_zoedepth()
            elif model_type == DepthModel.DEPTH_ANYTHING:
                self._load_depth_anything()
            else:
                self._load_midas()  # Default fallback

            self._is_loaded = True
            logger.info(f"Depth model loaded: {model_type.value}")

        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            # Create fallback
            self._create_fallback()
            self._is_loaded = True

    def _load_midas(self) -> None:
        """Load MiDaS depth model."""
        import torch

        self._torch = torch

        # Try loading from torch hub
        model_type = "DPT_Large" if self.config.model == DepthModel.MIDAS else "MiDaS_small"

        try:
            self._model = torch.hub.load("intel-isl/MiDaS", model_type)
            self._model = self._model.to(self.config.device)
            self._model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "DPT_Large":
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            self._use_torch = True

        except Exception as e:
            logger.warning(f"Could not load MiDaS from hub: {e}")
            self._create_fallback()

    def _load_zoedepth(self) -> None:
        """Load ZoeDepth model."""
        import torch

        self._torch = torch

        try:
            # Load ZoeDepth from torch hub
            self._model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_N",
                pretrained=True,
            )
            self._model = self._model.to(self.config.device)
            self._model.eval()
            self._use_torch = True

        except Exception as e:
            logger.warning(f"Could not load ZoeDepth: {e}, using MiDaS fallback")
            self._load_midas()

    def _load_depth_anything(self) -> None:
        """Load Depth Anything model."""
        import torch

        self._torch = torch

        try:
            # Try loading from transformers
            from transformers import pipeline

            self._model = pipeline(
                "depth-estimation",
                model="LiheYoung/depth-anything-base-hf",
                device=0 if self.config.device == "cuda" else -1,
            )
            self._use_torch = False
            self._use_pipeline = True

        except Exception as e:
            logger.warning(f"Could not load Depth Anything: {e}")
            self._load_midas()

    def _create_fallback(self) -> None:
        """Create fallback depth estimation."""
        logger.info("Using fallback depth estimation (blur-based)")
        self._use_torch = False
        self._use_fallback = True

    def _fallback_depth(self, image: ndarray) -> ndarray:
        """Fallback depth estimation using simple heuristics.

        Args:
            image: Input image.

        Returns:
            Estimated depth map.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use blur as depth proxy (sharper = closer)
        blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
        blur2 = cv2.GaussianBlur(gray, (15, 15), 0)

        # Sharpness = difference between original and blur
        sharpness = np.abs(gray.astype(np.float32) - blur1.astype(np.float32))

        # Normalize
        depth = cv2.normalize(sharpness, None, 0, 255, cv2.NORM_MINMAX)

        # Smooth
        depth = cv2.GaussianBlur(depth.astype(np.uint8), (21, 21), 0)

        return depth.astype(np.float32) / 255.0

    def estimate(self, image: ndarray) -> ndarray:
        """Estimate depth from image.

        Args:
            image: Input image (BGR).

        Returns:
            Depth map (0-1, closer=lower values).
        """
        if not self._is_loaded:
            self.load()

        # Scale if needed
        scale = self.config.process_scale
        if scale != 1.0:
            h, w = image.shape[:2]
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        if hasattr(self, '_use_fallback') and self._use_fallback:
            depth = self._fallback_depth(image)
        elif hasattr(self, '_use_pipeline') and self._use_pipeline:
            depth = self._estimate_pipeline(image)
        elif hasattr(self, '_use_torch') and self._use_torch:
            depth = self._estimate_torch(image)
        else:
            depth = self._fallback_depth(image)

        # Restore size if scaled
        if scale != 1.0:
            depth = cv2.resize(depth, (w, h))

        return depth

    def _estimate_torch(self, image: ndarray) -> ndarray:
        """Estimate depth using PyTorch model.

        Args:
            image: Input image.

        Returns:
            Depth map.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform
        if self._transform is not None:
            input_tensor = self._transform(image_rgb)
        else:
            # Manual transform
            input_tensor = self._torch.from_numpy(image_rgb).permute(2, 0, 1).float()
            input_tensor = input_tensor.unsqueeze(0) / 255.0

        input_tensor = input_tensor.to(self.config.device)

        # Inference
        with self._torch.no_grad():
            prediction = self._model(input_tensor)

            if isinstance(prediction, dict):
                prediction = prediction.get('out', prediction.get('depth', prediction))

            if hasattr(prediction, 'squeeze'):
                prediction = prediction.squeeze().cpu().numpy()

        # Normalize to 0-1
        depth = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)

        # Resize to original
        if depth.shape != image.shape[:2]:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        return depth.astype(np.float32)

    def _estimate_pipeline(self, image: ndarray) -> ndarray:
        """Estimate depth using HuggingFace pipeline.

        Args:
            image: Input image.

        Returns:
            Depth map.
        """
        from PIL import Image

        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Predict
        result = self._model(pil_image)
        depth_pil = result['depth']

        # Convert to numpy
        depth = np.array(depth_pil)

        # Normalize
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Resize if needed
        if depth.shape != image.shape[:2]:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        return depth

    def unload(self) -> None:
        """Unload model."""
        self._model = None
        self._transform = None
        self._is_loaded = False


class DepthAwareProcessor:
    """Depth-aware matting processor.

    Uses depth information to improve foreground/background
    separation and handle complex scenes.

    Example:
        >>> processor = DepthAwareProcessor(DepthConfig())
        >>> improved_alpha = processor.refine(image, alpha)
    """

    def __init__(self, config: DepthConfig | None = None) -> None:
        """Initialize depth-aware processor.

        Args:
            config: Depth configuration.
        """
        self.config = config or DepthConfig()
        self._depth_estimator = DepthEstimator(config)

    def load(self) -> None:
        """Load depth model."""
        self._depth_estimator.load()

    @property
    def is_loaded(self) -> bool:
        """Check if loaded."""
        return self._depth_estimator.is_loaded

    def estimate_depth(self, image: ndarray) -> ndarray:
        """Estimate depth for image.

        Args:
            image: Input image.

        Returns:
            Depth map.
        """
        return self._depth_estimator.estimate(image)

    def separate_by_depth(
        self,
        image: ndarray,
        depth: ndarray | None = None,
        threshold: float | None = None,
    ) -> ndarray:
        """Separate foreground by depth threshold.

        Args:
            image: Input image.
            depth: Optional pre-computed depth map.
            threshold: Depth threshold (auto if None).

        Returns:
            Foreground mask based on depth.
        """
        if depth is None:
            depth = self.estimate_depth(image)

        if threshold is None:
            threshold = self.config.foreground_depth_threshold

        # Create mask (closer objects = foreground)
        # Note: MiDaS outputs inverse depth (closer = higher values)
        mask = (depth > threshold).astype(np.float32)

        # Smooth
        if self.config.depth_blur_size > 0:
            ksize = self.config.depth_blur_size
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return (mask * 255).astype(np.uint8)

    def get_depth_edges(self, depth: ndarray) -> ndarray:
        """Extract edges from depth map.

        Args:
            depth: Depth map.

        Returns:
            Depth edge map.
        """
        # Convert to uint8
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Compute gradient
        grad_x = cv2.Sobel(depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        edges = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

        return edges.astype(np.uint8)

    def refine_alpha_with_depth(
        self,
        image: ndarray,
        alpha: ndarray,
        depth: ndarray | None = None,
    ) -> ndarray:
        """Refine alpha matte using depth information.

        Args:
            image: Input image.
            alpha: Initial alpha matte.
            depth: Optional pre-computed depth map.

        Returns:
            Refined alpha matte.
        """
        if depth is None:
            depth = self.estimate_depth(image)

        # Get depth-based mask
        depth_mask = self.separate_by_depth(image, depth)
        depth_mask_float = depth_mask.astype(np.float32) / 255.0

        # Get depth edges
        if self.config.use_depth_edges:
            depth_edges = self.get_depth_edges(depth)
            depth_edges = cv2.dilate(depth_edges, None, iterations=2)
            edge_mask = depth_edges.astype(np.float32) / 255.0
        else:
            edge_mask = np.zeros_like(depth)

        # Combine alpha with depth
        alpha_float = alpha.astype(np.float32) / 255.0

        # In edge regions, blend with depth mask
        edge_weight = self.config.edge_weight
        refined = (
            edge_mask * (edge_weight * depth_mask_float + (1 - edge_weight) * alpha_float) +
            (1 - edge_mask) * alpha_float
        )

        # Ensure consistency with depth
        # If depth says background but alpha says foreground, reduce alpha
        depth_bg = (depth < 0.2).astype(np.float32)
        alpha_high = (alpha_float > 0.8).astype(np.float32)
        inconsistent = depth_bg * alpha_high

        refined = refined * (1 - inconsistent * 0.5)

        return (np.clip(refined, 0, 1) * 255).astype(np.uint8)

    def segment_layers(
        self,
        image: ndarray,
        depth: ndarray | None = None,
        num_layers: int = 3,
    ) -> list[ndarray]:
        """Segment image into depth layers.

        Args:
            image: Input image.
            depth: Optional depth map.
            num_layers: Number of depth layers.

        Returns:
            List of masks for each layer (front to back).
        """
        if depth is None:
            depth = self.estimate_depth(image)

        layers = []
        thresholds = np.linspace(0, 1, num_layers + 1)

        for i in range(num_layers):
            low = thresholds[i]
            high = thresholds[i + 1]

            mask = ((depth >= low) & (depth < high)).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            layers.append((mask * 255).astype(np.uint8))

        return layers

    def refine(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Refine alpha (alias for refine_alpha_with_depth).

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Refined alpha.
        """
        return self.refine_alpha_with_depth(image, alpha)

    def __repr__(self) -> str:
        """String representation."""
        return f"DepthAwareProcessor(model={self.config.model.value})"


def estimate_depth(
    image: ndarray,
    model: DepthModel = DepthModel.MIDAS,
) -> ndarray:
    """Convenience function for depth estimation.

    Args:
        image: Input image.
        model: Depth model to use.

    Returns:
        Depth map.
    """
    config = DepthConfig(model=model)
    estimator = DepthEstimator(config)
    estimator.load()
    return estimator.estimate(image)


def refine_with_depth(
    image: ndarray,
    alpha: ndarray,
    model: DepthModel = DepthModel.MIDAS,
) -> ndarray:
    """Convenience function for depth-aware refinement.

    Args:
        image: Input image.
        alpha: Alpha matte.
        model: Depth model.

    Returns:
        Refined alpha.
    """
    config = DepthConfig(model=model)
    processor = DepthAwareProcessor(config)
    processor.load()
    return processor.refine(image, alpha)
