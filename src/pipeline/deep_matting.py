"""Deep learning matting models for high-quality alpha matte generation.

Provides integration with state-of-the-art matting models:
- MODNet: Real-time matting with portrait optimization
- ViTMatte: Vision Transformer based matting
- Matte Anything: SAM-based universal matting

These models provide significantly better edge quality than
traditional segmentation approaches.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class MattingModel(Enum):
    """Available matting models."""

    MODNET = "modnet"
    VITMATTE = "vitmatte"
    MATTE_ANYTHING = "matte_anything"


@dataclass
class MattingConfig:
    """Configuration for matting models."""

    model: MattingModel = MattingModel.MODNET
    device: str = "cuda"  # "cuda" or "cpu"

    # MODNet settings
    modnet_backbone: str = "mobilenetv2"  # "mobilenetv2", "resnet50", "hrnet"

    # ViTMatte settings
    vitmatte_backbone: str = "vit_b"  # "vit_b", "vit_l"

    # Processing settings
    input_size: int = 512  # Input resolution for inference
    use_trimap: bool = False  # Whether to use trimap guidance
    refine_foreground: bool = True  # Refine foreground colors


class BaseMattingModel(ABC):
    """Base class for matting models."""

    def __init__(self, config: MattingConfig) -> None:
        """Initialize matting model.

        Args:
            config: Matting configuration.
        """
        self.config = config
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def predict(
        self,
        image: ndarray,
        trimap: ndarray | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Predict alpha matte.

        Args:
            image: Input image (BGR).
            trimap: Optional trimap guidance.

        Returns:
            Tuple of (foreground, alpha).
        """
        pass

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._is_loaded = False


class MODNetMatting(BaseMattingModel):
    """MODNet matting model.

    MODNet is optimized for real-time portrait matting with
    excellent performance on hair and transparent objects.

    Reference: https://github.com/ZHKKKe/MODNet

    Example:
        >>> modnet = MODNetMatting(MattingConfig())
        >>> modnet.load()
        >>> fg, alpha = modnet.predict(image)
    """

    MODEL_URLS = {
        "mobilenetv2": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.ckpt",
        "hrnet": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_webcam_portrait_matting.ckpt",
    }

    def __init__(self, config: MattingConfig | None = None) -> None:
        """Initialize MODNet.

        Args:
            config: Matting configuration.
        """
        super().__init__(config or MattingConfig(model=MattingModel.MODNET))
        self._ref_size = 512

    def load(self) -> None:
        """Load MODNet model."""
        if self._is_loaded:
            return

        try:
            import torch
            import torch.nn.functional as F

            self._torch = torch
            self._F = F

            # Try to import MODNet
            try:
                from modnet.models.modnet import MODNet

                self._model = MODNet(backbone_pretrained=False)

                # Load weights
                backbone = self.config.modnet_backbone
                if backbone in self.MODEL_URLS:
                    # Download and load pretrained weights
                    weights_path = self._download_weights(self.MODEL_URLS[backbone])
                    state_dict = torch.load(weights_path, map_location=self.config.device)
                    self._model.load_state_dict(state_dict)

                self._model = self._model.to(self.config.device)
                self._model.eval()

            except ImportError:
                # Fallback: Create a simple MODNet-like model using ONNX
                logger.warning("MODNet not installed, using ONNX fallback")
                self._use_onnx_fallback()

            self._is_loaded = True
            logger.info("MODNet loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MODNet: {e}")
            raise

    def _use_onnx_fallback(self) -> None:
        """Use ONNX runtime as fallback."""
        try:
            import onnxruntime as ort

            # Check for existing ONNX model
            model_path = Path(__file__).parent / "weights" / "modnet.onnx"
            if model_path.exists():
                self._onnx_session = ort.InferenceSession(
                    str(model_path),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                self._use_onnx = True
            else:
                logger.warning("MODNet ONNX model not found, matting may be degraded")
                self._use_onnx = False

        except ImportError:
            self._use_onnx = False

    def _download_weights(self, url: str) -> Path:
        """Download model weights.

        Args:
            url: URL to download from.

        Returns:
            Path to downloaded weights.
        """
        import urllib.request

        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(exist_ok=True)

        filename = url.split("/")[-1]
        weights_path = weights_dir / filename

        if not weights_path.exists():
            logger.info(f"Downloading MODNet weights from {url}")
            urllib.request.urlretrieve(url, weights_path)

        return weights_path

    def _preprocess(self, image: ndarray) -> "torch.Tensor":
        """Preprocess image for MODNet.

        Args:
            image: Input BGR image.

        Returns:
            Preprocessed tensor.
        """
        import torch

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to reference size
        h, w = image.shape[:2]
        if max(h, w) > self._ref_size:
            if h > w:
                new_h = self._ref_size
                new_w = int(w * self._ref_size / h)
            else:
                new_w = self._ref_size
                new_h = int(h * self._ref_size / w)
            image = cv2.resize(image, (new_w, new_h))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        # To tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.config.device)

        return tensor

    def predict(
        self,
        image: ndarray,
        trimap: ndarray | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Predict alpha matte using MODNet.

        Args:
            image: Input image (BGR).
            trimap: Optional trimap (unused for MODNet).

        Returns:
            Tuple of (foreground, alpha).
        """
        if not self._is_loaded:
            self.load()

        h, w = image.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(image)

        # Inference
        with self._torch.no_grad():
            _, _, matte = self._model(input_tensor, True)

        # Postprocess
        matte = matte.squeeze().cpu().numpy()
        matte = cv2.resize(matte, (w, h))
        alpha = (matte * 255).astype(np.uint8)

        # Generate foreground
        foreground = self._extract_foreground(image, alpha)

        return foreground, alpha

    def _extract_foreground(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Extract foreground using alpha.

        Args:
            image: Input image.
            alpha: Alpha matte.

        Returns:
            Foreground image.
        """
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]

        foreground = (image * alpha_norm).astype(np.uint8)
        return foreground


class ViTMatteMatting(BaseMattingModel):
    """ViTMatte matting model.

    Vision Transformer based matting that achieves
    state-of-the-art quality on matting benchmarks.

    Reference: https://github.com/hustvl/ViTMatte

    Example:
        >>> vitmatte = ViTMatteMatting(MattingConfig())
        >>> vitmatte.load()
        >>> fg, alpha = vitmatte.predict(image, trimap)
    """

    def __init__(self, config: MattingConfig | None = None) -> None:
        """Initialize ViTMatte.

        Args:
            config: Matting configuration.
        """
        super().__init__(config or MattingConfig(model=MattingModel.VITMATTE))

    def load(self) -> None:
        """Load ViTMatte model."""
        if self._is_loaded:
            return

        try:
            import torch

            self._torch = torch

            # Try to load from transformers/huggingface
            try:
                from transformers import VitMatteForImageMatting, VitMatteImageProcessor

                model_name = f"hustvl/vitmatte-{self.config.vitmatte_backbone}-composition-1k"

                self._processor = VitMatteImageProcessor.from_pretrained(model_name)
                self._model = VitMatteForImageMatting.from_pretrained(model_name)
                self._model = self._model.to(self.config.device)
                self._model.eval()

                self._use_hf = True
                logger.info(f"ViTMatte ({self.config.vitmatte_backbone}) loaded from HuggingFace")

            except ImportError:
                logger.warning("transformers not installed, using fallback")
                self._use_hf = False
                self._create_fallback_model()

            self._is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load ViTMatte: {e}")
            raise

    def _create_fallback_model(self) -> None:
        """Create a fallback using simpler approach."""
        # Use guided filter based approach as fallback
        from src.pipeline.edge_refine import GuidedFilter
        self._guided_filter = GuidedFilter(radius=16, eps=0.001)
        logger.info("ViTMatte fallback: using guided filter refinement")

    def _generate_trimap(self, image: ndarray) -> ndarray:
        """Generate trimap from image using edge detection.

        Args:
            image: Input image.

        Returns:
            Generated trimap.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges for unknown region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        unknown = cv2.dilate(edges, kernel, iterations=2)

        # Simple foreground estimation using GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            foreground = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        except cv2.error:
            # Fallback: use center as foreground
            foreground = np.zeros_like(gray)
            h, w = gray.shape
            foreground[h//4:3*h//4, w//4:3*w//4] = 255

        # Create trimap
        trimap = np.zeros_like(gray)
        trimap[foreground > 0] = 255
        trimap[unknown > 0] = 128

        return trimap

    def predict(
        self,
        image: ndarray,
        trimap: ndarray | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Predict alpha matte using ViTMatte.

        Args:
            image: Input image (BGR).
            trimap: Optional trimap guidance.

        Returns:
            Tuple of (foreground, alpha).
        """
        if not self._is_loaded:
            self.load()

        # Generate trimap if not provided
        if trimap is None:
            trimap = self._generate_trimap(image)

        if hasattr(self, '_use_hf') and self._use_hf:
            return self._predict_hf(image, trimap)
        else:
            return self._predict_fallback(image, trimap)

    def _predict_hf(
        self,
        image: ndarray,
        trimap: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Predict using HuggingFace model.

        Args:
            image: Input image.
            trimap: Trimap.

        Returns:
            Tuple of (foreground, alpha).
        """
        from PIL import Image

        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_trimap = Image.fromarray(trimap)

        # Process
        inputs = self._processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Inference
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            alpha = outputs.alphas

        # Postprocess
        alpha = alpha.squeeze().cpu().numpy()
        alpha = (alpha * 255).astype(np.uint8)

        # Resize if needed
        if alpha.shape[:2] != image.shape[:2]:
            alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]))

        # Extract foreground
        alpha_norm = alpha.astype(np.float32) / 255.0
        foreground = (image * alpha_norm[:, :, np.newaxis]).astype(np.uint8)

        return foreground, alpha

    def _predict_fallback(
        self,
        image: ndarray,
        trimap: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Predict using fallback method.

        Args:
            image: Input image.
            trimap: Trimap.

        Returns:
            Tuple of (foreground, alpha).
        """
        # Use guided filter to refine trimap
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = self._guided_filter.filter(trimap, gray)

        # Extract foreground
        alpha_norm = alpha.astype(np.float32) / 255.0
        foreground = (image * alpha_norm[:, :, np.newaxis]).astype(np.uint8)

        return foreground, alpha


class MatteAnythingMatting(BaseMattingModel):
    """Matte Anything model using SAM.

    Combines Segment Anything Model (SAM) with matting
    for universal object matting.

    Reference: https://github.com/SHI-Labs/Matte-Anything

    Example:
        >>> matte_any = MatteAnythingMatting(MattingConfig())
        >>> matte_any.load()
        >>> fg, alpha = matte_any.predict(image)
    """

    def __init__(self, config: MattingConfig | None = None) -> None:
        """Initialize Matte Anything.

        Args:
            config: Matting configuration.
        """
        super().__init__(config or MattingConfig(model=MattingModel.MATTE_ANYTHING))
        self._sam = None
        self._matting_model = None

    def load(self) -> None:
        """Load Matte Anything model."""
        if self._is_loaded:
            return

        try:
            import torch
            self._torch = torch

            # Try to load SAM
            try:
                from segment_anything import SamPredictor, sam_model_registry

                # Load SAM
                sam_checkpoint = self._get_sam_checkpoint()
                if sam_checkpoint and sam_checkpoint.exists():
                    sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
                    sam = sam.to(self.config.device)
                    self._sam = SamPredictor(sam)
                    logger.info("SAM loaded for Matte Anything")
                else:
                    logger.warning("SAM checkpoint not found")
                    self._sam = None

            except ImportError:
                logger.warning("segment_anything not installed")
                self._sam = None

            # Load secondary matting model (MODNet for refinement)
            self._matting_model = MODNetMatting(self.config)
            try:
                self._matting_model.load()
            except Exception:
                self._matting_model = None

            self._is_loaded = True
            logger.info("Matte Anything loaded")

        except Exception as e:
            logger.error(f"Failed to load Matte Anything: {e}")
            raise

    def _get_sam_checkpoint(self) -> Path | None:
        """Get SAM checkpoint path.

        Returns:
            Path to checkpoint or None.
        """
        weights_dir = Path(__file__).parent / "weights"
        checkpoint = weights_dir / "sam_vit_h_4b8939.pth"
        return checkpoint if checkpoint.exists() else None

    def predict(
        self,
        image: ndarray,
        trimap: ndarray | None = None,
        point_coords: ndarray | None = None,
        point_labels: ndarray | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Predict alpha matte using Matte Anything.

        Args:
            image: Input image (BGR).
            trimap: Optional trimap.
            point_coords: Optional point prompts for SAM.
            point_labels: Optional point labels (1=fg, 0=bg).

        Returns:
            Tuple of (foreground, alpha).
        """
        if not self._is_loaded:
            self.load()

        # Use SAM for initial segmentation if available
        if self._sam is not None:
            mask = self._segment_with_sam(image, point_coords, point_labels)
        else:
            # Fallback: use simple segmentation
            mask = self._simple_segment(image)

        # Refine mask with matting
        if self._matting_model is not None:
            # Use mask as guidance for matting
            foreground, alpha = self._matting_model.predict(image)

            # Combine SAM mask with matting result
            mask_float = mask.astype(np.float32) / 255.0
            alpha_float = alpha.astype(np.float32) / 255.0

            # Blend: use matting in edge regions, SAM elsewhere
            edges = cv2.Canny(mask, 50, 150)
            edge_region = cv2.dilate(edges, None, iterations=5)
            edge_mask = edge_region.astype(np.float32) / 255.0

            combined = edge_mask * alpha_float + (1 - edge_mask) * mask_float
            alpha = (combined * 255).astype(np.uint8)
        else:
            alpha = mask

        # Extract foreground
        alpha_norm = alpha.astype(np.float32) / 255.0
        if len(alpha_norm.shape) == 2:
            alpha_norm = alpha_norm[:, :, np.newaxis]
        foreground = (image * alpha_norm).astype(np.uint8)

        return foreground, alpha

    def _segment_with_sam(
        self,
        image: ndarray,
        point_coords: ndarray | None = None,
        point_labels: ndarray | None = None,
    ) -> ndarray:
        """Segment using SAM.

        Args:
            image: Input image.
            point_coords: Point coordinates.
            point_labels: Point labels.

        Returns:
            Segmentation mask.
        """
        # Set image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._sam.set_image(image_rgb)

        # Generate prompts if not provided
        if point_coords is None:
            # Use center point as default prompt
            h, w = image.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])

        # Predict
        masks, scores, _ = self._sam.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Select best mask
        best_idx = np.argmax(scores)
        mask = (masks[best_idx] * 255).astype(np.uint8)

        return mask

    def _simple_segment(self, image: ndarray) -> ndarray:
        """Simple segmentation fallback.

        Args:
            image: Input image.

        Returns:
            Segmentation mask.
        """
        # Use GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        except cv2.error:
            mask = np.ones(image.shape[:2], np.uint8) * 255

        return mask


def create_matting_model(config: MattingConfig | None = None) -> BaseMattingModel:
    """Factory function to create matting model.

    Args:
        config: Matting configuration.

    Returns:
        Matting model instance.
    """
    config = config or MattingConfig()

    if config.model == MattingModel.MODNET:
        return MODNetMatting(config)
    elif config.model == MattingModel.VITMATTE:
        return ViTMatteMatting(config)
    elif config.model == MattingModel.MATTE_ANYTHING:
        return MatteAnythingMatting(config)
    else:
        raise ValueError(f"Unknown matting model: {config.model}")


def refine_alpha_with_matting(
    image: ndarray,
    alpha: ndarray,
    model: MattingModel = MattingModel.MODNET,
) -> ndarray:
    """Refine existing alpha using deep matting.

    Args:
        image: Input image.
        alpha: Existing alpha matte.
        model: Matting model to use.

    Returns:
        Refined alpha matte.
    """
    config = MattingConfig(model=model)
    matting = create_matting_model(config)
    matting.load()

    # Use existing alpha as trimap
    trimap = alpha.copy()
    trimap[(alpha > 10) & (alpha < 245)] = 128

    _, refined = matting.predict(image, trimap)

    # Blend with original
    alpha_float = alpha.astype(np.float32) / 255.0
    refined_float = refined.astype(np.float32) / 255.0

    # Use refined in edge regions
    edges = cv2.Canny(alpha, 50, 150)
    edge_region = cv2.dilate(edges, None, iterations=3)
    edge_mask = edge_region.astype(np.float32) / 255.0

    blended = edge_mask * refined_float + (1 - edge_mask) * alpha_float

    return (blended * 255).astype(np.uint8)
