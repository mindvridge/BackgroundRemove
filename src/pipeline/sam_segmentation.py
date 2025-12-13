"""Segment Anything Model (SAM) integration for video matting.

Provides high-quality segmentation using Meta's SAM model
for precise object boundaries.

Models supported:
- SAM (original)
- SAM 2
- FastSAM (lightweight)
- MobileSAM (mobile-optimized)
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


class SAMModel(Enum):
    """Available SAM model variants."""

    SAM_VIT_H = "sam_vit_h"  # Largest, best quality
    SAM_VIT_L = "sam_vit_l"  # Large
    SAM_VIT_B = "sam_vit_b"  # Base
    FAST_SAM = "fast_sam"  # Fast variant
    MOBILE_SAM = "mobile_sam"  # Mobile optimized
    SAM2 = "sam2"  # SAM 2


class PromptType(Enum):
    """Types of prompts for SAM."""

    POINT = "point"
    BOX = "box"
    MASK = "mask"
    AUTO = "auto"


@dataclass
class SAMConfig:
    """Configuration for SAM segmentation."""

    model: SAMModel = SAMModel.SAM_VIT_B
    device: str = "cuda"

    # Prompt settings
    prompt_type: PromptType = PromptType.AUTO
    auto_mask_threshold: float = 0.5

    # Processing settings
    process_scale: float = 1.0
    multimask_output: bool = True  # Output multiple mask candidates

    # Refinement
    refine_mask: bool = True
    refinement_iterations: int = 3

    # Model paths (optional)
    checkpoint_path: str | None = None


class SAMSegmenter:
    """SAM-based segmentation for precise object boundaries.

    Provides multiple modes for object segmentation using
    point prompts, box prompts, or automatic segmentation.

    Example:
        >>> sam = SAMSegmenter(SAMConfig(model=SAMModel.SAM_VIT_B))
        >>> sam.load()
        >>> mask = sam.segment(image, point=(100, 100))
    """

    CHECKPOINT_URLS = {
        SAMModel.SAM_VIT_H: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        SAMModel.SAM_VIT_L: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        SAMModel.SAM_VIT_B: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    def __init__(self, config: SAMConfig | None = None) -> None:
        """Initialize SAM segmenter.

        Args:
            config: SAM configuration.
        """
        self.config = config or SAMConfig()
        self._model = None
        self._predictor = None
        self._mask_generator = None
        self._is_loaded = False
        self._current_image = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load SAM model."""
        if self._is_loaded:
            return

        model_type = self.config.model

        try:
            if model_type in [SAMModel.SAM_VIT_H, SAMModel.SAM_VIT_L, SAMModel.SAM_VIT_B]:
                self._load_sam()
            elif model_type == SAMModel.FAST_SAM:
                self._load_fast_sam()
            elif model_type == SAMModel.MOBILE_SAM:
                self._load_mobile_sam()
            elif model_type == SAMModel.SAM2:
                self._load_sam2()
            else:
                self._load_sam()

            self._is_loaded = True
            logger.info(f"SAM loaded: {model_type.value}")

        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            self._create_fallback()
            self._is_loaded = True

    def _load_sam(self) -> None:
        """Load original SAM model."""
        try:
            from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

            # Get checkpoint
            checkpoint = self._get_checkpoint()
            if checkpoint is None:
                raise RuntimeError("SAM checkpoint not found")

            # Model type mapping
            model_map = {
                SAMModel.SAM_VIT_H: "vit_h",
                SAMModel.SAM_VIT_L: "vit_l",
                SAMModel.SAM_VIT_B: "vit_b",
            }
            model_type = model_map.get(self.config.model, "vit_b")

            # Load model
            self._model = sam_model_registry[model_type](checkpoint=str(checkpoint))
            self._model = self._model.to(self.config.device)

            # Create predictor and mask generator
            self._predictor = SamPredictor(self._model)
            self._mask_generator = SamAutomaticMaskGenerator(
                self._model,
                pred_iou_thresh=self.config.auto_mask_threshold,
            )

            self._use_sam = True

        except ImportError:
            logger.warning("segment_anything not installed")
            self._create_fallback()

    def _load_fast_sam(self) -> None:
        """Load FastSAM model."""
        try:
            from ultralytics import YOLO

            # Load FastSAM
            self._model = YOLO("FastSAM-x.pt")
            self._use_fastsam = True

        except ImportError:
            logger.warning("ultralytics not installed for FastSAM")
            self._load_sam()

    def _load_mobile_sam(self) -> None:
        """Load MobileSAM model."""
        try:
            from mobile_sam import SamPredictor, sam_model_registry

            # Load mobile checkpoint
            checkpoint = self._get_checkpoint("mobile_sam.pt")

            self._model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
            self._model = self._model.to(self.config.device)
            self._predictor = SamPredictor(self._model)
            self._use_sam = True

        except ImportError:
            logger.warning("mobile_sam not installed")
            self._load_sam()

    def _load_sam2(self) -> None:
        """Load SAM 2 model."""
        try:
            # Try loading SAM 2 from HuggingFace
            from transformers import SamModel, SamProcessor

            self._processor = SamProcessor.from_pretrained("facebook/sam2-hiera-base-plus")
            self._model = SamModel.from_pretrained("facebook/sam2-hiera-base-plus")
            self._model = self._model.to(self.config.device)
            self._use_sam2 = True

        except ImportError:
            logger.warning("SAM 2 not available, using SAM 1")
            self._load_sam()

    def _create_fallback(self) -> None:
        """Create fallback segmentation."""
        logger.info("Using GrabCut fallback for segmentation")
        self._use_fallback = True

    def _get_checkpoint(self, filename: str | None = None) -> Path | None:
        """Get checkpoint path.

        Args:
            filename: Optional specific filename.

        Returns:
            Path to checkpoint or None.
        """
        if self.config.checkpoint_path:
            return Path(self.config.checkpoint_path)

        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(exist_ok=True)

        if filename:
            return weights_dir / filename

        # Map model to checkpoint
        checkpoint_names = {
            SAMModel.SAM_VIT_H: "sam_vit_h_4b8939.pth",
            SAMModel.SAM_VIT_L: "sam_vit_l_0b3195.pth",
            SAMModel.SAM_VIT_B: "sam_vit_b_01ec64.pth",
        }

        name = checkpoint_names.get(self.config.model)
        if name:
            path = weights_dir / name
            if path.exists():
                return path

        return None

    def set_image(self, image: ndarray) -> None:
        """Set image for segmentation.

        Args:
            image: Input image (BGR).
        """
        if not self._is_loaded:
            self.load()

        self._current_image = image

        if hasattr(self, '_use_sam') and self._use_sam:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(image_rgb)

    def segment_point(
        self,
        point: tuple[int, int],
        label: int = 1,
    ) -> ndarray:
        """Segment using point prompt.

        Args:
            point: (x, y) coordinates.
            label: 1 for foreground, 0 for background.

        Returns:
            Segmentation mask.
        """
        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image first.")

        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([label])

        return self._segment_with_prompts(point_coords, point_labels)

    def segment_box(
        self,
        box: tuple[int, int, int, int],
    ) -> ndarray:
        """Segment using box prompt.

        Args:
            box: (x1, y1, x2, y2) bounding box.

        Returns:
            Segmentation mask.
        """
        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image first.")

        box_array = np.array([list(box)])

        return self._segment_with_box(box_array)

    def segment_auto(self) -> list[dict]:
        """Automatic segmentation of all objects.

        Returns:
            List of mask dictionaries with 'segmentation', 'area', 'bbox'.
        """
        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image first.")

        if hasattr(self, '_use_sam') and self._use_sam and self._mask_generator:
            image_rgb = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)
            return self._mask_generator.generate(image_rgb)
        elif hasattr(self, '_use_fastsam') and self._use_fastsam:
            return self._segment_fastsam_auto()
        else:
            return self._segment_fallback_auto()

    def _segment_with_prompts(
        self,
        point_coords: ndarray,
        point_labels: ndarray,
        box: ndarray | None = None,
    ) -> ndarray:
        """Segment with point/box prompts.

        Args:
            point_coords: Point coordinates.
            point_labels: Point labels.
            box: Optional box.

        Returns:
            Segmentation mask.
        """
        if hasattr(self, '_use_sam') and self._use_sam:
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=self.config.multimask_output,
            )

            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

        elif hasattr(self, '_use_sam2') and self._use_sam2:
            mask = self._segment_sam2(point_coords, point_labels)

        else:
            mask = self._segment_fallback(point_coords[0])

        # Refine if enabled
        if self.config.refine_mask:
            mask = self._refine_mask(mask)

        return (mask * 255).astype(np.uint8)

    def _segment_with_box(self, box: ndarray) -> ndarray:
        """Segment with box prompt.

        Args:
            box: Box array.

        Returns:
            Segmentation mask.
        """
        if hasattr(self, '_use_sam') and self._use_sam:
            masks, scores, _ = self._predictor.predict(
                box=box[0],
                multimask_output=self.config.multimask_output,
            )

            best_idx = np.argmax(scores)
            mask = masks[best_idx]

        else:
            mask = self._segment_fallback_box(box[0])

        if self.config.refine_mask:
            mask = self._refine_mask(mask)

        return (mask * 255).astype(np.uint8)

    def _segment_sam2(
        self,
        point_coords: ndarray,
        point_labels: ndarray,
    ) -> ndarray:
        """Segment using SAM 2.

        Args:
            point_coords: Points.
            point_labels: Labels.

        Returns:
            Mask.
        """
        from PIL import Image

        image_rgb = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self._processor(
            pil_image,
            input_points=[[point_coords.tolist()]],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        mask = masks[0][0][0].numpy()

        return mask

    def _segment_fastsam_auto(self) -> list[dict]:
        """Automatic segmentation with FastSAM.

        Returns:
            List of mask dicts.
        """
        results = self._model(
            self._current_image,
            device=self.config.device,
            retina_masks=True,
        )

        masks = []
        if results[0].masks is not None:
            for i, mask in enumerate(results[0].masks.data):
                mask_np = mask.cpu().numpy()
                masks.append({
                    'segmentation': mask_np,
                    'area': mask_np.sum(),
                    'bbox': results[0].boxes.xyxy[i].cpu().numpy().tolist(),
                })

        return masks

    def _segment_fallback(self, point: ndarray) -> ndarray:
        """Fallback segmentation using GrabCut.

        Args:
            point: Center point.

        Returns:
            Mask.
        """
        image = self._current_image
        h, w = image.shape[:2]

        # Create initial mask with point as foreground hint
        mask = np.zeros((h, w), np.uint8)
        mask[:] = cv2.GC_PR_BGD

        # Mark region around point as probable foreground
        x, y = int(point[0]), int(point[1])
        cv2.circle(mask, (x, y), 50, cv2.GC_FGD, -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0)
        except cv2.error:
            result = np.zeros((h, w))

        return result.astype(np.float32)

    def _segment_fallback_box(self, box: ndarray) -> ndarray:
        """Fallback segmentation with box using GrabCut.

        Args:
            box: Bounding box.

        Returns:
            Mask.
        """
        image = self._current_image
        h, w = image.shape[:2]

        mask = np.zeros((h, w), np.uint8)
        rect = tuple(map(int, box))

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0)
        except cv2.error:
            result = np.zeros((h, w))

        return result.astype(np.float32)

    def _segment_fallback_auto(self) -> list[dict]:
        """Fallback automatic segmentation.

        Returns:
            List of mask dicts.
        """
        # Use simple contour-based segmentation
        image = self._current_image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                masks.append({
                    'segmentation': mask,
                    'area': area,
                    'bbox': cv2.boundingRect(contour),
                })

        return masks

    def _refine_mask(self, mask: ndarray) -> ndarray:
        """Refine segmentation mask.

        Args:
            mask: Input mask.

        Returns:
            Refined mask.
        """
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for _ in range(self.config.refinement_iterations):
            # Close small holes
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            # Open to remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask.astype(np.float32)

    def segment(
        self,
        image: ndarray,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
        mask_hint: ndarray | None = None,
    ) -> ndarray:
        """Segment object in image.

        Args:
            image: Input image.
            point: Optional point prompt.
            box: Optional box prompt.
            mask_hint: Optional mask hint.

        Returns:
            Segmentation mask.
        """
        self.set_image(image)

        if point is not None:
            return self.segment_point(point)
        elif box is not None:
            return self.segment_box(box)
        elif mask_hint is not None:
            # Use mask centroid as point
            moments = cv2.moments(mask_hint)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                return self.segment_point((cx, cy))

        # Auto mode: get largest segment
        masks = self.segment_auto()
        if masks:
            largest = max(masks, key=lambda x: x['area'])
            return largest['segmentation']

        return np.zeros(image.shape[:2], np.uint8)

    def refine_alpha_with_sam(
        self,
        image: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Refine alpha matte using SAM segmentation.

        Args:
            image: Input image.
            alpha: Initial alpha matte.

        Returns:
            Refined alpha.
        """
        self.set_image(image)

        # Get SAM segmentation guided by alpha
        moments = cv2.moments(alpha)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            sam_mask = self.segment_point((cx, cy))
        else:
            return alpha

        # Combine SAM mask with alpha
        alpha_float = alpha.astype(np.float32) / 255.0
        sam_float = sam_mask.astype(np.float32) / 255.0

        # In uncertain regions, prefer SAM
        uncertain = (alpha > 30) & (alpha < 225)
        uncertain_float = uncertain.astype(np.float32)

        refined = (
            uncertain_float * sam_float +
            (1 - uncertain_float) * alpha_float
        )

        return (refined * 255).astype(np.uint8)

    def unload(self) -> None:
        """Unload model."""
        self._model = None
        self._predictor = None
        self._mask_generator = None
        self._is_loaded = False

    def __repr__(self) -> str:
        """String representation."""
        return f"SAMSegmenter(model={self.config.model.value})"


def segment_with_sam(
    image: ndarray,
    point: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    model: SAMModel = SAMModel.SAM_VIT_B,
) -> ndarray:
    """Convenience function for SAM segmentation.

    Args:
        image: Input image.
        point: Point prompt.
        box: Box prompt.
        model: SAM model variant.

    Returns:
        Segmentation mask.
    """
    config = SAMConfig(model=model)
    segmenter = SAMSegmenter(config)
    segmenter.load()
    return segmenter.segment(image, point=point, box=box)


def refine_alpha_sam(
    image: ndarray,
    alpha: ndarray,
    model: SAMModel = SAMModel.SAM_VIT_B,
) -> ndarray:
    """Convenience function for SAM-based alpha refinement.

    Args:
        image: Input image.
        alpha: Alpha matte.
        model: SAM model.

    Returns:
        Refined alpha.
    """
    config = SAMConfig(model=model)
    segmenter = SAMSegmenter(config)
    segmenter.load()
    return segmenter.refine_alpha_with_sam(image, alpha)
