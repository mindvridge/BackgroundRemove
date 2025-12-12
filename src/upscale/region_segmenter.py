"""Region-based image segmentation for selective upscaling.

Segments images into different regions for optimized upscaling:
- Face regions: Use CodeFormer for best face quality
- Text regions: Use SwinIR for sharp text
- Detail regions: Use HAT for fine details
- Background regions: Use Real-ESRGAN for speed

Supports multiple segmentation backends:
- SAM (Segment Anything Model) - best quality
- YOLO - fast object detection
- OpenCV - lightweight fallback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Types of image regions."""

    FACE = "face"
    TEXT = "text"
    DETAIL = "detail"  # High-frequency areas
    SMOOTH = "smooth"  # Low-frequency areas
    EDGE = "edge"
    BACKGROUND = "background"
    UNKNOWN = "unknown"


class SegmentationBackend(Enum):
    """Available segmentation backends."""

    OPENCV = "opencv"  # Lightweight, always available
    SAM = "sam"  # Segment Anything Model
    YOLO = "yolo"  # YOLO-based detection


@dataclass
class Region:
    """Represents a segmented region."""

    region_type: RegionType
    mask: "ndarray"  # Binary mask (H, W)
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        """Get region area in pixels."""
        return int(np.sum(self.mask > 0))

    def crop(self, image: "ndarray") -> "ndarray":
        """Crop image to this region's bounding box."""
        if self.bbox is None:
            return image
        x1, y1, x2, y2 = self.bbox
        return image[y1:y2, x1:x2].copy()

    def apply_mask(self, image: "ndarray") -> "ndarray":
        """Apply mask to image (set non-region pixels to 0)."""
        result = image.copy()
        mask_3ch = np.stack([self.mask] * 3, axis=-1)
        result = (result * (mask_3ch / 255.0)).astype(np.uint8)
        return result


@dataclass
class SegmentationResult:
    """Result of image segmentation."""

    regions: list[Region]
    combined_mask: "ndarray"  # Labeled mask where each pixel has region ID
    image_size: tuple[int, int]  # (height, width)

    def get_regions_by_type(self, region_type: RegionType) -> list[Region]:
        """Get all regions of a specific type."""
        return [r for r in self.regions if r.region_type == region_type]

    def get_region_mask(self, region_type: RegionType) -> "ndarray":
        """Get combined mask for all regions of a type."""
        masks = [r.mask for r in self.regions if r.region_type == region_type]
        if not masks:
            return np.zeros(self.image_size, dtype=np.uint8)
        combined = np.zeros_like(masks[0])
        for m in masks:
            combined = np.maximum(combined, m)
        return combined


@dataclass
class SegmenterConfig:
    """Configuration for region segmentation."""

    backend: SegmentationBackend = SegmentationBackend.OPENCV

    # Detection settings
    detect_faces: bool = True
    detect_text: bool = True
    detect_details: bool = True
    min_region_size: int = 32  # Minimum region size in pixels

    # Face detection
    face_detector: str = "haar"  # 'haar', 'dnn', 'retinaface'
    min_face_size: int = 32
    face_confidence: float = 0.5

    # Text detection
    text_detector: str = "east"  # 'east', 'craft'
    text_confidence: float = 0.5

    # Detail detection
    detail_threshold: float = 0.3  # Laplacian variance threshold
    edge_threshold: float = 100  # Canny edge threshold

    # SAM settings (if using SAM backend)
    sam_model: str = "vit_b"  # 'vit_b', 'vit_l', 'vit_h'
    sam_points_per_side: int = 32

    # Model paths
    model_dir: Path | None = None


class RegionSegmenter:
    """Segments images into different regions for selective processing.

    Example:
        >>> segmenter = RegionSegmenter()
        >>> result = segmenter.segment(image)
        >>> face_regions = result.get_regions_by_type(RegionType.FACE)
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()
        self._face_detector: Any = None
        self._text_detector: Any = None
        self._sam_predictor: Any = None
        self._loaded = False

    def load(self) -> None:
        """Load detection models."""
        if self._loaded:
            return

        logger.info(f"Loading segmenter with {self.config.backend.value} backend")

        # Load face detector
        if self.config.detect_faces:
            self._load_face_detector()

        # Load text detector
        if self.config.detect_text:
            self._load_text_detector()

        # Load SAM if using SAM backend
        if self.config.backend == SegmentationBackend.SAM:
            self._load_sam()

        self._loaded = True

    def _load_face_detector(self) -> None:
        """Load face detection model."""
        if self.config.face_detector == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_detector = cv2.CascadeClassifier(cascade_path)
        elif self.config.face_detector == "dnn":
            # Load DNN face detector
            proto_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            if Path(proto_path).exists() and Path(model_path).exists():
                self._face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        logger.info(f"Face detector loaded: {self.config.face_detector}")

    def _load_text_detector(self) -> None:
        """Load text detection model."""
        # EAST text detector
        model_path = "frozen_east_text_detection.pb"
        if Path(model_path).exists():
            self._text_detector = cv2.dnn.readNet(model_path)
            logger.info("EAST text detector loaded")

    def _load_sam(self) -> None:
        """Load Segment Anything Model."""
        try:
            from segment_anything import SamPredictor, sam_model_registry

            model_type = self.config.sam_model
            model_path = self._get_sam_path()

            if model_path and model_path.exists():
                sam = sam_model_registry[model_type](checkpoint=str(model_path))
                self._sam_predictor = SamPredictor(sam)
                logger.info(f"SAM loaded: {model_type}")
        except ImportError:
            logger.warning("segment_anything not installed, SAM unavailable")

    def _get_sam_path(self) -> Path | None:
        """Get SAM model path."""
        if self.config.model_dir:
            return self.config.model_dir / f"sam_{self.config.sam_model}.pth"
        return None

    def segment(self, image: "ndarray") -> SegmentationResult:
        """Segment image into regions.

        Args:
            image: Input BGR image.

        Returns:
            SegmentationResult with detected regions.
        """
        if not self._loaded:
            self.load()

        h, w = image.shape[:2]
        regions: list[Region] = []

        # Detect faces
        if self.config.detect_faces:
            face_regions = self._detect_faces(image)
            regions.extend(face_regions)

        # Detect text
        if self.config.detect_text:
            text_regions = self._detect_text(image)
            regions.extend(text_regions)

        # Detect detail/smooth regions
        if self.config.detect_details:
            detail_regions = self._detect_details(image)
            regions.extend(detail_regions)

        # Create combined mask
        combined_mask = self._create_combined_mask(regions, (h, w))

        # Fill remaining areas as background
        bg_mask = self._get_background_mask(combined_mask)
        if np.any(bg_mask):
            regions.append(Region(
                region_type=RegionType.BACKGROUND,
                mask=bg_mask,
                confidence=1.0,
            ))

        return SegmentationResult(
            regions=regions,
            combined_mask=combined_mask,
            image_size=(h, w),
        )

    def _detect_faces(self, image: "ndarray") -> list[Region]:
        """Detect face regions."""
        regions = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        if isinstance(self._face_detector, cv2.CascadeClassifier):
            # Haar cascade detection
            faces = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.config.min_face_size, self.config.min_face_size),
            )

            for (x, y, fw, fh) in faces:
                # Create mask
                mask = np.zeros((h, w), dtype=np.uint8)

                # Expand region slightly
                pad = int(max(fw, fh) * 0.2)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + fw + pad)
                y2 = min(h, y + fh + pad)

                mask[y1:y2, x1:x2] = 255

                regions.append(Region(
                    region_type=RegionType.FACE,
                    mask=mask,
                    bbox=(x1, y1, x2, y2),
                    confidence=0.9,
                ))

        return regions

    def _detect_text(self, image: "ndarray") -> list[Region]:
        """Detect text regions."""
        regions = []
        h, w = image.shape[:2]

        if self._text_detector is not None:
            # Use EAST detector
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (320, 320),
                (123.68, 116.78, 103.94), True, False
            )
            self._text_detector.setInput(blob)

            try:
                scores, geometry = self._text_detector.forward([
                    "feature_fusion/Conv_7/Sigmoid",
                    "feature_fusion/concat_3",
                ])

                boxes = self._decode_text_boxes(scores, geometry, self.config.text_confidence)

                for box in boxes:
                    x1, y1, x2, y2 = box
                    # Scale to original size
                    x1 = int(x1 * w / 320)
                    y1 = int(y1 * h / 320)
                    x2 = int(x2 * w / 320)
                    y2 = int(y2 * h / 320)

                    mask = np.zeros((h, w), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 255

                    regions.append(Region(
                        region_type=RegionType.TEXT,
                        mask=mask,
                        bbox=(x1, y1, x2, y2),
                        confidence=0.8,
                    ))
            except Exception as e:
                logger.warning(f"Text detection failed: {e}")
        else:
            # Fallback: use morphology to detect text-like regions
            regions.extend(self._detect_text_morph(image))

        return regions

    def _detect_text_morph(self, image: "ndarray") -> list[Region]:
        """Detect text using morphological operations (fallback)."""
        regions = []
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect high contrast regions (likely text)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5,
                               cv2.convertScaleAbs(grad_y), 0.5, 0)

        # Threshold
        _, binary = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Filter by aspect ratio (text is usually wider than tall)
            if cw > ch * 2 and cw * ch > self.config.min_region_size**2:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y:y+ch, x:x+cw] = 255

                regions.append(Region(
                    region_type=RegionType.TEXT,
                    mask=mask,
                    bbox=(x, y, x + cw, y + ch),
                    confidence=0.6,
                ))

        return regions

    def _decode_text_boxes(
        self,
        scores: "ndarray",
        geometry: "ndarray",
        threshold: float,
    ) -> list[tuple[int, int, int, int]]:
        """Decode EAST text detector output."""
        boxes = []
        rows, cols = scores.shape[2:4]

        for y in range(rows):
            for x in range(cols):
                if scores[0, 0, y, x] < threshold:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0

                # Get geometry
                h_top = geometry[0, 0, y, x]
                h_right = geometry[0, 1, y, x]
                h_bottom = geometry[0, 2, y, x]
                h_left = geometry[0, 3, y, x]

                x1 = int(offset_x - h_left)
                y1 = int(offset_y - h_top)
                x2 = int(offset_x + h_right)
                y2 = int(offset_y + h_bottom)

                boxes.append((x1, y1, x2, y2))

        return boxes

    def _detect_details(self, image: "ndarray") -> list[Region]:
        """Detect detail and smooth regions based on local variance."""
        regions = []
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate local variance using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)

        # Use block-based variance
        block_size = 32
        variance_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = laplacian_abs[y:y+block_size, x:x+block_size]
                var = np.var(block)
                variance_map[y:y+block_size, x:x+block_size] = max(
                    variance_map[y:y+block_size, x:x+block_size].max(),
                    var
                )

        # Normalize
        variance_map = variance_map / (variance_map.max() + 1e-6)

        # Create detail mask
        detail_threshold = self.config.detail_threshold
        detail_mask = (variance_map > detail_threshold).astype(np.uint8) * 255

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        detail_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_CLOSE, kernel)
        detail_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_OPEN, kernel)

        if np.any(detail_mask):
            regions.append(Region(
                region_type=RegionType.DETAIL,
                mask=detail_mask,
                confidence=0.8,
                metadata={"variance_threshold": detail_threshold},
            ))

        # Create smooth mask (inverse of detail)
        smooth_mask = 255 - detail_mask

        if np.any(smooth_mask):
            regions.append(Region(
                region_type=RegionType.SMOOTH,
                mask=smooth_mask,
                confidence=0.8,
            ))

        # Detect edges
        edges = cv2.Canny(gray, self.config.edge_threshold, self.config.edge_threshold * 2)
        edge_dilated = cv2.dilate(edges, kernel, iterations=2)

        if np.any(edge_dilated):
            regions.append(Region(
                region_type=RegionType.EDGE,
                mask=edge_dilated,
                confidence=0.7,
            ))

        return regions

    def _create_combined_mask(
        self,
        regions: list[Region],
        size: tuple[int, int],
    ) -> "ndarray":
        """Create combined labeled mask."""
        h, w = size
        combined = np.zeros((h, w), dtype=np.uint8)

        # Priority: face > text > detail > smooth > edge
        priority = {
            RegionType.FACE: 5,
            RegionType.TEXT: 4,
            RegionType.DETAIL: 3,
            RegionType.EDGE: 2,
            RegionType.SMOOTH: 1,
            RegionType.BACKGROUND: 0,
            RegionType.UNKNOWN: 0,
        }

        # Sort by priority (lowest first, so higher priority overwrites)
        sorted_regions = sorted(regions, key=lambda r: priority.get(r.region_type, 0))

        for i, region in enumerate(sorted_regions, 1):
            combined[region.mask > 0] = i

        return combined

    def _get_background_mask(self, combined_mask: "ndarray") -> "ndarray":
        """Get mask for unassigned (background) pixels."""
        return ((combined_mask == 0) * 255).astype(np.uint8)


# =============================================================================
# Region-based processing utilities
# =============================================================================


def get_optimal_upscaler(region_type: RegionType) -> str:
    """Get optimal upscaler for a region type.

    Returns:
        Recommended upscaler name.
    """
    mapping = {
        RegionType.FACE: "codeformer",  # Best for faces
        RegionType.TEXT: "swinir",  # Best for sharp text
        RegionType.DETAIL: "hat",  # Best for fine details
        RegionType.EDGE: "hat",  # Preserve edges
        RegionType.SMOOTH: "real_esrgan",  # Fast for smooth areas
        RegionType.BACKGROUND: "real_esrgan",  # Fast for background
        RegionType.UNKNOWN: "real_esrgan",
    }
    return mapping.get(region_type, "real_esrgan")


def blend_regions(
    base_image: "ndarray",
    region_images: list[tuple["ndarray", Region]],
    blend_radius: int = 16,
) -> "ndarray":
    """Blend multiple region images together.

    Args:
        base_image: Base/background image.
        region_images: List of (processed_image, region) tuples.
        blend_radius: Feathering radius for blending.

    Returns:
        Blended result image.
    """
    result = base_image.copy().astype(np.float32)
    weight_sum = np.ones(base_image.shape[:2], dtype=np.float32)

    for processed, region in region_images:
        # Create soft mask with feathered edges
        mask = region.mask.astype(np.float32) / 255.0

        if blend_radius > 0:
            mask = cv2.GaussianBlur(mask, (blend_radius * 2 + 1, blend_radius * 2 + 1), 0)

        # Expand mask to 3 channels
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Weighted blend
        result = result * (1 - mask_3ch) + processed.astype(np.float32) * mask_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


def segment_and_process(
    image: "ndarray",
    process_fn: dict[RegionType, callable],
    config: SegmenterConfig | None = None,
) -> "ndarray":
    """Segment image and process each region with appropriate function.

    Args:
        image: Input image.
        process_fn: Dict mapping RegionType to processing function.
        config: Segmenter configuration.

    Returns:
        Processed image.
    """
    segmenter = RegionSegmenter(config)
    result = segmenter.segment(image)

    processed_regions = []

    for region in result.regions:
        if region.region_type in process_fn:
            fn = process_fn[region.region_type]

            # Extract region
            if region.bbox:
                x1, y1, x2, y2 = region.bbox
                roi = image[y1:y2, x1:x2].copy()
                processed_roi = fn(roi)

                # Put back
                processed = image.copy()
                processed[y1:y2, x1:x2] = processed_roi
            else:
                processed = fn(image)

            processed_regions.append((processed, region))

    if not processed_regions:
        return image

    return blend_regions(image, processed_regions)
