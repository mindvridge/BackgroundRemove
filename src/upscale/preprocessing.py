"""Image preprocessing for upscaling.

Provides preprocessing functions to improve upscaling quality:
- Denoising: Remove noise before upscaling to prevent amplification
- Sharpening: Enhance edges for better detail preservation
- Histogram equalization: Improve contrast
- Color correction: Fix color casts
- JPEG artifact removal: Clean compression artifacts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class DenoiseMethod(Enum):
    """Denoising methods."""

    NONE = "none"
    BILATERAL = "bilateral"  # Edge-preserving, good for faces
    NLM = "nlm"  # Non-local means, best quality
    FAST_NLM = "fast_nlm"  # Fast non-local means
    GAUSSIAN = "gaussian"  # Simple gaussian blur
    MEDIAN = "median"  # Good for salt-and-pepper noise
    BM3D = "bm3d"  # Block-matching 3D (requires additional lib)


class SharpenMethod(Enum):
    """Sharpening methods."""

    NONE = "none"
    UNSHARP_MASK = "unsharp_mask"  # Classic unsharp masking
    LAPLACIAN = "laplacian"  # Laplacian-based
    HIGH_PASS = "high_pass"  # High-pass filter
    ADAPTIVE = "adaptive"  # Adaptive sharpening


@dataclass
class PreprocessConfig:
    """Configuration for image preprocessing."""

    # Denoising
    denoise_method: DenoiseMethod = DenoiseMethod.FAST_NLM
    denoise_strength: float = 0.5  # 0.0 - 1.0
    denoise_preserve_detail: bool = True

    # Sharpening
    sharpen_method: SharpenMethod = SharpenMethod.UNSHARP_MASK
    sharpen_amount: float = 0.3  # 0.0 - 1.0
    sharpen_radius: float = 1.0  # Kernel radius

    # Color correction
    auto_white_balance: bool = False
    auto_contrast: bool = False
    gamma_correction: float = 1.0  # 1.0 = no change

    # JPEG artifact removal
    remove_jpeg_artifacts: bool = False
    jpeg_quality_estimate: int = 0  # 0 = auto-detect

    # General
    normalize_histogram: bool = False
    clahe_enabled: bool = False  # Contrast Limited Adaptive Histogram Equalization
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8


class ImagePreprocessor:
    """Image preprocessor for upscaling quality improvement."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration.
        """
        self.config = config or PreprocessConfig()

    def process(self, image: "ndarray") -> "ndarray":
        """Apply all preprocessing steps.

        Args:
            image: Input BGR image.

        Returns:
            Preprocessed image.
        """
        result = image.copy()

        # Step 1: JPEG artifact removal (if enabled)
        if self.config.remove_jpeg_artifacts:
            result = self.remove_jpeg_artifacts(result)

        # Step 2: Denoising
        if self.config.denoise_method != DenoiseMethod.NONE:
            result = self.denoise(result)

        # Step 3: Color correction
        if self.config.auto_white_balance:
            result = self.auto_white_balance(result)

        if self.config.gamma_correction != 1.0:
            result = self.gamma_correct(result, self.config.gamma_correction)

        # Step 4: Contrast enhancement
        if self.config.auto_contrast:
            result = self.auto_contrast(result)

        if self.config.clahe_enabled:
            result = self.apply_clahe(result)

        if self.config.normalize_histogram:
            result = self.normalize_histogram(result)

        # Step 5: Sharpening (last step before upscale)
        if self.config.sharpen_method != SharpenMethod.NONE:
            result = self.sharpen(result)

        return result

    def denoise(self, image: "ndarray") -> "ndarray":
        """Apply denoising.

        Args:
            image: Input BGR image.

        Returns:
            Denoised image.
        """
        method = self.config.denoise_method
        strength = self.config.denoise_strength

        if method == DenoiseMethod.NONE:
            return image

        # Scale strength to appropriate parameter values
        h = int(strength * 20) + 3  # Filter strength

        if method == DenoiseMethod.BILATERAL:
            # Bilateral filter - preserves edges well
            d = int(strength * 7) + 5
            sigma_color = strength * 75 + 25
            sigma_space = strength * 75 + 25
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif method == DenoiseMethod.NLM:
            # Non-local means for color images
            template_window = 7
            search_window = 21
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_window, search_window
            )

        elif method == DenoiseMethod.FAST_NLM:
            # Fast NLM with smaller windows
            template_window = 5
            search_window = 15
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_window, search_window
            )

        elif method == DenoiseMethod.GAUSSIAN:
            # Simple Gaussian blur
            kernel_size = int(strength * 4) * 2 + 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        elif method == DenoiseMethod.MEDIAN:
            # Median filter - good for salt-and-pepper noise
            kernel_size = int(strength * 2) * 2 + 3
            return cv2.medianBlur(image, kernel_size)

        elif method == DenoiseMethod.BM3D:
            # BM3D requires additional library
            logger.warning("BM3D requires bm3d library, falling back to NLM")
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

        return image

    def sharpen(self, image: "ndarray") -> "ndarray":
        """Apply sharpening.

        Args:
            image: Input BGR image.

        Returns:
            Sharpened image.
        """
        method = self.config.sharpen_method
        amount = self.config.sharpen_amount
        radius = self.config.sharpen_radius

        if method == SharpenMethod.NONE or amount <= 0:
            return image

        if method == SharpenMethod.UNSHARP_MASK:
            return self._unsharp_mask(image, amount, radius)

        elif method == SharpenMethod.LAPLACIAN:
            return self._laplacian_sharpen(image, amount)

        elif method == SharpenMethod.HIGH_PASS:
            return self._high_pass_sharpen(image, amount, radius)

        elif method == SharpenMethod.ADAPTIVE:
            return self._adaptive_sharpen(image, amount)

        return image

    def _unsharp_mask(
        self, image: "ndarray", amount: float, radius: float
    ) -> "ndarray":
        """Apply unsharp mask sharpening.

        Args:
            image: Input image.
            amount: Sharpening amount (0-1).
            radius: Blur radius.

        Returns:
            Sharpened image.
        """
        # Create blurred version
        kernel_size = int(radius * 4) * 2 + 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(
            image, 1.0 + amount, blurred, -amount, 0
        )

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _laplacian_sharpen(self, image: "ndarray", amount: float) -> "ndarray":
        """Apply Laplacian-based sharpening.

        Args:
            image: Input image.
            amount: Sharpening amount (0-1).

        Returns:
            Sharpened image.
        """
        # Convert to float
        img_float = image.astype(np.float32)

        # Apply Laplacian
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F)

        # Add back to original
        sharpened = img_float - amount * laplacian

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _high_pass_sharpen(
        self, image: "ndarray", amount: float, radius: float
    ) -> "ndarray":
        """Apply high-pass filter sharpening.

        Args:
            image: Input image.
            amount: Sharpening amount (0-1).
            radius: Filter radius.

        Returns:
            Sharpened image.
        """
        # Create low-pass version
        kernel_size = int(radius * 6) * 2 + 1
        low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius * 2)

        # High-pass = original - low_pass
        high_pass = cv2.subtract(image, low_pass)

        # Add high-pass back with amount
        sharpened = cv2.addWeighted(image, 1.0, high_pass, amount * 2, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _adaptive_sharpen(self, image: "ndarray", amount: float) -> "ndarray":
        """Apply adaptive sharpening based on local variance.

        Args:
            image: Input image.
            amount: Sharpening amount (0-1).

        Returns:
            Sharpened image.
        """
        # Convert to LAB for luminance processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Calculate local variance for edge detection
        l_float = l.astype(np.float32)
        mean = cv2.blur(l_float, (5, 5))
        sqr_mean = cv2.blur(l_float**2, (5, 5))
        variance = np.sqrt(np.maximum(sqr_mean - mean**2, 0))

        # Normalize variance to create adaptive mask
        variance_norm = variance / (variance.max() + 1e-6)
        mask = np.clip(variance_norm * 2, 0, 1)

        # Apply unsharp mask
        blurred = cv2.GaussianBlur(l_float, (5, 5), 1.0)
        sharpened = l_float + amount * mask * (l_float - blurred)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Merge back
        lab_sharpened = cv2.merge([sharpened, a, b])
        return cv2.cvtColor(lab_sharpened, cv2.COLOR_LAB2BGR)

    def remove_jpeg_artifacts(self, image: "ndarray") -> "ndarray":
        """Remove JPEG compression artifacts.

        Args:
            image: Input image with JPEG artifacts.

        Returns:
            Cleaned image.
        """
        # Use bilateral filter to reduce blocking while preserving edges
        # This is a simple approach; more advanced would use deep learning

        # Detect quality level if not specified
        quality = self.config.jpeg_quality_estimate
        if quality == 0:
            quality = self._estimate_jpeg_quality(image)

        # Stronger filtering for lower quality
        if quality < 30:
            d = 9
            sigma_color = 75
            sigma_space = 75
        elif quality < 50:
            d = 7
            sigma_color = 50
            sigma_space = 50
        elif quality < 70:
            d = 5
            sigma_color = 35
            sigma_space = 35
        else:
            d = 3
            sigma_color = 25
            sigma_space = 25

        # Apply bilateral filter
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        # Apply slight median filter to reduce mosquito noise
        if quality < 50:
            result = cv2.medianBlur(result, 3)

        return result

    def _estimate_jpeg_quality(self, image: "ndarray") -> int:
        """Estimate JPEG quality from image.

        Args:
            image: Input image.

        Returns:
            Estimated quality (0-100).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect blockiness using 8x8 block analysis
        h, w = gray.shape
        h8, w8 = h // 8 * 8, w // 8 * 8
        gray_cropped = gray[:h8, :w8]

        # Calculate block boundary differences
        block_diff_h = np.abs(
            gray_cropped[7::8, :].astype(np.float32) - gray_cropped[8::8, :].astype(np.float32)
        )
        block_diff_v = np.abs(
            gray_cropped[:, 7::8].astype(np.float32) - gray_cropped[:, 8::8].astype(np.float32)
        )

        # Calculate internal differences
        internal_diff_h = np.abs(
            gray_cropped[3::8, :].astype(np.float32) - gray_cropped[4::8, :].astype(np.float32)
        )
        internal_diff_v = np.abs(
            gray_cropped[:, 3::8].astype(np.float32) - gray_cropped[:, 4::8].astype(np.float32)
        )

        # Blockiness measure
        block_mean = (np.mean(block_diff_h) + np.mean(block_diff_v)) / 2
        internal_mean = (np.mean(internal_diff_h) + np.mean(internal_diff_v)) / 2

        if internal_mean > 0:
            blockiness = block_mean / internal_mean
        else:
            blockiness = 0

        # Convert blockiness to quality estimate
        # Higher blockiness = lower quality
        quality = int(100 - min(blockiness * 50, 80))
        return max(10, min(100, quality))

    def auto_white_balance(self, image: "ndarray") -> "ndarray":
        """Apply automatic white balance.

        Args:
            image: Input BGR image.

        Returns:
            White-balanced image.
        """
        # Simple gray world assumption
        result = image.astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg = (avg_b + avg_g + avg_r) / 3

        result[:, :, 0] *= avg / (avg_b + 1e-6)
        result[:, :, 1] *= avg / (avg_g + 1e-6)
        result[:, :, 2] *= avg / (avg_r + 1e-6)

        return np.clip(result, 0, 255).astype(np.uint8)

    def gamma_correct(self, image: "ndarray", gamma: float) -> "ndarray":
        """Apply gamma correction.

        Args:
            image: Input image.
            gamma: Gamma value (>1 darkens, <1 brightens).

        Returns:
            Gamma-corrected image.
        """
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        return cv2.LUT(image, table)

    def auto_contrast(self, image: "ndarray") -> "ndarray":
        """Apply automatic contrast adjustment.

        Args:
            image: Input image.

        Returns:
            Contrast-adjusted image.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Stretch luminance histogram
        l_min, l_max = np.percentile(l, [1, 99])
        l_stretched = np.clip((l - l_min) * 255.0 / (l_max - l_min + 1e-6), 0, 255)
        l_stretched = l_stretched.astype(np.uint8)

        # Merge back
        lab_stretched = cv2.merge([l_stretched, a, b])
        return cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)

    def apply_clahe(self, image: "ndarray") -> "ndarray":
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input image.

        Returns:
            CLAHE-enhanced image.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to luminance channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size),
        )
        l_clahe = clahe.apply(l)

        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def normalize_histogram(self, image: "ndarray") -> "ndarray":
        """Normalize histogram for each channel.

        Args:
            image: Input image.

        Returns:
            Histogram-normalized image.
        """
        # Apply histogram equalization to each channel
        b, g, r = cv2.split(image)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge([b_eq, g_eq, r_eq])


# =============================================================================
# Convenience functions
# =============================================================================


def quick_denoise(image: "ndarray", strength: float = 0.5) -> "ndarray":
    """Quick denoising with default settings.

    Args:
        image: Input image.
        strength: Denoising strength (0-1).

    Returns:
        Denoised image.
    """
    config = PreprocessConfig(
        denoise_method=DenoiseMethod.FAST_NLM,
        denoise_strength=strength,
    )
    preprocessor = ImagePreprocessor(config)
    return preprocessor.denoise(image)


def quick_sharpen(image: "ndarray", amount: float = 0.3) -> "ndarray":
    """Quick sharpening with default settings.

    Args:
        image: Input image.
        amount: Sharpening amount (0-1).

    Returns:
        Sharpened image.
    """
    config = PreprocessConfig(
        sharpen_method=SharpenMethod.UNSHARP_MASK,
        sharpen_amount=amount,
    )
    preprocessor = ImagePreprocessor(config)
    return preprocessor.sharpen(image)


def preprocess_for_upscale(
    image: "ndarray",
    denoise_strength: float = 0.3,
    sharpen_amount: float = 0.2,
    remove_artifacts: bool = False,
) -> "ndarray":
    """Recommended preprocessing for upscaling.

    Args:
        image: Input image.
        denoise_strength: Denoising strength.
        sharpen_amount: Sharpening amount.
        remove_artifacts: Whether to remove JPEG artifacts.

    Returns:
        Preprocessed image.
    """
    config = PreprocessConfig(
        denoise_method=DenoiseMethod.FAST_NLM,
        denoise_strength=denoise_strength,
        sharpen_method=SharpenMethod.UNSHARP_MASK,
        sharpen_amount=sharpen_amount,
        remove_jpeg_artifacts=remove_artifacts,
        auto_contrast=False,  # Let upscaler handle this
    )
    preprocessor = ImagePreprocessor(config)
    return preprocessor.process(image)
