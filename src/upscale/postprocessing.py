"""Image postprocessing after upscaling.

Provides postprocessing functions to enhance upscaled images:
- Color correction and grading
- Artifact removal (halos, ringing)
- Detail enhancement
- Texture refinement
- Final sharpening
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


class ArtifactType(Enum):
    """Types of upscaling artifacts."""

    HALO = "halo"  # Edge halos
    RINGING = "ringing"  # Gibbs phenomenon
    CHECKERBOARD = "checkerboard"  # From upsampling
    BLUR = "blur"  # Over-smoothed areas
    NOISE = "noise"  # Amplified noise


class ColorGradingPreset(Enum):
    """Color grading presets."""

    NONE = "none"
    NATURAL = "natural"  # Slight saturation boost
    VIVID = "vivid"  # Enhanced colors
    CINEMATIC = "cinematic"  # Film-like tones
    WARM = "warm"  # Warm color cast
    COOL = "cool"  # Cool color cast
    HDR = "hdr"  # HDR-like effect


@dataclass
class PostprocessConfig:
    """Configuration for image postprocessing."""

    # Artifact removal
    remove_halos: bool = True
    halo_threshold: float = 0.3  # Sensitivity for halo detection
    remove_ringing: bool = True
    remove_checkerboard: bool = True

    # Detail enhancement
    enhance_details: bool = True
    detail_strength: float = 0.3  # 0.0 - 1.0
    texture_enhancement: bool = True
    texture_strength: float = 0.2

    # Final sharpening
    final_sharpen: bool = True
    sharpen_amount: float = 0.2
    sharpen_radius: float = 0.5

    # Color grading
    color_grading: ColorGradingPreset = ColorGradingPreset.NONE
    saturation_boost: float = 0.0  # -1.0 to 1.0
    vibrance: float = 0.0  # -1.0 to 1.0

    # Color correction
    auto_color_correct: bool = False
    preserve_skin_tones: bool = True

    # Quality
    reduce_banding: bool = True
    output_bit_depth: int = 8  # 8 or 16


class ImagePostprocessor:
    """Image postprocessor for upscaling quality improvement."""

    def __init__(self, config: PostprocessConfig | None = None) -> None:
        """Initialize postprocessor.

        Args:
            config: Postprocessing configuration.
        """
        self.config = config or PostprocessConfig()

    def process(
        self,
        upscaled: "ndarray",
        original: "ndarray" | None = None,
        scale: int = 4,
    ) -> "ndarray":
        """Apply all postprocessing steps.

        Args:
            upscaled: Upscaled image.
            original: Original image (optional, for reference).
            scale: Upscaling factor used.

        Returns:
            Postprocessed image.
        """
        result = upscaled.copy()

        # Step 1: Artifact removal
        if self.config.remove_halos:
            result = self.remove_halos(result, original, scale)

        if self.config.remove_ringing:
            result = self.remove_ringing(result)

        if self.config.remove_checkerboard:
            result = self.remove_checkerboard(result)

        # Step 2: Detail enhancement
        if self.config.enhance_details:
            result = self.enhance_details(result)

        if self.config.texture_enhancement:
            result = self.enhance_texture(result)

        # Step 3: Color processing
        if self.config.auto_color_correct:
            result = self.auto_color_correct(result)

        if self.config.saturation_boost != 0:
            result = self.adjust_saturation(result, self.config.saturation_boost)

        if self.config.vibrance != 0:
            result = self.adjust_vibrance(result, self.config.vibrance)

        if self.config.color_grading != ColorGradingPreset.NONE:
            result = self.apply_color_grading(result, self.config.color_grading)

        # Step 4: Final touches
        if self.config.reduce_banding:
            result = self.reduce_banding(result)

        if self.config.final_sharpen:
            result = self.final_sharpen(result)

        return result

    def remove_halos(
        self,
        image: "ndarray",
        original: "ndarray" | None = None,
        scale: int = 4,
    ) -> "ndarray":
        """Remove edge halos from upscaled image.

        Halos appear as bright/dark bands along edges, common in
        AI upscalers that over-enhance edges.

        Args:
            image: Upscaled image.
            original: Original image for reference.
            scale: Upscaling factor.

        Returns:
            Image with reduced halos.
        """
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0

        # Detect edges using Sobel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-6)

        # Create edge mask (where halos typically occur)
        edge_mask = (edge_magnitude > self.config.halo_threshold).astype(np.float32)

        # Dilate edge mask to cover halo region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        halo_region = cv2.dilate(edge_mask, kernel, iterations=2)

        # Apply guided filter to smooth halos while preserving edges
        # Using bilateral filter as an approximation
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)

        # Blend original and smoothed based on halo region
        halo_region_3ch = np.stack([halo_region] * 3, axis=-1)
        blend_strength = 0.5  # How much to blend smoothed version in halo regions

        result = image.astype(np.float32)
        smoothed_float = smoothed.astype(np.float32)

        # Only apply in halo regions, away from actual edges
        edge_mask_3ch = np.stack([edge_mask] * 3, axis=-1)
        halo_only = halo_region_3ch * (1 - edge_mask_3ch)

        result = result * (1 - halo_only * blend_strength) + smoothed_float * (halo_only * blend_strength)

        return np.clip(result, 0, 255).astype(np.uint8)

    def remove_ringing(self, image: "ndarray") -> "ndarray":
        """Remove ringing artifacts (Gibbs phenomenon).

        Ringing appears as repeated oscillations near sharp edges.

        Args:
            image: Upscaled image.

        Returns:
            Image with reduced ringing.
        """
        # Use median filter to reduce ringing while preserving edges
        # Apply only to high-frequency components

        # Extract high-frequency
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        high_freq = cv2.subtract(image, blurred)

        # Apply median filter to high-frequency
        high_freq_filtered = cv2.medianBlur(high_freq, 3)

        # Reconstruct
        result = cv2.add(blurred, high_freq_filtered)

        return result

    def remove_checkerboard(self, image: "ndarray") -> "ndarray":
        """Remove checkerboard artifacts from pixel shuffle upsampling.

        Args:
            image: Upscaled image.

        Returns:
            Image with reduced checkerboard patterns.
        """
        # Apply slight Gaussian blur to reduce checkerboard
        # while preserving most detail

        # Detect checkerboard pattern using DFT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)

        # Check for peaks at checkerboard frequencies
        h, w = gray.shape
        magnitude = np.abs(fshift)

        # Create notch filter for checkerboard frequencies
        # Checkerboard appears at (h/2, w/2) ± small offset
        crow, ccol = h // 2, w // 2
        mask = np.ones((h, w), np.float32)

        # Suppress potential checkerboard frequencies
        notch_size = 3
        for dy in [-1, 1]:
            for dx in [-1, 1]:
                cy = crow + dy * (h // 4)
                cx = ccol + dx * (w // 4)
                if 0 <= cy < h and 0 <= cx < w:
                    y1 = max(0, cy - notch_size)
                    y2 = min(h, cy + notch_size)
                    x1 = max(0, cx - notch_size)
                    x2 = min(w, cx + notch_size)
                    mask[y1:y2, x1:x2] = 0.5

        # Apply filter if significant checkerboard detected
        # For now, use simple spatial filtering
        result = cv2.bilateralFilter(image, 5, 30, 30)

        return result

    def enhance_details(self, image: "ndarray") -> "ndarray":
        """Enhance fine details in upscaled image.

        Args:
            image: Upscaled image.

        Returns:
            Detail-enhanced image.
        """
        strength = self.config.detail_strength

        # Convert to LAB for luminance processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_float = l.astype(np.float32)

        # Multi-scale detail enhancement
        details = []
        current = l_float.copy()

        for scale in [3, 5, 7]:
            blurred = cv2.GaussianBlur(current, (scale, scale), 0)
            detail = current - blurred
            details.append(detail * (1.0 / scale))
            current = blurred

        # Combine details with boosting
        total_detail = sum(details)
        enhanced_l = l_float + total_detail * strength * 50

        # Clamp and convert back
        enhanced_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)
        lab_enhanced = cv2.merge([enhanced_l, a, b])

        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def enhance_texture(self, image: "ndarray") -> "ndarray":
        """Enhance texture details using local contrast.

        Args:
            image: Upscaled image.

        Returns:
            Texture-enhanced image.
        """
        strength = self.config.texture_strength

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_float = l.astype(np.float32)

        # Calculate local mean and variance
        kernel_size = 15
        local_mean = cv2.blur(l_float, (kernel_size, kernel_size))
        local_sqr_mean = cv2.blur(l_float**2, (kernel_size, kernel_size))
        local_var = np.sqrt(np.maximum(local_sqr_mean - local_mean**2, 0))

        # Enhance local contrast
        # Normalize: (pixel - local_mean) / local_std * target_std + local_mean
        target_std = 30  # Target local standard deviation
        normalized = (l_float - local_mean) / (local_var + 1e-6)
        enhanced = normalized * (local_var + strength * target_std) + local_mean

        # Blend with original
        enhanced_l = l_float * (1 - strength) + enhanced * strength
        enhanced_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)

        lab_enhanced = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def auto_color_correct(self, image: "ndarray") -> "ndarray":
        """Automatic color correction.

        Args:
            image: Upscaled image.

        Returns:
            Color-corrected image.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Center a and b channels (remove color cast)
        a_centered = a.astype(np.float32) - 128
        b_centered = b.astype(np.float32) - 128

        # Preserve skin tones if enabled
        if self.config.preserve_skin_tones:
            # Detect skin-like regions (roughly)
            skin_mask = self._detect_skin(image)
            skin_mask = skin_mask.astype(np.float32) / 255.0

            # Only correct non-skin regions fully
            a_correction = np.mean(a_centered * (1 - skin_mask))
            b_correction = np.mean(b_centered * (1 - skin_mask))
        else:
            a_correction = np.mean(a_centered)
            b_correction = np.mean(b_centered)

        # Apply correction
        a_corrected = (a.astype(np.float32) - a_correction * 0.5).clip(0, 255).astype(np.uint8)
        b_corrected = (b.astype(np.float32) - b_correction * 0.5).clip(0, 255).astype(np.uint8)

        lab_corrected = cv2.merge([l, a_corrected, b_corrected])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    def _detect_skin(self, image: "ndarray") -> "ndarray":
        """Detect skin-like regions.

        Args:
            image: BGR image.

        Returns:
            Binary mask of skin regions.
        """
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)

        # Skin color range in YCrCb
        skin_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        return skin_mask

    def adjust_saturation(self, image: "ndarray", amount: float) -> "ndarray":
        """Adjust color saturation.

        Args:
            image: BGR image.
            amount: Saturation adjustment (-1.0 to 1.0).

        Returns:
            Saturation-adjusted image.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # Adjust saturation
        s = s * (1 + amount)
        s = np.clip(s, 0, 255)

        hsv_adjusted = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    def adjust_vibrance(self, image: "ndarray", amount: float) -> "ndarray":
        """Adjust vibrance (smart saturation that protects skin tones).

        Args:
            image: BGR image.
            amount: Vibrance adjustment (-1.0 to 1.0).

        Returns:
            Vibrance-adjusted image.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # Calculate saturation boost based on current saturation
        # Less saturated pixels get more boost
        saturation_factor = 1 - (s / 255.0)
        boost = amount * saturation_factor * 0.5

        # Apply boost
        s = s * (1 + boost)
        s = np.clip(s, 0, 255)

        hsv_adjusted = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    def apply_color_grading(
        self, image: "ndarray", preset: ColorGradingPreset
    ) -> "ndarray":
        """Apply color grading preset.

        Args:
            image: BGR image.
            preset: Color grading preset.

        Returns:
            Color-graded image.
        """
        if preset == ColorGradingPreset.NONE:
            return image

        result = image.copy()

        if preset == ColorGradingPreset.NATURAL:
            # Slight saturation boost, subtle contrast
            result = self.adjust_saturation(result, 0.1)
            result = self._adjust_contrast(result, 1.05)

        elif preset == ColorGradingPreset.VIVID:
            # Strong saturation and contrast
            result = self.adjust_saturation(result, 0.3)
            result = self._adjust_contrast(result, 1.15)
            result = self.adjust_vibrance(result, 0.2)

        elif preset == ColorGradingPreset.CINEMATIC:
            # Teal/orange split toning, lifted blacks
            result = self._apply_split_toning(result, (30, 150, 180), (255, 200, 150), 0.2)
            result = self._lift_blacks(result, 10)
            result = self._adjust_contrast(result, 1.1)

        elif preset == ColorGradingPreset.WARM:
            # Add warmth
            result = self._apply_color_temperature(result, 0.15)
            result = self.adjust_saturation(result, 0.05)

        elif preset == ColorGradingPreset.COOL:
            # Add coolness
            result = self._apply_color_temperature(result, -0.15)
            result = self.adjust_saturation(result, 0.05)

        elif preset == ColorGradingPreset.HDR:
            # HDR-like: local contrast, vibrance
            result = self.enhance_details(result)
            result = self.adjust_vibrance(result, 0.3)
            result = self._adjust_contrast(result, 1.1)

        return result

    def _adjust_contrast(self, image: "ndarray", factor: float) -> "ndarray":
        """Adjust contrast around midpoint.

        Args:
            image: BGR image.
            factor: Contrast factor (>1 increases, <1 decreases).

        Returns:
            Contrast-adjusted image.
        """
        result = image.astype(np.float32)
        result = (result - 127.5) * factor + 127.5
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_split_toning(
        self,
        image: "ndarray",
        shadow_color: tuple[int, int, int],
        highlight_color: tuple[int, int, int],
        strength: float,
    ) -> "ndarray":
        """Apply split toning (different colors for shadows/highlights).

        Args:
            image: BGR image.
            shadow_color: BGR color for shadows.
            highlight_color: BGR color for highlights.
            strength: Effect strength (0-1).

        Returns:
            Split-toned image.
        """
        # Create luminance mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Create shadow and highlight masks
        shadow_mask = 1 - gray
        highlight_mask = gray

        # Create color overlays
        result = image.astype(np.float32)
        shadow_overlay = np.full_like(result, shadow_color, dtype=np.float32)
        highlight_overlay = np.full_like(result, highlight_color, dtype=np.float32)

        # Apply overlays
        for i in range(3):
            result[:, :, i] += (shadow_overlay[:, :, i] - result[:, :, i]) * shadow_mask * strength * 0.5
            result[:, :, i] += (highlight_overlay[:, :, i] - result[:, :, i]) * highlight_mask * strength * 0.5

        return np.clip(result, 0, 255).astype(np.uint8)

    def _lift_blacks(self, image: "ndarray", amount: int) -> "ndarray":
        """Lift black levels.

        Args:
            image: BGR image.
            amount: Amount to lift (0-50).

        Returns:
            Image with lifted blacks.
        """
        result = image.astype(np.float32)
        result = result + amount
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_color_temperature(self, image: "ndarray", amount: float) -> "ndarray":
        """Apply color temperature adjustment.

        Args:
            image: BGR image.
            amount: Temperature adjustment (-1=cool, +1=warm).

        Returns:
            Temperature-adjusted image.
        """
        result = image.astype(np.float32)

        # Adjust blue and red channels
        result[:, :, 0] -= amount * 30  # Blue
        result[:, :, 2] += amount * 30  # Red

        return np.clip(result, 0, 255).astype(np.uint8)

    def reduce_banding(self, image: "ndarray") -> "ndarray":
        """Reduce color banding artifacts.

        Args:
            image: Upscaled image.

        Returns:
            Image with reduced banding.
        """
        # Add slight dithering noise to break up bands
        noise = np.random.normal(0, 1.5, image.shape).astype(np.float32)
        result = image.astype(np.float32) + noise

        return np.clip(result, 0, 255).astype(np.uint8)

    def final_sharpen(self, image: "ndarray") -> "ndarray":
        """Apply final sharpening pass.

        Args:
            image: Processed image.

        Returns:
            Sharpened image.
        """
        amount = self.config.sharpen_amount
        radius = self.config.sharpen_radius

        # Unsharp mask
        kernel_size = int(radius * 4) * 2 + 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)

        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)


# =============================================================================
# Convenience functions
# =============================================================================


def postprocess_upscaled(
    image: "ndarray",
    original: "ndarray" | None = None,
    scale: int = 4,
    preset: str = "balanced",
) -> "ndarray":
    """Apply recommended postprocessing for upscaled images.

    Args:
        image: Upscaled image.
        original: Original image (optional).
        scale: Upscaling factor used.
        preset: Preset name - "minimal", "balanced", "quality".

    Returns:
        Postprocessed image.
    """
    if preset == "minimal":
        config = PostprocessConfig(
            remove_halos=True,
            remove_ringing=False,
            enhance_details=False,
            final_sharpen=True,
            sharpen_amount=0.1,
        )
    elif preset == "balanced":
        config = PostprocessConfig(
            remove_halos=True,
            remove_ringing=True,
            enhance_details=True,
            detail_strength=0.2,
            final_sharpen=True,
            sharpen_amount=0.15,
        )
    elif preset == "quality":
        config = PostprocessConfig(
            remove_halos=True,
            remove_ringing=True,
            remove_checkerboard=True,
            enhance_details=True,
            detail_strength=0.3,
            texture_enhancement=True,
            texture_strength=0.2,
            final_sharpen=True,
            sharpen_amount=0.2,
            reduce_banding=True,
        )
    else:
        config = PostprocessConfig()

    postprocessor = ImagePostprocessor(config)
    return postprocessor.process(image, original, scale)


def apply_color_grade(
    image: "ndarray",
    preset: str | ColorGradingPreset = "natural",
) -> "ndarray":
    """Apply color grading to image.

    Args:
        image: Input image.
        preset: Preset name or ColorGradingPreset.

    Returns:
        Color-graded image.
    """
    if isinstance(preset, str):
        preset_map = {
            "none": ColorGradingPreset.NONE,
            "natural": ColorGradingPreset.NATURAL,
            "vivid": ColorGradingPreset.VIVID,
            "cinematic": ColorGradingPreset.CINEMATIC,
            "warm": ColorGradingPreset.WARM,
            "cool": ColorGradingPreset.COOL,
            "hdr": ColorGradingPreset.HDR,
        }
        preset = preset_map.get(preset.lower(), ColorGradingPreset.NONE)

    config = PostprocessConfig(
        remove_halos=False,
        remove_ringing=False,
        enhance_details=False,
        final_sharpen=False,
        color_grading=preset,
    )

    postprocessor = ImagePostprocessor(config)
    return postprocessor.apply_color_grading(image, preset)
