"""Ultimate video pipeline with all quality enhancements.

Integrates all advanced techniques for maximum video matting quality:
- Deep learning matting (MODNet, ViTMatte)
- Multi-model ensemble
- Temporal consistency
- Edge refinement
- Depth-aware processing
- SAM segmentation
- Alpha super-resolution

Target quality: 9.8/10
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from numpy import ndarray
    from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class UltimatePreset(Enum):
    """Quality presets for ultimate pipeline."""

    FAST = "fast"  # Real-time capable (~25fps)
    BALANCED = "balanced"  # Good balance (~10fps)
    QUALITY = "quality"  # High quality (~5fps)
    MAXIMUM = "maximum"  # Maximum quality (~2fps)
    ULTRA = "ultra"  # Extreme quality (~0.5fps)


@dataclass
class UltimateConfig:
    """Configuration for ultimate video pipeline."""

    preset: UltimatePreset = UltimatePreset.BALANCED

    # Component toggles
    enable_deep_matting: bool = True
    enable_ensemble: bool = True
    enable_temporal: bool = True
    enable_edge_refine: bool = True
    enable_depth: bool = True
    enable_sam: bool = True
    enable_alpha_sr: bool = False  # For upscaling output

    # Base model
    use_base_model: bool = True  # Use RVM as base

    # Deep matting settings
    deep_matting_model: str = "modnet"  # "modnet", "vitmatte"

    # Ensemble settings
    ensemble_method: str = "quality_guided"

    # Temporal settings
    temporal_method: str = "optical_flow"

    # Edge refinement settings
    edge_method: str = "combined"

    # Depth settings
    depth_model: str = "midas"

    # SAM settings
    sam_model: str = "sam_vit_b"

    # Output settings
    output_scale: float = 1.0  # Output resolution scale


def get_preset_config(preset: UltimatePreset) -> UltimateConfig:
    """Get configuration for preset.

    Args:
        preset: Quality preset.

    Returns:
        Configured UltimateConfig.
    """
    if preset == UltimatePreset.FAST:
        return UltimateConfig(
            preset=preset,
            enable_deep_matting=False,
            enable_ensemble=False,
            enable_temporal=True,
            enable_edge_refine=True,
            enable_depth=False,
            enable_sam=False,
            temporal_method="ema",
            edge_method="fast",
        )

    elif preset == UltimatePreset.BALANCED:
        return UltimateConfig(
            preset=preset,
            enable_deep_matting=True,
            enable_ensemble=False,
            enable_temporal=True,
            enable_edge_refine=True,
            enable_depth=False,
            enable_sam=False,
            deep_matting_model="modnet",
            temporal_method="optical_flow",
            edge_method="guided_filter",
        )

    elif preset == UltimatePreset.QUALITY:
        return UltimateConfig(
            preset=preset,
            enable_deep_matting=True,
            enable_ensemble=True,
            enable_temporal=True,
            enable_edge_refine=True,
            enable_depth=True,
            enable_sam=False,
            deep_matting_model="modnet",
            ensemble_method="quality_guided",
            temporal_method="bidirectional",
            edge_method="combined",
            depth_model="midas",
        )

    elif preset == UltimatePreset.MAXIMUM:
        return UltimateConfig(
            preset=preset,
            enable_deep_matting=True,
            enable_ensemble=True,
            enable_temporal=True,
            enable_edge_refine=True,
            enable_depth=True,
            enable_sam=True,
            deep_matting_model="vitmatte",
            ensemble_method="laplacian_pyramid",
            temporal_method="adaptive",
            edge_method="combined",
            depth_model="midas",
            sam_model="sam_vit_b",
        )

    else:  # ULTRA
        return UltimateConfig(
            preset=preset,
            enable_deep_matting=True,
            enable_ensemble=True,
            enable_temporal=True,
            enable_edge_refine=True,
            enable_depth=True,
            enable_sam=True,
            enable_alpha_sr=True,
            deep_matting_model="vitmatte",
            ensemble_method="laplacian_pyramid",
            temporal_method="adaptive",
            edge_method="combined",
            depth_model="midas",
            sam_model="sam_vit_h",
        )


@dataclass
class UltimateFrame:
    """Container for ultimate pipeline frame result."""

    frame_index: int
    original: ndarray
    foreground: ndarray
    alpha: ndarray
    refined_alpha: ndarray
    output: ndarray
    depth: ndarray | None = None
    processing_time: float = 0.0


class UltimateVideoProcessor:
    """Ultimate video processor with all enhancements.

    Combines all available techniques for maximum quality
    video background removal.

    Example:
        >>> processor = UltimateVideoProcessor(model, UltimatePreset.QUALITY)
        >>> for result in processor.process_frames(frames):
        ...     save_frame(result.output)
    """

    def __init__(
        self,
        base_model: BaseModel,
        config: UltimateConfig | None = None,
        preset: UltimatePreset | None = None,
    ) -> None:
        """Initialize ultimate processor.

        Args:
            base_model: Base background removal model (e.g., RVM).
            config: Configuration (overrides preset).
            preset: Quality preset.
        """
        self.base_model = base_model

        if config is not None:
            self.config = config
        elif preset is not None:
            self.config = get_preset_config(preset)
        else:
            self.config = UltimateConfig()

        self._components_loaded = False
        self._frame_index = 0
        self._processed_count = 0

        # Component instances
        self._deep_matting = None
        self._ensemble = None
        self._temporal = None
        self._edge_refiner = None
        self._depth_processor = None
        self._sam_segmenter = None
        self._alpha_sr = None

        logger.info(f"Ultimate processor initialized: {self.config.preset.value}")

    def _load_components(self) -> None:
        """Load all enabled components."""
        if self._components_loaded:
            return

        logger.info("Loading ultimate pipeline components...")

        # Deep matting
        if self.config.enable_deep_matting:
            try:
                from src.pipeline.deep_matting import MattingConfig, MattingModel, create_matting_model

                model_map = {
                    "modnet": MattingModel.MODNET,
                    "vitmatte": MattingModel.VITMATTE,
                }
                model_type = model_map.get(self.config.deep_matting_model, MattingModel.MODNET)

                config = MattingConfig(model=model_type)
                self._deep_matting = create_matting_model(config)
                self._deep_matting.load()
                logger.info(f"Deep matting loaded: {self.config.deep_matting_model}")

            except Exception as e:
                logger.warning(f"Could not load deep matting: {e}")
                self._deep_matting = None

        # Ensemble
        if self.config.enable_ensemble:
            try:
                from src.pipeline.matting_ensemble import EnsembleConfig, EnsembleMethod, MattingEnsemble

                method_map = {
                    "weighted_average": EnsembleMethod.WEIGHTED_AVERAGE,
                    "quality_guided": EnsembleMethod.QUALITY_GUIDED,
                    "laplacian_pyramid": EnsembleMethod.LAPLACIAN_PYRAMID,
                }
                method = method_map.get(self.config.ensemble_method, EnsembleMethod.QUALITY_GUIDED)

                config = EnsembleConfig(method=method)
                self._ensemble = MattingEnsemble(config)

                # Add base model
                if self.config.use_base_model:
                    self._ensemble.add_model("base", self.base_model, weight=0.4)

                # Add deep matting if available
                if self._deep_matting:
                    self._ensemble.add_model("deep", self._deep_matting, weight=0.6)

                logger.info(f"Ensemble loaded: {self.config.ensemble_method}")

            except Exception as e:
                logger.warning(f"Could not load ensemble: {e}")
                self._ensemble = None

        # Temporal consistency
        if self.config.enable_temporal:
            try:
                from src.pipeline.temporal import TemporalConfig, TemporalConsistency, TemporalMethod

                method_map = {
                    "ema": TemporalMethod.EMA,
                    "optical_flow": TemporalMethod.OPTICAL_FLOW,
                    "bidirectional": TemporalMethod.BIDIRECTIONAL,
                    "adaptive": TemporalMethod.ADAPTIVE,
                }
                method = method_map.get(self.config.temporal_method, TemporalMethod.OPTICAL_FLOW)

                config = TemporalConfig(method=method)
                self._temporal = TemporalConsistency(config)
                logger.info(f"Temporal loaded: {self.config.temporal_method}")

            except Exception as e:
                logger.warning(f"Could not load temporal: {e}")
                self._temporal = None

        # Edge refinement
        if self.config.enable_edge_refine:
            try:
                from src.pipeline.edge_refine import EdgeConfig, EdgeMethod, EdgeRefiner

                method_map = {
                    "fast": EdgeMethod.FAST,
                    "guided_filter": EdgeMethod.GUIDED_FILTER,
                    "combined": EdgeMethod.COMBINED,
                }
                method = method_map.get(self.config.edge_method, EdgeMethod.COMBINED)

                config = EdgeConfig(method=method)
                self._edge_refiner = EdgeRefiner(config)
                logger.info(f"Edge refiner loaded: {self.config.edge_method}")

            except Exception as e:
                logger.warning(f"Could not load edge refiner: {e}")
                self._edge_refiner = None

        # Depth processing
        if self.config.enable_depth:
            try:
                from src.pipeline.depth_aware import DepthAwareProcessor, DepthConfig, DepthModel

                model_map = {
                    "midas": DepthModel.MIDAS,
                    "midas_small": DepthModel.MIDAS_SMALL,
                    "zoedepth": DepthModel.ZOEDEPTH,
                }
                model = model_map.get(self.config.depth_model, DepthModel.MIDAS)

                config = DepthConfig(model=model)
                self._depth_processor = DepthAwareProcessor(config)
                self._depth_processor.load()
                logger.info(f"Depth processor loaded: {self.config.depth_model}")

            except Exception as e:
                logger.warning(f"Could not load depth processor: {e}")
                self._depth_processor = None

        # SAM segmentation
        if self.config.enable_sam:
            try:
                from src.pipeline.sam_segmentation import SAMConfig, SAMModel, SAMSegmenter

                model_map = {
                    "sam_vit_b": SAMModel.SAM_VIT_B,
                    "sam_vit_l": SAMModel.SAM_VIT_L,
                    "sam_vit_h": SAMModel.SAM_VIT_H,
                    "fast_sam": SAMModel.FAST_SAM,
                }
                model = model_map.get(self.config.sam_model, SAMModel.SAM_VIT_B)

                config = SAMConfig(model=model)
                self._sam_segmenter = SAMSegmenter(config)
                self._sam_segmenter.load()
                logger.info(f"SAM loaded: {self.config.sam_model}")

            except Exception as e:
                logger.warning(f"Could not load SAM: {e}")
                self._sam_segmenter = None

        # Alpha super-resolution
        if self.config.enable_alpha_sr:
            try:
                from src.pipeline.alpha_sr import AlphaSRConfig, AlphaSRMethod, AlphaSuperResolution

                config = AlphaSRConfig(method=AlphaSRMethod.GUIDED)
                self._alpha_sr = AlphaSuperResolution(config)
                logger.info("Alpha SR loaded")

            except Exception as e:
                logger.warning(f"Could not load alpha SR: {e}")
                self._alpha_sr = None

        self._components_loaded = True
        logger.info("All components loaded")

    def reset(self) -> None:
        """Reset processor state for new video."""
        self._frame_index = 0
        self._processed_count = 0
        self.base_model.reset_state()

        if self._temporal:
            self._temporal.reset()

        logger.debug("Ultimate processor reset")

    @property
    def processed_count(self) -> int:
        """Number of frames processed."""
        return self._processed_count

    def _create_output(
        self,
        foreground: ndarray,
        alpha: ndarray,
    ) -> ndarray:
        """Create RGBA output.

        Args:
            foreground: Foreground image.
            alpha: Alpha matte.

        Returns:
            RGBA output.
        """
        if len(foreground.shape) == 2:
            foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)

        return np.dstack([foreground, alpha])

    @torch.inference_mode()
    def process_frame(self, frame: ndarray) -> UltimateFrame:
        """Process single frame with all enhancements.

        Args:
            frame: Input frame (BGR).

        Returns:
            UltimateFrame with all results.
        """
        if not self._components_loaded:
            self._load_components()

        start_time = time.time()

        # Step 1: Base inference
        if self._ensemble:
            foreground, alpha = self._ensemble.predict(frame)
        else:
            foreground, alpha = self.base_model.inference(frame)

        # Step 2: Deep matting refinement (if not in ensemble)
        if self._deep_matting and not self._ensemble:
            try:
                _, deep_alpha = self._deep_matting.predict(frame)
                # Blend in edge regions
                edges = cv2.Canny(alpha, 50, 150)
                edge_region = cv2.dilate(edges, None, iterations=3)
                edge_mask = edge_region.astype(np.float32) / 255.0

                alpha_float = alpha.astype(np.float32) / 255.0
                deep_float = deep_alpha.astype(np.float32) / 255.0
                alpha = ((edge_mask * deep_float + (1 - edge_mask) * alpha_float) * 255).astype(np.uint8)
            except Exception as e:
                logger.debug(f"Deep matting failed: {e}")

        # Step 3: Depth-aware refinement
        depth = None
        if self._depth_processor:
            try:
                depth = self._depth_processor.estimate_depth(frame)
                alpha = self._depth_processor.refine_alpha_with_depth(frame, alpha, depth)
            except Exception as e:
                logger.debug(f"Depth processing failed: {e}")

        # Step 4: SAM refinement
        if self._sam_segmenter:
            try:
                alpha = self._sam_segmenter.refine_alpha_with_sam(frame, alpha)
            except Exception as e:
                logger.debug(f"SAM refinement failed: {e}")

        # Step 5: Edge refinement
        if self._edge_refiner:
            alpha = self._edge_refiner.refine(frame, alpha)

        # Step 6: Temporal consistency
        if self._temporal:
            alpha = self._temporal.process(frame, alpha)

        refined_alpha = alpha

        # Step 7: Alpha super-resolution (if enabled)
        if self._alpha_sr and self.config.output_scale > 1.0:
            target_h = int(frame.shape[0] * self.config.output_scale)
            target_w = int(frame.shape[1] * self.config.output_scale)
            refined_alpha = self._alpha_sr.upscale(
                refined_alpha,
                guide=frame,
                target_size=(target_w, target_h),
            )
            foreground = cv2.resize(foreground, (target_w, target_h))
            frame = cv2.resize(frame, (target_w, target_h))

        # Create output
        output = self._create_output(foreground, refined_alpha)

        processing_time = time.time() - start_time

        result = UltimateFrame(
            frame_index=self._frame_index,
            original=frame,
            foreground=foreground,
            alpha=alpha,
            refined_alpha=refined_alpha,
            output=output,
            depth=depth,
            processing_time=processing_time,
        )

        self._frame_index += 1
        self._processed_count += 1

        return result

    def process_frames(
        self,
        frame_iterator: Iterator[ndarray],
        progress_callback: Callable[[int, int | None, float], None] | None = None,
        total_frames: int | None = None,
    ) -> Iterator[UltimateFrame]:
        """Process frames with ultimate pipeline.

        Args:
            frame_iterator: Frame iterator.
            progress_callback: Optional callback(current, total, fps).
            total_frames: Optional total count.

        Yields:
            UltimateFrame for each processed frame.
        """
        start_time = time.time()

        for frame in frame_iterator:
            result = self.process_frame(frame)

            if progress_callback:
                elapsed = time.time() - start_time
                fps = self._processed_count / elapsed if elapsed > 0 else 0
                progress_callback(self._processed_count, total_frames, fps)

            yield result

        logger.info(
            f"Ultimate processing complete: {self._processed_count} frames, "
            f"avg {self._processed_count / (time.time() - start_time):.1f} fps"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"UltimateVideoProcessor(preset={self.config.preset.value})"


class UltimateVideoPipeline:
    """End-to-end ultimate video pipeline.

    Complete pipeline for highest quality video background removal.

    Example:
        >>> pipeline = UltimateVideoPipeline(model, UltimatePreset.QUALITY)
        >>> result = pipeline.process("input.mp4", "output.mp4")
    """

    def __init__(
        self,
        model: BaseModel,
        preset: UltimatePreset = UltimatePreset.BALANCED,
        config: UltimateConfig | None = None,
    ) -> None:
        """Initialize ultimate pipeline.

        Args:
            model: Base model.
            preset: Quality preset.
            config: Optional config override.
        """
        self.model = model
        self._processor = UltimateVideoProcessor(
            model,
            config=config,
            preset=preset if config is None else None,
        )
        self._progress_callback = None

    def set_progress_callback(
        self,
        callback: Callable[[int, int, float], None] | None,
    ) -> None:
        """Set progress callback.

        Args:
            callback: Function(current, total, fps).
        """
        self._progress_callback = callback

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> dict:
        """Process video file.

        Args:
            input_path: Input video.
            output_path: Output video.

        Returns:
            Processing statistics.
        """
        from src.pipeline.reader import ReaderConfig, VideoReader
        from src.pipeline.writer import VideoWriter, WriterConfig

        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Ultimate processing: {input_path}")
        start_time = time.time()

        reader = None
        writer = None
        frames_processed = 0

        try:
            if not self.model.is_loaded:
                self.model.load()

            self._processor.reset()

            # Setup reader
            reader = VideoReader(input_path, ReaderConfig(queue_size=32))
            reader.start()

            while reader.metadata is None and reader.is_running:
                time.sleep(0.01)

            if reader.metadata is None:
                raise RuntimeError("Failed to read video")

            metadata = reader.metadata

            # Setup writer
            writer_config = WriterConfig(queue_size=32, fps=metadata.fps)
            writer = VideoWriter(
                output_path,
                metadata.width,
                metadata.height,
                writer_config,
                has_alpha=True,
            )
            writer.start()

            # Process
            def progress_cb(current, total, fps):
                nonlocal frames_processed
                frames_processed = current
                if self._progress_callback:
                    self._progress_callback(current, total or 0, fps)

            for result in self._processor.process_frames(
                iter(reader),
                progress_callback=progress_cb,
                total_frames=metadata.frame_count,
            ):
                writer.write(result.output)

            frames_processed = self._processor.processed_count

        finally:
            if reader:
                reader.stop()
            if writer:
                writer.stop()

        elapsed = time.time() - start_time
        fps = frames_processed / elapsed if elapsed > 0 else 0

        logger.info(
            f"Ultimate processing complete: {frames_processed} frames "
            f"in {elapsed:.1f}s ({fps:.1f} fps)"
        )

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "frames_processed": frames_processed,
            "duration_seconds": elapsed,
            "fps_achieved": fps,
            "preset": self._processor.config.preset.value,
            "success": True,
        }


def process_video_ultimate(
    model: BaseModel,
    input_path: str | Path,
    output_path: str | Path,
    preset: UltimatePreset = UltimatePreset.BALANCED,
) -> dict:
    """Convenience function for ultimate video processing.

    Args:
        model: Base model.
        input_path: Input video.
        output_path: Output video.
        preset: Quality preset.

    Returns:
        Processing statistics.
    """
    pipeline = UltimateVideoPipeline(model, preset=preset)
    return pipeline.process(input_path, output_path)
