"""Video processing pipeline module.

This module provides a comprehensive video processing pipeline with:

Core Components:
- VideoReader: Thread-based frame reading with queue buffering
- FrameProcessor: Batch inference with seq_chunk processing
- VideoWriter: Thread-based FFmpeg encoding
- VideoPipeline: End-to-end pipeline orchestration

Quality Enhancement:
- TemporalConsistency: Frame-to-frame consistency for flicker-free video
- EdgeRefiner: Advanced edge handling for hair/fur details
- DeepMatting: MODNet, ViTMatte for high-quality alpha
- MattingEnsemble: Multi-model combination for best results
- DepthAwareProcessor: Depth-guided foreground separation
- SAMSegmenter: Segment Anything Model integration
- AlphaSuperResolution: High-quality alpha upscaling

Output formats supported:
- MP4 (H.264)
- WebM (VP9 with alpha)
- ProRes 4444 (with alpha)
- Green screen compositing
- Custom background replacement

Ultimate Pipeline (9.8/10 quality):
- Combines all techniques for maximum video matting quality
- Multiple presets: Fast, Balanced, Quality, Maximum, Ultra
"""

from src.pipeline.output import (
    CODEC_CONFIGS,
    CodecConfig,
    Compositor,
    FFmpegEncoder,
    GreenScreenCompositor,
    ImageBackgroundCompositor,
    OutputConfig,
    OutputFormat,
    VideoBackgroundCompositor,
    create_compositor,
)
from src.pipeline.pipeline import PipelineConfig, PipelineResult, VideoPipeline
from src.pipeline.processor import (
    FrameProcessor,
    OutputType,
    ProcessedFrame,
    ProcessorConfig,
)
from src.pipeline.reader import ReaderConfig, VideoMetadata, VideoReader
from src.pipeline.writer import (
    AdvancedVideoWriter,
    ImageSequenceWriter,
    VideoWriter,
    WriterConfig,
)
from src.pipeline.temporal import (
    TemporalConfig,
    TemporalConsistency,
    TemporalFilter,
    TemporalMethod,
    TemporalState,
    apply_temporal_consistency,
    temporal_smooth,
)
from src.pipeline.edge_refine import (
    EdgeConfig,
    EdgeMethod,
    EdgeRefiner,
    GuidedFilter,
    KNNMatting,
    TrimapGenerator,
    generate_trimap,
    guided_filter,
    refine_edges,
)
from src.pipeline.enhanced import (
    EnhancedConfig,
    EnhancedFrame,
    EnhancedPipeline,
    EnhancedProcessor,
    QualityPreset,
    get_preset_config,
    process_video_enhanced,
)

# Lazy imports for optional advanced modules
def __getattr__(name: str):
    """Lazy import for optional modules."""
    # Deep matting
    if name in ("MattingConfig", "MattingModel", "MODNetMatting", "ViTMatteMatting",
                "MatteAnythingMatting", "create_matting_model", "refine_alpha_with_matting"):
        from src.pipeline import deep_matting
        return getattr(deep_matting, name)

    # Ensemble
    if name in ("EnsembleConfig", "EnsembleMethod", "MattingEnsemble", "MattingResult",
                "QualityAssessor", "create_ensemble_from_models"):
        from src.pipeline import matting_ensemble
        return getattr(matting_ensemble, name)

    # Alpha SR
    if name in ("AlphaSRConfig", "AlphaSRMethod", "AlphaSuperResolution",
                "LowResProcessor", "upscale_alpha"):
        from src.pipeline import alpha_sr
        return getattr(alpha_sr, name)

    # Depth
    if name in ("DepthConfig", "DepthModel", "DepthEstimator", "DepthAwareProcessor",
                "estimate_depth", "refine_with_depth"):
        from src.pipeline import depth_aware
        return getattr(depth_aware, name)

    # SAM
    if name in ("SAMConfig", "SAMModel", "PromptType", "SAMSegmenter",
                "segment_with_sam", "refine_alpha_sam"):
        from src.pipeline import sam_segmentation
        return getattr(sam_segmentation, name)

    # Ultimate pipeline
    if name in ("UltimateConfig", "UltimatePreset", "UltimateFrame",
                "UltimateVideoProcessor", "UltimateVideoPipeline", "process_video_ultimate"):
        from src.pipeline import ultimate_video
        return getattr(ultimate_video, name)

    # Iterative refinement
    if name in ("IterativeConfig", "IterativeRefinement", "RefinementStrategy",
                "MultiPassRefiner", "CoarseToFineRefiner", "FeedbackLoopRefiner",
                "CascadeRefiner", "ProgressiveRefiner", "iterative_refine"):
        from src.pipeline import iterative_refine
        return getattr(iterative_refine, name)

    # GAN refinement
    if name in ("GANConfig", "GANModel", "GANRefiner", "HairRefiner",
                "EdgeDetector", "SimpleGenerator", "PatchDiscriminator",
                "gan_refine", "refine_hair"):
        from src.pipeline import gan_refine
        return getattr(gan_refine, name)

    # Video enhancement
    if name in ("VideoEnhanceConfig", "VideoEnhancer", "VideoMethod",
                "FrameMemory", "TemporalAttention", "MotionCompensator",
                "MultiFrameFusion", "video_enhance"):
        from src.pipeline import video_enhance
        return getattr(video_enhance, name)

    # Neural rendering
    if name in ("NeuralRenderConfig", "NeuralRenderer", "NeuralMethod",
                "EdgeInpainter", "HaloRemover", "DiffusionRefiner",
                "LatentBlender", "ForegroundEnhancer",
                "neural_refine", "remove_halos"):
        from src.pipeline import neural_render
        return getattr(neural_render, name)

    # Perfect pipeline
    if name in ("PerfectConfig", "PerfectPreset", "PerfectFrame",
                "PerfectProcessor", "PerfectPipeline", "QualityEstimator",
                "process_video_perfect", "get_perfect_config"):
        from src.pipeline import perfect_pipeline
        return getattr(perfect_pipeline, name)

    raise AttributeError(f"module 'src.pipeline' has no attribute '{name}'")


__all__ = [
    # Pipeline
    "VideoPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Reader
    "VideoReader",
    "ReaderConfig",
    "VideoMetadata",
    # Processor
    "FrameProcessor",
    "ProcessorConfig",
    "ProcessedFrame",
    "OutputType",
    # Writer
    "VideoWriter",
    "WriterConfig",
    "ImageSequenceWriter",
    "AdvancedVideoWriter",
    # Output formats
    "OutputFormat",
    "OutputConfig",
    "CodecConfig",
    "CODEC_CONFIGS",
    "FFmpegEncoder",
    # Compositors
    "Compositor",
    "GreenScreenCompositor",
    "ImageBackgroundCompositor",
    "VideoBackgroundCompositor",
    "create_compositor",
    # Temporal Consistency
    "TemporalConsistency",
    "TemporalConfig",
    "TemporalMethod",
    "TemporalState",
    "TemporalFilter",
    "apply_temporal_consistency",
    "temporal_smooth",
    # Edge Refinement
    "EdgeRefiner",
    "EdgeConfig",
    "EdgeMethod",
    "GuidedFilter",
    "TrimapGenerator",
    "KNNMatting",
    "refine_edges",
    "guided_filter",
    "generate_trimap",
    # Enhanced Pipeline
    "EnhancedPipeline",
    "EnhancedProcessor",
    "EnhancedConfig",
    "EnhancedFrame",
    "QualityPreset",
    "get_preset_config",
    "process_video_enhanced",
    # Deep Matting (lazy)
    "MattingConfig",
    "MattingModel",
    "MODNetMatting",
    "ViTMatteMatting",
    "MatteAnythingMatting",
    "create_matting_model",
    "refine_alpha_with_matting",
    # Matting Ensemble (lazy)
    "EnsembleConfig",
    "EnsembleMethod",
    "MattingEnsemble",
    "MattingResult",
    "QualityAssessor",
    "create_ensemble_from_models",
    # Alpha SR (lazy)
    "AlphaSRConfig",
    "AlphaSRMethod",
    "AlphaSuperResolution",
    "LowResProcessor",
    "upscale_alpha",
    # Depth (lazy)
    "DepthConfig",
    "DepthModel",
    "DepthEstimator",
    "DepthAwareProcessor",
    "estimate_depth",
    "refine_with_depth",
    # SAM (lazy)
    "SAMConfig",
    "SAMModel",
    "PromptType",
    "SAMSegmenter",
    "segment_with_sam",
    "refine_alpha_sam",
    # Ultimate Pipeline (lazy)
    "UltimateConfig",
    "UltimatePreset",
    "UltimateFrame",
    "UltimateVideoProcessor",
    "UltimateVideoPipeline",
    "process_video_ultimate",
    # Iterative Refinement (lazy)
    "IterativeConfig",
    "IterativeRefinement",
    "RefinementStrategy",
    "MultiPassRefiner",
    "CoarseToFineRefiner",
    "FeedbackLoopRefiner",
    "CascadeRefiner",
    "ProgressiveRefiner",
    "iterative_refine",
    # GAN Refinement (lazy)
    "GANConfig",
    "GANModel",
    "GANRefiner",
    "HairRefiner",
    "gan_refine",
    "refine_hair",
    # Video Enhancement (lazy)
    "VideoEnhanceConfig",
    "VideoEnhancer",
    "VideoMethod",
    "FrameMemory",
    "TemporalAttention",
    "MotionCompensator",
    "MultiFrameFusion",
    "video_enhance",
    # Neural Rendering (lazy)
    "NeuralRenderConfig",
    "NeuralRenderer",
    "NeuralMethod",
    "EdgeInpainter",
    "HaloRemover",
    "DiffusionRefiner",
    "LatentBlender",
    "ForegroundEnhancer",
    "neural_refine",
    "remove_halos",
    # Perfect Pipeline (lazy)
    "PerfectConfig",
    "PerfectPreset",
    "PerfectFrame",
    "PerfectProcessor",
    "PerfectPipeline",
    "QualityEstimator",
    "process_video_perfect",
    "get_perfect_config",
]
