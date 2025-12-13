"""Video processing pipeline module.

This module provides a Producer-Consumer pattern pipeline for video processing:
- VideoReader: Thread-based frame reading with queue buffering
- FrameProcessor: Batch inference with seq_chunk processing
- VideoWriter: Thread-based FFmpeg encoding
- VideoPipeline: End-to-end pipeline orchestration
- TemporalConsistency: Frame-to-frame consistency for flicker-free video
- EdgeRefiner: Advanced edge handling for hair/fur details

Output formats supported:
- MP4 (H.264)
- WebM (VP9 with alpha)
- ProRes 4444 (with alpha)
- Green screen compositing
- Custom background replacement

Quality Enhancement:
- Temporal consistency via optical flow and EMA smoothing
- Edge refinement via guided filter and alpha matting
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
]
