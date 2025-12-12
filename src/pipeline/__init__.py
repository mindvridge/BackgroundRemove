"""Video processing pipeline module.

This module provides a Producer-Consumer pattern pipeline for video processing:
- VideoReader: Thread-based frame reading with queue buffering
- FrameProcessor: Batch inference with seq_chunk processing
- VideoWriter: Thread-based FFmpeg encoding
- VideoPipeline: End-to-end pipeline orchestration

Output formats supported:
- MP4 (H.264)
- WebM (VP9 with alpha)
- ProRes 4444 (with alpha)
- Green screen compositing
- Custom background replacement
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
]
