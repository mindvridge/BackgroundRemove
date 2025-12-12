"""Video processing pipeline module.

This module provides a Producer-Consumer pattern pipeline for video processing:
- VideoReader: Thread-based frame reading with queue buffering
- FrameProcessor: Batch inference with seq_chunk processing
- VideoWriter: Thread-based FFmpeg encoding
- VideoPipeline: End-to-end pipeline orchestration
"""

from src.pipeline.pipeline import PipelineConfig, PipelineResult, VideoPipeline
from src.pipeline.processor import (
    FrameProcessor,
    OutputType,
    ProcessedFrame,
    ProcessorConfig,
)
from src.pipeline.reader import ReaderConfig, VideoMetadata, VideoReader
from src.pipeline.writer import ImageSequenceWriter, VideoWriter, WriterConfig

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
]
