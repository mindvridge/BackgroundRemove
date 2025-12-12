"""Main entry point for the background removal application."""

from __future__ import annotations

import argparse
import logging
import sys


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Video Background Removal Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of GUI",
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input video file (CLI mode)",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output video file (CLI mode)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["mp4", "webm", "prores", "green"],
        default="mp4",
        help="Output format",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["mobilenetv3", "resnet50"],
        default="mobilenetv3",
        help="Model variant",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 inference (CUDA only)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """Run in CLI mode.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from pathlib import Path

    from src.models.rvm import RVMConfig, RVMModel
    from src.pipeline.output import OutputConfig, OutputFormat
    from src.pipeline.pipeline import PipelineConfig, VideoPipeline

    # Validate input/output
    if not args.input:
        print("Error: --input is required in CLI mode", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path.with_stem(
        f"{input_path.stem}_removed"
    )

    # Map format string to OutputFormat
    format_map = {
        "mp4": OutputFormat.MP4_H264,
        "webm": OutputFormat.WEBM_VP9,
        "prores": OutputFormat.PRORES_4444,
        "green": OutputFormat.GREEN_SCREEN,
    }
    output_format = format_map.get(args.format, OutputFormat.MP4_H264)

    # Create model
    print(f"Loading model ({args.model})...")
    model_config = RVMConfig(variant=args.model, fp16=args.fp16)
    model = RVMModel(model_config)
    model.load()

    # Create pipeline
    from src.pipeline.processor import OutputType

    pipeline_config = PipelineConfig(
        output_type=OutputType.COMPOSITE,
    )
    pipeline = VideoPipeline(model, pipeline_config)

    # Process
    print(f"Processing: {input_path} -> {output_path}")
    result = pipeline.process(input_path, output_path)

    if result.success:
        print(f"\nComplete!")
        print(f"  Frames: {result.frames_processed}")
        print(f"  Time: {result.duration_seconds:.1f}s")
        print(f"  Speed: {result.fps_achieved:.1f} fps")
        print(f"  Output: {result.output_path}")
        return 0
    else:
        print(f"\nError: {result.error}", file=sys.stderr)
        return 1


def run_gui(args: argparse.Namespace) -> int:
    """Run in GUI mode.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from src.gui.app import run_app
    from src.models.rvm import RVMConfig, RVMModel

    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Create model
    model_config = RVMConfig(variant=args.model, fp16=args.fp16)
    model = RVMModel(model_config)

    return run_app(model=model, log_level=log_level)


def main() -> int:
    """Application entry point.

    Returns:
        Exit code.
    """
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cli:
        return run_cli(args)
    else:
        return run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
