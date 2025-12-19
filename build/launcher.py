#!/usr/bin/env python3
"""Auto-install launcher for BackgroundRemove application.

This launcher:
1. Checks for required dependencies
2. Downloads missing model files
3. Verifies FFmpeg installation
4. Launches the main application
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


# Application directories
APP_NAME = "BackgroundRemove"
if getattr(sys, "frozen", False):
    # Running as compiled executable
    APP_DIR = Path(sys.executable).parent
    DATA_DIR = APP_DIR / "_internal"
else:
    # Running as script
    APP_DIR = Path(__file__).parent.parent
    DATA_DIR = APP_DIR

# User data directory for models
USER_DATA_DIR = Path.home() / ".backgroundremove"
MODELS_DIR = USER_DATA_DIR / "models"
CACHE_DIR = USER_DATA_DIR / "cache"


# Model definitions with download URLs
MODELS = {
    "rvm_mobilenetv3": {
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.onnx",
        "filename": "rvm_mobilenetv3.onnx",
        "size_mb": 14.2,
        "sha256": None,  # Add hash for verification
    },
    "rvm_resnet50": {
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.onnx",
        "filename": "rvm_resnet50.onnx",
        "size_mb": 106.7,
        "sha256": None,
    },
    "modnet": {
        "url": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.onnx",
        "filename": "modnet.onnx",
        "size_mb": 25.0,
        "sha256": None,
    },
}

# FFmpeg download URLs
FFMPEG_URLS = {
    "Windows": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
    "Darwin": "https://evermeet.cx/ffmpeg/getrelease/zip",
    "Linux": None,  # Usually installed via package manager
}


def ensure_directories():
    """Create necessary directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, progress_callback: Callable[[float], None] | None = None) -> bool:
    """Download a file with progress reporting.

    Args:
        url: Download URL
        dest: Destination path
        progress_callback: Optional callback for progress updates (0-1)

    Returns:
        True if successful
    """
    try:
        print(f"Downloading: {url}")
        print(f"Destination: {dest}")

        # Create request with headers
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "BackgroundRemove/1.0"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback and total_size:
                        progress_callback(downloaded / total_size)

                    # Print progress
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\rProgress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="", flush=True)

            print()  # New line after progress
            return True

    except Exception as e:
        print(f"Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def verify_file(path: Path, expected_hash: str | None) -> bool:
    """Verify file integrity using SHA256.

    Args:
        path: File path
        expected_hash: Expected SHA256 hash (optional)

    Returns:
        True if valid
    """
    if not path.exists():
        return False

    if expected_hash is None:
        return True

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest().lower() == expected_hash.lower()


def check_model(model_name: str) -> Path | None:
    """Check if a model is available.

    Args:
        model_name: Model identifier

    Returns:
        Path to model if available, None otherwise
    """
    if model_name not in MODELS:
        return None

    model_info = MODELS[model_name]
    model_path = MODELS_DIR / model_info["filename"]

    if model_path.exists():
        if verify_file(model_path, model_info.get("sha256")):
            return model_path
        else:
            print(f"Model file corrupted: {model_path}")
            model_path.unlink()

    return None


def download_model(model_name: str, progress_callback: Callable[[float], None] | None = None) -> Path | None:
    """Download a model if not present.

    Args:
        model_name: Model identifier
        progress_callback: Optional progress callback

    Returns:
        Path to model if successful
    """
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        return None

    # Check if already available
    existing = check_model(model_name)
    if existing:
        return existing

    model_info = MODELS[model_name]
    model_path = MODELS_DIR / model_info["filename"]

    print(f"\nDownloading model: {model_name}")
    print(f"Size: ~{model_info['size_mb']:.1f} MB")

    if download_file(model_info["url"], model_path, progress_callback):
        if verify_file(model_path, model_info.get("sha256")):
            print(f"Model downloaded successfully: {model_path}")
            return model_path
        else:
            print("Downloaded file failed verification")
            model_path.unlink()

    return None


def check_ffmpeg() -> str | None:
    """Check if FFmpeg is available.

    Returns:
        Path to FFmpeg executable or None
    """
    # Check bundled FFmpeg first
    if platform.system() == "Windows":
        bundled_ffmpeg = APP_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"
    else:
        bundled_ffmpeg = APP_DIR / "ffmpeg" / "ffmpeg"

    if bundled_ffmpeg.exists():
        return str(bundled_ffmpeg)

    # Check system FFmpeg
    ffmpeg_cmd = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    ffmpeg_path = shutil.which(ffmpeg_cmd)

    if ffmpeg_path:
        return ffmpeg_path

    return None


def install_ffmpeg() -> str | None:
    """Install FFmpeg if not present.

    Returns:
        Path to FFmpeg executable or None
    """
    existing = check_ffmpeg()
    if existing:
        return existing

    system = platform.system()
    url = FFMPEG_URLS.get(system)

    if not url:
        print(f"\nFFmpeg not found. Please install FFmpeg manually:")
        if system == "Linux":
            print("  Ubuntu/Debian: sudo apt install ffmpeg")
            print("  Fedora: sudo dnf install ffmpeg")
            print("  Arch: sudo pacman -S ffmpeg")
        return None

    print("\nDownloading FFmpeg...")

    # Download to temp file
    temp_zip = CACHE_DIR / "ffmpeg.zip"
    if not download_file(url, temp_zip):
        return None

    # Extract
    ffmpeg_dir = APP_DIR / "ffmpeg"
    try:
        with zipfile.ZipFile(temp_zip, "r") as zf:
            # Find the ffmpeg executable in the archive
            for name in zf.namelist():
                if name.endswith("ffmpeg.exe") or name.endswith("ffmpeg"):
                    # Extract maintaining structure
                    zf.extractall(CACHE_DIR / "ffmpeg_extract")
                    break

            # Move to final location
            extracted = CACHE_DIR / "ffmpeg_extract"
            if extracted.exists():
                # Find bin directory
                for item in extracted.rglob("bin"):
                    if item.is_dir():
                        ffmpeg_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(item, ffmpeg_dir / "bin", dirs_exist_ok=True)
                        break

                shutil.rmtree(extracted, ignore_errors=True)

        temp_zip.unlink(missing_ok=True)

        return check_ffmpeg()

    except Exception as e:
        print(f"Failed to extract FFmpeg: {e}")
        return None


def check_gpu() -> dict:
    """Check GPU availability.

    Returns:
        Dictionary with GPU info
    """
    info = {
        "cuda_available": False,
        "cuda_version": None,
        "device_name": None,
        "vram_gb": 0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass

    return info


def setup_environment():
    """Set up environment variables for the application."""
    # Add models directory to environment
    os.environ["BACKGROUNDREMOVE_MODELS_DIR"] = str(MODELS_DIR)
    os.environ["BACKGROUNDREMOVE_CACHE_DIR"] = str(CACHE_DIR)

    # FFmpeg path
    ffmpeg = check_ffmpeg()
    if ffmpeg:
        ffmpeg_dir = Path(ffmpeg).parent
        current_path = os.environ.get("PATH", "")
        if str(ffmpeg_dir) not in current_path:
            os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}"


def show_splash():
    """Show startup splash/info."""
    print("=" * 60)
    print("  BackgroundRemove - Video Background Removal")
    print("=" * 60)
    print()


def run_startup_checks() -> bool:
    """Run all startup checks.

    Returns:
        True if all checks passed
    """
    print("Running startup checks...\n")

    # Create directories
    ensure_directories()

    # Check GPU
    print("Checking GPU...")
    gpu_info = check_gpu()
    if gpu_info["cuda_available"]:
        print(f"  CUDA: {gpu_info['cuda_version']}")
        print(f"  GPU: {gpu_info['device_name']}")
        print(f"  VRAM: {gpu_info['vram_gb']:.1f} GB")
    else:
        print("  No CUDA GPU detected - will use CPU")
    print()

    # Check/install FFmpeg
    print("Checking FFmpeg...")
    ffmpeg = install_ffmpeg()
    if ffmpeg:
        print(f"  FFmpeg: {ffmpeg}")
    else:
        print("  WARNING: FFmpeg not found - video processing may fail")
    print()

    # Check/download default model
    print("Checking models...")
    model = download_model("rvm_mobilenetv3")
    if model:
        print(f"  Default model: {model}")
    else:
        print("  WARNING: Could not download default model")
        return False
    print()

    # Set up environment
    setup_environment()

    print("Startup checks complete!\n")
    return True


def launch_app(args: list[str] | None = None):
    """Launch the main application.

    Args:
        args: Optional command line arguments
    """
    # Add src to path
    sys.path.insert(0, str(APP_DIR))

    # Import and run
    from src.main import main

    if args:
        sys.argv = ["BackgroundRemove"] + args

    return main()


def main():
    """Main entry point."""
    show_splash()

    # Run checks
    if not run_startup_checks():
        print("Startup checks failed. Please check the errors above.")
        input("Press Enter to exit...")
        sys.exit(1)

    print("Starting application...\n")

    # Launch app with remaining arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    exit_code = launch_app(args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
