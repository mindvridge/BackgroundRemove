#!/usr/bin/env python3
"""Single executable launcher for BackgroundRemove application.

This launcher handles:
1. First-run setup (downloads models, FFmpeg)
2. Environment configuration
3. Application launch

All in a single .exe file.
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
from typing import Callable, Optional

# Determine if running as frozen executable
FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    # Running as compiled executable
    APP_DIR = Path(sys.executable).parent
    # PyInstaller extracts to _MEIPASS
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    # Running as script
    APP_DIR = Path(__file__).parent.parent
    BUNDLE_DIR = APP_DIR

# User data directory
APP_NAME = "BackgroundRemove"
if platform.system() == "Windows":
    USER_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
else:
    USER_DATA_DIR = Path.home() / f".{APP_NAME.lower()}"

MODELS_DIR = USER_DATA_DIR / "models"
CACHE_DIR = USER_DATA_DIR / "cache"
LOGS_DIR = USER_DATA_DIR / "logs"
FFMPEG_DIR = USER_DATA_DIR / "ffmpeg"


# Model definitions
MODELS = {
    "rvm_mobilenetv3": {
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.onnx",
        "filename": "rvm_mobilenetv3.onnx",
        "size_mb": 14.2,
    },
    "rvm_resnet50": {
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.onnx",
        "filename": "rvm_resnet50.onnx",
        "size_mb": 106.7,
    },
}

# FFmpeg URLs
FFMPEG_URLS = {
    "Windows": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
    "Darwin": "https://evermeet.cx/ffmpeg/getrelease/zip",
}


class SetupWindow:
    """Simple setup/progress window using PySide6."""

    def __init__(self):
        self.app = None
        self.window = None
        self.progress = None
        self.status = None
        self._initialized = False

    def init(self):
        """Initialize the Qt application."""
        if self._initialized:
            return True

        try:
            from PySide6.QtWidgets import (
                QApplication, QWidget, QVBoxLayout,
                QProgressBar, QLabel, QPushButton
            )
            from PySide6.QtCore import Qt

            self.app = QApplication.instance() or QApplication(sys.argv)

            self.window = QWidget()
            self.window.setWindowTitle("BackgroundRemove - Setup")
            self.window.setFixedSize(450, 150)
            self.window.setWindowFlags(Qt.WindowStaysOnTopHint)

            layout = QVBoxLayout()

            title = QLabel("BackgroundRemove Setup")
            title.setStyleSheet("font-size: 16px; font-weight: bold;")
            layout.addWidget(title)

            self.status = QLabel("Initializing...")
            layout.addWidget(self.status)

            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            layout.addWidget(self.progress)

            self.window.setLayout(layout)
            self._initialized = True
            return True

        except ImportError:
            return False

    def show(self):
        """Show the window."""
        if self.window:
            self.window.show()
            self.app.processEvents()

    def close(self):
        """Close the window."""
        if self.window:
            self.window.close()

    def set_status(self, text: str):
        """Update status text."""
        if self.status:
            self.status.setText(text)
            self.app.processEvents()
        else:
            print(text)

    def set_progress(self, value: int):
        """Update progress bar (0-100)."""
        if self.progress:
            self.progress.setValue(value)
            self.app.processEvents()


def ensure_directories():
    """Create necessary directories."""
    for d in [USER_DATA_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR, FFMPEG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_file(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[float], None]] = None
) -> bool:
    """Download a file with progress.

    Args:
        url: Download URL
        dest: Destination path
        progress_callback: Progress callback (0.0-1.0)

    Returns:
        True if successful
    """
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "BackgroundRemove/1.0"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536  # 64KB chunks

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback and total_size:
                        progress_callback(downloaded / total_size)

        return True

    except Exception as e:
        print(f"Download error: {e}")
        if dest.exists():
            dest.unlink()
        return False


def check_model(model_name: str) -> Optional[Path]:
    """Check if model exists.

    Returns:
        Path to model or None
    """
    if model_name not in MODELS:
        return None

    model_path = MODELS_DIR / MODELS[model_name]["filename"]
    return model_path if model_path.exists() else None


def download_model(
    model_name: str,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Optional[Path]:
    """Download model if needed.

    Returns:
        Path to model or None
    """
    existing = check_model(model_name)
    if existing:
        return existing

    if model_name not in MODELS:
        return None

    model_info = MODELS[model_name]
    model_path = MODELS_DIR / model_info["filename"]

    if download_file(model_info["url"], model_path, progress_callback):
        return model_path
    return None


def check_ffmpeg() -> Optional[str]:
    """Check if FFmpeg is available.

    Returns:
        Path to ffmpeg or None
    """
    # Check our installed FFmpeg
    if platform.system() == "Windows":
        our_ffmpeg = FFMPEG_DIR / "bin" / "ffmpeg.exe"
    else:
        our_ffmpeg = FFMPEG_DIR / "ffmpeg"

    if our_ffmpeg.exists():
        return str(our_ffmpeg)

    # Check system FFmpeg
    ffmpeg_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    system_ffmpeg = shutil.which(ffmpeg_name)

    return system_ffmpeg


def install_ffmpeg(progress_callback: Optional[Callable[[float], None]] = None) -> Optional[str]:
    """Install FFmpeg if needed.

    Returns:
        Path to ffmpeg or None
    """
    existing = check_ffmpeg()
    if existing:
        return existing

    system = platform.system()
    url = FFMPEG_URLS.get(system)

    if not url:
        print("Please install FFmpeg manually:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  Fedora: sudo dnf install ffmpeg")
        return None

    # Download
    zip_path = CACHE_DIR / "ffmpeg.zip"
    if not download_file(url, zip_path, progress_callback):
        return None

    # Extract
    try:
        extract_dir = CACHE_DIR / "ffmpeg_extract"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Find and copy ffmpeg binaries
        bin_dir = FFMPEG_DIR / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        for item in extract_dir.rglob("*"):
            if item.is_file() and item.stem in ["ffmpeg", "ffprobe"]:
                dest = bin_dir / item.name
                shutil.copy2(item, dest)
                if platform.system() != "Windows":
                    dest.chmod(0o755)

        # Cleanup
        shutil.rmtree(extract_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

        return check_ffmpeg()

    except Exception as e:
        print(f"FFmpeg extraction error: {e}")
        return None


def check_gpu() -> dict:
    """Check GPU availability.

    Returns:
        GPU info dict
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
    except:
        pass

    return info


def setup_environment():
    """Configure environment variables."""
    os.environ["BACKGROUNDREMOVE_MODELS_DIR"] = str(MODELS_DIR)
    os.environ["BACKGROUNDREMOVE_CACHE_DIR"] = str(CACHE_DIR)

    # Add FFmpeg to PATH
    ffmpeg = check_ffmpeg()
    if ffmpeg:
        ffmpeg_bin = Path(ffmpeg).parent
        current_path = os.environ.get("PATH", "")
        if str(ffmpeg_bin) not in current_path:
            os.environ["PATH"] = f"{ffmpeg_bin}{os.pathsep}{current_path}"

    # Add bundle directory to Python path for imports
    if str(BUNDLE_DIR) not in sys.path:
        sys.path.insert(0, str(BUNDLE_DIR))


def run_setup(setup_ui: Optional[SetupWindow] = None) -> bool:
    """Run first-time setup.

    Args:
        setup_ui: Optional UI for progress display

    Returns:
        True if successful
    """
    def update_progress(pct: float):
        if setup_ui:
            setup_ui.set_progress(int(pct * 100))

    # Create directories
    if setup_ui:
        setup_ui.set_status("Creating directories...")
        setup_ui.set_progress(5)
    ensure_directories()

    # Check GPU
    if setup_ui:
        setup_ui.set_status("Checking GPU...")
        setup_ui.set_progress(10)
    gpu = check_gpu()
    if gpu["cuda_available"]:
        print(f"GPU: {gpu['device_name']} ({gpu['vram_gb']:.1f} GB)")
    else:
        print("No CUDA GPU - using CPU")

    # Check/install FFmpeg
    if setup_ui:
        setup_ui.set_status("Checking FFmpeg...")
        setup_ui.set_progress(15)

    ffmpeg = check_ffmpeg()
    if not ffmpeg:
        if setup_ui:
            setup_ui.set_status("Downloading FFmpeg...")
        ffmpeg = install_ffmpeg(lambda p: update_progress(0.15 + p * 0.35))
        if not ffmpeg:
            print("Warning: FFmpeg not available")

    if setup_ui:
        setup_ui.set_progress(50)

    # Download default model
    if setup_ui:
        setup_ui.set_status("Downloading model (14 MB)...")

    model = download_model(
        "rvm_mobilenetv3",
        lambda p: update_progress(0.50 + p * 0.45)
    )

    if not model:
        if setup_ui:
            setup_ui.set_status("Failed to download model!")
        return False

    if setup_ui:
        setup_ui.set_progress(100)
        setup_ui.set_status("Setup complete!")

    return True


def needs_setup() -> bool:
    """Check if first-time setup is needed.

    Returns:
        True if setup needed
    """
    # Check if default model exists
    model = check_model("rvm_mobilenetv3")
    return model is None


def launch_app():
    """Launch the main application."""
    setup_environment()

    # Import and run main app
    from src.main import main
    return main()


def main():
    """Main entry point."""
    print("=" * 50)
    print("  BackgroundRemove")
    print("=" * 50)
    print()

    # Check if setup needed
    if needs_setup():
        print("First-time setup required...")
        print()

        # Try to show UI
        setup_ui = SetupWindow()
        has_ui = setup_ui.init()

        if has_ui:
            setup_ui.show()

        success = run_setup(setup_ui if has_ui else None)

        if has_ui:
            setup_ui.close()

        if not success:
            print("\nSetup failed!")
            input("Press Enter to exit...")
            return 1

        print("\nSetup complete!")
        print()

    # Launch application
    print("Starting application...")
    return launch_app()


if __name__ == "__main__":
    sys.exit(main())
