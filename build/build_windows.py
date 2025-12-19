#!/usr/bin/env python3
"""Build script for creating single Windows executable.

Usage:
    python build/build_windows.py [--debug] [--clean]

Options:
    --debug     Enable debug mode (shows console window)
    --clean     Clean build directories before building
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Build configuration
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"
SPEC_FILE = BUILD_DIR / "BackgroundRemove.spec"


def run_command(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a command and return success status."""
    print(f"\n> {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False


def check_requirements() -> bool:
    """Check if all build requirements are installed."""
    print("Checking build requirements...")

    if sys.version_info < (3, 10):
        print(f"Error: Python 3.10+ required, got {sys.version}")
        return False
    print(f"  Python: {sys.version}")

    try:
        import PyInstaller
        print(f"  PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("  PyInstaller: NOT FOUND")
        print("\nInstall with: pip install pyinstaller")
        return False

    required = ["torch", "onnxruntime", "cv2", "PySide6", "numpy"]
    missing = []

    for module in required:
        try:
            __import__(module)
            print(f"  {module}: OK")
        except ImportError:
            missing.append(module)
            print(f"  {module}: NOT FOUND")

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("Install with: pip install -e .")
        return False

    return True


def clean_build():
    """Clean previous build artifacts."""
    print("\nCleaning build directories...")

    dirs_to_clean = [
        DIST_DIR,
        BUILD_DIR / "BackgroundRemove",
        PROJECT_ROOT / "__pycache__",
    ]

    for d in dirs_to_clean:
        if d.exists():
            print(f"  Removing: {d}")
            shutil.rmtree(d, ignore_errors=True)

    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)

    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def create_version_info():
    """Create Windows version info file."""
    version_file = BUILD_DIR / "version_info.txt"

    version_info = """
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(0, 1, 0, 0),
    prodvers=(0, 1, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'BackgroundRemove'),
            StringStruct(u'FileDescription', u'Video Background Removal'),
            StringStruct(u'FileVersion', u'0.1.0'),
            StringStruct(u'InternalName', u'BackgroundRemove'),
            StringStruct(u'LegalCopyright', u'MIT License'),
            StringStruct(u'OriginalFilename', u'BackgroundRemove.exe'),
            StringStruct(u'ProductName', u'BackgroundRemove'),
            StringStruct(u'ProductVersion', u'0.1.0'),
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    version_file.write_text(version_info.strip())
    print(f"  Created: {version_file}")


def build_executable(debug: bool = False) -> bool:
    """Build the single Windows executable."""
    print("\nBuilding single executable...")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        str(SPEC_FILE),
    ]

    return run_command(cmd, cwd=PROJECT_ROOT)


def print_summary(success: bool):
    """Print build summary."""
    print("\n" + "=" * 60)
    if success:
        exe_path = DIST_DIR / "BackgroundRemove.exe"
        print("BUILD SUCCESSFUL!")
        print("=" * 60)
        print(f"\nOutput: {exe_path}")

        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"Size: {size_mb:.1f} MB")

        print("\nUsage:")
        print("  1. Copy BackgroundRemove.exe to any location")
        print("  2. Double-click to run")
        print("  3. First run will download required files (~15 MB)")
    else:
        print("BUILD FAILED!")
        print("=" * 60)
        print("\nCheck the error messages above.")


def main():
    """Main build entry point."""
    parser = argparse.ArgumentParser(description="Build BackgroundRemove")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--clean", action="store_true", help="Clean first")
    args = parser.parse_args()

    print("=" * 60)
    print("  BackgroundRemove - Single Executable Build")
    print("=" * 60)

    if not check_requirements():
        print_summary(False)
        return 1

    if args.clean:
        clean_build()

    create_version_info()

    success = build_executable(debug=args.debug)
    print_summary(success)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
