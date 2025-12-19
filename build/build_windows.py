#!/usr/bin/env python3
"""Build script for creating Windows executable.

Usage:
    python build/build_windows.py [--onefile] [--debug]

Options:
    --onefile   Create single executable instead of directory
    --debug     Enable debug mode (shows console)
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
    """Run a command and return success status.

    Args:
        cmd: Command and arguments
        cwd: Working directory

    Returns:
        True if successful
    """
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
    """Check if all build requirements are installed.

    Returns:
        True if all requirements are met
    """
    print("Checking build requirements...")

    # Check Python version
    if sys.version_info < (3, 10):
        print(f"Error: Python 3.10+ required, got {sys.version}")
        return False
    print(f"  Python: {sys.version}")

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("  PyInstaller: NOT FOUND")
        print("\nInstall with: pip install pyinstaller")
        return False

    # Check project dependencies
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
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -e .")
        return False

    return True


def clean_build():
    """Clean previous build artifacts."""
    print("\nCleaning build directories...")

    dirs_to_clean = [
        DIST_DIR,
        PROJECT_ROOT / "build" / "BackgroundRemove",
        PROJECT_ROOT / "__pycache__",
    ]

    for d in dirs_to_clean:
        if d.exists():
            print(f"  Removing: {d}")
            shutil.rmtree(d, ignore_errors=True)

    # Clean .pyc files
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
            StringStruct(u'FileDescription', u'Video Background Removal Application'),
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


def build_executable(onefile: bool = False, debug: bool = False) -> bool:
    """Build the Windows executable.

    Args:
        onefile: Create single file executable
        debug: Enable debug mode

    Returns:
        True if successful
    """
    print("\nBuilding executable...")

    # Base PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    if debug:
        cmd.extend(["--debug", "all", "--console"])
    else:
        cmd.append("--windowed")

    # Use spec file
    cmd.append(str(SPEC_FILE))

    return run_command(cmd, cwd=PROJECT_ROOT)


def copy_resources():
    """Copy additional resources to dist directory."""
    print("\nCopying resources...")

    dist_app = DIST_DIR / "BackgroundRemove"
    if not dist_app.exists():
        print("  Dist directory not found")
        return

    # Copy configs
    configs_src = PROJECT_ROOT / "configs"
    configs_dst = dist_app / "configs"
    if configs_src.exists() and not configs_dst.exists():
        shutil.copytree(configs_src, configs_dst)
        print(f"  Copied: configs/")

    # Copy launcher
    launcher_src = BUILD_DIR / "launcher.py"
    launcher_dst = dist_app / "launcher.py"
    if launcher_src.exists():
        shutil.copy(launcher_src, launcher_dst)
        print(f"  Copied: launcher.py")

    # Create README for distribution
    readme = dist_app / "README.txt"
    readme.write_text("""BackgroundRemove - Video Background Removal Application
=====================================================

QUICK START:
1. Run BackgroundRemove.exe
2. On first run, the application will download required model files
3. Select a video file and choose your output settings
4. Click "Process" to remove the background

REQUIREMENTS:
- Windows 10/11 64-bit
- NVIDIA GPU with CUDA support (recommended for best performance)
- FFmpeg (will be downloaded automatically if not found)

TROUBLESHOOTING:
- If the application crashes, try running from command prompt to see error messages
- For GPU issues, ensure you have the latest NVIDIA drivers installed
- Check the logs in %USERPROFILE%\\.backgroundremove\\logs

For more information, visit: https://github.com/your-repo/BackgroundRemove
""")
    print(f"  Created: README.txt")


def create_installer_script():
    """Create an Inno Setup script for proper installer."""
    iss_file = BUILD_DIR / "installer.iss"

    iss_content = """; Inno Setup Script for BackgroundRemove
; Download Inno Setup from: https://jrsoftware.org/isinfo.php

[Setup]
AppName=BackgroundRemove
AppVersion=0.1.0
AppPublisher=BackgroundRemove
DefaultDirName={autopf}\\BackgroundRemove
DefaultGroupName=BackgroundRemove
OutputDir=..\\dist
OutputBaseFilename=BackgroundRemove_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\\dist\\BackgroundRemove\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\BackgroundRemove"; Filename: "{app}\\BackgroundRemove.exe"
Name: "{group}\\{cm:UninstallProgram,BackgroundRemove}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\\BackgroundRemove"; Filename: "{app}\\BackgroundRemove.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\\BackgroundRemove.exe"; Description: "{cm:LaunchProgram,BackgroundRemove}"; Flags: nowait postinstall skipifsilent
"""
    iss_file.write_text(iss_content)
    print(f"\n  Created Inno Setup script: {iss_file}")
    print("  To create installer: Install Inno Setup and compile the .iss file")


def print_summary(success: bool):
    """Print build summary."""
    print("\n" + "=" * 60)
    if success:
        print("BUILD SUCCESSFUL!")
        print("=" * 60)
        print(f"\nOutput directory: {DIST_DIR / 'BackgroundRemove'}")
        print("\nTo run the application:")
        print(f"  {DIST_DIR / 'BackgroundRemove' / 'BackgroundRemove.exe'}")
        print("\nTo create an installer:")
        print("  1. Install Inno Setup from https://jrsoftware.org/isinfo.php")
        print(f"  2. Open and compile: {BUILD_DIR / 'installer.iss'}")
    else:
        print("BUILD FAILED!")
        print("=" * 60)
        print("\nCheck the error messages above for details.")


def main():
    """Main build entry point."""
    parser = argparse.ArgumentParser(description="Build BackgroundRemove for Windows")
    parser.add_argument("--onefile", action="store_true", help="Create single executable")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--clean", action="store_true", help="Clean build directories")
    args = parser.parse_args()

    print("=" * 60)
    print("  BackgroundRemove Windows Build")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        print_summary(False)
        return 1

    # Clean if requested
    if args.clean:
        clean_build()

    # Create version info
    create_version_info()

    # Build
    success = build_executable(onefile=args.onefile, debug=args.debug)

    if success:
        # Copy resources
        copy_resources()

        # Create installer script
        create_installer_script()

    print_summary(success)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
