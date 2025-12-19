# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for BackgroundRemove application.

This spec file creates a Windows executable with:
- All necessary Python dependencies bundled
- Model files included or downloaded on first run
- FFmpeg bundled for video processing
"""

import os
import sys
from pathlib import Path

# Project root
project_root = Path(SPECPATH).parent
src_path = project_root / "src"

block_cipher = None

# Collect all source files
a = Analysis(
    [str(project_root / "src" / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Include config files
        (str(project_root / "configs"), "configs"),
    ],
    hiddenimports=[
        # Core dependencies
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torchvision",
        "torchvision.transforms",
        "torchvision.models",
        # ONNX
        "onnxruntime",
        "onnxruntime.capi",
        # OpenCV
        "cv2",
        # PySide6 / Qt
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        # Numpy
        "numpy",
        # FFmpeg
        "ffmpeg",
        # Rembg and its dependencies
        "rembg",
        "rembg.bg",
        "rembg.session_factory",
        # PIL
        "PIL",
        "PIL.Image",
        # Scipy (for some image processing)
        "scipy",
        "scipy.ndimage",
        # Skimage (optional)
        "skimage",
        "skimage.transform",
        # Project modules
        "src",
        "src.main",
        "src.models",
        "src.models.rvm",
        "src.models.rvm_onnx",
        "src.models.rvm_optimized",
        "src.models.base",
        "src.models.session_manager",
        "src.pipeline",
        "src.pipeline.reader",
        "src.pipeline.processor",
        "src.pipeline.pipeline",
        "src.pipeline.output",
        "src.pipeline.writer",
        "src.pipeline.temporal",
        "src.pipeline.edge_refine",
        "src.pipeline.enhanced",
        "src.pipeline.deep_matting",
        "src.pipeline.matting_ensemble",
        "src.pipeline.alpha_sr",
        "src.pipeline.depth_aware",
        "src.pipeline.sam_segmentation",
        "src.pipeline.ultimate_video",
        "src.pipeline.iterative_refine",
        "src.pipeline.gan_refine",
        "src.pipeline.video_enhance",
        "src.pipeline.neural_render",
        "src.pipeline.perfect_pipeline",
        "src.gui",
        "src.gui.app",
        "src.gui.main_window",
        "src.gui.preview",
        "src.gui.worker",
        "src.upscale",
        "src.upscale.base",
        "src.upscale.real_esrgan",
        "src.upscale.swinir",
        "src.upscale.hat",
        "src.upscale.pipeline",
        "src.upscale.preprocessing",
        "src.upscale.postprocessing",
        "src.upscale.sd_upscaler",
        "src.upscale.supir_upscaler",
        "src.upscale.advanced_pipeline",
        "src.upscale.codeformer",
        "src.upscale.region_segmenter",
        "src.upscale.ensemble",
        "src.upscale.ultimate_pipeline",
        "src.utils",
        "src.utils.gpu_monitor",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        "matplotlib",
        "tkinter",
        "unittest",
        "test",
        "tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files
def filter_binaries(binaries):
    """Remove unnecessary binaries to reduce size."""
    exclude_patterns = [
        "api-ms-win",  # Windows API sets
        "ucrtbase",
        "vcruntime",
    ]
    filtered = []
    for name, path, type_ in binaries:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in name.lower():
                exclude = True
                break
        if not exclude:
            filtered.append((name, path, type_))
    return filtered

# a.binaries = filter_binaries(a.binaries)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BackgroundRemove",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "build" / "icon.ico") if (project_root / "build" / "icon.ico").exists() else None,
    version=str(project_root / "build" / "version_info.txt") if (project_root / "build" / "version_info.txt").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BackgroundRemove",
)
