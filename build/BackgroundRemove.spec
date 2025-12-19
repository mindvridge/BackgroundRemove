# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BackgroundRemove single executable."""

from pathlib import Path

project_root = Path(SPECPATH).parent

a = Analysis(
    [str(project_root / "BackgroundRemove.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "configs"), "configs"),
        (str(project_root / "src"), "src"),
    ],
    hiddenimports=[
        "torch", "torch.nn", "torch.nn.functional", "torch.cuda",
        "torchvision", "torchvision.transforms",
        "onnxruntime", "cv2", "numpy", "PIL", "PIL.Image",
        "PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
        "scipy", "scipy.ndimage", "ffmpeg", "rembg",
        "urllib.request", "ssl", "certifi", "zipfile",
    ],
    excludes=["matplotlib", "tkinter", "IPython", "jupyter"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="BackgroundRemove",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(project_root / "build" / "icon.ico") if (project_root / "build" / "icon.ico").exists() else None,
)
