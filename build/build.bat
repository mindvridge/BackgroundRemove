@echo off
REM Build script for BackgroundRemove Windows executable
REM Run this script from the project root directory

echo ============================================================
echo   BackgroundRemove Windows Build Script
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist ".venv" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat

    echo Installing dependencies...
    pip install -r build\requirements-build.txt
    pip install -e .
)

REM Run build script
echo.
echo Starting build...
python build\build_windows.py %*

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build complete!
echo Output: dist\BackgroundRemove\BackgroundRemove.exe
pause
