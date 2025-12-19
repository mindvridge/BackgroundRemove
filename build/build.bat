@echo off
REM Build script for BackgroundRemove Windows executable
REM Can be run from anywhere - will automatically find project root

echo ============================================================
echo   BackgroundRemove Windows Build Script
echo ============================================================
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
REM Get project root (parent of build directory)
for %%I in ("%SCRIPT_DIR%") do set "PROJECT_ROOT=%%~dpI"
REM Remove trailing backslash
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Project root: %PROJECT_ROOT%
echo.

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "%PROJECT_ROOT%\.venv" (
    echo Activating virtual environment...
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
) else (
    echo Creating virtual environment...
    python -m venv "%PROJECT_ROOT%\.venv"
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"

    echo.
    echo Installing dependencies...
    pip install --upgrade pip
    pip install -r "%PROJECT_ROOT%\build\requirements-build.txt"
    pip install -e "%PROJECT_ROOT%"
)

REM Run build script
echo.
echo Starting build...
python "%PROJECT_ROOT%\build\build_windows.py" %*

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build complete!
echo Output: %PROJECT_ROOT%\dist\BackgroundRemove\BackgroundRemove.exe
pause
