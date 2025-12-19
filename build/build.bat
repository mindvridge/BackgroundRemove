@echo off
REM Build BackgroundRemove single executable
REM Output: dist\BackgroundRemove.exe

echo ============================================================
echo   BackgroundRemove - Single Executable Build
echo ============================================================
echo.

REM Get script directory and project root
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%") do set "PROJECT_ROOT=%%~dpI"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Project: %PROJECT_ROOT%
echo.

cd /d "%PROJECT_ROOT%"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Setup virtual environment
if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    echo Using existing virtual environment...
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
) else (
    echo Creating virtual environment...
    python -m venv "%PROJECT_ROOT%\.venv"
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"

    echo.
    echo Installing dependencies (this may take a few minutes)...
    pip install --upgrade pip
    pip install -r "%PROJECT_ROOT%\build\requirements-build.txt"
    pip install -e "%PROJECT_ROOT%"
)

REM Build
echo.
echo Building executable...
python "%PROJECT_ROOT%\build\build_windows.py" %*

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   BUILD COMPLETE!
echo ============================================================
echo.
echo Output: %PROJECT_ROOT%\dist\BackgroundRemove.exe
echo.
echo Copy this single file anywhere and run it.
echo First run will automatically download required files.
echo.
pause
