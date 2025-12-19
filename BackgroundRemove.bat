@echo off
chcp 65001 >nul
title BackgroundRemove
cd /d "%~dp0"

REM ============================================================
REM   BackgroundRemove - 원클릭 실행
REM   더블클릭만 하면 자동 설치 후 실행됩니다.
REM ============================================================

echo.
echo ============================================================
echo   BackgroundRemove - Video Background Removal
echo ============================================================
echo.

REM Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python이 설치되어 있지 않습니다.
    echo.
    echo     Python 설치:
    echo     1. https://python.org 접속
    echo     2. Downloads - Python 3.12 다운로드
    echo     3. 설치시 "Add Python to PATH" 반드시 체크!
    echo.
    pause
    exit /b 1
)

echo [OK] Python 확인됨
echo.

REM 필수 패키지 확인 및 설치
echo 패키지 확인 중...

python -c "import torch" 2>nul
if errorlevel 1 (
    echo   torch 설치 중... (시간이 걸립니다)
    pip install torch torchvision --quiet
)

python -c "import PySide6" 2>nul
if errorlevel 1 (
    echo   PySide6 설치 중...
    pip install PySide6 --quiet
)

python -c "import cv2" 2>nul
if errorlevel 1 (
    echo   opencv 설치 중...
    pip install opencv-python --quiet
)

python -c "import onnxruntime" 2>nul
if errorlevel 1 (
    echo   onnxruntime 설치 중...
    pip install onnxruntime --quiet
)

python -c "import rembg" 2>nul
if errorlevel 1 (
    echo   rembg 설치 중...
    pip install rembg --quiet
)

python -c "import ffmpeg" 2>nul
if errorlevel 1 (
    echo   ffmpeg-python 설치 중...
    pip install ffmpeg-python --quiet
)

echo.
echo [OK] 패키지 준비 완료
echo.

REM 모델 디렉토리 설정
set "MODEL_DIR=%LOCALAPPDATA%\BackgroundRemove\models"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

REM 모델 다운로드 (없으면)
if not exist "%MODEL_DIR%\rvm_mobilenetv3.onnx" (
    echo 모델 다운로드 중 (14MB)...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.onnx' -OutFile '%MODEL_DIR%\rvm_mobilenetv3.onnx'}"
    if exist "%MODEL_DIR%\rvm_mobilenetv3.onnx" (
        echo [OK] 모델 다운로드 완료
    ) else (
        echo [!] 모델 다운로드 실패
    )
) else (
    echo [OK] 모델 파일 확인됨
)

echo.
set "BACKGROUNDREMOVE_MODELS_DIR=%MODEL_DIR%"

REM 애플리케이션 실행
echo ============================================================
echo   애플리케이션 시작
echo ============================================================
echo.

python -c "import sys; sys.path.insert(0, r'%~dp0'); from src.main import main; main()"

if errorlevel 1 (
    echo.
    echo [오류] 실행 중 문제가 발생했습니다.
    pause
)
