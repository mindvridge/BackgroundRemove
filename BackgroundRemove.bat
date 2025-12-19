@echo off
chcp 65001 >nul
title BackgroundRemove
cd /d "%~dp0"

REM ============================================================
REM   BackgroundRemove - 원클릭 실행
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
    pip install torch torchvision
    if errorlevel 1 (
        echo [!] torch 설치 실패
        pause
        exit /b 1
    )
)

python -c "import PySide6" 2>nul
if errorlevel 1 (
    echo   PySide6 설치 중...
    pip install PySide6
    if errorlevel 1 (
        echo [!] PySide6 설치 실패
        pause
        exit /b 1
    )
)

python -c "import cv2" 2>nul
if errorlevel 1 (
    echo   opencv 설치 중...
    pip install opencv-python
)

python -c "import onnxruntime" 2>nul
if errorlevel 1 (
    echo   onnxruntime 설치 중...
    pip install onnxruntime
)

python -c "import rembg" 2>nul
if errorlevel 1 (
    echo   rembg 설치 중...
    pip install rembg
)

python -c "import ffmpeg" 2>nul
if errorlevel 1 (
    echo   ffmpeg-python 설치 중...
    pip install ffmpeg-python
)

python -c "import numpy" 2>nul
if errorlevel 1 (
    echo   numpy 설치 중...
    pip install numpy
)

python -c "import PIL" 2>nul
if errorlevel 1 (
    echo   pillow 설치 중...
    pip install pillow
)

python -c "import scipy" 2>nul
if errorlevel 1 (
    echo   scipy 설치 중...
    pip install scipy
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
        pause
        exit /b 1
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
echo 창이 닫히면 아래에 오류 메시지가 표시됩니다.
echo.

REM Python 스크립트 실행 (오류 출력 포함)
python "%~dp0src\main.py"

echo.
echo ============================================================
if errorlevel 1 (
    echo [!] 오류가 발생했습니다. 위의 메시지를 확인하세요.
) else (
    echo 애플리케이션이 종료되었습니다.
)
echo ============================================================
echo.
pause
