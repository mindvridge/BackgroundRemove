@echo off
chcp 65001 >nul
title BackgroundRemove
cd /d "%~dp0"

echo.
echo ============================================================
echo   BackgroundRemove - Video Background Removal
echo ============================================================
echo.

REM ============================================================
REM   Python 확인
REM ============================================================
echo [1/5] Python 확인 중...
python --version
if errorlevel 1 (
    echo.
    echo ************************************************************
    echo   [오류] Python이 설치되어 있지 않습니다!
    echo ************************************************************
    echo.
    echo   Python 설치 방법:
    echo     1. https://python.org 접속
    echo     2. Downloads - Python 3.12 다운로드
    echo     3. 설치시 "Add Python to PATH" 반드시 체크!
    echo.
    goto :END
)
echo.

REM ============================================================
REM   패키지 설치
REM ============================================================
echo [2/5] 패키지 확인 중...
echo.

echo   - torch 확인...
python -c "import torch; print('    버전:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo     설치 중... (5-10분 소요)
    pip install torch torchvision
    if errorlevel 1 (
        echo.
        echo ************************************************************
        echo   [오류] torch 설치 실패!
        echo ************************************************************
        goto :END
    )
)

echo   - PySide6 확인...
python -c "import PySide6; print('    버전:', PySide6.__version__)" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install PySide6
    if errorlevel 1 (
        echo.
        echo ************************************************************
        echo   [오류] PySide6 설치 실패!
        echo ************************************************************
        goto :END
    )
)

echo   - opencv 확인...
python -c "import cv2; print('    버전:', cv2.__version__)" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install opencv-python
)

echo   - onnxruntime 확인...
python -c "import onnxruntime; print('    버전:', onnxruntime.__version__)" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install onnxruntime
)

echo   - rembg 확인...
python -c "import rembg" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install rembg
)

echo   - ffmpeg-python 확인...
python -c "import ffmpeg" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install ffmpeg-python
)

echo   - numpy 확인...
python -c "import numpy" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install numpy
)

echo   - pillow 확인...
python -c "import PIL" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install pillow
)

echo   - scipy 확인...
python -c "import scipy" 2>nul
if errorlevel 1 (
    echo     설치 중...
    pip install scipy
)

echo.
echo   패키지 준비 완료!
echo.

REM ============================================================
REM   모델 다운로드
REM ============================================================
echo [3/5] 모델 파일 확인 중...

set "MODEL_DIR=%LOCALAPPDATA%\BackgroundRemove\models"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if not exist "%MODEL_DIR%\rvm_mobilenetv3.onnx" (
    echo   모델 다운로드 중 (14MB)...
    echo   URL: https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.onnx
    echo.
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.onnx' -OutFile '%MODEL_DIR%\rvm_mobilenetv3.onnx'}"
    if exist "%MODEL_DIR%\rvm_mobilenetv3.onnx" (
        echo   모델 다운로드 완료!
    ) else (
        echo.
        echo ************************************************************
        echo   [오류] 모델 다운로드 실패!
        echo ************************************************************
        echo   수동 다운로드:
        echo   위 URL에서 파일을 다운로드하여 아래 경로에 저장하세요:
        echo   %MODEL_DIR%\rvm_mobilenetv3.onnx
        echo.
        goto :END
    )
) else (
    echo   모델 파일 있음: %MODEL_DIR%\rvm_mobilenetv3.onnx
)
echo.

REM ============================================================
REM   환경 설정
REM ============================================================
echo [4/5] 환경 설정 중...
set "BACKGROUNDREMOVE_MODELS_DIR=%MODEL_DIR%"
echo   모델 경로: %MODEL_DIR%
echo.

REM ============================================================
REM   애플리케이션 실행
REM ============================================================
echo [5/5] 애플리케이션 시작...
echo.
echo ============================================================
echo.

python "%~dp0src\main.py"
set EXITCODE=%errorlevel%

echo.
echo ============================================================
echo.

if %EXITCODE% neq 0 (
    echo ************************************************************
    echo   [오류] 애플리케이션 오류 발생! (코드: %EXITCODE%)
    echo ************************************************************
    echo.
    echo   위의 오류 메시지를 확인하세요.
    echo.
) else (
    echo   애플리케이션이 정상 종료되었습니다.
    echo.
)

:END
echo.
echo ============================================================
echo   아무 키나 누르면 창이 닫힙니다...
echo ============================================================
pause >nul
