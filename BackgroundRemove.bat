@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
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
python --version 2>&1
if !errorlevel! neq 0 (
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

call :CHECK_INSTALL torch "pip install torch torchvision"
if !errorlevel! neq 0 goto :END

call :CHECK_INSTALL PySide6 "pip install PySide6"
if !errorlevel! neq 0 goto :END

call :CHECK_INSTALL cv2 "pip install opencv-python"
call :CHECK_INSTALL onnxruntime "pip install onnxruntime"
call :CHECK_INSTALL rembg "pip install rembg"
call :CHECK_INSTALL ffmpeg "pip install ffmpeg-python"
call :CHECK_INSTALL numpy "pip install numpy"
call :CHECK_INSTALL PIL "pip install pillow"
call :CHECK_INSTALL scipy "pip install scipy"

echo.
echo   패키지 준비 완료!
echo.

REM ============================================================
REM   모델 다운로드
REM ============================================================
echo [3/5] 모델 파일 확인 중...

set "MODEL_DIR=%LOCALAPPDATA%\BackgroundRemove\models"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%" 2>nul

if not exist "%MODEL_DIR%\rvm_mobilenetv3.onnx" (
    echo   모델 다운로드 중 ^(14MB^)...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.onnx' -OutFile '%MODEL_DIR%\rvm_mobilenetv3.onnx'" 2>&1
    if exist "%MODEL_DIR%\rvm_mobilenetv3.onnx" (
        echo   모델 다운로드 완료!
    ) else (
        echo.
        echo ************************************************************
        echo   [오류] 모델 다운로드 실패!
        echo ************************************************************
        goto :END
    )
) else (
    echo   모델 파일 있음
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

python "%~dp0src\main.py" 2>&1
set "APP_EXIT=!errorlevel!"

echo.
echo ============================================================
echo.

if !APP_EXIT! neq 0 (
    echo ************************************************************
    echo   [오류] 애플리케이션 오류 발생! ^(코드: !APP_EXIT!^)
    echo ************************************************************
    echo.
    echo   위의 오류 메시지를 확인하세요.
    echo.
)

goto :END

REM ============================================================
REM   패키지 확인/설치 함수
REM ============================================================
:CHECK_INSTALL
set "PKG_NAME=%~1"
set "INSTALL_CMD=%~2"

echo   - %PKG_NAME% 확인...
python -c "import %PKG_NAME%" 2>nul
if !errorlevel! neq 0 (
    echo     설치 중...
    %INSTALL_CMD% 2>&1
    python -c "import %PKG_NAME%" 2>nul
    if !errorlevel! neq 0 (
        echo.
        echo ************************************************************
        echo   [오류] %PKG_NAME% 설치 실패!
        echo ************************************************************
        exit /b 1
    )
)
exit /b 0

REM ============================================================
REM   종료
REM ============================================================
:END
echo.
echo ============================================================
echo   아무 키나 누르면 창이 닫힙니다...
echo ============================================================
pause >nul
endlocal
exit /b 0
