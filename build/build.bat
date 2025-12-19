@echo off
chcp 65001 >nul
REM BackgroundRemove.exe 빌드 스크립트

echo ============================================================
echo   BackgroundRemove.exe 빌드
echo ============================================================
echo.

REM 경로 설정
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%") do set "PROJECT_ROOT=%%~dpI"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

cd /d "%PROJECT_ROOT%"

REM Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo        python.org에서 Python 3.10 이상을 설치하세요.
    pause
    exit /b 1
)

REM PyInstaller 확인/설치
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo PyInstaller 설치 중...
    pip install pyinstaller
)

REM 빌드 실행
echo.
echo 빌드 시작...
echo.

pyinstaller --noconfirm --clean "%PROJECT_ROOT%\build\BackgroundRemove.spec"

if errorlevel 1 (
    echo.
    echo [오류] 빌드 실패!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   빌드 완료!
echo ============================================================
echo.
echo   실행파일: %PROJECT_ROOT%\dist\BackgroundRemove.exe
echo.
echo   이 파일을 아무 곳에나 복사해서 실행하세요.
echo   첫 실행시 필요한 파일을 자동으로 다운로드합니다.
echo.
pause
