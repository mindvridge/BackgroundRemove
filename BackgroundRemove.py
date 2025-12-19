#!/usr/bin/env python3
"""
BackgroundRemove - Single File Launcher
=======================================

이 파일 하나로 모든 것을 실행합니다:
1. 필요한 패키지 자동 설치
2. 모델 파일 자동 다운로드
3. FFmpeg 자동 설치
4. 애플리케이션 실행

사용법:
    python BackgroundRemove.py
    또는 더블클릭 (Windows에서 .py 파일 연결 시)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# ============================================================================
# 설정
# ============================================================================

APP_NAME = "BackgroundRemove"
REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "onnxruntime",
    "opencv-python",
    "PySide6",
    "ffmpeg-python",
    "rembg",
    "numpy",
    "pillow",
    "scipy",
]

# 사용자 데이터 디렉토리
if platform.system() == "Windows":
    USER_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
else:
    USER_DATA_DIR = Path.home() / f".{APP_NAME.lower()}"

MODELS_DIR = USER_DATA_DIR / "models"
CACHE_DIR = USER_DATA_DIR / "cache"
FFMPEG_DIR = USER_DATA_DIR / "ffmpeg"

# 모델 정보
MODELS = {
    "rvm_mobilenetv3": {
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.onnx",
        "filename": "rvm_mobilenetv3.onnx",
    },
}

# FFmpeg URL
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"


# ============================================================================
# 유틸리티 함수
# ============================================================================

def print_header(text: str):
    """헤더 출력"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(text: str):
    """단계 출력"""
    print(f"\n>> {text}")


def ensure_dirs():
    """디렉토리 생성"""
    for d in [USER_DATA_DIR, MODELS_DIR, CACHE_DIR, FFMPEG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run_pip(args: list[str]) -> bool:
    """pip 명령 실행"""
    cmd = [sys.executable, "-m", "pip"] + args
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def is_package_installed(package: str) -> bool:
    """패키지 설치 확인"""
    # 패키지 이름 정규화
    pkg_name = package.replace("-", "_").lower()
    if pkg_name == "opencv_python":
        pkg_name = "cv2"
    elif pkg_name == "pillow":
        pkg_name = "PIL"
    elif pkg_name == "ffmpeg_python":
        pkg_name = "ffmpeg"

    try:
        __import__(pkg_name)
        return True
    except ImportError:
        return False


def install_packages():
    """필요한 패키지 설치"""
    print_step("패키지 확인 중...")

    missing = []
    for pkg in REQUIRED_PACKAGES:
        if not is_package_installed(pkg):
            missing.append(pkg)
            print(f"  [ ] {pkg}")
        else:
            print(f"  [✓] {pkg}")

    if not missing:
        print("\n모든 패키지가 설치되어 있습니다.")
        return True

    print(f"\n{len(missing)}개 패키지 설치 필요: {', '.join(missing)}")
    print_step("패키지 설치 중... (시간이 걸릴 수 있습니다)")

    # pip 업그레이드
    run_pip(["install", "--upgrade", "pip"])

    # 패키지 설치
    for pkg in missing:
        print(f"  설치 중: {pkg}...")
        if not run_pip(["install", pkg]):
            print(f"  경고: {pkg} 설치 실패")

    # 다시 확인
    still_missing = [pkg for pkg in missing if not is_package_installed(pkg)]
    if still_missing:
        print(f"\n설치 실패: {', '.join(still_missing)}")
        return False

    print("\n패키지 설치 완료!")
    return True


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """파일 다운로드 (진행률 표시)"""
    try:
        print(f"  다운로드: {desc or url}")

        request = urllib.request.Request(
            url, headers={"User-Agent": "BackgroundRemove/1.0"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        pct = downloaded / total * 100
                        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                        print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

            print()  # 줄바꿈
        return True

    except Exception as e:
        print(f"\n  다운로드 실패: {e}")
        if dest.exists():
            dest.unlink()
        return False


def check_model() -> Path | None:
    """모델 파일 확인"""
    model_path = MODELS_DIR / "rvm_mobilenetv3.onnx"
    return model_path if model_path.exists() else None


def download_model() -> bool:
    """모델 다운로드"""
    print_step("모델 파일 확인 중...")

    model_path = MODELS_DIR / "rvm_mobilenetv3.onnx"
    if model_path.exists():
        print("  [✓] 모델 파일 있음")
        return True

    print("  모델 파일 다운로드 필요 (~14MB)")

    info = MODELS["rvm_mobilenetv3"]
    if download_file(info["url"], model_path, "rvm_mobilenetv3.onnx"):
        print("  모델 다운로드 완료!")
        return True

    return False


def check_ffmpeg() -> str | None:
    """FFmpeg 확인"""
    # 설치된 FFmpeg 확인
    if platform.system() == "Windows":
        our_ffmpeg = FFMPEG_DIR / "bin" / "ffmpeg.exe"
    else:
        our_ffmpeg = FFMPEG_DIR / "ffmpeg"

    if our_ffmpeg.exists():
        return str(our_ffmpeg)

    # 시스템 FFmpeg 확인
    system_ffmpeg = shutil.which("ffmpeg")
    return system_ffmpeg


def install_ffmpeg() -> bool:
    """FFmpeg 설치"""
    print_step("FFmpeg 확인 중...")

    existing = check_ffmpeg()
    if existing:
        print(f"  [✓] FFmpeg: {existing}")
        return True

    if platform.system() != "Windows":
        print("  FFmpeg 없음. 수동 설치 필요:")
        print("    Ubuntu: sudo apt install ffmpeg")
        print("    Mac: brew install ffmpeg")
        return False

    print("  FFmpeg 다운로드 중... (~100MB)")

    zip_path = CACHE_DIR / "ffmpeg.zip"
    if not download_file(FFMPEG_URL, zip_path, "FFmpeg"):
        return False

    print("  압축 해제 중...")
    try:
        extract_dir = CACHE_DIR / "ffmpeg_extract"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # ffmpeg 바이너리 찾아서 복사
        bin_dir = FFMPEG_DIR / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        for item in extract_dir.rglob("*.exe"):
            if item.stem in ["ffmpeg", "ffprobe"]:
                shutil.copy2(item, bin_dir / item.name)

        # 정리
        shutil.rmtree(extract_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

        print("  FFmpeg 설치 완료!")
        return True

    except Exception as e:
        print(f"  FFmpeg 설치 실패: {e}")
        return False


def setup_environment():
    """환경 설정"""
    # 모델 디렉토리 환경변수
    os.environ["BACKGROUNDREMOVE_MODELS_DIR"] = str(MODELS_DIR)

    # FFmpeg PATH 추가
    ffmpeg = check_ffmpeg()
    if ffmpeg:
        ffmpeg_bin = Path(ffmpeg).parent
        current_path = os.environ.get("PATH", "")
        if str(ffmpeg_bin) not in current_path:
            os.environ["PATH"] = f"{ffmpeg_bin}{os.pathsep}{current_path}"


def check_gpu():
    """GPU 확인"""
    print_step("GPU 확인 중...")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  [✓] CUDA GPU: {name} ({vram:.1f}GB)")
            return True
        else:
            print("  [!] CUDA GPU 없음 - CPU 모드로 실행")
            return False
    except:
        print("  [!] GPU 확인 실패 - CPU 모드로 실행")
        return False


def launch_app():
    """애플리케이션 실행"""
    print_step("애플리케이션 시작...")

    # 현재 스크립트 위치에서 src 찾기
    script_dir = Path(__file__).parent
    src_dir = script_dir / "src"

    if not src_dir.exists():
        print(f"  오류: src 디렉토리를 찾을 수 없습니다: {src_dir}")
        print("  이 파일을 프로젝트 루트 디렉토리에서 실행해주세요.")
        return 1

    # src를 import 경로에 추가
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        from src.main import main
        return main()
    except Exception as e:
        print(f"  실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
# 메인
# ============================================================================

def main():
    """메인 함수"""
    print_header("BackgroundRemove")
    print(f"Python {sys.version}")
    print(f"데이터 디렉토리: {USER_DATA_DIR}")

    # 디렉토리 생성
    ensure_dirs()

    # 1. 패키지 설치
    if not install_packages():
        print("\n패키지 설치에 실패했습니다.")
        input("Enter 키를 눌러 종료...")
        return 1

    # 2. GPU 확인
    check_gpu()

    # 3. FFmpeg 설치
    if not install_ffmpeg():
        print("\n경고: FFmpeg가 없으면 비디오 처리가 제한됩니다.")

    # 4. 모델 다운로드
    if not download_model():
        print("\n모델 다운로드에 실패했습니다.")
        input("Enter 키를 눌러 종료...")
        return 1

    # 5. 환경 설정
    setup_environment()

    # 6. 앱 실행
    print_header("준비 완료!")
    return launch_app()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n취소됨")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        input("Enter 키를 눌러 종료...")
        sys.exit(1)
