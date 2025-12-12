"""Default configuration settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """Application configuration."""

    app_name: str = "Background Remove"
    version: str = "0.1.0"

    # Video processing settings
    default_output_format: str = "mp4"
    default_codec: str = "libx264"
    max_resolution: tuple[int, int] = (1920, 1080)

    # Model settings
    model_name: str = "u2net"
    use_gpu: bool = True

    # Paths
    cache_dir: Path = Path.home() / ".cache" / "background-remove"
    models_dir: Path = Path.home() / ".cache" / "background-remove" / "models"


# Default configuration instance
config = AppConfig()
