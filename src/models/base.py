"""Base model interface for background removal models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


class BaseModel(ABC):
    """Abstract base class for background removal models."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        ...

    @abstractmethod
    def reset_state(self) -> None:
        """Reset any internal state (e.g., recurrent states)."""
        ...

    @abstractmethod
    def inference(
        self,
        frame: ndarray,
        downsample_ratio: float | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Run inference on a single frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format.
            downsample_ratio: Optional downsample ratio override.

        Returns:
            Tuple of (foreground, alpha_matte) as numpy arrays.
        """
        ...

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload()
