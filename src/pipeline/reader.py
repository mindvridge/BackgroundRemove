"""Video reader with thread-based frame reading.

Implements producer pattern for video frame extraction using
a background thread and queue-based buffering.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from typing import TYPE_CHECKING, Iterator

import cv2

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video file metadata."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    codec: str

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) tuple."""
        return (self.width, self.height)


@dataclass
class ReaderConfig:
    """Configuration for VideoReader."""

    queue_size: int = 64
    timeout: float = 1.0  # Queue put/get timeout
    loop: bool = False  # Loop video infinitely
    start_frame: int = 0
    end_frame: int | None = None  # None = read to end


class VideoReader:
    """Thread-based video frame reader (Producer).

    Reads video frames in a background thread and buffers them
    in a queue for consumption by the processing pipeline.

    Example:
        >>> reader = VideoReader("input.mp4")
        >>> reader.start()
        >>> for frame in reader:
        ...     process(frame)
        >>> reader.stop()
    """

    # Sentinel value to signal end of stream
    END_OF_STREAM = None

    def __init__(
        self,
        video_path: str | Path,
        config: ReaderConfig | None = None,
    ) -> None:
        """Initialize video reader.

        Args:
            video_path: Path to input video file.
            config: Reader configuration.
        """
        self.video_path = Path(video_path)
        self.config = config or ReaderConfig()

        self._queue: Queue[ndarray | None] = Queue(maxsize=self.config.queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._metadata: VideoMetadata | None = None
        self._frames_read = 0
        self._error: Exception | None = None

    @property
    def metadata(self) -> VideoMetadata | None:
        """Get video metadata (available after start)."""
        return self._metadata

    @property
    def frames_read(self) -> int:
        """Number of frames read so far."""
        return self._frames_read

    @property
    def is_running(self) -> bool:
        """Check if reader thread is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def queue_size(self) -> int:
        """Current number of frames in queue."""
        return self._queue.qsize()

    def _extract_metadata(self, cap: cv2.VideoCapture) -> VideoMetadata:
        """Extract video metadata from capture object.

        Args:
            cap: OpenCV VideoCapture object.

        Returns:
            VideoMetadata instance.
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=self.video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
        )

    def _reader_thread(self) -> None:
        """Background thread for reading video frames."""
        cap = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {self.video_path}")

            # Extract metadata
            self._metadata = self._extract_metadata(cap)
            logger.info(
                f"Video opened: {self._metadata.width}x{self._metadata.height} "
                f"@ {self._metadata.fps:.2f}fps, {self._metadata.frame_count} frames"
            )

            # Seek to start frame if specified
            if self.config.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
                logger.debug(f"Seeked to frame {self.config.start_frame}")

            # Determine end frame
            end_frame = self.config.end_frame
            if end_frame is None:
                end_frame = self._metadata.frame_count

            while not self._stop_event.is_set():
                ret, frame = cap.read()

                if not ret:
                    if self.config.loop:
                        # Reset to start for looping
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
                        continue
                    else:
                        # End of video
                        break

                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame > end_frame:
                    break

                # Put frame in queue with timeout
                try:
                    self._queue.put(frame, timeout=self.config.timeout)
                    self._frames_read += 1
                except Full:
                    if self._stop_event.is_set():
                        break
                    # Queue full, retry
                    continue

        except Exception as e:
            logger.error(f"Reader thread error: {e}")
            self._error = e
        finally:
            if cap is not None:
                cap.release()

            # Signal end of stream
            try:
                self._queue.put(self.END_OF_STREAM, timeout=self.config.timeout)
            except Full:
                pass

            logger.debug("Reader thread finished")

    def start(self) -> None:
        """Start the reader thread.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            RuntimeError: If reader is already started.
        """
        if self._started:
            raise RuntimeError("Reader already started")

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()
        self._started = True

        logger.info(f"Reader started for: {self.video_path}")

    def stop(self) -> None:
        """Stop the reader thread."""
        if not self._started:
            return

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Reader thread did not stop gracefully")

        self._started = False
        logger.info("Reader stopped")

    def get_frame(self, timeout: float | None = None) -> ndarray | None:
        """Get next frame from queue.

        Args:
            timeout: Timeout in seconds. Uses config timeout if None.

        Returns:
            Frame as numpy array, or None if end of stream.

        Raises:
            RuntimeError: If reader is not started or an error occurred.
        """
        if not self._started:
            raise RuntimeError("Reader not started. Call start() first.")

        if self._error is not None:
            raise RuntimeError(f"Reader error: {self._error}")

        timeout = timeout if timeout is not None else self.config.timeout

        try:
            frame = self._queue.get(timeout=timeout)
            return frame
        except Exception:
            return None

    def __iter__(self) -> Iterator[ndarray]:
        """Iterate over video frames.

        Yields:
            Video frames as numpy arrays.
        """
        if not self._started:
            self.start()

        while True:
            frame = self.get_frame()
            if frame is None:
                break
            yield frame

    def __enter__(self) -> "VideoReader":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        """String representation."""
        status = "running" if self.is_running else "stopped"
        return f"VideoReader(path={self.video_path!r}, status={status})"
