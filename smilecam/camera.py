import platform
import threading

import cv2
import numpy as np

from .devices import CAMERA_BACKENDS

PERMISSION_HINTS = {
    "Darwin": "System Settings > Privacy & Security > Camera, then allow your terminal or editor",
    "Windows": "Settings > Privacy & security > Camera, then allow desktop apps",
    "Linux": "check that your user is in the 'video' group and /dev/video* exists",
}


class CameraError(RuntimeError):
    pass


class Camera:
    """Reads frames from a webcam, video file, or stream URL.

    Live sources are read on a background thread that keeps only the newest frame,
    so a slow device drops frames instead of falling further and further behind.
    Video files are read in order, one frame per read().
    """

    def __init__(self, source: int | str, backend: str, width: int, height: int):
        self.live = isinstance(source, int) or "://" in source
        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source, CAMERA_BACKENDS[backend])
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            message = f"Could not open video source {source!r}"
            if isinstance(source, int):
                hint = PERMISSION_HINTS.get(platform.system(), "check camera permissions")
                message += f" (backend {backend}). Try another --source index, or {hint}."
            raise CameraError(message)

        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_read = 0
        self._ended = False
        self._cond = threading.Condition()
        if self.live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _reader(self) -> None:
        while not self._ended:
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._ended = True
                else:
                    self._frame = frame
                    self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """Return the next frame, or None once the source has ended or stalled."""
        if not self.live:
            ok, frame = self._cap.read()
            return frame if ok else None
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._last_read or self._ended, timeout)
            if self._seq == self._last_read:
                return None
            self._last_read = self._seq
            return self._frame

    def close(self) -> None:
        with self._cond:
            self._ended = True
        if self.live:
            self._thread.join(timeout=1.0)
        self._cap.release()
