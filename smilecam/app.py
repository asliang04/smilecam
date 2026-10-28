import platform
import time
from collections import Counter

import cv2
import numpy as np

from . import overlay
from .camera import Camera
from .config import Config
from .counting import SeenCounter
from .devices import camera_backend, torch_device
from .faces import Face, FaceAnalyzer
from .objects import Detection, ObjectDetector
from .smile import SmileTrigger

WINDOW = "smilecam"


class PausableClock:
    """Monotonic seconds that stand still while paused.

    Every timer in the loop (smile hold, cooldown, photo display, counting) reads this
    clock, so after a resume they all pick up exactly where they stopped.
    """

    def __init__(self) -> None:
        self._paused_at: float | None = None
        self._paused_total = 0.0

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def now(self) -> float:
        t = time.monotonic() if self._paused_at is None else self._paused_at
        return t - self._paused_total

    def toggle(self) -> None:
        if self._paused_at is None:
            self._paused_at = time.monotonic()
        else:
            self._paused_total += time.monotonic() - self._paused_at
            self._paused_at = None


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # macOS raises instead of returning 0 once the window is gone
        return True


def run(cfg: Config) -> None:
    device = torch_device(cfg.device)
    backend = camera_backend(cfg.camera_backend)
    objects = ObjectDetector(cfg.yolo_model, device, cfg.object_conf, cfg.imgsz)
    faces = FaceAnalyzer(cfg.face_model, cfg.max_faces)
    trigger = SmileTrigger(cfg.smile_hold, cfg.cooldown)
    seen = SeenCounter(cfg.count_after)
    camera = Camera(cfg.source, backend, cfg.width, cfg.height)
    w, h = camera.resolution
    via = f" via {backend}" if isinstance(cfg.source, int) else ""
    print(f"{platform.system()} | YOLO on {device} | source {cfg.source!r}{via} at {w}x{h}")
    print("In the video window: space or the Pause button pauses, q or Esc quits.")

    clock = PausableClock()
    button = (0, 0, 0, 0)  # pause button in window pixels, refreshed every frame

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        x1, y1, x2, y2 = button
        if event == cv2.EVENT_LBUTTONDOWN and x1 <= x < x2 and y1 <= y < y2:
            clock.toggle()

    detections: list[Detection] = []
    found: list[Face] = []
    photo: np.ndarray | None = None
    photo_taken = 0.0
    captures = 0
    fps = 0.0
    last = clock.now()
    frame_index = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w + overlay.panel_width(h), h)
    cv2.setMouseCallback(WINDOW, on_mouse)  # OpenCV reports clicks in image pixels, even when resized
    try:
        while True:
            # While paused, nothing is read or processed and the last frame stays on screen.
            if not clock.paused:
                frame = camera.read()
                if frame is None:
                    if camera.live:
                        print("The camera stopped sending frames.")
                    break
                if cfg.mirror:
                    frame = cv2.flip(frame, 1)
                now = clock.now()

                if frame_index % cfg.detect_every == 0:
                    detections = objects.detect(frame)
                    seen.update(detections, now)
                found = faces.analyze(frame, int(now * 1000))
                head_on = [abs(f.yaw) <= cfg.max_yaw and abs(f.pitch) <= cfg.max_pitch for f in found]
                smiling = [ok and f.smile >= cfg.smile_threshold for f, ok in zip(found, head_on)]

                # No new captures while a photo is up, and a smile still held when it
                # goes away counts as the same smile, so it isn't captured twice.
                if photo is None and trigger.update(any(smiling), now):
                    photo = frame.copy()  # taken before any boxes are drawn
                    photo_taken = now
                    captures += 1
                    score = max(f.smile for f, s in zip(found, smiling) if s)
                    print(f"[{time.strftime('%H:%M:%S')}] smile captured (score {score:.2f})")

                for det in detections:
                    overlay.draw_detection(frame, det)
                for face, ok, smile in zip(found, head_on, smiling):
                    overlay.draw_face(frame, face, ok, smile)
                dt, last = now - last, now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt

                view = frame
                if photo is not None:
                    progress = overlay.inset_progress(now - photo_taken, cfg.slide_seconds, cfg.photo_seconds)
                    if progress is None:
                        photo = None
                    else:
                        view = overlay.picture_in_picture(photo, frame, progress)
                frame_index += 1

            paused = clock.paused
            in_view = Counter(d.label for d in detections)
            panel = overlay.stats_panel(
                view.shape[0], in_view, seen.totals, len(found), captures, fps, device, paused
            )
            x1, y1, x2, y2 = overlay.pause_button_rect(view.shape[0])
            button = (x1 + view.shape[1], y1, x2 + view.shape[1], y2)
            cv2.imshow(WINDOW, np.hstack([overlay.draw_paused(view) if paused else view, panel]))
            # Nothing changes while paused, so poll less often and leave the CPU idle.
            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), 27) or _window_closed():
                break
            if key == ord(" "):
                clock.toggle()
    finally:
        camera.close()
        faces.close()
        cv2.destroyAllWindows()
