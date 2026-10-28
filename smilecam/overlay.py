from collections import Counter

import cv2
import numpy as np

from .faces import Face
from .objects import TARGET_CLASSES, Detection

# OpenCV colors are BGR.
OBJECT_COLORS = {
    "person": (255, 160, 0),
    "backpack": (0, 140, 255),
    "handbag": (0, 200, 255),
    "suitcase": (180, 80, 255),
}
DEFAULT_COLOR = (200, 200, 200)
FACE_TURNED_AWAY = (128, 128, 128)
FACE_HEAD_ON = (255, 255, 255)
FACE_SMILING = (80, 220, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_BG = (32, 32, 32)
PANEL_TEXT = (235, 235, 235)
PANEL_MUTED = (150, 150, 150)
BUTTON_IDLE = (70, 70, 70)
BUTTON_ACTIVE = (40, 150, 230)


def _label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw text on a filled background whose bottom-left corner is (x, y)."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
    y = min(max(y, th + baseline + 2), frame.shape[0])  # keep it on screen
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 4, y), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 2, y - baseline), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, det: Detection) -> None:
    color = OBJECT_COLORS.get(det.label, DEFAULT_COLOR)
    x1, y1, x2, y2 = det.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    track = f" #{det.track_id}" if det.track_id is not None else ""
    _label(frame, f"{det.label}{track} {det.conf:.2f}", x1, y1, color)


def draw_face(frame: np.ndarray, face: Face, head_on: bool, smiling: bool) -> None:
    color = FACE_SMILING if smiling else FACE_HEAD_ON if head_on else FACE_TURNED_AWAY
    x1, y1, x2, y2 = face.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Below the box, since the person label usually sits at the top of the frame.
    _label(frame, f"smile {face.smile:.2f}  yaw {face.yaw:+.0f}  pitch {face.pitch:+.0f}", x1, y2 + 20, color)


def inset_progress(elapsed: float, slide: float, hold: float) -> float | None:
    """How far the live feed has moved into the corner `elapsed` seconds after a capture.

    0 is full screen and 1 is fully in the corner. It slides in, holds for `hold` seconds,
    slides back, then returns None once the photo should be dropped.
    """
    back_at = slide + hold
    if elapsed >= back_at + slide:
        return None
    if elapsed < slide:
        t = elapsed / slide
    elif elapsed < back_at:
        t = 1.0
    else:
        t = (back_at + slide - elapsed) / slide
    return t * t * (3 - 2 * t)  # smoothstep: ease in and out of each slide


def picture_in_picture(
    photo: np.ndarray, live: np.ndarray, progress: float, scale: float = 0.3, margin: int = 16
) -> np.ndarray:
    """Show `photo` full screen with `live` shrunk toward the bottom-right corner by `progress`."""
    if progress <= 0:
        return live
    h, w = live.shape[:2]
    # Interpolate each edge between the full frame and the corner inset.
    x1 = round((w - margin - w * scale) * progress)
    y1 = round((h - margin - h * scale) * progress)
    x2 = round(w - margin * progress)
    y2 = round(h - margin * progress)
    canvas = photo.copy()
    cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 2, y2 + 2), (255, 255, 255), cv2.FILLED)
    canvas[y1:y2, x1:x2] = cv2.resize(live, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    return canvas


def panel_width(height: int) -> int:
    return int(300 * height / 720)


def pause_button_rect(height: int) -> tuple[int, int, int, int]:
    """Pause button position (x1, y1, x2, y2) inside the side panel, for drawing and click tests."""
    s = height / 720
    return int(20 * s), height - int(104 * s), int(270 * s), height - int(56 * s)


def draw_paused(frame: np.ndarray) -> np.ndarray:
    """Return a dimmed copy of `frame` with a pause symbol in the middle."""
    out = cv2.convertScaleAbs(frame, alpha=0.5)
    h, w = out.shape[:2]
    bar_w, bar_h, gap = h // 24, h // 6, h // 24
    cx, cy = w // 2, h // 2
    for x in (cx - gap // 2 - bar_w, cx + gap // 2):
        cv2.rectangle(out, (x, cy - bar_h // 2), (x + bar_w, cy + bar_h // 2), (255, 255, 255), cv2.FILLED)
    return out


def stats_panel(
    height: int,
    in_view: Counter,
    seen: Counter,
    faces: int,
    captures: int,
    fps: float,
    device: str,
    paused: bool,
) -> np.ndarray:
    """Side panel: what's in view right now and how many distinct objects have been seen."""
    s = height / 720  # layout is designed at 720p and scaled to the video's height
    panel = np.full((height, panel_width(height), 3), PANEL_BG, np.uint8)
    size, weight = 0.6 * s, max(1, round(s))
    col_now, col_seen = int(200 * s), int(270 * s)

    def text(value: str, x: int, y: int, color=PANEL_TEXT, right: bool = False) -> None:
        if right:
            x -= cv2.getTextSize(value, FONT, size, weight)[0][0]
        cv2.putText(panel, value, (x, y), FONT, size, color, weight, cv2.LINE_AA)

    y = int(40 * s)
    text("now", col_now, y, PANEL_MUTED, right=True)
    text("seen", col_seen, y, PANEL_MUTED, right=True)
    for label in TARGET_CLASSES:
        y += int(36 * s)
        swatch = int(14 * s)
        cv2.rectangle(panel, (int(20 * s), y - swatch), (int(20 * s) + swatch, y), OBJECT_COLORS[label], cv2.FILLED)
        text(label, int(44 * s), y)
        text(str(in_view[label]), col_now, y, right=True)
        text(str(seen[label]), col_seen, y, right=True)

    y += int(24 * s)
    cv2.line(panel, (int(20 * s), y), (col_seen, y), PANEL_MUTED, 1)
    y += int(36 * s)
    text("faces", int(44 * s), y)
    text(str(faces), col_now, y, right=True)
    y += int(36 * s)
    text("smiles captured", int(44 * s), y)
    text(str(captures), col_seen, y, right=True)

    x1, y1, x2, y2 = pause_button_rect(height)
    cv2.rectangle(panel, (x1, y1), (x2, y2), BUTTON_ACTIVE if paused else BUTTON_IDLE, cv2.FILLED)
    text("Resume  [space]" if paused else "Pause  [space]", x1 + int(16 * s), y2 - int(16 * s))

    status = "paused" if paused else f"{fps:4.1f} fps  {device}"
    text(status, int(20 * s), height - int(20 * s), PANEL_MUTED)
    return panel
