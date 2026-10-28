import math
import os
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe's C++ startup logging

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    smile: float  # 0..1
    yaw: float  # degrees; 0 means facing straight down the camera axis
    pitch: float


def ensure_face_model(path: Path) -> None:
    if path.exists():
        return
    import certifi  # python.org macOS builds ship without system root certificates

    print(f"Downloading face model to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(FACE_MODEL_URL, context=context) as response:
        data = response.read()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)


class FaceAnalyzer:
    """MediaPipe Face Landmarker: per-face smile score and head orientation."""

    def __init__(self, model_path: Path, max_faces: int):
        ensure_face_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def analyze(self, frame: np.ndarray, timestamp_ms: int) -> list[Face]:
        # VIDEO mode needs strictly increasing timestamps; it uses them to track faces between frames.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        h, w = frame.shape[:2]
        faces = []
        for landmarks, blendshapes, matrix in zip(
            result.face_landmarks, result.face_blendshapes, result.facial_transformation_matrixes
        ):
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            box = (int(min(xs) * w), int(min(ys) * h), int(max(xs) * w), int(max(ys) * h))

            scores = {c.category_name: c.score for c in blendshapes}
            smile = (scores["mouthSmileLeft"] + scores["mouthSmileRight"]) / 2

            # Third column of the rotation is the direction the face points, in camera
            # coordinates; a face looking straight at the camera gives (0, 0, 1).
            fx, fy, fz = np.asarray(matrix)[:3, 2]
            yaw = math.degrees(math.atan2(fx, fz))
            pitch = math.degrees(math.atan2(fy, fz))
            faces.append(Face(box, smile, yaw, pitch))
        return faces

    def close(self) -> None:
        self._landmarker.close()
