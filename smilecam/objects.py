from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Looked up by name rather than COCO index so a custom or fine-tuned model also works.
TARGET_CLASSES = ("person", "backpack", "handbag", "suitcase")
TRACKER_CONFIG = Path(__file__).with_name("bytetrack.yaml")


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    track_id: int | None  # stays the same across frames for the same object; None until confirmed


class ObjectDetector:
    """YOLO detection plus ByteTrack, which links detections across frames into tracks."""

    def __init__(self, model_path: Path, device: str, conf: float, imgsz: int):
        from ultralytics import YOLO

        # Ultralytics downloads official weights (yolo11n.pt, yolo11s.pt, ...) to this path if missing.
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(model_path))
        self._names = self._model.names
        self._class_ids = [i for i, name in self._names.items() if name in TARGET_CLASSES]
        if not self._class_ids:
            raise ValueError(f"{model_path} has none of the classes {TARGET_CLASSES}")
        self._device = device
        self._conf = conf
        self._imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self._model.track(
            frame,
            persist=True,  # keep tracker state between calls
            tracker=str(TRACKER_CONFIG),
            classes=self._class_ids,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        # No IDs on frames where the tracker has nothing confirmed yet (e.g. the first frame).
        ids = [int(i) for i in boxes.id.tolist()] if boxes.id is not None else [None] * len(boxes)
        return [
            Detection(self._names[int(c)], float(s), tuple(int(v) for v in xyxy), track_id)
            for xyxy, c, s, track_id in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), ids)
        ]
