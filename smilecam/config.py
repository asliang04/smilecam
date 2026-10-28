from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass
class Config:
    # Video input: a webcam index, a video file path, or a stream URL (rtsp://, http://).
    source: int | str = 0
    camera_backend: str = "auto"  # auto | any | avfoundation | dshow | msmf | v4l2
    width: int = 1280
    height: int = 720
    mirror: bool = True

    # Object detection. Any format Ultralytics can load works here (.pt, .onnx,
    # an OpenVINO dir, a CoreML .mlpackage), so other devices can use an export.
    yolo_model: Path = MODELS_DIR / "yolo11n.pt"
    device: str = "auto"  # auto | cpu | mps | cuda | cuda:N
    object_conf: float = 0.4
    imgsz: int = 640
    detect_every: int = 1  # run YOLO on every Nth frame and reuse boxes in between
    count_after: float = 0.5  # seconds a tracked object must last before it counts as "seen"

    # Face analysis and the smile trigger.
    face_model: Path = MODELS_DIR / "face_landmarker.task"
    max_faces: int = 4
    smile_threshold: float = 0.7  # mean of MediaPipe's mouthSmileLeft/Right blendshapes
    max_yaw: float = 15.0  # degrees of left/right turn still counted as head-on
    max_pitch: float = 20.0  # looser: laptop webcams sit above the eyes
    smile_hold: float = 0.25  # seconds a smile must last before it counts
    cooldown: float = 3.0  # minimum seconds between captures

    # After a capture the photo fills the window and the live feed slides into a corner.
    photo_seconds: float = 3.0  # how long the photo stays up, not counting the slides
    slide_seconds: float = 0.4  # length of each slide animation
