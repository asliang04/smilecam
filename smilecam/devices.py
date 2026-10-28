"""Per-platform choices: which accelerator YOLO runs on and which camera API OpenCV uses."""

import platform

import cv2

CAMERA_BACKENDS = {
    "any": cv2.CAP_ANY,
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
    "dshow": cv2.CAP_DSHOW,  # Windows DirectShow
    "msmf": cv2.CAP_MSMF,  # Windows Media Foundation
    "v4l2": cv2.CAP_V4L2,  # Linux, including Raspberry Pi
}


def torch_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def camera_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    system = platform.system()
    if system == "Darwin":
        return "avfoundation"
    if system == "Windows":
        # OpenCV defaults to MSMF on Windows, which can take 10+ seconds to open some webcams.
        return "dshow"
    if system == "Linux":
        return "v4l2"
    return "any"
