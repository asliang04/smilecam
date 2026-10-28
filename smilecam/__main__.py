import argparse
from pathlib import Path

from .app import run
from .camera import CameraError
from .config import MODELS_DIR, Config
from .devices import CAMERA_BACKENDS


def parse_args() -> Config:
    d = Config()
    p = argparse.ArgumentParser(
        prog="python -m smilecam",
        description="Live video with boxes around people, bags, and faces; captures head-on smiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default=str(d.source), help="webcam index, video file, or stream URL")
    p.add_argument("--camera-backend", default=d.camera_backend, choices=["auto", *CAMERA_BACKENDS])
    p.add_argument("--width", type=int, default=d.width, help="requested webcam width")
    p.add_argument("--height", type=int, default=d.height, help="requested webcam height")
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=d.mirror, help="flip like a mirror")
    p.add_argument("--yolo-model", type=Path, default=d.yolo_model, help="weights or exported model")
    p.add_argument("--device", default=d.device, help="auto, cpu, mps, cuda, or cuda:N")
    p.add_argument("--conf", dest="object_conf", type=float, default=d.object_conf, help="min object confidence")
    p.add_argument("--imgsz", type=int, default=d.imgsz, help="YOLO input size; 320 is much faster")
    p.add_argument("--detect-every", type=int, default=d.detect_every, help="run YOLO every Nth frame")
    p.add_argument("--max-faces", type=int, default=d.max_faces, help="faces analyzed per frame")
    p.add_argument("--smile-threshold", type=float, default=d.smile_threshold, help="0..1; watch the live score")
    p.add_argument("--max-yaw", type=float, default=d.max_yaw, help="max head turn, degrees")
    p.add_argument("--max-pitch", type=float, default=d.max_pitch, help="max head tilt, degrees")
    p.add_argument("--cooldown", type=float, default=d.cooldown, help="seconds between captures")
    p.add_argument("--photo-seconds", type=float, default=d.photo_seconds, help="how long a capture is shown")
    args = p.parse_args()

    if args.detect_every < 1:
        p.error("--detect-every must be at least 1")
    args.source = int(args.source) if args.source.isdigit() else args.source
    # Bare names like yolo11s.pt are kept in models/ alongside the default weights.
    if not args.yolo_model.exists() and args.yolo_model.parent == Path("."):
        args.yolo_model = MODELS_DIR / args.yolo_model
    return Config(**vars(args))


def main() -> None:
    cfg = parse_args()
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass
    except CameraError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
