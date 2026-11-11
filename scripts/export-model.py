"""Export COCO-pretrained YOLO11 weights to the browser's ONNX model asset."""
from pathlib import Path
import shutil
from ultralytics import YOLO

root = Path(__file__).resolve().parent.parent
source = root / "models" / "yolo11n.pt"
if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    # Official weights are downloaded by Ultralytics when missing.
model = YOLO(str(source))
exported = model.export(format="onnx", imgsz=320, opset=17, simplify=False,
                        dynamic=False, nms=False, device="cpu")
destination = root / "public" / "models" / "yolo11n.onnx"
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(exported, destination)
print(f"Browser model: {destination}")
