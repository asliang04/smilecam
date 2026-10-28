# smilecam

Live webcam window with boxes around people, bags (backpack, handbag, suitcase) and faces.
When someone smiles at the camera, the frame is captured and fills the window for a few
seconds while the live feed slides into the bottom-right corner. Then the live feed slides back
to full screen and the photo is discarded. Nothing is written to disk.

- Objects: [Ultralytics YOLO11](https://docs.ultralytics.com) (COCO-pretrained, AGPL-3.0)
- Faces, smile score, head pose: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Apache-2.0)

## Setup

Needs Python 3.10+. Model files (~10 MB) download into `models/` on first run.

**macOS / Linux**
```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m smilecam
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m smilecam
```
For an NVIDIA GPU on Windows, replace the CPU-only PyTorch that pip installs by default
with a CUDA build. Get the exact command from <https://pytorch.org/get-started/locally/>, e.g.
`.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
See [Windows](#windows) for known issues.

The first run asks for camera permission. If it was denied, allow your terminal or editor under
System Settings > Privacy & Security > Camera (macOS) or Settings > Privacy & security > Camera (Windows).

In the video window, press `Space` or click **Pause** in the side panel to pause, and `q` or `Esc`
to quit. Pausing freezes the picture and stops all detection. The smile timers and any photo
on screen pick up where they stopped when you resume. The webcam itself stays on while paused.
People who move a lot during a pause may get new track IDs and be counted again.

## Reading the window

| Box | Meaning |
|---|---|
| blue / orange / yellow / pink | person / backpack / handbag / suitcase, with track ID and confidence |
| gray face | face turned away (outside `--max-yaw` / `--max-pitch`) |
| white face | head-on, not smiling enough |
| green face | head-on smile; holding it for 0.25 s triggers a capture |

Under each face: `smile` (0–1), `yaw` (left/right turn) and `pitch` (up/down tilt) in degrees.
Watch those numbers to tune `--smile-threshold` for your face and lighting. A big smile scores
about 0.9; a closed-mouth half smile about 0.6. Each smile is captured once. Smiles are ignored
while a photo is up (`--photo-seconds`, plus 0.4 s for each slide), and a smile still held when the
photo goes away doesn't count again. Every capture prints a line to the terminal.

The panel on the right shows how many of each object are in view **now**, and how many distinct
ones have been **seen** this session. Distinct means distinct track IDs: the ByteTrack tracker
follows each object from frame to frame. An object counts once it has been tracked for 0.5 s,
and a track survives being hidden for about 2.5 s (`track_buffer` in `smilecam/bytetrack.yaml`).
Someone who leaves the frame for longer and comes back is counted again.

## Running on other machines

Each per-platform choice is made automatically, and each can be overridden:

| | macOS (Apple Silicon) | Windows | Linux / Raspberry Pi |
|---|---|---|---|
| YOLO device (`--device`) | `mps` | `cuda:0` if NVIDIA + CUDA torch, else `cpu` | `cuda:0` or `cpu` |
| Camera API (`--camera-backend`) | `avfoundation` | `dshow` (try `msmf` if the image is wrong) | `v4l2` |

The face model always runs on the CPU. It's fast enough everywhere.

**Slow CPU-only devices:** `--imgsz 320 --detect-every 3` runs YOLO at a quarter of the pixels
on every third frame, reusing boxes in between. The smile trigger is timed in seconds, so it
behaves the same at any frame rate.

**Exported models:** `--yolo-model` accepts any format Ultralytics can load, so you can export once
for the target hardware:
```sh
.venv/bin/yolo export model=models/yolo11n.pt format=openvino   # Intel CPUs/iGPUs
.venv/bin/yolo export model=models/yolo11n.pt format=onnx       # portable
.venv/bin/yolo export model=models/yolo11n.pt format=coreml     # Apple Neural Engine
.venv/bin/python -m smilecam --yolo-model models/yolo11n_openvino_model
```
For better accuracy at some speed cost, use `--yolo-model yolo11s.pt` (downloads automatically).

**Other video sources:** `--source 1` picks another webcam. `--source clip.mp4` or
`--source rtsp://...` reads a file or IP camera.

**Intel Macs:** current MediaPipe has no Intel-Mac build. With Python 3.12 or older, pip falls
back to MediaPipe 0.10.21, which uses the same API but hasn't been tested with this project.

### Windows

Windows is supported by design but **hasn't been tested on a real Windows machine yet**. Every
dependency has a Windows build, `devices.py` picks CUDA or the CPU and the DirectShow camera API,
and nothing in the code is Mac-specific. On a normal Intel/AMD PC it should run after the setup
steps above. These are the problems most likely to come up:

- **Wrong resolution or a low frame rate from the webcam.** DirectShow sometimes ignores the
  720p request or delivers slow uncompressed video. Try `--camera-backend msmf` first. If that
  doesn't help, the usual fix is to ask the camera for MJPG in `camera.py`, before the width and
  height are set: `self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))`.
- **NVIDIA GPU not used.** pip installs a CPU-only PyTorch on Windows; see the CUDA command under
  Setup. The first line the app prints shows which device YOLO ended up on (`YOLO on cuda:0`
  or `YOLO on cpu`).
- **Slow on laptops without an NVIDIA GPU.** YOLO runs on the CPU, likely 2–3× slower than on an
  Apple Silicon GPU. Use `--imgsz 320 --detect-every 2`, or on Intel CPUs export to OpenVINO:
  ```powershell
  .venv\Scripts\yolo export model=models\yolo11n.pt format=openvino
  .venv\Scripts\python -m smilecam --yolo-model models\yolo11n_openvino_model
  ```
- **Jumpy FPS readout on Python 3.12 or older.** There, `time.monotonic()` on Windows only ticks
  about every 15 ms. This is cosmetic. Use Python 3.13+ (`py -3.13 -m venv .venv`) or switch the
  FPS timer in `app.py` to `time.perf_counter()`.
- **Windows on ARM (Snapdragon laptops).** MediaPipe has ARM builds, but PyTorch doesn't publish
  Windows-on-ARM builds on PyPI, so `pip install` will likely fail. This is the one Windows
  setup that may need real extra work.

## Layout

```
smilecam/
  __main__.py  command-line flags -> Config
  config.py    every tunable default
  devices.py   per-platform accelerator and camera-API choice
  camera.py    webcam/file/stream reader (keeps only the newest live frame)
  objects.py   YOLO + ByteTrack wrapper -> Detection(label, conf, box, track_id)
  bytetrack.yaml  tracker settings
  counting.py  distinct-object totals from track IDs
  faces.py     MediaPipe wrapper -> Face(box, smile, yaw, pitch)
  smile.py     smile-start / hold / cooldown logic
  overlay.py   all drawing: boxes, side panel, photo/live-feed slide animation
  app.py       main loop and display window
```
Only `app.py` and `overlay.py` touch the display. To run headless on another device (e.g.
sending capture events over the network), replace the display part of `app.py` and reuse the rest.
