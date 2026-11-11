# smilecam

Live webcam with boxes around people, backpacks, handbags, suitcases, and faces.
A head-on smile held for 0.25 seconds captures a photo. The live feed slides into
the corner for three seconds, then returns to full screen. Photos are discarded.

## Run locally

All camera processing runs locally in the browser. The host serves static code,
models, and inference runtimes. Camera frames and photos are never uploaded,
saved to disk, or sent to analytics. Camera permission is requested only when
**Start camera** is clicked. **Stop**, `Esc`, or `q` releases the camera and clears
temporary image buffers. Closing the page also releases it.

Use a current Chrome or Edge on Windows. WebGPU is attempted automatically; if
unavailable, processing falls back to the local CPU. **CPU only** forces the CPU
backend before starting. Both backends run in a worker so inference does not
block the interface. The face model uses the CPU.

Requires Node.js 22.12 or later.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. Camera access requires **HTTPS or
localhost**; opening `index.html` from disk or using ordinary HTTP on another PC
will not work. Models and runtimes load when starting the camera and may take a
moment on the first run. Reliable offline use is not implemented.

- **Space / Pause** freezes the picture and all session timers. The webcam stays
  active while paused. Hiding the tab pauses automatically.
- **Smile threshold** defaults to `0.70`; smaller values accept smaller smiles.
- Faces are gray when turned away, white when head-on, and green when smiling.
  Labels show smile score, yaw, and pitch. Head-on means yaw within 15° and pitch
  within 20°.
- Object labels show confidence and track ID. **now** counts visible detections;
  **seen** counts confirmed tracks present for at least 0.5 seconds. A track can
  survive about 2.5 seconds out of view. These are tracked visits, not unique
  identities. A two-stage IoU/motion tracker uses global assignment to match
  detections across frames.
- A held smile captures once. It must be released before another capture, with
  a minimum three-second cooldown. Captures are suppressed while a photo is up.
- Camera or mirror changes start a new session and reset counts.

## Build and host

```sh
npm test
npm run build
npm run preview
```

Deploy **`dist/`** to any HTTPS static host. Model files are committed in
`public/models/`. `npm ci` copies the pinned runtime assets from `node_modules/`
into `public/vendor/`; no CDN is used at runtime. ONNX WebAssembly binaries are
stored as `.wasm.gz` and decompressed locally to keep individual assets below
common static-host file limits. The loader accepts both ordinary gzip files
and hosts that transparently decompress them through `Content-Encoding: gzip`.

The default relative base supports both a root URL and a `/smilecam/` subpath.
For a specific base path, set `BASE_PATH` when building.

For **GitHub Pages**, open the repository's **Settings → Pages**, select
**GitHub Actions** as the source, then run **Deploy web app** from the Actions
tab. The workflow builds and publishes `dist/`. Pages availability for a private
repository depends on the GitHub account plan. The separate CI workflow runs
tests and a production build on pushes and pull requests.

## Verification

`npm test` covers capture timing, cooldown, paused clocks, tracking after
occlusion, track expiry, low-confidence matching, assignment, YOLO decoding,
class filtering, suppression, and pose conversion.

`npm run test:models` checks the committed ONNX model with the exact shipped
CPU WebAssembly runtime. For both models in a real browser, run the dev server,
open `/test/browser.html`, and click **Run model test**. This uses Google's
sample portrait and Ultralytics' bus image without requesting camera access.
Append `?gpu=1` to exercise automatic WebGPU selection and fallback. The fixture
page is a development test and is not included in `dist/`.

Windows webcam operation and WebGPU performance still need testing on an actual
Windows PC. Performance and supported GPU adapters vary by device and browser.

## Models and licenses

- `public/models/yolo11n.onnx`: exported from the COCO-pretrained
  Ultralytics `yolo11n.pt`, input `1×3×320×320`, ONNX opset 17, raw output
  `1×84×2100`. Model is **AGPL-3.0**; see
  [Ultralytics licensing](https://www.ultralytics.com/license).
- `public/models/face_landmarker.task`: the Google MediaPipe face
  landmarker, **Apache-2.0**; see
  [MediaPipe](https://github.com/google-ai-edge/mediapipe).
- MediaPipe runtime: Apache-2.0. ONNX Runtime: MIT. Their license notices are
  included in `public/licenses/` and the built `dist/licenses/`.
- Test images: [Google MediaPipe portrait](https://storage.googleapis.com/mediapipe-assets/portrait.jpg)
  and [Ultralytics bus](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/assets).

### Regenerate the object model (optional)

Development and production builds use the committed ONNX model. To regenerate
it, install the Python export tools in a virtual environment:

```sh
python -m pip install 'ultralytics>=8.3' onnx
python scripts/export-model.py
```

The script reads `models/yolo11n.pt`, downloading the official weights if
missing, and writes `public/models/yolo11n.onnx` with the browser's expected
input and output shapes. Commit the exported asset after regeneration.
