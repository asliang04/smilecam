import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { ObjectTracker, decodeYolo, letterbox, faceResults } from './core.js';
import { unpackWasm } from './wasm.js';

let landmarker, session, ort, tracker;
let backend = 'CPU', lastTimestamp = -1;
const inputSize = 320;
const inputCanvas = new OffscreenCanvas(inputSize, inputSize);
const inputContext = inputCanvas.getContext('2d', { willReadFrequently: true });

async function checkedFetch(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${new URL(url).pathname} (${response.status}).`);
  return new Uint8Array(await response.arrayBuffer());
}

async function initialize({ baseUrl, forceCpu = false }) {
  self.postMessage({ type: 'status', message: 'Loading local face model…' });
  const files = await FilesetResolver.forVisionTasks(new URL('vendor/mediapipe', baseUrl).href, true);
  landmarker = await FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetBuffer: await checkedFetch(new URL('models/face_landmarker.task', baseUrl)), delegate: 'CPU' },
    runningMode: 'VIDEO', numFaces: 4, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  });
  self.postMessage({ type: 'status', message: 'Loading local object model…' });
  const wantGpu = !forceCpu && !!self.navigator.gpu;
  const runtimeUrl = new URL(`vendor/onnx/ort.${wantGpu ? 'webgpu' : 'wasm'}.bundle.min.mjs`, baseUrl).href;
  ort = await import(/* @vite-ignore */ runtimeUrl);
  ort.env.wasm.wasmPaths = new URL('vendor/onnx/', baseUrl).href;
  const binaryUrl = new URL(`vendor/onnx/ort-wasm-simd-threaded${wantGpu ? '.asyncify' : ''}.wasm.gz`, baseUrl);
  const compressed = await checkedFetch(binaryUrl);
  ort.env.wasm.wasmBinary = await unpackWasm(compressed);
  // Avoid cross-origin isolation requirements; inference already runs off the UI thread.
  ort.env.wasm.numThreads = 1;
  const model = await checkedFetch(new URL('models/yolo11n.onnx', baseUrl));
  if (wantGpu) {
    try {
      session = await ort.InferenceSession.create(model, { executionProviders: ['webgpu'] });
      backend = 'WebGPU';
      // Exercise the graph before declaring success: some adapters fail on the first run.
      const outputs = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * inputSize ** 2), [1, 3, inputSize, inputSize]) });
      for (const value of Object.values(outputs)) value.dispose();
    } catch (error) {
      await session?.release();
      session = null;
      console.warn('WebGPU unavailable; using local CPU inference.', error);
      self.postMessage({ type: 'status', message: 'Using local CPU processing…' });
    }
  }
  if (!session) {
    session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
    backend = 'CPU';
  }
  tracker = new ObjectTracker();
  return { backend };
}

async function infer({ bitmap, timestamp, now }) {
  try {
    const { width, height } = bitmap;
    const transform = letterbox(width, height, inputSize);
    inputContext.fillStyle = 'rgb(114,114,114)';
    inputContext.fillRect(0, 0, inputSize, inputSize);
    inputContext.drawImage(bitmap, transform.left, transform.top, transform.drawWidth, transform.drawHeight);
    const pixels = inputContext.getImageData(0, 0, inputSize, inputSize).data;
    const plane = inputSize ** 2, input = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      input[i] = pixels[i * 4] / 255;
      input[plane + i] = pixels[i * 4 + 1] / 255;
      input[2 * plane + i] = pixels[i * 4 + 2] / 255;
    }
    const tensor = new ort.Tensor('float32', input, [1, 3, inputSize, inputSize]);
    let outputs;
    try {
      outputs = await session.run({ [session.inputNames[0]]: tensor });
      const output = outputs[session.outputNames[0]];
      const decoded = decodeYolo(await output.getData(), output.dims, transform, width, height);
      const detections = tracker.update(decoded, now);
      lastTimestamp = Math.max(timestamp, lastTimestamp + 1);
      const faces = faceResults(landmarker.detectForVideo(bitmap, lastTimestamp), width, height);
      return { detections, faces, totals: { ...tracker.totals }, backend, width, height };
    } finally {
      tensor.dispose();
      if (outputs) for (const value of Object.values(outputs)) value.dispose();
    }
  } finally { bitmap.close(); }
}

self.onmessage = async ({ data }) => {
  try {
    let result;
    if (data.type === 'init') result = await initialize(data);
    else if (data.type === 'infer') result = await infer(data);
    else if (data.type === 'reset-tracks') { tracker = new ObjectTracker(); result = {}; }
    else throw new Error(`Unknown worker request: ${data.type}`);
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message || String(error) });
  }
};
