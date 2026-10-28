import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import * as ort from '../public/vendor/onnx/ort.wasm.bundle.min.mjs';
import { decodeYolo, letterbox } from '../src/core.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmBinary = gunzipSync(await readFile(new URL('../public/vendor/onnx/ort-wasm-simd-threaded.wasm.gz', import.meta.url)));
const session = await ort.InferenceSession.create(new Uint8Array(await readFile(new URL('../public/models/yolo11n.onnx', import.meta.url))), { executionProviders: ['wasm'] });
const tensor = new ort.Tensor('float32', new Float32Array(3 * 320 ** 2), [1, 3, 320, 320]);
try {
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const output = results[session.outputNames[0]];
  assert.deepEqual(output.dims, [1, 84, 2100]);
  assert.ok(output.data.every(Number.isFinite));
  assert.deepEqual(decodeYolo(output.data, output.dims, letterbox(1280, 720), 1280, 720, 0.4), []);
  for (const value of Object.values(results)) value.dispose();
  console.log('PASS: shipped compressed WebAssembly runtime loads and runs the committed YOLO model.');
} finally { tensor.dispose(); await session.release(); }
