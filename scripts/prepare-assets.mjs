import { cp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { gzip } from 'node:zlib';
import { promisify } from 'node:util';

const root = fileURLToPath(new URL('../', import.meta.url));
const vendor = path.join(root, 'public/vendor');
await rm(vendor, { recursive: true, force: true });
await mkdir(path.join(vendor, 'onnx'), { recursive: true });
await mkdir(path.join(vendor, 'mediapipe'), { recursive: true });
for (const extension of ['js', 'wasm']) {
  const file = `vision_wasm_module_internal.${extension}`;
  await cp(path.join(root, 'node_modules/@mediapipe/tasks-vision/wasm', file), path.join(vendor, 'mediapipe', file));
}
const ort = path.join(root, 'node_modules/onnxruntime-web/dist');
for (const file of ['ort.wasm.bundle.min.mjs', 'ort.webgpu.bundle.min.mjs',
  'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.asyncify.mjs']) {
  await cp(path.join(ort, file), path.join(vendor, 'onnx', file));
}
// Keep every static asset below common hosts' 25 MiB per-file limit.
// Workers decompress these locally before passing the bytes to ONNX Runtime.
for (const file of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  await writeFile(path.join(vendor, 'onnx', `${file}.gz`), await promisify(gzip)(await readFile(path.join(ort, file)), { level: 9 }));
}
console.log('Browser inference runtimes copied to public/vendor (no CDN requests).');
