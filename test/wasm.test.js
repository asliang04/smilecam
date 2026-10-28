import test from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { unpackWasm } from '../src/wasm.js';

test('runtime loading accepts gzip bytes and transparently decompressed HTTP responses', async () => {
  const binary = new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0]);
  assert.deepEqual(await unpackWasm(binary), binary);
  assert.deepEqual(await unpackWasm(gzipSync(binary)), binary);
  await assert.rejects(unpackWasm(new TextEncoder().encode('<html>Not found</html>')), /not a valid WebAssembly/);
});
