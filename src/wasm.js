export async function unpackWasm(bytes) {
  // Some static hosts (including Vite) transparently decompress .wasm.gz in fetch.
  // Others serve the gzip bytes verbatim. Accept both without double decoding.
  let binary = bytes;
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    binary = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  if (binary[0] !== 0 || binary[1] !== 0x61 || binary[2] !== 0x73 || binary[3] !== 0x6d) {
    throw new Error('The inference runtime download is not a valid WebAssembly binary.');
  }
  return binary;
}
