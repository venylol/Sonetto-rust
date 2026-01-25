// -----------------------------------------------------------------------------
// STUB MODULE (SIMD build placeholder)
// -----------------------------------------------------------------------------
//
// This repository expects `wasm-pack` build artifacts in `web/pkg_simd/` for the
// simd128 build:
//   - sonetto_wasm.js
//   - sonetto_wasm_bg.wasm
//
// In minimal/source-only distributions (like the sandbox ZIP), those artifacts
// may be missing. Historically that caused:
//   - 404 for ./pkg/sonetto_wasm.js
//   - Worker crashed at module load time (often with an unhelpful `undefined`)
//
// This stub ensures the module import always succeeds so the worker can report
// a *clear* actionable error through its normal `type:'error'` message path.
//
// To build the real WASM SIMD engine, run:
//   cd crates/sonetto_wasm
//   RUSTFLAGS="-C target-feature=+simd128" \
//     wasm-pack build --release --target web --out-dir ../../web/pkg_simd --features wasm_simd
//
// When you build, wasm-pack will overwrite this file with the real loader.

function buildMissingWasmError(extra) {
  const msg = [
    'Sonetto WASM SIMD artifacts are missing.',
    'Build them with:',
    '  cd crates/sonetto_wasm',
    '  RUSTFLAGS="-C target-feature=+simd128" \\\n  wasm-pack build --release --target web --out-dir ../../web/pkg_simd --features wasm_simd',
    extra ? String(extra) : null,
  ].filter(Boolean).join('\n');
  return new Error(msg);
}

// wasm-pack's default export is an async init function.
export default async function initWasm() {
  // Give a more helpful hint when people open the HTML directly.
  const proto = (typeof self !== 'undefined' && self.location && self.location.protocol)
    ? self.location.protocol
    : '';
  if (proto === 'file:') {
    throw buildMissingWasmError('Tip: open Sonetto.html via an HTTP server (file:// blocks module workers).');
  }

  throw buildMissingWasmError();
}

// wasm-bindgen glue also exports a named `init()` (we use it for panic hook).
export function init() {
  // no-op in stub
}

// -----------------------------------------------------------------------------
// Engine API stubs: the worker imports these by name.
// -----------------------------------------------------------------------------

function stub() {
  throw buildMissingWasmError();
}

export function engine_new() { return stub(); }
export function engine_analyze() { return stub(); }
export function engine_analyze_v2() { return stub(); }
export function engine_best_move() { return stub(); }
export function engine_set_weights_flat() { return stub(); }
export function engine_get_weights_flat() { return stub(); }
export function engine_set_weights_egev2() { return stub(); }
export function engine_get_weights_egev2() { return stub(); }
