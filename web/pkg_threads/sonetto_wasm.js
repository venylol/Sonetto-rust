// -----------------------------------------------------------------------------
// STUB MODULE (WASM threads build placeholder)
// -----------------------------------------------------------------------------
//
// This repository expects `wasm-pack` build artifacts in `web/pkg_threads/`:
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
// To build the real WASM threads engine, run:
//   cd crates/sonetto_wasm
//   RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals" \
//     wasm-pack build --release --target web --out-dir ../../web/pkg_threads --features wasm-bindgen-rayon
//
// When you build, wasm-pack will overwrite this file with the real loader.

function buildMissingWasmError(extra) {
  const msg = [
    'Sonetto WASM (threads) artifacts are missing.',
    'Build them with:',
    '  cd crates/sonetto_wasm',
    '  RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals" wasm-pack build --release --target web --out-dir ../../web/pkg_threads --features wasm-bindgen-rayon',
    'Note: WASM threads requires Cross-Origin isolation (COOP/COEP) so SharedArrayBuffer is available.',
    'See web/_headers and README_WASM_BUILD.md.',
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
