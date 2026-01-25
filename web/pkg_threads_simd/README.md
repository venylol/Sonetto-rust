This directory is intended to hold the **WebAssembly threads + SIMD (simd128)** build output.

Generate it via:

    ./scripts/build_wasm_threads_dual.sh

The web worker (web/engine_worker.js) will try to load `./pkg_threads_simd/sonetto_wasm.js`
when SharedArrayBuffer + crossOriginIsolated are available and SIMD is supported.
It will fall back to `./pkg_threads/`, then non-threaded builds (`./pkg_simd/` / `./pkg/`).
