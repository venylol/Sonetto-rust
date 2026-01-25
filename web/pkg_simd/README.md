This directory is intended to hold the **WebAssembly SIMD (simd128)** build output.

Generate it via:

    ./scripts/build_wasm_dual.sh

The web worker (web/engine_worker.js) will try to load `./pkg_simd/sonetto_wasm.js`
when SIMD is supported by the runtime, and fall back to the scalar build at `./pkg/`.
