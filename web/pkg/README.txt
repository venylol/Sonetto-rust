This folder is expected to contain wasm-pack build artifacts:
  - sonetto_wasm.js
  - sonetto_wasm_bg.wasm

Normally generated via:
  cd crates/sonetto_wasm
  wasm-pack build --release --target web --out-dir ../../web/pkg

Notes:
  - In some source-only/sandbox distributions we cannot run wasm-pack here.
  - For convenience, the repo ships a small *stub* `sonetto_wasm.js` that
    exports the expected symbols and throws a clear "please build wasm" error.
    When you run wasm-pack, it overwrites the stub with the real loader.
