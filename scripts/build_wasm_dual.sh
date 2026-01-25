#!/usr/bin/env bash
set -euo pipefail

# Build BOTH:
#  - scalar wasm (no simd128) -> web/pkg
#  - simd128 wasm            -> web/pkg_simd
#
# The web worker (web/engine_worker.js) will auto-detect SIMD support at runtime and
# load ./pkg_simd when available, falling back to ./pkg otherwise.
#
# Requirements:
#  - wasm-pack
#  - Rust toolchain with wasm32-unknown-unknown target installed
#
# NOTE:
#  The SIMD build requires WebAssembly SIMD support in the runtime.
#  If you deploy only the SIMD build, older browsers will fail to load it.
#  This dual-build keeps a safe fallback.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

find_wasm_crate_dir() {
  local root="$1"
  local candidates=(
    "$root/crates/sonetto_wasm"
    "$root/sonetto_wasm"
  )
  for c in "${candidates[@]}"; do
    if [[ -f "$c/Cargo.toml" ]]; then
      echo "$c"
      return 0
    fi
  done

  # Fallback: search for a Cargo.toml that declares `name = "sonetto_wasm"`.
  local cargo_toml
  cargo_toml="$(grep -Rsl --include 'Cargo.toml' 'name = "sonetto_wasm"' "$root" 2>/dev/null | head -n 1 || true)"
  if [[ -n "${cargo_toml:-}" ]]; then
    dirname "$cargo_toml"
    return 0
  fi

  return 1
}

WASM_CRATE_DIR="$(find_wasm_crate_dir "$ROOT_DIR" || true)"
if [[ -z "${WASM_CRATE_DIR:-}" || ! -d "$WASM_CRATE_DIR" ]]; then
  echo "::error::Could not find the sonetto_wasm crate directory." >&2
  echo "Expected at: $ROOT_DIR/crates/sonetto_wasm" >&2
  echo "" >&2
  echo "Repository root: $ROOT_DIR" >&2
  echo "Contents of $ROOT_DIR/crates (if present):" >&2
  ls -la "$ROOT_DIR/crates" 2>/dev/null || true
  exit 1
fi

OUT_SCALAR="$ROOT_DIR/web/pkg"
OUT_SIMD="$ROOT_DIR/web/pkg_simd"

echo "[1/2] Building scalar wasm -> ${OUT_SCALAR}"
(cd "$WASM_CRATE_DIR" &&   RUSTUP_TOOLCHAIN=stable   wasm-pack build --release --no-opt --target web --out-dir "$OUT_SCALAR")

echo "[2/2] Building wasm SIMD (simd128) -> ${OUT_SIMD}"
(cd "$WASM_CRATE_DIR" &&   RUSTUP_TOOLCHAIN=stable   RUSTFLAGS="-C target-feature=+simd128"   wasm-pack build --release --no-opt --target web --out-dir "$OUT_SIMD" --features wasm_simd)

echo "[post] wasm-opt (if available)"
bash scripts/wasm_opt_dir.sh "$OUT_SCALAR"
bash scripts/wasm_opt_dir.sh "$OUT_SIMD"

echo "Done."
