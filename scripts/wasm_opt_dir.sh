#!/usr/bin/env bash
set -euo pipefail

# Optimize the wasm-bindgen output wasm in a wasm-pack package directory.
#
# Usage:
#   scripts/wasm_opt_dir.sh <pkg_dir>
#
# Example:
#   scripts/wasm_opt_dir.sh web/pkg_simd

DIR="${1:?pkg dir required}"
WASM="${DIR}/sonetto_wasm_bg.wasm"

if ! command -v wasm-opt >/dev/null 2>&1; then
  echo "[wasm-opt] wasm-opt not found; skipping (${DIR})" >&2
  exit 0
fi

if [[ ! -f "${WASM}" ]]; then
  echo "[wasm-opt] no wasm file at ${WASM}; skipping" >&2
  exit 0
fi

# Rust (via LLVM) emits post-MVP Wasm ops by default (e.g. bulk-memory,
# sign-ext, non-trapping float-to-int). wasm-opt validates modules and will
# fail unless these features are explicitly enabled.
#
# We always enable the commonly required features; then add SIMD/threads flags
# based on the package directory name.
FLAGS=(
  -O3
  --enable-bulk-memory
  --enable-sign-ext
  --enable-nontrapping-float-to-int
)

# Package directory naming convention:
# - web/pkg_simd
# - web/pkg_threads
# - web/pkg_threads_simd
if [[ "${DIR}" == *simd* ]]; then
  FLAGS+=(--enable-simd)
fi
if [[ "${DIR}" == *threads* ]]; then
  FLAGS+=(--enable-threads --enable-mutable-globals)
  # Older binaryen releases don't recognize --enable-atomics; threads validation still works without it.
  if wasm-opt --help 2>/dev/null | grep -q -- '--enable-atomics'; then
    FLAGS+=(--enable-atomics)
  fi
fi

TMP="${WASM}.opt"
echo "[wasm-opt] optimizing ${WASM} -> ${TMP}  flags: ${FLAGS[*]}"
wasm-opt "${FLAGS[@]}" -o "${TMP}" "${WASM}"
mv "${TMP}" "${WASM}"
