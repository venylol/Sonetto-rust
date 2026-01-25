#!/usr/bin/env bash
set -euo pipefail

# Builds 4 variants:
#   - scalar         -> web/pkg
#   - SIMD (simd128) -> web/pkg_simd
#   - threads (rayon)        -> web/pkg_threads
#   - threads + SIMD (rayon) -> web/pkg_threads_simd
#
# Notes:
# - The "threads" variants require WebAssembly threads support:
#   you must serve the site with COOP/COEP to enable SharedArrayBuffer.
#   See web/_headers and README_WASM_BUILD.md.

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

# wasm-bindgen-rayon needs a nightly toolchain + build-std.
#
# Newer nightlies can occasionally regress shared-memory initialization
# (manifesting as: `DataCloneError: Failed to execute 'postMessage' on 'Worker':
# #<Memory> could not be cloned` during initThreadPool). To keep CI reliable we
# default to `nightly` (override via env if you need to pin a specific date).
THREADS_TOOLCHAIN_DEFAULT="nightly"
THREADS_TOOLCHAIN="${SONETTO_THREADS_TOOLCHAIN:-$THREADS_TOOLCHAIN_DEFAULT}"

OUT_SCALAR="$ROOT_DIR/web/pkg"
OUT_SIMD="$ROOT_DIR/web/pkg_simd"
OUT_THREADS="$ROOT_DIR/web/pkg_threads"
OUT_THREADS_SIMD="$ROOT_DIR/web/pkg_threads_simd"

printf '\n== Sonetto: building wasm-pack outputs (scalar + SIMD + threads) ==\n\n'

echo "[1/4] Scalar (no threads) -> ${OUT_SCALAR}"
# NOTE: wasm-pack resolves paths relative to the current directory, but it accepts
# absolute output directories. Using absolute paths makes CI robust if the wasm
# crate is moved within the repo.
(cd "$WASM_CRATE_DIR" && \
  RUSTUP_TOOLCHAIN=stable \
  rustup run "$THREADS_TOOLCHAIN" wasm-pack build --release --no-opt --target web --out-dir "$OUT_SCALAR")

echo "[2/4] SIMD (no threads) -> ${OUT_SIMD}"
(cd "$WASM_CRATE_DIR" && \
  RUSTUP_TOOLCHAIN=stable \
  RUSTFLAGS="-C target-feature=+simd128" \
  rustup run "$THREADS_TOOLCHAIN" wasm-pack build --release --no-opt --target web --out-dir "$OUT_SIMD" --features wasm_simd)

HAVE_THREADS_TOOLCHAIN=0
NEED_RUST_SRC=0
if command -v rustup >/dev/null 2>&1; then
  if rustup toolchain list | grep -q "^${THREADS_TOOLCHAIN}"; then
    HAVE_THREADS_TOOLCHAIN=1
  elif rustup toolchain list | grep -q '^nightly'; then
    # Fallback for local environments that have `nightly` but not the pinned toolchain.
    THREADS_TOOLCHAIN="nightly"
    HAVE_THREADS_TOOLCHAIN=1
  fi

  if [[ "$HAVE_THREADS_TOOLCHAIN" -eq 1 ]]; then
    # build-std requires rust-src.
    if ! rustup component list --toolchain "$THREADS_TOOLCHAIN" --installed 2>/dev/null | grep -q '^rust-src'; then
      NEED_RUST_SRC=1
      HAVE_THREADS_TOOLCHAIN=0
    fi
  fi
fi

if [[ "$HAVE_THREADS_TOOLCHAIN" -ne 1 ]]; then
  if [[ "$NEED_RUST_SRC" -eq 1 ]]; then
    SKIP_REASON="rust-src component is missing for toolchain '$THREADS_TOOLCHAIN'"
  else
    SKIP_REASON="the required Rust nightly toolchain is not installed"
  fi
  cat <<EOF

[3/4] Threads (rayon) -> ${OUT_THREADS}
[4/4] Threads + SIMD (rayon) -> ${OUT_THREADS_SIMD}

NOTE: Skipping threads builds because $SKIP_REASON.
To enable wasm threads builds locally:
  rustup toolchain install ${THREADS_TOOLCHAIN_DEFAULT}
  rustup component add rust-src --toolchain ${THREADS_TOOLCHAIN_DEFAULT}
  rustup target add wasm32-unknown-unknown --toolchain ${THREADS_TOOLCHAIN_DEFAULT}

If you already have the toolchain but not rust-src:
  rustup component add rust-src --toolchain ${THREADS_TOOLCHAIN_DEFAULT}

The scalar/SIMD builds are still available (${OUT_SCALAR}, ${OUT_SIMD}).
EOF
else
  echo "[3/4] Threads (rayon) -> ${OUT_THREADS}"
  echo "Using threads toolchain: $THREADS_TOOLCHAIN"
  # Ensure the cargo wrapper is executable in CI (git mode bits can be lost when copying files).
  chmod +x "$ROOT_DIR/scripts/cargo_with_toolchain.sh" || true
  ls -l "$ROOT_DIR/scripts/cargo_with_toolchain.sh" || true
  rustup run "$THREADS_TOOLCHAIN" cargo --version
  echo "cargo on PATH: $(command -v cargo || true)"
  cargo --version || true

  # wasm-pack cannot reliably forward `-Z build-std=...` to Cargo across all wrappers/subcommands.
  # Cargo allows configuring any `-Z` flag via `.cargo/config.toml` in the `[unstable]` table.
  # We generate a temporary config for the threads builds only, so scalar/SIMD builds stay unchanged.
  WASM_CARGO_CONFIG_DIR="$WASM_CRATE_DIR/.cargo"
  WASM_CARGO_CONFIG_FILE="$WASM_CARGO_CONFIG_DIR/config.toml"
  WASM_CARGO_CONFIG_BACKUP=""
  mkdir -p "$WASM_CARGO_CONFIG_DIR"
  if [[ -f "$WASM_CARGO_CONFIG_FILE" ]]; then
    WASM_CARGO_CONFIG_BACKUP="$WASM_CARGO_CONFIG_FILE.bak_$(date +%s)"
    cp "$WASM_CARGO_CONFIG_FILE" "$WASM_CARGO_CONFIG_BACKUP"
  fi
  cat >"$WASM_CARGO_CONFIG_FILE" <<'EOF'
[unstable]
build-std = ["std", "panic_abort"]
# NOTE: Newer Rust nightlies treat `panic_immediate_abort` as a *panic strategy*.
# Enabling the old std feature causes `core` to emit a compile_error asking for
# `panic = "immediate-abort"`. We build with `panic = abort` for compatibility.
EOF
  echo "Wrote temporary Cargo config for build-std: $WASM_CARGO_CONFIG_FILE"
  sed -n '1,120p' "$WASM_CARGO_CONFIG_FILE" || true

  (cd "$WASM_CRATE_DIR" && \
    # wasm-pack spawns `cargo` internally. `rustup run <toolchain> wasm-pack ...`
    # does *not* reliably force the same toolchain for the spawned cargo on all
    # environments. Pin the toolchain explicitly so `cargo build ... -Z ...`
    # is executed on nightly and accepts `-Z build-std=...`.
    PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH" \
    SONETTO_THREADS_TOOLCHAIN="$THREADS_TOOLCHAIN" \
    CARGO="$ROOT_DIR/scripts/cargo_with_toolchain.sh" \
    RUSTUP_TOOLCHAIN="$THREADS_TOOLCHAIN" \
    CARGO_PROFILE_RELEASE_PANIC=abort \
    RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base" \
    wasm-pack build --release --no-opt --target web --out-dir "$OUT_THREADS" --features wasm-bindgen-rayon)

  echo "[4/4] Threads + SIMD (rayon) -> ${OUT_THREADS_SIMD}"
  (cd "$WASM_CRATE_DIR" && \
    PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH" \
    SONETTO_THREADS_TOOLCHAIN="$THREADS_TOOLCHAIN" \
    CARGO="$ROOT_DIR/scripts/cargo_with_toolchain.sh" \
    RUSTUP_TOOLCHAIN="$THREADS_TOOLCHAIN" \
    CARGO_PROFILE_RELEASE_PANIC=abort \
    RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base" \
    wasm-pack build --release --no-opt --target web --out-dir "$OUT_THREADS_SIMD" --features "wasm_simd wasm-bindgen-rayon")

  # Post-build patch: some wasm-bindgen-rayon versions generate worker snippets that
  # import the package directory (e.g. "../.."), which fails in browsers when
  # served as plain ES modules. Rewrite those imports to point to sonetto_wasm.js.
  echo "Patching wasm-bindgen-rayon worker snippet imports (web target)..."
  python3 "$ROOT_DIR/scripts/patch_rayon_worker_imports.py" "$OUT_THREADS" "$OUT_THREADS_SIMD"

  # Cleanup temporary Cargo config.
  if [[ -n "${WASM_CARGO_CONFIG_BACKUP:-}" ]]; then
    mv "$WASM_CARGO_CONFIG_BACKUP" "$WASM_CARGO_CONFIG_FILE"
    echo "Restored previous Cargo config: $WASM_CARGO_CONFIG_FILE"
  else
    rm -f "$WASM_CARGO_CONFIG_FILE"
    # Remove the directory if empty (best-effort).
    rmdir "$WASM_CARGO_CONFIG_DIR" 2>/dev/null || true
    echo "Removed temporary Cargo config: $WASM_CARGO_CONFIG_FILE"
  fi
fi

echo "[post] wasm-opt (if available)"
bash scripts/wasm_opt_dir.sh "$OUT_SCALAR"
bash scripts/wasm_opt_dir.sh "$OUT_SIMD"
bash scripts/wasm_opt_dir.sh "$OUT_THREADS"
bash scripts/wasm_opt_dir.sh "$OUT_THREADS_SIMD"

printf '\nDone.\n'
