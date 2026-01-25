#!/usr/bin/env bash
set -euo pipefail

# A tiny wrapper so tools like wasm-pack can be forced to use a specific rustup toolchain
# even if PATH contains a direct stable toolchain cargo binary.
#
# Toolchain selection:
#   - SONETTO_THREADS_TOOLCHAIN (preferred)
#   - SONETTO_CARGO_TOOLCHAIN
#   - RUSTUP_TOOLCHAIN
#   - default: nightly
TOOLCHAIN="${SONETTO_THREADS_TOOLCHAIN:-${SONETTO_CARGO_TOOLCHAIN:-${RUSTUP_TOOLCHAIN:-nightly}}}"

exec rustup run "$TOOLCHAIN" cargo "$@"
