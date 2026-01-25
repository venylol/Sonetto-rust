#!/usr/bin/env bash
set -euo pipefail

# Build a Cloudflare Pages-friendly ZIP where the archive root contains:
#   index.html, Sonetto.html, _headers, pkg/, ...
# (i.e. no extra top-level folder inside the ZIP)

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

  local cargo_toml
  cargo_toml="$(grep -Rsl --include 'Cargo.toml' 'name = "sonetto_wasm"' "$root" 2>/dev/null | head -n 1 || true)"
  if [[ -n "${cargo_toml:-}" ]]; then
    dirname "$cargo_toml"
    return 0
  fi

  return 1
}

# If the wasm output isn't present yet, build it.
if [ ! -f "web/pkg/sonetto_wasm_bg.wasm" ] || [ ! -f "web/pkg/sonetto_wasm.js" ]; then
  if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "ERROR: wasm-pack is not installed. Install it with: cargo install wasm-pack" >&2
    exit 1
  fi
  WASM_CRATE_DIR="$(find_wasm_crate_dir "$ROOT_DIR" || true)"
  if [[ -z "${WASM_CRATE_DIR:-}" || ! -d "$WASM_CRATE_DIR" ]]; then
    echo "ERROR: Could not find the sonetto_wasm crate directory. Expected at: $ROOT_DIR/crates/sonetto_wasm" >&2
    ls -la "$ROOT_DIR/crates" 2>/dev/null || true
    exit 1
  fi
  echo "[build_pages_zip] wasm output not found; running wasm-pack build..."
  (cd "$WASM_CRATE_DIR" && wasm-pack build --release --target web --out-dir "$ROOT_DIR/web/pkg")
fi

# Prepare dist folder
rm -rf dist
mkdir -p dist
cp -r web/* dist/

# Optional large assets
# --------------------
# We keep the opening book + weights as raw .gz files to respect Cloudflare Pages'
# per-file asset size limit (25 MiB). If these files live at the repo root (common
# when you don't want to commit them under web/), copy them into the Pages root.
#
# Sonetto.html will probe multiple paths, but the canonical ones are:
#   ./eval.egev2.gz
#   ./book.egbk3.gz

# IMPORTANT:
# We **prefer .gz** assets. If both the compressed file and the plain file exist,
# we copy only the .gz file. This avoids accidentally bundling the uncompressed
# blobs which can exceed Cloudflare Pages' per-file size limit and cause uploads
# to fail.
copy_asset_prefer_gz() {
  local base="$1"
  if [ -f "${base}.gz" ]; then
    cp "${base}.gz" "dist/${base}.gz"
    return 0
  fi
  if [ -f "${base}" ]; then
    cp "${base}" "dist/${base}"
    return 0
  fi
  return 0
}

copy_asset_prefer_gz "eval.egev2"
copy_asset_prefer_gz "book.egbk3"

# Guardrail: Cloudflare Pages (direct upload) has a per-file asset size limit.
# Fail early if any single file exceeds 25 MiB, because the upload will be
# rejected even if the ZIP itself is accepted.
MAX_BYTES=$((25 * 1024 * 1024))
too_big=0
while IFS= read -r -d '' f; do
  sz=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
  if [ "${sz}" -gt "${MAX_BYTES}" ]; then
    echo "ERROR: dist artifact too large (>25 MiB): ${f} (${sz} bytes)" >&2
    too_big=1
  fi
done < <(find dist -type f -print0)

if [ "${too_big}" -ne 0 ]; then
  echo "ERROR: Refusing to build Pages ZIP because one or more files exceed the per-file limit." >&2
  echo "Tip: keep large assets compressed (.gz) and do not include the plain eval.egev2 / book.egbk3." >&2
  exit 1
fi

# Create ZIP with files at the archive root (no dist/ prefix)
rm -f sonetto_pages_root.zip
(
  cd dist
  if ! command -v zip >/dev/null 2>&1; then
    echo "ERROR: zip is not installed. (Ubuntu/Debian: sudo apt-get install -y zip)" >&2
    exit 1
  fi
  zip -r ../sonetto_pages_root.zip .
)

echo "[build_pages_zip] wrote: ${ROOT_DIR}/sonetto_pages_root.zip"
