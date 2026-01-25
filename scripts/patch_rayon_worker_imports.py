#!/usr/bin/env python3
"""Patch wasm-bindgen-rayon worker snippet imports for --target web builds.

Problem
-------
In some wasm-bindgen/wasm-bindgen-rayon combinations, the Rayon worker helper JS
uses bundler-style resolution like:

  import init, { wbg_rayon_start_worker } from '../..';
  wasm_bindgen = await import('../..');

When served as plain ES modules (Cloudflare Pages ZIP + <script type=module>),
this becomes a network request to the *directory* URL (e.g. /pkg_threads_simd/),
which fails and prevents thread pool initialization.

Fix
---
For any JS file in a wasm-pack output directory that appears to be part of
wasm-bindgen-rayon (heuristic: contains `wbg_rayon_start_worker`), rewrite any
*directory-ish* module specifier ('.', '..', '../..', '.../' etc.) used by
static or dynamic `import` to an explicit relative path to `sonetto_wasm.js`
inside the same wasm-pack output directory.

This script is intended to be run as a post-build step on:
  - web/pkg_threads
  - web/pkg_threads_simd
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


# Matches: import ... from '<spec>'
STATIC_IMPORT_RE = re.compile(
    r"(import\s+[^;\n]*?\sfrom\s*['\"])([^'\"]+)(['\"])",
    re.MULTILINE,
)

# Matches: import('<spec>') or import(/*comment*/ '<spec>')
DYNAMIC_IMPORT_RE = re.compile(
    r"(import\(\s*(?:/\*.*?\*/\s*)?['\"])([^'\"]+)(['\"]\s*\))",
    re.MULTILINE | re.DOTALL,
)


def _is_problematic_spec(spec: str) -> bool:
    """Return True if `spec` looks like a directory-ish/bundler-only specifier."""
    s = spec.strip()

    # Absolute/bare specifiers are not safe to touch here.
    if s.startswith("http://") or s.startswith("https://") or s.startswith("//"):
        return False
    if s.startswith("/"):
        # absolute path could be intentional; leave it alone
        return False

    # The problematic cases are relative specs that don't name a concrete JS file.
    if s in {".", "./", "..", "../"}:
        return True
    if s.endswith("/"):
        return True

    # Things like '../..', '../', './pkg_threads_simd' (no .js) are also problematic
    # when served directly, because they resolve to directory URLs.
    if s.startswith(".") and not s.endswith(".js"):
        return True

    return False


def _ensure_relative_es_module_path(rel: str) -> str:
    # ESM relative specifiers must start with './' or '../'.
    if rel.startswith("."):
        return rel
    return "./" + rel


def _patch_text(text: str, rel_to_wasm_js: str) -> str:
    def repl_static(m: re.Match[str]) -> str:
        old = m.group(2)
        if not _is_problematic_spec(old):
            return m.group(0)
        return f"{m.group(1)}{rel_to_wasm_js}{m.group(3)}"

    def repl_dynamic(m: re.Match[str]) -> str:
        old = m.group(2)
        if not _is_problematic_spec(old):
            return m.group(0)
        return f"{m.group(1)}{rel_to_wasm_js}{m.group(3)}"

    text2 = STATIC_IMPORT_RE.sub(repl_static, text)
    text3 = DYNAMIC_IMPORT_RE.sub(repl_dynamic, text2)
    return text3


def patch_pkg(pkg_dir: Path) -> int:
    wasm_js = pkg_dir / "sonetto_wasm.js"
    if not wasm_js.is_file():
        return 0

    changed = 0
    for js in pkg_dir.rglob("*.js"):
        try:
            text = js.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = js.read_text(encoding="utf-8", errors="replace")

        # Only touch files that look like wasm-bindgen-rayon worker helpers.
        if "wbg_rayon_start_worker" not in text:
            continue

        rel = os.path.relpath(wasm_js, js.parent).replace(os.sep, "/")
        rel = _ensure_relative_es_module_path(rel)

        new_text = _patch_text(text, rel)
        if new_text != text:
            js.write_text(new_text, encoding="utf-8")
            changed += 1

    return changed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <pkg_dir> [<pkg_dir> ...]", file=sys.stderr)
        return 2

    for p in argv[1:]:
        pkg_dir = Path(p).resolve()
        n = patch_pkg(pkg_dir)
        if n:
            print(f"[patch_rayon_worker_imports] patched {n} file(s) under {pkg_dir}")

    # Exit 0 even if nothing changed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
