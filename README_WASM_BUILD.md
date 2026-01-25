# Sonetto (Front-end WASM) Build Guide

This repository is set up for **Mode A**: the engine runs in the browser as WebAssembly.

## Folder layout

- `crates/sonetto_core`: engine core logic (pure Rust)
- `crates/sonetto_wasm`: `wasm-bindgen` wrapper exporting JS-callable functions
- `web/`: your front-end files (`Sonetto.html`, `engine_worker.js`)

`engine_worker.js` expects the generated wasm-pack output at:

- `web/pkg/sonetto_wasm.js`
- `web/pkg/sonetto_wasm_bg.wasm`

## 1) Install toolchain

```bash
# install Rust (if you don't have it)
# https://rustup.rs

rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## 2) Build WASM into the front-end folder

From the repo root:

```bash
cd crates/sonetto_wasm
wasm-pack build --release --target web --out-dir ../../web/pkg
```

After building, you should see `web/pkg/` contain the `.js` and `.wasm` files.

## 3) Run locally (must use HTTP, don't double-click the HTML)

```bash
cd web
python -m http.server 8080
```

Open in your browser:

- http://localhost:8080/Sonetto.html

## 3.5) Build a Cloudflare Pages-friendly ZIP (drag & drop)

If you want to deploy via **Cloudflare Pages** (direct ZIP upload), you need a ZIP whose *archive root*
contains `index.html` (no extra top-level folder inside the ZIP).

From the repo root:

```bash
bash scripts/build_pages_zip.sh
```

This will produce:

- `sonetto_pages_root.zip`  (upload this)

## 4) Weights ZIP format (eval.egev2)

Sonetto uses **Egaroucid-compatible** compressed weights (`eval.egev2`) as the canonical format.

When you use the UI "Export" buttons or the native trainer, the exported ZIP is expected to contain:

- `eval.egev2` (canonical weights blob)
- `manifest.json` (metadata, including the uncompressed parameter count)

Legacy `eval.egev` is not used and should not be included in the exported artifacts.

## 5) Native trainer crate (not WASM)

This repo also contains a native-only CLI tool:

- `crates/sonetto_trainer`

It is intended for offline dataset streaming, training, and teacher validation. It depends on
native HTTP/filesystem crates and is **not** meant to be built for `wasm32-unknown-unknown`.

To reduce accidental cross-target builds, the workspace is configured with `default-members`
so that a plain `cargo build` from the repo root does not try to build the native trainer.

## WebAssembly SIMD (simd128) 双构建（推荐）

本仓库支持“标量版 + SIMD版”两套 wasm 并存，并在运行时自动选择：

- 标量版输出：`web/pkg`（兼容性最好）
- SIMD版输出：`web/pkg_simd`（需要浏览器支持 WebAssembly SIMD / `simd128`）

Web Worker (`web/engine_worker.js`) 会在运行时探测 SIMD 支持并优先加载 `pkg_simd`，
若不存在或不支持则自动回退到 `pkg`。

### 构建命令

在仓库根目录执行：

```bash
./scripts/build_wasm_dual.sh
```

如果你只想单独构建 SIMD 版本（会导致不支持 SIMD 的浏览器无法加载），可参考：

```bash
RUSTFLAGS="-C target-feature=+simd128"   wasm-pack build crates/sonetto_wasm -t web -d web/pkg_simd --release --features wasm_simd
```

> 说明：`--features wasm_simd` 会开启 `sonetto_core` 中的 `wasm_simd` 路径（P2-1）。

## WebAssembly Threads (rayon + SharedArrayBuffer)（推荐）

本仓库现在支持 **“WASM 内部线程”**（Rayon + SharedArrayBuffer）：

- 线程标量版输出：`web/pkg_threads`
- 线程 SIMD版输出：`web/pkg_threads_simd`

`web/engine_worker.js` 会在运行时探测：

- 浏览器是否支持 `SharedArrayBuffer`，以及 `crossOriginIsolated` 是否为 true
- 如满足条件，会优先加载 `pkg_threads_simd` / `pkg_threads` 并调用 `initThreadPool(...)`
- 否则自动回退到 `pkg_simd` / `pkg`

### 重要：需要 COOP/COEP（Cross-Origin Isolation）

WASM threads 只能在 **cross-origin isolated** 的页面上工作。
这通常需要 HTTP 响应头：

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

本仓库已在 `web/_headers` 内加入这些头，适用于 Cloudflare Pages。

### 构建命令（四套输出）

在仓库根目录执行：

```bash
./scripts/build_wasm_threads_dual.sh
```

> 备注：threads 构建依赖 `+atomics` / `+bulk-memory` 等 wasm32 target-feature。
> Rust 官方并不提供默认启用线程的 wasm32 标准库，因此 threads 构建需要：
>
> - **nightly 工具链**（CI 默认 pin 到 `nightly-2024-08-02`）
> - **rust-src 组件**（用于 `-Z build-std` 重新编译 `std`）
>
> 脚本会对 threads 两个变体使用 `-Z build-std=panic_abort,std` 来保证 threads 能正常运行，
> 而标量/SIMD 仍保持 stable。
>
> 如你想切换 nightly 版本，可通过环境变量覆盖：
>
> ```bash
> SONETTO_THREADS_TOOLCHAIN=nightly   ./scripts/build_wasm_threads_dual.sh
> ```
>
> 若你在 initThreadPool 时遇到 `DataCloneError: #<Memory> could not be cloned`，通常意味着 threads 产物没有正确初始化 shared memory（shared:true）。此时建议回到默认 pin 的 nightly，或确保 build-std + wasm32 atomics 配置正确。

该脚本会构建：

- `web/pkg`（标量）
- `web/pkg_simd`（SIMD）
- `web/pkg_threads`（线程标量）
- `web/pkg_threads_simd`（线程 SIMD）

> 注意：线程构建使用 `+atomics` 等 target-feature，浏览器端必须有 SharedArrayBuffer 支持。

