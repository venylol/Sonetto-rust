# Phase 5 — Performance & WASM specialization

This phase focuses on making the Sensei backend and the derivative scheduler practical on `wasm32`:

- **budget-friendly** execution (bounded by `SearchLimits`)
- **no unnecessary heap allocation** in hot paths
- optional **SIMD** (`wasm_simd`) and **threads** (`wasm-bindgen-rayon` / `parallel_rayon`) remain feature-gated

## New benchmark: `nps_compare`

Added in two places (kept in sync):

- `crates/sonetto_core/examples/nps_compare.rs`
- `examples/nps_compare.rs`

Run it with:

```bash
cargo run -p sonetto_core --example nps_compare --release
```

It prints (per backend) at least:

- elapsed time
- `nodes` used
- `nodes/sec`
- whether the search aborted due to node budget

The comparison includes:

- **Sonetto** (Searcher)
- **SenseiAB** (ported alpha-beta)
- **Derivative + SenseiAB** (derivative scheduler driving SenseiAB)

To tweak depth/budget/repetitions, edit the constants near the top of the example.

## Key code changes

### 1) Derivative propagation stack reuse (no per-update Vec alloc)

File: `crates/sonetto_core/src/derivative.rs`

- `TreeNodeSupplier` now owns a reusable `stack: Vec<NodeId>`.
- `update_fathers_from_child` and `propagate_descendants` use `core::mem::take(&mut self.stack)` and return it on exit.

This removes repeated heap allocations from hot propagation paths.

### 2) Derivative arena reuse across calls

File: `crates/sonetto_core/src/derivative.rs`

- Added `DerivativeEvaluator::reconfigure(cfg)`.
- If the requested arena sizes fit within the already-allocated buffers, the evaluator updates its config and **reuses** the existing arena allocations.
- If not, it rebuilds the evaluator with the new config.

### 3) Searcher caches DerivativeEvaluator (important for WASM)

File: `crates/sonetto_core/src/search.rs`

- `Searcher` gained an optional `derivative_cache: Option<DerivativeEvaluator>`.
- The `AnalyzeTopNStrategy::Derivative` path now `take()`s the cached evaluator, calls `reconfigure`, runs the evaluation, then stores it back.

This avoids large per-call heap allocations when derivative analysis is invoked repeatedly (typical in a web UI).

### 4) Fix + stabilize `analyze_top_n_with_derivative_backend`

File: `crates/sonetto_core/src/search.rs`

- Fixed parameter wiring (`req.top_n`, `req.params.tree_node_cap`) and applied the same derivative evaluator cache.

## WASM build recipes

### No threads

```bash
cargo build -p sonetto_wasm --target wasm32-unknown-unknown --release
```

### SIMD (simd128)

```bash
RUSTFLAGS='-C target-feature=+simd128' \
  cargo build -p sonetto_wasm --target wasm32-unknown-unknown --release --features wasm_simd
```

### Threads (Rayon)

This requires atomics/bulk-memory/mutable-globals and the host serving with COOP/COEP for `SharedArrayBuffer`.

```bash
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
  cargo build -p sonetto_wasm --target wasm32-unknown-unknown --release --features wasm-bindgen-rayon
```

(You can combine with SIMD by adding `--features wasm-bindgen-rayon,wasm_simd` and enabling `+simd128`.)

## Static checks (suggested)

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test --workspace
```
