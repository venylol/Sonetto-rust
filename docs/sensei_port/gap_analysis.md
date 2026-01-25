# Sensei → Sonetto-rust Port: Gap Analysis

This document tracks notable gaps between:

- **Sensei (native)** layout: `sensei_engine/engine/...`
- **Sonetto-rust** layout: `sonetto/crates/sonetto_core/src/...`

It is a practical checklist for "port closeness" (behavior + performance).

Hard constraints that should continue to hold:
- **Keep** compatibility with Sensei-style assets and formats already wired in:
  - `egbk3` opening book flow in the web UI.
  - `egev2` evaluation weights / feature layout.
- **Keep** the current web UI / wasm interface surface used by
  `sonetto/js/src/engine_worker.js`.

## High-impact gaps (still open)

### 1. Depth-one incremental evaluator used for disproof-number move ordering

Sensei's native engine uses a specialized **DepthOneEvaluator** to cheaply
compute a child evaluation for each candidate move during **disproof-number**
ordering (without recomputing all features from scratch).

Sonetto-rust now uses Sensei's **EndgameTime** regression model for the
`DisproofNumberOverProb` metric, but the "approx_eval" input is still not as
cheap as native in all backends:

- In `sonetto_core/src/search.rs`, we compute an approximate disc-eval via
  `score_disc` on a temporary board state.
- In `sonetto_core/src/sensei_ab/move_iter.rs`, we use a very cheap disc-count
  approximation.

This is logically aligned with Sensei, but may still be slower than the native
DepthOneEvaluator in tight endgame searches.

### 2. Sensei AB backend is intentionally simplified (no transposition table)

Sonetto's main `Searcher` has a transposition table and modern move ordering.
The optional `sensei_ab` backend is closer in structure to Sensei's alpha-beta
but remains a **minimal** implementation (no TT). If the UI uses `sensei_ab`,
performance will remain far from native Sensei.

### 3. Remaining Sensei utility modules not yet ported

The following Sensei modules remain unported (or not wired):

- `error_margin.h` / endgame error-margins
- `opening_options.h` / `endgame_options.h`

## Recently closed gaps

### Stable discs

`sonetto_core/src/stability.rs` now follows Sensei's full stable-disc approach:

1. Table-driven exact **edge stability**.
2. Add squares that lie on a **full row/column/diagonal** (no empties).
3. Iteratively propagate stability inward along fully occupied lines.

(Adapted for Sonetto's bitboard mapping: bit 0 == A1.)

### EndgameTime estimators

`sonetto_core/src/sensei_extras/endgame_time.rs` ports Sensei's regression models
and precomputes the `disproof_number_over_prob` table, which is now used in:

- `sonetto_core/src/search.rs` disproof-number move ordering.
- `sonetto_core/src/sensei_ab/move_iter.rs` disproof-number move ordering.

## Mapping summary (selected)

| Sensei module | Sonetto-rust analogue | Status | Notes |
|---|---|---:|---|
| `stable.h` | `stability.rs` | ✅ | Full Sensei-style stable propagation (mapping-adapted). |
| `win_probability.h` | `sensei_extras/win_probability.rs` | ✅ | Table-backed + explicit probability function. |
| `endgame_time.h` | `sensei_extras/endgame_time.rs` | ✅ | Ported + wired into disproof ordering. |
| `evaluator_alpha_beta.*` | `search.rs` | ⚠️ | Structure differs; key heuristics/ordering align where possible. |
| `depth_one_evaluator.*` | (none) | ❌ | Not yet replicated as an incremental evaluator. |
| `opening_book.*` | JS UI (`egbk3_book.js`) + internal book code | ✅ | Must remain compatible. |
