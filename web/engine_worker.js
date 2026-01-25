// engine_worker.js (Web Worker, ES module)
//
// Phase4 goal: keep only inference paths.
// - analyze / best_move
// - set_weights_egev2 / get_weights_egev2
// - (optional legacy) set_weights (flat)
//
// This build is inference-only.
//
// P2-1 (wasm SIMD): this worker supports an optional SIMD build living at
// `./pkg_simd/sonetto_wasm.js`. At runtime we detect SIMD support and
// prefer the SIMD build when available, falling back to the scalar build
// (`./pkg/sonetto_wasm.js`) otherwise.


// -----------------------------------------------------------------------------
// Wasm module loader
//
// Variants (optional):
//  - scalar           : ./pkg
//  - simd128          : ./pkg_simd
//  - threads (rayon)  : ./pkg_threads
//  - threads + simd   : ./pkg_threads_simd
//
// Threads require:
//  - SharedArrayBuffer
//  - crossOriginIsolated === true (COOP/COEP headers)
//  - a wasm build with +atomics,+bulk-memory,+mutable-globals
//  - JS calls initThreadPool(N) once after init()
// -----------------------------------------------------------------------------

let wasmMod = null;
let wasmReady = false;

// Runtime feature flags (best-effort; may still fail at import/instantiation time).
function detectWasmThreads() {
  // wasm threads requires ALL of:
  //   - SharedArrayBuffer
  //   - crossOriginIsolated (COOP/COEP headers)
  //   - Atomics
  //   - shared WebAssembly.Memory support
  //
  // NOTE:
  //  We keep this as a *best-effort* runtime probe. Even if this returns true,
  //  the threads build can still fail to instantiate (e.g. browser bugs / policy
  //  changes). loadSonettoWasmModule() always includes non-threads fallbacks.
  try {
    if (typeof SharedArrayBuffer === 'undefined') return false;
    if (self.crossOriginIsolated !== true) return false;
    if (typeof Atomics === 'undefined') return false;
    // Some environments expose SAB but don't allow shared wasm memories.
    // Constructing a shared Memory is the most reliable probe.
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
    return (mem && mem.buffer instanceof SharedArrayBuffer);
  } catch (_) {
    return false;
  }
}

// Cap the number of Rayon workers in automated/CI ("CL") environments to keep
// startup latency and resource usage predictable on high-core machines.
//
// In real deployments, we do NOT cap (beyond hardwareConcurrency itself), so
// users can take advantage of their available CPU.
//
// Detection is best-effort:
//  - Explicit: URL query ?cl=1 / ?ci=1 / ?selfcheck=1
//  - Automated browser: navigator.webdriver === true
//  - Headless tooling UA hints (Playwright / HeadlessChrome)
const MAX_RAYON_WORKERS_CL = 8;

function isCLLikeEnv() {
  try {
    const qs = (typeof location !== 'undefined' && location && typeof location.search === 'string') ? location.search : '';
    if (qs && /(?:\?|&)(?:cl|ci|selfcheck)=1(?:&|$)/.test(qs)) return true;

    if (typeof navigator !== 'undefined' && navigator && navigator.webdriver) return true;

    const ua = (typeof navigator !== 'undefined' && navigator && typeof navigator.userAgent === 'string') ? navigator.userAgent : '';
    if (ua && /HeadlessChrome|Playwright|Puppeteer/i.test(ua)) return true;
  } catch (_) {}
  return false;
}

function pickThreadCount(requested) {
  const hw = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) ? (navigator.hardwareConcurrency | 0) : 4;
  const hint = (requested != null) ? (requested | 0) : (hw | 0);
  const n0 = (hint > 0) ? hint : (hw | 0);

  const cap = isCLLikeEnv() ? MAX_RAYON_WORKERS_CL : Number.POSITIVE_INFINITY;
  const n = Math.min(n0, hw > 0 ? hw : n0, cap);
  return Math.max(1, n);
}

let wasmInfo = {
  simd: false,
  threads: false,
  threadsRequested: 0,
  threadsUsed: 0,
  threadPoolError: null,
  variant: 'scalar',
};

// Minimal SIMD probe module.
//
// This module is taken from wasm-feature-detect's SIMD detector (v1.8.0) and is
// a small, valid module that requires the fixed-width SIMD proposal.
// If the runtime doesn't support SIMD, validation/instantiation will fail.
const SIMD_PROBE_WASM = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 5, 1, 96, 0, 1, 123,
  3, 2, 1, 0,
  10, 10, 1, 8, 0,
  65, 0,
  253, 15,
  253, 98,
  11,
]);

async function detectWasmSimd() {
  if (typeof WebAssembly === 'undefined') return false;
  try {
    // Prefer validate() (fast) when available; fall back to instantiate().
    if (typeof WebAssembly.validate === 'function') {
      return WebAssembly.validate(SIMD_PROBE_WASM);
    }
    await WebAssembly.instantiate(SIMD_PROBE_WASM);
    return true;
  } catch (_) {
    return false;
  }
}

async function loadSonettoWasmModule() {
  if (wasmMod) return wasmMod;

  const hasSimd = await detectWasmSimd();
  const hasThreads = detectWasmThreads();

  // Prefer SIMD build if the runtime supports it. If the SIMD build isn't present
  // (e.g., repo not built with the dual-build script), fall back to scalar.
  // Prefer threads build when runtime supports it, then SIMD, then scalar fallback.
  // Each candidate import is tried in order until one succeeds.
  const candidates = (hasThreads && hasSimd)
    ? [
        "./pkg_threads_simd/sonetto_wasm.js",
        "./pkg_threads/sonetto_wasm.js",
        "./pkg_simd/sonetto_wasm.js",
        "./pkg/sonetto_wasm.js",
      ]
    : (hasThreads && !hasSimd)
      ? [
          "./pkg_threads/sonetto_wasm.js",
          "./pkg/sonetto_wasm.js",
        ]
      : (!hasThreads && hasSimd)
        ? ["./pkg_simd/sonetto_wasm.js", "./pkg/sonetto_wasm.js"]
        : ["./pkg/sonetto_wasm.js"];

  let lastErr = null;
  for (const path of candidates) {
    try {
      const mod = await import(path);
      // wasm-pack default export is the async init() function
      await mod.default();
      // optional init panic hook
      if (typeof mod.init === 'function') mod.init();

      // Track which variant was actually loaded.
      wasmInfo.simd = hasSimd && (path.includes('pkg_simd') || path.includes('pkg_threads_simd'));
      wasmInfo.threads = path.includes('pkg_threads');
      wasmInfo.variant = wasmInfo.threads
        ? (wasmInfo.simd ? 'threads_simd' : 'threads')
        : (wasmInfo.simd ? 'simd' : 'scalar');

      wasmMod = mod;
      return wasmMod;
    } catch (err) {
      lastErr = err;
      // eslint-disable-next-line no-console
      console.warn(`[sonetto] failed to load wasm module ${path}`, err);
    }
  }

  throw lastErr ?? new Error("Failed to load sonetto_wasm module");
}

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------

let eng = null;
let currentHashSizeMb = null;
let currentBackend = 0; // 0=Sonetto(Searcher), 1=SenseiAlphaBeta

let threadPoolReady = false;

async function ensureThreadPool(threadCount) {
  if (threadPoolReady) return;

  // Only try when:
  // - we actually loaded a threads-enabled wasm module
  // - runtime environment provides SAB + crossOriginIsolated
  if (!wasmInfo.threads) {
    threadPoolReady = true; // nothing to do
    return;
  }

  const canThreads = detectWasmThreads();
  if (!canThreads) {
    // Threads module loaded but runtime isn't isolated: it will likely fail anyway.
    // Mark ready to avoid retry loops.
    wasmInfo.threadPoolError = 'threads_requested_but_runtime_not_isolated';
    wasmInfo.threadsUsed = 1;
    threadPoolReady = true;
    return;
  }

  const n = pickThreadCount(threadCount);
  wasmInfo.threadsRequested = n;

  // wasm-bindgen-rayon exports `initThreadPool` (camelCase). Some builds may
  // still expose snake_case; support both.
  const initFn = wasmMod?.initThreadPool || wasmMod?.init_thread_pool;
  if (typeof initFn !== 'function') {
    wasmInfo.threadPoolError = 'initThreadPool_export_missing';
    wasmInfo.threadsUsed = 1;
    threadPoolReady = true;
    return;
  }

  try {
    await initFn(n);
    wasmInfo.threadsUsed = n;
    wasmInfo.threadPoolError = null;
  } catch (err) {
    // If init fails, fall back to single-thread behavior.
    // eslint-disable-next-line no-console
    console.warn('[sonetto] initThreadPool failed; running single-threaded', err);
    wasmInfo.threadsUsed = 1;
    wasmInfo.threadPoolError = err?.message ? String(err.message) : String(err);
  }

  threadPoolReady = true;
}

function ensureEngine(hashSizeMb) {
  const mb = (hashSizeMb | 0) > 0 ? (hashSizeMb | 0) : 64;
  if (!eng || currentHashSizeMb !== mb) {
    eng = wasmMod.engine_new(mb);
    currentHashSizeMb = mb;
    // Re-apply backend selection when recreating the engine.
    if (typeof wasmMod.engine_set_backend === 'function') {
      try {
        currentBackend = wasmMod.engine_set_backend(eng, currentBackend) | 0;
      } catch (_) {
        // ignore
      }
    }
  }
}

async function ensureReady(hashSizeMb, threadCount) {
  if (!wasmReady) {
    wasmMod = await loadSonettoWasmModule();
    wasmReady = true;
  }

  // Threads: initialize Rayon pool exactly once (no-op for non-thread builds).
  await ensureThreadPool(threadCount);

  ensureEngine(hashSizeMb);
}

function postError(err, context = '') {
  const msg = (err && (err.message || err.error)) ? (err.message || err.error) : String(err);
  postMessage({
    type: 'error',
    message: context ? `${context}: ${msg}` : msg,
    stack: err?.stack ?? null,
  });
}

function unsupported(type) {
  postError(new Error(`Unsupported message type: ${type}. This build supports inference only.`));
}

// -----------------------------------------------------------------------------
// Message handler
// -----------------------------------------------------------------------------

// IMPORTANT:
//   Do NOT use an `async` onmessage handler directly.
//   If the main thread posts multiple messages quickly (init + set_weights + analyze...),
//   an `async` handler can yield at `await` points and allow later messages to run
//   concurrently. That can lead to multiple overlapping wasm init() calls and
//   invalidate wasm-bindgen objects, surfacing as runtime panics like:
//     "recursive use of an object detected which would lead to unsafe aliasing in rust"
//
//   We serialize message processing explicitly.
let __msgQueue = Promise.resolve();

async function __handleMessage(e) {
  const msg = e.data || {};
  const type = msg?.type;

  switch (type) {
    case 'init':
    case 'init_wasm': {
      const hashSizeMb = msg?.hashSizeMb ?? 64;
      const threadCount = msg?.threadCount;
      await ensureReady(hashSizeMb, threadCount);
      // Keep message name compatible with Sonetto.html
      postMessage({ type: 'inited', wasmInfo });
      break;
    }

    case 'set_backend': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      const threadCount = msg?.threadCount;
      await ensureReady(hashSizeMb, threadCount);

      const raw = (msg?.backend ?? msg?.kind ?? 0);
      let backendId = 0;
      if (typeof raw === 'string') {
        const s = raw.toLowerCase();
        if (s === 'sensei' || s === 'sensei_ab' || s === 'senseialphabeta') backendId = 1;
        else backendId = 0;
      } else {
        backendId = (raw | 0);
      }

      if (typeof wasmMod.engine_set_backend === 'function') {
        currentBackend = wasmMod.engine_set_backend(eng, backendId) | 0;
      } else {
        currentBackend = backendId | 0;
      }

      postMessage({ type: 'backend_set', backend: currentBackend });
      break;
    }

    // Legacy flat-weights path (kept for backwards compatibility with older zips)
    case 'set_weights': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      await ensureReady(hashSizeMb);
      wasmMod.engine_set_weights_flat(eng, msg.weightsFlat);
      break;
    }

    // Canonical weights: Egaroucid eval.egev2
    case 'set_weights_egev2': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      await ensureReady(hashSizeMb);
      const ok = wasmMod.engine_set_weights_egev2(eng, msg.evalEgev2Bytes);
      if (!ok) throw new Error('engine_set_weights_egev2 failed (invalid eval.egev2?)');
      break;
    }

    case 'get_weights_egev2': {
      // If the page requests weights early (e.g. export), ensure the engine exists.
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      await ensureReady(hashSizeMb);
      const reqId = msg?.reqId;
      const evalEgev2Bytes = wasmMod.engine_get_weights_egev2(eng);
      postMessage({ type: 'get_weights_egev2_result', reqId, evalEgev2Bytes });
      break;
    }

    // Optional legacy export (not used by phase4 front-end)
    case 'get_weights_flat': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      await ensureReady(hashSizeMb);
      const reqId = msg?.reqId;
      const weightsFlat = wasmMod.engine_get_weights_flat(eng);
      postMessage({ type: 'get_weights_flat_result', reqId, weightsFlat });
      break;
    }

    case 'analyze': {
      // Robustness: if analysis is requested before init completes, do not drop it.
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      const threadCount = msg?.threadCount;
      await ensureReady(hashSizeMb, threadCount);

      const { reqId, boardArr, player, midDepth, endStart, topN, tag } = msg;
      const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const useV3 = (currentBackend !== 0) && (typeof wasmMod.engine_analyze_v3 === 'function');
      const out = useV3
        ? wasmMod.engine_analyze_v3(eng, boardArr, player, midDepth, endStart, topN)
        : wasmMod.engine_analyze(eng, boardArr, player, midDepth, endStart, topN);
      const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const ms = (t1 - t0);
      const nodes = (useV3 && typeof wasmMod.engine_last_nodes_v3 === 'function')
        ? (wasmMod.engine_last_nodes_v3(eng) >>> 0)
        : ((typeof wasmMod.engine_last_nodes === 'function') ? (wasmMod.engine_last_nodes(eng) >>> 0) : 0);
      const nps = (ms > 0) ? Math.floor((nodes * 1000) / ms) : 0;

      // Sonetto.html expects a plain Array, not a TypedArray.
      const resultsFlat = Array.isArray(out) ? out : Array.from(out || []);
      postMessage({ type: 'analysis', reqId, resultsFlat, ms, nodes, nps, tag, wasmInfo });
      break;
    }

    case 'analyze_v3': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      const threadCount = msg?.threadCount;
      await ensureReady(hashSizeMb, threadCount);

      // Optional one-shot override (persisted within this worker instance).
      if (msg?.backend != null && typeof wasmMod.engine_set_backend === 'function') {
        try { currentBackend = wasmMod.engine_set_backend(eng, (msg.backend | 0)) | 0; } catch (_) {}
      }

      const { reqId, boardArr, player, midDepth, endStart, topN, tag } = msg;
      const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const out = wasmMod.engine_analyze_v3(eng, boardArr, player, midDepth, endStart, topN);
      const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const ms = (t1 - t0);
      const nodes = (typeof wasmMod.engine_last_nodes_v3 === 'function')
        ? (wasmMod.engine_last_nodes_v3(eng) >>> 0)
        : ((typeof wasmMod.engine_last_nodes === 'function') ? (wasmMod.engine_last_nodes(eng) >>> 0) : 0);
      const nps = (ms > 0) ? Math.floor((nodes * 1000) / ms) : 0;

      const resultsFlat = Array.isArray(out) ? out : Array.from(out || []);
      postMessage({ type: 'analysis', reqId, resultsFlat, ms, nodes, nps, tag, wasmInfo, backend: currentBackend });
      break;
    }

    case 'analyze_v2': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      const threadCount = msg?.threadCount;
      await ensureReady(hashSizeMb, threadCount);

      const {
        reqId,
        boardArr,
        player,
        mode,
        midDepth,
        endStart,
        topN,
        seedDepth,
        aspirationWidth,
        nodeBudget,
        treeNodeCap,
      } = msg;

      // Stage 6 unified analysis entry. All tuning knobs are optional;
      // pass 0 / null / undefined to fall back to Rust-side defaults.
      const m = (mode ?? 0) | 0;
      const d = (midDepth ?? 0) | 0;
      const e = (endStart ?? 0) | 0;
      const n = (topN ?? 0) | 0;
      const sd = (seedDepth ?? 0) | 0;
      const aw = (aspirationWidth ?? 0) | 0;
      const nb = (nodeBudget ?? 0) >>> 0;
      const tc = (treeNodeCap ?? 0) >>> 0;

      const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const out = wasmMod.engine_analyze_v2(eng, boardArr, player, m, d, e, n, sd, aw, nb, tc);
      const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const ms = (t1 - t0);
      const nodes = (typeof wasmMod.engine_last_nodes === 'function') ? (wasmMod.engine_last_nodes(eng) >>> 0) : 0;
      const nps = (ms > 0) ? Math.floor((nodes * 1000) / ms) : 0;

      const resultsFlat = Array.isArray(out) ? out : Array.from(out || []);
      postMessage({ type: 'analysis', reqId, resultsFlat, ms, nodes, nps, mode: m });
      break;
    }

    case 'best_move': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      const threadCount = msg?.threadCount;
      await ensureReady(hashSizeMb, threadCount);

      const { reqId, boardArr, player, depth } = msg;
      const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const useV3 = (currentBackend !== 0) && (typeof wasmMod.engine_best_move_v3 === 'function');
      const best = useV3
        ? wasmMod.engine_best_move_v3(eng, boardArr, player, depth)
        : wasmMod.engine_best_move(eng, boardArr, player, depth);
      const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const ms = (t1 - t0);
      const nodes = (useV3 && typeof wasmMod.engine_last_nodes_v3 === 'function')
        ? (wasmMod.engine_last_nodes_v3(eng) >>> 0)
        : ((typeof wasmMod.engine_last_nodes === 'function') ? (wasmMod.engine_last_nodes(eng) >>> 0) : 0);
      const nps = (ms > 0) ? Math.floor((nodes * 1000) / ms) : 0;
      postMessage({ type: 'best_move', reqId, best, ms, nodes, nps });
      break;
    }

    case 'best_move_v3': {
      const hashSizeMb = msg?.hashSizeMb ?? currentHashSizeMb ?? 64;
      const threadCount = msg?.threadCount;
      await ensureReady(hashSizeMb, threadCount);

      if (msg?.backend != null && typeof wasmMod.engine_set_backend === 'function') {
        try { currentBackend = wasmMod.engine_set_backend(eng, (msg.backend | 0)) | 0; } catch (_) {}
      }

      const { reqId, boardArr, player, depth } = msg;
      const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const best = wasmMod.engine_best_move_v3(eng, boardArr, player, depth);
      const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      const ms = (t1 - t0);
      const nodes = (typeof wasmMod.engine_last_nodes_v3 === 'function')
        ? (wasmMod.engine_last_nodes_v3(eng) >>> 0)
        : ((typeof wasmMod.engine_last_nodes === 'function') ? (wasmMod.engine_last_nodes(eng) >>> 0) : 0);
      const nps = (ms > 0) ? Math.floor((nodes * 1000) / ms) : 0;
      postMessage({ type: 'best_move', reqId, best, ms, nodes, nps, backend: currentBackend });
      break;
    }

    default: {
      // Unknown messages: treat as unsupported to surface hidden call paths.
      unsupported(type);
      break;
    }
  }
}

self.onmessage = (e) => {
  __msgQueue = __msgQueue
    .then(() => __handleMessage(e))
    .catch((err) => {
      postError(err);
    });
};
