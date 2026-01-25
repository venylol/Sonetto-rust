#!/usr/bin/env node
/*
  scripts/self_check_pages.mjs
  --------------------------
  Cloudflare Pages smoke-test for the built "sonetto_pages_root.zip".

  Goals (strict):
    - If this self-check passes, the uploaded Cloudflare Pages ZIP should run
      without runtime errors.
    - Catch the most common Pages deployment foot-guns:
        * missing files / wrong ZIP root layout
        * missing COOP/COEP headers (threads)
        * missing/incorrect wasm MIME types
        * missing large assets (weights/book) or per-file size limit violations
        * UI fails to render or becomes unresponsive
        * WASM engine fails to initialize (single-thread and threads)
        * midgame + endgame analysis paths fail
        * basic interaction + AI move loop fails

  What this script does:
    1) Extract the ZIP to a temp dir
    2) Validate required files & size guardrails
    3) Start a local HTTP server (two runs):
         - without COOP/COEP (single-thread path)
         - with COOP/COEP    (threads path)
    4) Use Playwright (Chromium) to open the UI and run in-page checks.

  Usage:
    node scripts/self_check_pages.mjs \
      --zip sonetto_pages_root.zip \
      --out selfcheck_report

  Notes:
    - Requires Playwright:
        npm install
        npx playwright install --with-deps chromium
*/

import fs from 'fs';
import fsp from 'fs/promises';
import os from 'os';
import path from 'path';
import http from 'http';
import { spawn } from 'child_process';

// Playwright is a dev dependency used only for self-check.
// eslint-disable-next-line import/no-extraneous-dependencies
import { chromium } from 'playwright';

// -----------------------------
// Small utilities
// -----------------------------

function nowIso() {
  return new Date().toISOString();
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function parseArgs(argv) {
  const out = {
    zip: 'sonetto_pages_root.zip',
    outDir: 'selfcheck_report',
    htmlPath: '/Sonetto.html',
    timeoutMs: 90_000,
    runNoCoi: true,
    runCoi: true,
    verbose: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const take = () => {
      if (i + 1 >= argv.length) throw new Error(`Missing value after ${a}`);
      return argv[++i];
    };
    if (a === '--zip') out.zip = take();
    else if (a === '--out') out.outDir = take();
    else if (a === '--html') out.htmlPath = take();
    else if (a === '--timeout-ms') out.timeoutMs = Number(take());
    else if (a === '--only-no-coi') { out.runNoCoi = true; out.runCoi = false; }
    else if (a === '--only-coi') { out.runNoCoi = false; out.runCoi = true; }
    else if (a === '--verbose') out.verbose = true;
    else if (a === '--help' || a === '-h') {
      console.log(`\nUsage: node scripts/self_check_pages.mjs [options]\n\nOptions:\n  --zip <path>          Pages ZIP path (default: sonetto_pages_root.zip)\n  --out <dir>           Output report directory (default: selfcheck_report)\n  --html <path>         Entry HTML path (default: /Sonetto.html)\n  --timeout-ms <ms>     Per-scenario timeout (default: 90000)\n  --only-no-coi         Only run single-thread scenario\n  --only-coi            Only run COOP/COEP (threads) scenario\n  --verbose             Print extra debug logs\n`);
      process.exit(0);
    } else {
      throw new Error(`Unknown arg: ${a} (use --help)`);
    }
  }

  if (!Number.isFinite(out.timeoutMs) || out.timeoutMs <= 0) {
    throw new Error('--timeout-ms must be a positive number');
  }
  if (!out.htmlPath.startsWith('/')) out.htmlPath = `/${out.htmlPath}`;
  return out;
}

async function mkdirp(p) {
  await fsp.mkdir(p, { recursive: true });
}

function rel(p) {
  return path.relative(process.cwd(), p);
}

function bytesToMiB(n) {
  return n / (1024 * 1024);
}

function statSizeBytes(filePath) {
  // Cross-platform-ish: try GNU stat first, then BSD stat.
  try {
    const st = fs.statSync(filePath);
    return st.size;
  } catch {
    return null;
  }
}

function mimeTypeForPath(p) {
  const ext = path.extname(p).toLowerCase();
  if (ext === '.html') return 'text/html; charset=utf-8';
  if (ext === '.js' || ext === '.mjs') return 'text/javascript; charset=utf-8';
  if (ext === '.css') return 'text/css; charset=utf-8';
  if (ext === '.json') return 'application/json; charset=utf-8';
  if (ext === '.wasm') return 'application/wasm';
  if (ext === '.gz') return 'application/octet-stream';
  if (ext === '.png') return 'image/png';
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
  if (ext === '.svg') return 'image/svg+xml';
  if (ext === '.txt') return 'text/plain; charset=utf-8';
  if (ext === '.ico') return 'image/x-icon';
  return 'application/octet-stream';
}

function normalizeUrlPath(u) {
  const p = (u || '/').split('?')[0].split('#')[0];
  // Ensure decode doesn't throw for malformed URLs.
  try {
    return decodeURIComponent(p);
  } catch {
    return p;
  }
}

async function runCmd(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      ...opts,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('close', (code) => resolve({ code, stdout, stderr }));
  });
}

function formatConsoleMsg(msg) {
  const loc = msg.location();
  const where = loc && loc.url ? `${loc.url}:${loc.lineNumber || 0}:${loc.columnNumber || 0}` : '';
  return {
    type: msg.type(),
    text: msg.text(),
    where,
  };
}

// -----------------------------
// ZIP extraction + static checks
// -----------------------------

async function extractZipToTemp(zipPath) {
  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'sonetto-pages-'));
  const { code, stdout, stderr } = await runCmd('unzip', ['-q', zipPath, '-d', tmp]);
  if (code !== 0) {
    throw new Error(`Failed to unzip '${zipPath}' (code=${code}).\n${stdout}\n${stderr}`);
  }
  return tmp;
}

async function fileExists(p) {
  try {
    await fsp.access(p, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function readTextIfExists(p) {
  if (!(await fileExists(p))) return null;
  return fsp.readFile(p, 'utf8');
}

async function listFilesRecursive(dir) {
  const out = [];
  async function walk(d) {
    const entries = await fsp.readdir(d, { withFileTypes: true });
    for (const e of entries) {
      const fp = path.join(d, e.name);
      if (e.isDirectory()) await walk(fp);
      else if (e.isFile()) out.push(fp);
    }
  }
  await walk(dir);
  return out;
}

function parseHeadersFile(text) {
  // Very small parser: just extract whether COOP/COEP appear anywhere.
  // (Cloudflare Pages parses the file; our smoke test just verifies intent.)
  const lower = (text || '').toLowerCase();
  return {
    hasCoop: lower.includes('cross-origin-opener-policy:') && lower.includes('same-origin'),
    hasCoep: lower.includes('cross-origin-embedder-policy:') && lower.includes('require-corp'),
    hasWasmMime: lower.includes('content-type: application/wasm'),
    hasGzOctet: lower.includes('application/octet-stream'),
  };
}

async function staticSanityChecks(extractedRoot, report, args) {
  // These checks are meant to fail fast with actionable messages.
  const required = [
    'index.html',
    'Sonetto.html',
    '_headers',
    'engine_worker.js',
    // At least the scalar wasm build must exist.
    'pkg/sonetto_wasm.js',
    'pkg/sonetto_wasm_bg.wasm',
  ];

  // When running the COOP/COEP (threads) scenario, require threads artifacts
  // up-front so failures are obvious (instead of timing out in runtime checks).
  if (args?.runCoi) {
    required.push(
      'pkg_threads/sonetto_wasm.js',
      'pkg_threads/sonetto_wasm_bg.wasm',
      // Chromium supports SIMD and will prefer the threads+SIMD build.
      'pkg_threads_simd/sonetto_wasm.js',
      'pkg_threads_simd/sonetto_wasm_bg.wasm',
    );
  }

  const missing = [];
  for (const r of required) {
    const fp = path.join(extractedRoot, r);
    // eslint-disable-next-line no-await-in-loop
    if (!(await fileExists(fp))) missing.push(r);
  }
  if (missing.length > 0) {
    report.staticChecks.ok = false;
    report.staticChecks.errors.push({
      kind: 'missing_files',
      message: `Missing required files in ZIP root: ${missing.join(', ')}`,
      missing,
    });
  }

  // Extra threads-specific shape checks (helps catch partial or wrongly-located builds)
  // without being overly strict across wasm-bindgen versions.
  if (args?.runCoi) {
    const threadPkgs = ['pkg_threads', 'pkg_threads_simd'];
    for (const pkg of threadPkgs) {
      try {
        // 1) If the generated glue references snippets/, the directory must exist.
        const jsPath = path.join(extractedRoot, pkg, 'sonetto_wasm.js');
        const threadsJs = await readTextIfExists(jsPath);
        if (threadsJs && threadsJs.includes('snippets/')) {
          const snippetsDir = path.join(extractedRoot, pkg, 'snippets');
          const hasSnippets = await fileExists(snippetsDir) && (await fsp.stat(snippetsDir)).isDirectory();
          if (!hasSnippets) {
            report.staticChecks.ok = false;
            report.staticChecks.errors.push({
              kind: 'missing_threads_snippets',
              message: `Threads build appears incomplete: ${pkg}/snippets directory is missing (sonetto_wasm.js references snippets/*).`,
            });
          }
        }

        // 2) Guardrail for a common failure mode:
        // If the wasm-bindgen glue doesn't initialize shared memory, initThreadPool will fail at runtime.
        // This often shows up as: DataCloneError: #<Memory> could not be cloned.
        if (threadsJs) {
          const hasInitMemory = threadsJs.includes('__wbg_init_memory');
          const hasWasmMemoryCtor = threadsJs.includes('new WebAssembly.Memory');
          const hasSharedFlag = threadsJs.includes('shared:true') || threadsJs.includes('shared: true');
          const emptyInitMemory = /function\s+__wbg_init_memory\([^)]*\)\s*\{\s*\}/m.test(threadsJs);

          if (hasInitMemory && (emptyInitMemory || !hasWasmMemoryCtor || !hasSharedFlag)) {
            report.staticChecks.ok = false;
            report.staticChecks.errors.push({
              kind: 'threads_memory_init_missing',
              message: `Threads build glue for ${pkg} does not appear to initialize shared WebAssembly.Memory. This usually breaks initThreadPool at runtime (DataCloneError: Memory could not be cloned). Ensure the threads builds are produced with nightly + -Z build-std and wasm32 atomics enabled.`,
            });
          }
        }
      } catch (_) {
        // Ignore: we already fail hard on missing core files.
      }
    }
  }

  const headersText = await readTextIfExists(path.join(extractedRoot, '_headers'));
  if (!headersText) {
    report.staticChecks.ok = false;
    report.staticChecks.errors.push({
      kind: 'missing_headers_file',
      message: 'Missing _headers file (required for COOP/COEP threads + wasm MIME types on Cloudflare Pages).',
    });
  } else {
    const parsed = parseHeadersFile(headersText);
    report.staticChecks.headers = parsed;
    if (!parsed.hasCoop || !parsed.hasCoep) {
      report.staticChecks.ok = false;
      report.staticChecks.errors.push({
        kind: 'missing_coop_coep',
        message: 'The _headers file does not appear to set COOP/COEP (threads will not work on Cloudflare Pages).',
      });
    }
    if (!parsed.hasWasmMime) {
      report.staticChecks.ok = false;
      report.staticChecks.errors.push({
        kind: 'missing_wasm_mime',
        message: 'The _headers file does not appear to set Content-Type: application/wasm for .wasm files (some browsers may refuse to load).',
      });
    }
  }

  // Cloudflare Pages direct upload per-file size limit (~25 MiB).
  const MAX_BYTES = 25 * 1024 * 1024;
  const files = await listFilesRecursive(extractedRoot);
  const tooBig = [];
  for (const f of files) {
    const sz = statSizeBytes(f);
    if (typeof sz === 'number' && sz > MAX_BYTES) {
      tooBig.push({
        file: path.relative(extractedRoot, f),
        bytes: sz,
        mib: Number(bytesToMiB(sz).toFixed(2)),
      });
    }
  }
  report.staticChecks.tooBigFiles = tooBig;
  if (tooBig.length > 0) {
    report.staticChecks.ok = false;
    report.staticChecks.errors.push({
      kind: 'pages_file_size_limit',
      message: `One or more files exceed 25 MiB (Cloudflare Pages direct upload limit).`,
      tooBig,
    });
  }

  // Ensure the big assets are present (strict requirement in the user request).
  const wantAssets = [
    'eval.egev2.gz',
    'book.egbk3.gz',
  ];
  const missingAssets = [];
  for (const a of wantAssets) {
    // eslint-disable-next-line no-await-in-loop
    if (!(await fileExists(path.join(extractedRoot, a)))) missingAssets.push(a);
  }
  report.staticChecks.assets = {
    expected: wantAssets,
    missing: missingAssets,
  };
  if (missingAssets.length > 0) {
    report.staticChecks.ok = false;
    report.staticChecks.errors.push({
      kind: 'missing_assets',
      message: `Missing required large assets (weights/book): ${missingAssets.join(', ')}`,
      missing: missingAssets,
    });
  }
}

// -----------------------------
// Local HTTP server (Pages-ish)
// -----------------------------

async function startStaticServer(root, { coi }) {
  // Choose a random free port.
  const server = http.createServer((req, res) => {
    const urlPath = normalizeUrlPath(req.url);
    let p = urlPath;
    if (p.endsWith('/')) p = `${p}index.html`;
    if (p === '') p = '/index.html';

    // Path traversal guard.
    const fsPath = path.normalize(path.join(root, p));
    if (!fsPath.startsWith(path.normalize(root))) {
      res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Forbidden');
      return;
    }

    if (!fs.existsSync(fsPath) || !fs.statSync(fsPath).isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not Found');
      return;
    }

    const headers = {
      'Content-Type': mimeTypeForPath(fsPath),
      'Cache-Control': 'no-store',
    };

    if (coi) {
      headers['Cross-Origin-Opener-Policy'] = 'same-origin';
      headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
    }

    res.writeHead(200, headers);
    const s = fs.createReadStream(fsPath);
    s.on('error', (e) => {
      res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(`Read error: ${e.message || String(e)}`);
    });
    s.pipe(res);
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address();
  const port = typeof addr === 'object' && addr ? addr.port : null;
  if (!port) throw new Error('Failed to bind local server port');

  return {
    port,
    urlBase: `http://127.0.0.1:${port}`,
    close: async () => new Promise((resolve) => server.close(() => resolve())),
  };
}

// -----------------------------
// Failure hinting (actionable)
// -----------------------------

function inferScenarioHints(scenario) {
  const hints = [];

  const consoleText = (scenario.console || [])
    .map((c) => `${c.type}: ${c.text}`)
    .join('\n');

  // wasm-bindgen-rayon initThreadPool failures.
  if (scenario?.coi) {
    const wi = scenario.wasmInfo;
    const looksLikeThreadsMisinit = wi && wi.threads && (!wi.threadsUsed || wi.threadsUsed < 2);

    if (/DataCloneError:.*Memory.*could not be cloned/i.test(consoleText)) {
      hints.push({
        kind: 'wasm_threads_datacloneerror',
        message:
          'initThreadPool failed with DataCloneError (#<Memory> could not be cloned). This usually means the generated wasm-bindgen glue did not initialize shared WebAssembly.Memory (shared:true), so the threads pool cannot share memory between workers. Fix: build threads variants with nightly + `-Z build-std=panic_abort,std` and wasm atomics enabled; pinning nightly can avoid regressions.',
      });
    } else if (looksLikeThreadsMisinit) {
      hints.push({
        kind: 'wasm_threads_not_initialized',
        message:
          'Threads build loaded but threadsUsed<2. Either initThreadPool was not exported, was not called, or failed early. Check for warnings around wasm-bindgen-rayon initialization and confirm the threads pkg contains snippets/ and shared memory init in sonetto_wasm.js.',
      });
    }
  }

  // Common network-level issues.
  const wasm404 = (scenario.badResponses || []).find((r) => /\.wasm($|\?)/.test(r.url) && r.status === 404);
  if (wasm404) {
    hints.push({
      kind: 'missing_wasm_file',
      message: `A .wasm request returned 404 (${wasm404.url}). The ZIP likely missed a wasm artifact or paths are wrong.`,
    });
  }

  const blocked = (scenario.requestFailed || []).find((r) => /blocked|CORS|CORP/i.test(r.failure || ''));
  if (scenario?.coi && blocked) {
    hints.push({
      kind: 'coi_resource_blocked',
      message: `A request was blocked under COI mode (${blocked.url}; ${blocked.failure}). This can happen if COOP/COEP is enabled but some resources are served without CORP-compatible headers.`,
    });
  }

  return hints;
}

// -----------------------------
// Playwright scenario runner
// -----------------------------

async function runScenario({
  name,
  coi,
  extractedRoot,
  htmlPath,
  timeoutMs,
  outDir,
  verbose,
}) {
  const scenario = {
    name,
    coi,
    startedAt: nowIso(),
    finishedAt: null,
    ok: true,
    checks: [],
    console: [],
    pageErrors: [],
    requestFailed: [],
    badResponses: [],
    dialogs: [],
    screenshots: [],
    wasmInfo: null,
    hints: [],
    notes: [],
  };

  const server = await startStaticServer(extractedRoot, { coi });
  scenario.notes.push(`Local server: ${server.urlBase} (coi=${coi})`);

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
    ],
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });

  const page = await context.newPage();
  const navUrl = `${server.urlBase}${htmlPath}?selfcheck=1&coi=${coi ? 1 : 0}`;

  // Capture diagnostics.
  page.on('console', (msg) => {
    const o = formatConsoleMsg(msg);
    scenario.console.push(o);
    if (verbose) {
      // eslint-disable-next-line no-console
      console.log(`[${name}][console.${o.type}] ${o.text}${o.where ? ` (${o.where})` : ''}`);
    }
  });
  page.on('pageerror', (err) => {
    scenario.pageErrors.push({
      message: err?.message || String(err),
      stack: err?.stack || null,
    });
  });
  page.on('requestfailed', (req) => {
    scenario.requestFailed.push({
      url: req.url(),
      method: req.method(),
      failure: req.failure()?.errorText || 'unknown',
      resourceType: req.resourceType(),
    });
  });
  page.on('response', (resp) => {
    const st = resp.status();
    if (st >= 400) {
      scenario.badResponses.push({
        url: resp.url(),
        status: st,
        statusText: resp.statusText(),
      });
    }
  });
  page.on('dialog', async (dlg) => {
    scenario.dialogs.push({ type: dlg.type(), message: dlg.message() });
    // Avoid hanging headless runs.
    await dlg.dismiss().catch(() => {});
  });

  async function recordScreenshot(tag) {
    const file = path.join(outDir, `${name}_${tag}.png`);
    await page.screenshot({ path: file, fullPage: true }).catch(() => {});
    scenario.screenshots.push(rel(file));
  }

  const check = async (title, fn) => {
    const entry = { title, ok: true, details: null };
    try {
      entry.details = await fn();
    } catch (e) {
      entry.ok = false;
      entry.details = {
        error: e?.message || String(e),
        stack: e?.stack || null,
      };
    }
    scenario.checks.push(entry);
    if (!entry.ok) scenario.ok = false;
    return entry;
  };

  try {
    await check('Navigate', async () => {
      const resp = await page.goto(navUrl, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
      const st = resp?.status() || null;
      if (st && st >= 400) throw new Error(`Navigation HTTP ${st}`);
      return { url: page.url(), status: st };
    });

    await check('No fatal JS errors during early load', async () => {
      // Give it a small moment to emit early errors.
      await sleep(250);
      if (scenario.pageErrors.length > 0) {
        throw new Error(`pageerror: ${scenario.pageErrors[0].message}`);
      }
      return { pageErrors: scenario.pageErrors.length };
    });

    await check('Wait for engine ready', async () => {
      await page.waitForFunction(() => window.__SONETTO_ENGINE_READY__ === true, { timeout: timeoutMs });
      const info = await page.evaluate(() => ({
        engineReady: !!window.__SONETTO_ENGINE_READY__,
        crossOriginIsolated: !!window.crossOriginIsolated,
        wasmInfo: window.__SONETTO_WASM_INFO__ || (window.AnalysisManager && window.AnalysisManager.wasmInfo) || null,
      }));
      scenario.wasmInfo = info.wasmInfo;
      if (coi && !info.crossOriginIsolated) {
        throw new Error('Expected crossOriginIsolated=true under COOP/COEP scenario');
      }
      if (!coi && info.crossOriginIsolated) {
        throw new Error('Expected crossOriginIsolated=false under no-COOP/COEP scenario');
      }
      return info;
    });

    await check('UI: board rendered (64 cells, visible)', async () => {
      await page.waitForFunction(() => {
        const b = document.getElementById('board');
        if (!b) return false;
        const cells = b.querySelectorAll('.cell');
        if (cells.length !== 64) return false;
        const r = b.getBoundingClientRect();
        return r.width > 10 && r.height > 10;
      }, { timeout: timeoutMs });

      return page.evaluate(() => {
        const b = document.getElementById('board');
        const r = b.getBoundingClientRect();
        return {
          cells: b.querySelectorAll('.cell').length,
          rect: { x: r.x, y: r.y, w: r.width, h: r.height },
        };
      });
    });

    await check('Assets: weights loaded', async () => {
      await page.waitForFunction(() => {
        const u = window.__SONETTO_EGEV2_BYTES__;
        return (u && (u instanceof Uint8Array) && u.byteLength > 1024);
      }, { timeout: timeoutMs });
      return page.evaluate(() => ({
        bytes: window.__SONETTO_EGEV2_BYTES__ ? window.__SONETTO_EGEV2_BYTES__.byteLength : 0,
      }));
    });

    await check('Assets: opening book loaded and indexed', async () => {
      // The book loads async; wait for the internal promise if present.
      await page.waitForFunction(() => !!window.__EGBK3_BOOK_PROMISE__, { timeout: timeoutMs });
      const ok = await page.evaluate(async () => {
        const p = window.__EGBK3_BOOK_PROMISE__;
        if (!p || typeof p.then !== 'function') return { ok: false, reason: 'no_promise' };
        const book = await p;
        if (!book) return { ok: false, reason: 'book_null' };
        if (!book.ready) return { ok: false, reason: 'not_ready' };
        return { ok: true, nBoards: book.nBoards || null, tableSize: book.tableSize || null };
      });
      if (!ok.ok) throw new Error(`Book not ready: ${ok.reason}`);
      return ok;
    });

    await check('Engine: single-thread vs threads selection is correct', async () => {
      const info = await page.evaluate(() => {
        const wi = window.__SONETTO_WASM_INFO__ || (window.AnalysisManager && window.AnalysisManager.wasmInfo) || null;
        return {
          wi,
          crossOriginIsolated: !!window.crossOriginIsolated,
          sab: typeof SharedArrayBuffer !== 'undefined',
        };
      });

      if (!info.wi) throw new Error('No wasmInfo reported from worker');
      if (coi) {
        if (!info.wi.threads) throw new Error('COI scenario expected a threads-capable build');
        if (!info.wi.threadsUsed || info.wi.threadsUsed < 2) {
          throw new Error(`COI scenario expected threadsUsed>=2 (got ${info.wi.threadsUsed})`);
        }
      } else {
        if (info.wi.threads) {
          // In no-COI, the worker should not attempt to use threads.
          // It may still report threads=false, which is expected.
          throw new Error('No-COI scenario should not report threads=true');
        }
      }

      return info;
    });

    await check('Self-check API exported (window bindings)', async () => {
      const api = await page.evaluate(() => ({
        hasUserConfig: !!window.UserConfig,
        hasGameHistory: Array.isArray(window.gameHistory),
        hasCurrentStepIndex: Number.isInteger(window.currentStepIndex),
        hasAnalysisManager: !!window.AnalysisManager,
        canStartAnalysis: !!(window.AnalysisManager && typeof window.AnalysisManager.startAnalysis === 'function'),
        hasEnsureMoveCache: typeof window.ensureStateMoveCache === 'function',
      }));
      if (!api.hasUserConfig) throw new Error('window.UserConfig is not exposed (selfcheck needs live config access)');
      if (!api.hasGameHistory) throw new Error('window.gameHistory is not exposed (selfcheck needs history access)');
      if (!api.hasCurrentStepIndex) throw new Error('window.currentStepIndex is not exposed (selfcheck needs current step)');
      if (!api.hasAnalysisManager || !api.canStartAnalysis) throw new Error('window.AnalysisManager.startAnalysis is not exposed (selfcheck needs to drive analysis)');
      if (!api.hasEnsureMoveCache) throw new Error('window.ensureStateMoveCache is not exposed');
      return api;
    });

    await check('Midgame analysis works (Top-N results)', async () => {
      const result = await page.evaluate(async () => {
        // Force engine path (not book) for the test.
        window.UserConfig = window.UserConfig || {};
        window.UserConfig.useBook = false;
        window.UserConfig.midDepth = 1;
        window.UserConfig.endStart = 12;

        const st = window.gameHistory[window.currentStepIndex];
        const cache = window.ensureStateMoveCache(st);
        const moves = cache.moves;
        if (!Array.isArray(moves) || moves.length === 0) {
          return { ok: false, reason: 'no_legal_moves' };
        }

        const board = st.board;
        const player = st.player;

        const res = await new Promise((resolve, reject) => {
          let done = false;
          const t = setTimeout(() => {
            if (done) return;
            done = true;
            reject(new Error('midgame analysis timeout'));
          }, 45_000);
          try {
            window.AnalysisManager.startAnalysis(board, player, moves, (r) => {
              if (done) return;
              done = true;
              clearTimeout(t);
              resolve(r);
            }, false);
          } catch (e) {
            if (done) return;
            done = true;
            clearTimeout(t);
            reject(e);
          }
        });

        const ok = Array.isArray(res) && res.length > 0 && res.every((x) => x && Number.isFinite(x.x) && Number.isFinite(x.y) && Number.isFinite(x.val));
        return { ok, n: Array.isArray(res) ? res.length : 0, sample: Array.isArray(res) ? res.slice(0, 3) : null };
      });

      if (!result.ok) throw new Error(`Midgame analysis failed (reason=${result.reason || 'bad_result'})`);
      return result;
    });

    await check('Endgame analysis works (Exact solver path)', async () => {
      const result = await page.evaluate(async () => {
        // Force engine exact path: empties <= endStart AND <= 30.
        window.UserConfig = window.UserConfig || {};
        window.UserConfig.useBook = false;
        window.UserConfig.midDepth = 1;
        window.UserConfig.endStart = 30;

        function computeMoves(board, player) {
          const r = window.computeValidMovesAndMask(board, player);
          return r && Array.isArray(r.moves) ? r.moves : [];
        }

        function applyMove(board, mv, player) {
          const flips = window.getFlippableDiscs(board, mv.x, mv.y, player);
          if (!flips || flips.length === 0) return false;
          board[mv.x][mv.y] = player;
          for (const f of flips) board[f.x][f.y] = player;
          return true;
        }

        function genReachablePosition(targetEmpties) {
          let board = window.createInitialBoard();
          let player = window.BLACK;
          let pass = 0;
          let guard = 0;
          while (window.countEmptySquares(board) > targetEmpties && pass < 2 && guard < 200) {
            guard++;
            const moves = computeMoves(board, player);
            if (!moves.length) {
              player = window.getOpponent(player);
              pass++;
              continue;
            }
            pass = 0;
            const mv = moves[(Math.random() * moves.length) | 0];
            applyMove(board, mv, player);
            player = window.getOpponent(player);
          }
          return { board, player, empties: window.countEmptySquares(board) };
        }

        const pos = genReachablePosition(8);
        if (pos.empties > 30) return { ok: false, reason: 'failed_to_reduce_empties', empties: pos.empties };
        const moves = computeMoves(pos.board, pos.player);
        if (!moves.length) return { ok: false, reason: 'no_legal_moves', empties: pos.empties };

        const res = await new Promise((resolve, reject) => {
          let done = false;
          const t = setTimeout(() => {
            if (done) return;
            done = true;
            reject(new Error('endgame analysis timeout'));
          }, 60_000);
          try {
            window.AnalysisManager.startAnalysis(pos.board, pos.player, moves, (r) => {
              if (done) return;
              done = true;
              clearTimeout(t);
              resolve(r);
            }, false);
          } catch (e) {
            if (done) return;
            done = true;
            clearTimeout(t);
            reject(e);
          }
        });

        const mode = window.AnalysisManager && window.AnalysisManager._lastAnalysisMode ? window.AnalysisManager._lastAnalysisMode : null;
        const ok = (mode === 'endgame') && Array.isArray(res) && res.length > 0;
        return { ok, mode, empties: pos.empties, n: Array.isArray(res) ? res.length : 0, sample: Array.isArray(res) ? res.slice(0, 3) : null };
      });

      if (!result.ok) throw new Error(`Endgame analysis failed (mode=${result.mode || 'unknown'}, reason=${result.reason || 'bad_result'})`);
      return result;
    });

    await check('User interaction: making a legal move updates state', async () => {
      const result = await page.evaluate(() => {
        const before = window.currentStepIndex;
        const st = window.gameHistory[before];
        const moves = window.ensureStateMoveCache(st).moves;
        if (!moves || !moves.length) return { ok: false, reason: 'no_moves' };
        const mv = moves[0];
        window.makeMove(mv.x, mv.y);
        const after = window.currentStepIndex;
        return { ok: after === before + 1, before, after, mv };
      });
      if (!result.ok) throw new Error(`Move did not apply (before=${result.before}, after=${result.after})`);
      return result;
    });

    await check('AI: can make at least one move (self-play BOTH)', async () => {
      const result = await page.evaluate(async () => {
        // Keep it fast.
        window.UserConfig = window.UserConfig || {};
        window.UserConfig.useBook = false;
        window.UserConfig.midDepth = 1;
        window.UserConfig.endStart = 12;

        if (typeof window.stopAiGame === 'function') window.stopAiGame();

        const startIdx = window.currentStepIndex;
        if (typeof window.startAiGame !== 'function') return { ok: false, reason: 'no_startAiGame' };
        window.startAiGame('BOTH');

        const ok = await new Promise((resolve) => {
          const deadline = Date.now() + 60_000;
          const tick = () => {
            if (window.currentStepIndex >= startIdx + 1) return resolve(true);
            if (Date.now() > deadline) return resolve(false);
            setTimeout(tick, 200);
          };
          tick();
        });

        if (typeof window.stopAiGame === 'function') window.stopAiGame();

        return { ok, startIdx, endIdx: window.currentStepIndex };
      });

      if (!result.ok) throw new Error(`AI did not play a move (start=${result.startIdx}, end=${result.endIdx})`);
      return result;
    });

    // Final diagnostics: ensure no network failures or JS crashes.
    await check('No request failures (404/blocked/etc)', async () => {
      if (scenario.requestFailed.length > 0) {
        throw new Error(`requestfailed: ${scenario.requestFailed[0].url} (${scenario.requestFailed[0].failure})`);
      }
      if (scenario.badResponses.length > 0) {
        throw new Error(`HTTP ${scenario.badResponses[0].status} for ${scenario.badResponses[0].url}`);
      }
      return {
        requestFailed: scenario.requestFailed.length,
        badResponses: scenario.badResponses.length,
      };
    });

    await check('No console.error messages', async () => {
      const errs = scenario.console.filter((c) => c.type === 'error');
      if (errs.length > 0) {
        throw new Error(`console.error: ${errs[0].text}`);
      }
      return { consoleErrors: 0 };
    });
  } catch (e) {
    scenario.ok = false;
    scenario.notes.push(`Scenario exception: ${e?.message || String(e)}`);
  } finally {
    // Always snapshot on failure for debugging.
    if (!scenario.ok) {
      await recordScreenshot('failed');
    }

    scenario.finishedAt = nowIso();

    // Add human-actionable hints based on collected diagnostics.
    scenario.hints = inferScenarioHints(scenario);

    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  }

  return scenario;
}

// -----------------------------
// Main
// -----------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const report = {
    startedAt: nowIso(),
    finishedAt: null,
    ok: true,
    args,
    platform: {
      node: process.version,
      os: `${os.platform()} ${os.release()} (${os.arch()})`,
    },
    staticChecks: {
      ok: true,
      headers: null,
      assets: null,
      tooBigFiles: [],
      errors: [],
    },
    scenarios: [],
  };

  const zipPath = path.resolve(process.cwd(), args.zip);
  if (!fs.existsSync(zipPath)) {
    throw new Error(`ZIP not found: ${zipPath}`);
  }

  const outDirAbs = path.resolve(process.cwd(), args.outDir);
  await mkdirp(outDirAbs);

  // 1) Extract.
  const extractedRoot = await extractZipToTemp(zipPath);

  // Some zips accidentally contain a top-level folder. Detect that and treat as error.
  // Cloudflare Pages direct upload requires the ZIP root to contain index.html.
  if (!fs.existsSync(path.join(extractedRoot, 'index.html'))) {
    // Try a single nested folder.
    const kids = (await fsp.readdir(extractedRoot, { withFileTypes: true }))
      .filter((d) => d.isDirectory())
      .map((d) => d.name);
    if (kids.length === 1 && fs.existsSync(path.join(extractedRoot, kids[0], 'index.html'))) {
      report.staticChecks.ok = false;
      report.staticChecks.errors.push({
        kind: 'zip_has_top_level_folder',
        message: `ZIP root does not contain index.html (found nested folder '${kids[0]}'). Cloudflare Pages drag&drop requires index.html at the archive root.`,
      });
    } else {
      report.staticChecks.ok = false;
      report.staticChecks.errors.push({
        kind: 'zip_root_missing_index',
        message: 'ZIP root does not contain index.html. Cloudflare Pages drag&drop requires index.html at the archive root.',
      });
    }
  }

  // 2) Static sanity checks.
  await staticSanityChecks(extractedRoot, report, args);
  if (!report.staticChecks.ok) {
    report.ok = false;
  }

  // 3) Runtime checks (Playwright) — only if static checks passed.
  if (report.staticChecks.ok) {
    if (args.runNoCoi) {
      report.scenarios.push(await runScenario({
        name: 'no_coi',
        coi: false,
        extractedRoot,
        htmlPath: args.htmlPath,
        timeoutMs: args.timeoutMs,
        outDir: outDirAbs,
        verbose: args.verbose,
      }));
    }
    if (args.runCoi) {
      report.scenarios.push(await runScenario({
        name: 'coi',
        coi: true,
        extractedRoot,
        htmlPath: args.htmlPath,
        timeoutMs: args.timeoutMs,
        outDir: outDirAbs,
        verbose: args.verbose,
      }));
    }
  }

  // 4) Summarize.
  for (const s of report.scenarios) {
    if (!s.ok) report.ok = false;
  }

  report.finishedAt = nowIso();

  const reportPath = path.join(outDirAbs, 'selfcheck_report.json');
  await fsp.writeFile(reportPath, JSON.stringify(report, null, 2), 'utf8');

  // Human-friendly summary.
  // eslint-disable-next-line no-console
  console.log(`\n[SelfCheck] Report: ${rel(reportPath)}`);

  if (!report.staticChecks.ok) {
    // eslint-disable-next-line no-console
    console.error('\n[SelfCheck] FAILED: static checks failed');
    for (const e of report.staticChecks.errors) {
      // eslint-disable-next-line no-console
      console.error(` - ${e.kind}: ${e.message}`);
    }
  }

  for (const s of report.scenarios) {
    // eslint-disable-next-line no-console
    console.log(`\n[SelfCheck] Scenario '${s.name}': ${s.ok ? 'PASS' : 'FAIL'}`);
    if (!s.ok) {
      // eslint-disable-next-line no-console
      console.error(`  - Screenshots: ${s.screenshots.join(', ') || '(none)'}`);
      const failed = s.checks.filter((c) => !c.ok);
      for (const c of failed) {
        // eslint-disable-next-line no-console
        console.error(`  - FAIL: ${c.title}: ${c.details?.error || 'unknown error'}`);
      }
      if (s.pageErrors.length > 0) {
        // eslint-disable-next-line no-console
        console.error(`  - pageErrors: ${s.pageErrors[0].message}`);
      }
      const errs = s.console.filter((c) => c.type === 'error');
      if (errs.length > 0) {
        // eslint-disable-next-line no-console
        console.error(`  - console.error: ${errs[0].text}`);
      }
      if (s.requestFailed.length > 0) {
        // eslint-disable-next-line no-console
        console.error(`  - requestFailed: ${s.requestFailed[0].url} (${s.requestFailed[0].failure})`);
      }
      if (s.badResponses.length > 0) {
        // eslint-disable-next-line no-console
        console.error(`  - badResponses: HTTP ${s.badResponses[0].status} for ${s.badResponses[0].url}`);
      }

      if (s.hints && s.hints.length) {
        // eslint-disable-next-line no-console
        console.error('  - hints:', JSON.stringify(s.hints, null, 2));
      }

      // Extra verbose diagnostics on failure (bounded to keep logs readable).
      // This is intentionally printed to the CI log so failures are actionable
      // even without downloading the JSON artifact.
      try {
        const consoleCounts = s.console.reduce((m, c) => {
          m[c.type] = (m[c.type] || 0) + 1;
          return m;
        }, {});
        const consoleErrors = s.console.filter((c) => c.type === 'error').slice(0, 20);
        const consoleWarnings = s.console.filter((c) => c.type === 'warning').slice(0, 20);

        // eslint-disable-next-line no-console
        console.error('  - wasmInfo:', JSON.stringify(s.wasmInfo || null));
        // eslint-disable-next-line no-console
        console.error('  - consoleCounts:', JSON.stringify(consoleCounts));
        if (consoleErrors.length) {
          // eslint-disable-next-line no-console
          console.error('  - console.error (first 20):', JSON.stringify(consoleErrors, null, 2));
        }
        if (consoleWarnings.length) {
          // eslint-disable-next-line no-console
          console.error('  - console.warning (first 20):', JSON.stringify(consoleWarnings, null, 2));
        }
        if (s.pageErrors.length) {
          // eslint-disable-next-line no-console
          console.error('  - pageErrors:', JSON.stringify(s.pageErrors.slice(0, 10), null, 2));
        }
        if (s.requestFailed.length) {
          // eslint-disable-next-line no-console
          console.error('  - requestFailed (first 20):', JSON.stringify(s.requestFailed.slice(0, 20), null, 2));
        }
        if (s.badResponses.length) {
          // eslint-disable-next-line no-console
          console.error('  - badResponses (first 20):', JSON.stringify(s.badResponses.slice(0, 20), null, 2));
        }
        if (s.dialogs.length) {
          // eslint-disable-next-line no-console
          console.error('  - dialogs:', JSON.stringify(s.dialogs.slice(0, 10), null, 2));
        }
      } catch (_) {}
    }
  }

  if (!report.ok) {
    // eslint-disable-next-line no-console
    console.error('\n[SelfCheck] OVERALL: FAIL');
    process.exit(1);
  }

  // eslint-disable-next-line no-console
  console.log('\n[SelfCheck] OVERALL: PASS');
}

main().catch((e) => {
  // eslint-disable-next-line no-console
  console.error('[SelfCheck] Fatal:', e?.stack || e?.message || String(e));
  process.exit(1);
});
