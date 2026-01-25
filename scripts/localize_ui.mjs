#!/usr/bin/env node
/*
  localize_ui.mjs
  ----------------
  Goal: keep Sonetto web UI fully local (COOP+COEP friendly) while staying flexible
  for future UI changes.

  What it does (opt-in via flags):
    --build-tailwind  Generate vendor/tailwind.css from Sonetto.html using Tailwind CLI.
    --with-fflate     Download fflate UMD bundle to vendor/fflate.umd.js (gzip fallback).
    --with-phosphor   (Best-effort) Download Phosphor web assets into vendor/ (optional).

  Recommended:
    cd sonetto/web
    npm install
    npm run localize:ui

  Notes:
    - The repo ships with offline fallbacks:
        vendor/tailwind.css   (minimal subset)
        vendor/phosphor.css   (unicode fallback)
        vendor/fflate.umd.js  (stub)
      This script upgrades them to full originals when desired.
*/

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function hasFlag(name) {
  return process.argv.includes(name);
}

async function ensureDir(p) {
  await fs.mkdir(p, { recursive: true });
}

async function writeFileAtomic(filepath, data) {
  const tmp = filepath + '.tmp';
  await fs.writeFile(tmp, data);
  await fs.rename(tmp, filepath);
}

async function download(url, dest) {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`Download failed: ${url} (${res.status})`);
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  await writeFileAtomic(dest, buf);
}

async function buildTailwind(webDir) {
  const input = path.join(webDir, 'tailwind.input.css');
  const output = path.join(webDir, 'vendor', 'tailwind.css');

  // Prefer local tailwindcss binary if installed.
  const localBin = path.join(webDir, 'node_modules', '.bin', process.platform === 'win32' ? 'tailwindcss.cmd' : 'tailwindcss');
  let cmd = localBin;
  let args = ['-i', input, '-o', output, '--minify'];

  try {
    await fs.access(localBin);
  } catch {
    // Fall back to npx if local bin isn't available.
    cmd = process.platform === 'win32' ? 'npx.cmd' : 'npx';
    args = ['tailwindcss', ...args];
  }

  console.log(`[localize_ui] Building Tailwind CSS -> ${path.relative(process.cwd(), output)}`);
  await execFileAsync(cmd, args, { cwd: webDir, stdio: 'inherit' });
}

async function fetchFflate(webDir) {
  const dest = path.join(webDir, 'vendor', 'fflate.umd.js');
  const url = 'https://cdn.jsdelivr.net/npm/fflate@0.8.2/umd/index.js';
  console.log(`[localize_ui] Downloading fflate -> ${path.relative(process.cwd(), dest)}`);
  await download(url, dest);
}

async function fetchPhosphor(webDir) {
  // Best-effort only. The repo already ships with a unicode fallback.
  // This tries a few known unpkg entry points.
  const destCss = path.join(webDir, 'vendor', 'phosphor.css');
  const candidates = [
    'https://unpkg.com/@phosphor-icons/web@latest/src/index.css',
    'https://unpkg.com/@phosphor-icons/web@latest/css/phosphor.css',
    'https://unpkg.com/@phosphor-icons/web@latest',
  ];

  console.log('[localize_ui] Attempting to download Phosphor web CSS (best-effort)...');
  let cssText = null;
  let pickedUrl = null;
  let lastErr = null;

  for (const url of candidates) {
    try {
      const res = await fetch(url, { redirect: 'follow' });
      if (!res.ok) throw new Error(`${res.status}`);
      const ct = res.headers.get('content-type') || '';
      const text = await res.text();
      // Heuristic: CSS files should include @font-face or .ph class.
      if (ct.includes('text/css') || text.includes('@font-face') || text.includes('.ph')) {
        cssText = text;
        pickedUrl = url;
        console.log(`[localize_ui] Picked Phosphor source: ${url}`);
        break;
      }
      lastErr = new Error(`Not CSS: ${url}`);
    } catch (e) {
      lastErr = e;
    }
  }

  if (!cssText) {
    console.warn('[localize_ui] Could not fetch Phosphor CSS automatically. Keeping unicode fallback.');
    if (lastErr) console.warn(`[localize_ui] Last error: ${lastErr.message || String(lastErr)}`);
    return;
  }

  // If the CSS references font URLs, try to download them and rewrite paths.
  // We look for url(...) occurrences and download relative assets.
  const baseUrl = new URL(pickedUrl || candidates[0]);
  const urlMatches = [...cssText.matchAll(/url\(([^)]+)\)/g)].map(m => m[1].trim().replace(/^['"]|['"]$/g, ''));
  const fontUrls = [...new Set(urlMatches)].filter(u => !u.startsWith('data:'));

  if (fontUrls.length > 0) {
    const fontDir = path.join(webDir, 'vendor', 'phosphor-fonts');
    await ensureDir(fontDir);
    for (const u of fontUrls) {
      try {
        const abs = new URL(u, baseUrl).toString();
        const filename = path.basename(new URL(abs).pathname);
        const out = path.join(fontDir, filename);
        console.log(`[localize_ui] Downloading font: ${filename}`);
        await download(abs, out);
        // Rewrite CSS to local relative path.
        cssText = cssText.split(u).join(`./phosphor-fonts/${filename}`);
      } catch (e) {
        console.warn(`[localize_ui] Failed to download font url(${u}): ${e.message || String(e)}`);
      }
    }
  }

  await writeFileAtomic(destCss, cssText);
  console.log(`[localize_ui] Wrote Phosphor CSS -> ${path.relative(process.cwd(), destCss)}`);
}

async function main() {
  const webDir = path.resolve(__dirname, '..', 'web');
  const vendorDir = path.join(webDir, 'vendor');
  await ensureDir(vendorDir);

  const doTailwind = hasFlag('--build-tailwind');
  const doFflate = hasFlag('--with-fflate');
  const doPhosphor = hasFlag('--with-phosphor');

  if (!doTailwind && !doFflate && !doPhosphor) {
    console.log('Usage: node ../scripts/localize_ui.mjs [--build-tailwind] [--with-fflate] [--with-phosphor]');
    process.exit(0);
  }

  if (doTailwind) {
    try {
      await buildTailwind(webDir);
    } catch (e) {
      console.warn('[localize_ui] Tailwind build failed. Keeping minimal vendor/tailwind.css');
      console.warn(`[localize_ui] ${e.message || String(e)}`);
    }
  }

  if (doFflate) {
    try {
      await fetchFflate(webDir);
    } catch (e) {
      console.warn('[localize_ui] fflate download failed. Keeping stub vendor/fflate.umd.js');
      console.warn(`[localize_ui] ${e.message || String(e)}`);
    }
  }

  if (doPhosphor) {
    try {
      await fetchPhosphor(webDir);
    } catch (e) {
      console.warn('[localize_ui] Phosphor localization failed. Keeping unicode fallback.');
      console.warn(`[localize_ui] ${e.message || String(e)}`);
    }
  }

  console.log('[localize_ui] Done.');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
