/* Sonetto PWA Service Worker
 * Notes:
 * - Keep COOP/COEP headers intact (required for WASM threads). We cache the *network responses*
 *   so headers are preserved.
 * - We avoid rewriting requests; navigation requests fall back to network first.
 */
const CACHE_NAME = 'sonetto-pwa-v1';

const CORE_ASSETS = [
  './',
  './index.html',
  './Sonetto.html',
  './manifest.webmanifest',
  './service-worker.js',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/apple-touch-icon.png',
  './vendor/tailwind.css',
  './vendor/phosphor.css',
  './vendor/fflate.umd.js',
  './zip_local.js',
  './egbk3_book.js',
  './engine_worker.js',
  './sonetto_assets.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    try {
      await cache.addAll(CORE_ASSETS);
    } catch (_) {
      // Some optional assets may be missing depending on build variant; ignore.
    }
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => (k !== CACHE_NAME ? caches.delete(k) : Promise.resolve())));
    self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // Only handle same-origin
  if (url.origin !== self.location.origin) return;

  const isNavigation = req.mode === 'navigate';

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);

    if (isNavigation) {
      // Network-first for HTML to keep updates + headers correct
      try {
        const fresh = await fetch(req);
        cache.put(req, fresh.clone());
        return fresh;
      } catch (_) {
        return (await cache.match(req)) || (await cache.match('./')) || (await cache.match('./index.html'));
      }
    }

    // Cache-first for static assets
    const cached = await cache.match(req);
    if (cached) return cached;

    try {
      const fresh = await fetch(req);
      // Only cache successful responses
      if (fresh && fresh.ok) cache.put(req, fresh.clone());
      return fresh;
    } catch (_) {
      return cached || Response.error();
    }
  })());
});
