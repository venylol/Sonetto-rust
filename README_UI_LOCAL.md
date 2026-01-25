# Sonetto Web UI: Local assets (COOP/COEP friendly)

This repo is configured for WebAssembly threads (rayon) using:
- Cross-Origin-Opener-Policy: same-origin
- Cross-Origin-Embedder-Policy: require-corp

When these headers are enabled, cross-origin CDN resources are often blocked by
browsers (you'll see `NotSameOriginAfterDefaultedToSameOriginByCoep`).

To keep the UI flexible while remaining fully local/offline, `web/Sonetto.html`
loads its UI resources from `web/vendor/`.

## Out-of-the-box behavior

The repo ships with **offline fallbacks**:
- `web/vendor/tailwind.css` — a minimal Tailwind-compatible utility subset
- `web/vendor/phosphor.css` — unicode fallback for icons (keeps the existing markup)
- `web/vendor/fflate.umd.js` — a small stub; gzip fallback is optional

## Upgrading to full Tailwind / vendor bundles (recommended if you change UI)

From `sonetto/web`:

```bash
npm install
npm run localize:ui
```

This runs `scripts/localize_ui.mjs` which:
- generates a full `vendor/tailwind.css` using Tailwind's content scan
- downloads the real `fflate` UMD bundle (gzip fallback)
- (best-effort) downloads Phosphor icon CSS/fonts

If a download/build step fails, the app keeps working with the included fallbacks.
