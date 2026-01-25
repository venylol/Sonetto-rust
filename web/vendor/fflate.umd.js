/*
  fflate UMD placeholder.
  - Sonetto uses this ONLY as a fallback when DecompressionStream is not available.
  - For full gzip support on older browsers, run:
      node ../scripts/localize_ui.mjs --with-fflate
    which will download the official fflate UMD build into this file.
*/

// Intentionally minimal: presence of this script avoids a 404.
// If you need gzip fallback, replace with the real fflate bundle.
window.fflate = window.fflate || undefined;
