/**
 * Defensive gtag bootstrap.
 *
 * The `@docusaurus/plugin-google-gtag` client module calls `window.gtag(...)`
 * on every client-side route change (see plugin-google-gtag/lib/gtag.js). It
 * assumes the analytics snippet has already defined `window.gtag`. When that
 * snippet is missing — e.g. the gtag <script> was blocked by a privacy
 * extension, or it has not run yet — that call throws the runtime error:
 *
 *     Uncaught TypeError: window.gtag is not a function
 *
 * Defining the standard Google Analytics stub here, at client-module
 * evaluation time (which runs before any route-update event fires), guarantees
 * `window.gtag` is always a function. When the real analytics library loads it
 * drains `window.dataLayer`, so this stub is a no-op on the normal path.
 *
 * Note: analytics itself is only enabled for production builds (see the
 * `gtag` preset option in docusaurus.config.js). This module just makes the
 * plugin's route hook crash-proof.
 */
if (typeof window !== 'undefined') {
  window.dataLayer = window.dataLayer || [];
  if (typeof window.gtag !== 'function') {
    window.gtag = function gtag() {
      window.dataLayer.push(arguments);
    };
  }
}
