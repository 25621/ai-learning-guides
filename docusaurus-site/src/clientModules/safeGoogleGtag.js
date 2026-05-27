const trackingID = 'G-LEDRXS502H';

function ensureGtag() {
  window.dataLayer = window.dataLayer || [];

  if (typeof window.gtag !== 'function') {
    window.gtag = function gtag() {
      window.dataLayer.push(arguments);
    };
  }

  return window.gtag;
}

export function onRouteDidUpdate({location, previousLocation}) {
  if (typeof window === 'undefined' || !location) {
    return;
  }

  if (
    previousLocation &&
    location.pathname === previousLocation.pathname &&
    location.search === previousLocation.search &&
    location.hash === previousLocation.hash
  ) {
    return;
  }

  ensureGtag()('config', trackingID, {
    page_path: `${location.pathname}${location.search}${location.hash}`,
    anonymize_ip: true,
  });
}
