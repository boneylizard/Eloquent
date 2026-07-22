const EXTERNAL_PROTOCOL = /^(https?:\/\/|mailto:|tel:)/i;

export function isExternalHref(href) {
  return typeof href === 'string' && EXTERNAL_PROTOCOL.test(href.trim());
}

export function installExternalLinkHandler() {
  if (typeof window === 'undefined' || !window.__TAURI_INTERNALS__) return () => {};

  const handleClick = async (event) => {
    if (event.defaultPrevented || event.button !== 0) return;
    const anchor = event.target?.closest?.('a[href]');
    if (!anchor || anchor.hasAttribute('download')) return;
    const href = anchor.getAttribute('href')?.trim();
    if (!isExternalHref(href)) return;

    event.preventDefault();
    try {
      const { open } = await import('@tauri-apps/plugin-shell');
      await open(href);
    } catch (error) {
      console.error(`Mirid could not open ${href}`, error);
    }
  };

  document.addEventListener('click', handleClick);
  return () => document.removeEventListener('click', handleClick);
}
