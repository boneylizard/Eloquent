/** Web search fields for /generate — separate from Transcript search tab. */

const MODE_KEY = 'eloquent:webSearchMode';
const URLS_KEY = 'eloquent:webSearchUrls';
const SITE_KEY = 'eloquent:webSearchSite';
const STRATEGY_KEY = 'eloquent:webSearchStrategy';

/** Global strategy: auto | eloquent | native | off (default auto). */
export const WEB_SEARCH_STRATEGIES = ['auto', 'eloquent', 'native', 'off'];

export function loadWebSearchStrategy() {
  try {
    const v = localStorage.getItem(STRATEGY_KEY);
    if (v && WEB_SEARCH_STRATEGIES.includes(v)) return v;
    return 'auto';
  } catch {
    return 'auto';
  }
}

export function saveWebSearchStrategy(strategy) {
  try {
    const v = WEB_SEARCH_STRATEGIES.includes(strategy) ? strategy : 'auto';
    localStorage.setItem(STRATEGY_KEY, v);
  } catch {
    /* ignore */
  }
}

export function loadWebSearchMode() {
  try {
    return localStorage.getItem(MODE_KEY) || '';
  } catch {
    return '';
  }
}

export function saveWebSearchMode(mode) {
  try {
    if (mode) localStorage.setItem(MODE_KEY, mode);
    else localStorage.removeItem(MODE_KEY);
  } catch {
    /* ignore */
  }
}

export function loadWebSearchArticleUrls() {
  try {
    return localStorage.getItem(URLS_KEY) || '';
  } catch {
    return '';
  }
}

export function saveWebSearchArticleUrls(text) {
  try {
    localStorage.setItem(URLS_KEY, text);
  } catch {
    /* ignore */
  }
}

export function loadWebSearchSite() {
  try {
    return localStorage.getItem(SITE_KEY) || '';
  } catch {
    return '';
  }
}

export function saveWebSearchSite(text) {
  try {
    if (text) localStorage.setItem(SITE_KEY, text);
    else localStorage.removeItem(SITE_KEY);
  } catch {
    /* ignore */
  }
}

/** Sent with chat only when Web Search globe is on. Does NOT read transcript tab settings. */
export function getWebSearchResearchPayload(settings) {
  try {
    const mode = loadWebSearchMode();
    const out = {};
    const strategy =
      settings?.webSearchStrategy && WEB_SEARCH_STRATEGIES.includes(settings.webSearchStrategy)
        ? settings.webSearchStrategy
        : loadWebSearchStrategy();
    out.web_search_strategy = strategy;
    if (mode) out.web_search_mode = mode;

    if (mode === 'articles' || mode === 'deep') {
      const site = loadWebSearchSite();
      const urlsRaw = loadWebSearchArticleUrls();
      const research_urls = urlsRaw
        .split('\n')
        .map((s) => s.trim())
        .filter((u) => u.startsWith('http'));
      if (site) out.research_site = site;
      if (research_urls.length) out.research_urls = research_urls;
    }
    return out;
  } catch {
    return { web_search_strategy: 'auto' };
  }
}

/** Label for search path from backend meta. */
export function webSearchPathLabel(meta) {
  if (!meta) return '';
  const path = meta.path || '';
  if (path === 'provider_native') return 'Native search';
  if (path === 'eloquent_prefetch') return 'Eloquent search';
  if (meta.status === 'searching') return 'Searching…';
  if (meta.status === 'native_delegated') return 'Native (model)';
  const n = meta.source_count ?? (meta.sources?.length ?? 0);
  if (n > 0) return `${n} source${n === 1 ? '' : 's'}`;
  return meta.status === 'error' ? 'Search failed' : '';
}
