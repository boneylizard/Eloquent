function makeId() {
  return `v-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Parse playlist from JSON, plain URL lines, or simple M3U (EXTINF lines ignored).
 * @param {string} raw
 * @returns {{ id: string, url: string, title: string }[]}
 */
export function parsePlaylistText(raw) {
  const text = String(raw || '').trim();
  if (!text) return [];
  try {
    const j = JSON.parse(text);
    if (Array.isArray(j)) {
      return j
        .map((entry) => {
          if (typeof entry === 'string') {
            const u = entry.trim();
            if (!u) return null;
            return { id: makeId(), url: u, title: u.split(/[/\\?#]/).filter(Boolean).pop() || u };
          }
          if (entry && typeof entry.url === 'string') {
            const u = entry.url.trim();
            if (!u) return null;
            const title = String(entry.title || u).trim() || u.split(/[/\\?#]/).filter(Boolean).pop() || u;
            return { id: makeId(), url: u, title };
          }
          return null;
        })
        .filter(Boolean);
    }
  } catch {
    /* not JSON */
  }
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const out = [];
  for (const line of lines) {
    if (line.startsWith('#')) continue;
    if (
      line.startsWith('http://') ||
      line.startsWith('https://') ||
      line.startsWith('/') ||
      line.startsWith('file:')
    ) {
      out.push({
        id: makeId(),
        url: line,
        title: line.split(/[/\\?#]/).filter(Boolean).pop() || line,
      });
    }
  }
  return out;
}
