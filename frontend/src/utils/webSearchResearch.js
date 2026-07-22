/** Automatic web-search fields for chat requests. */

/** Web search has one routing mode: automatic. */
export function getWebSearchResearchPayload() {
  return { web_search_strategy: 'auto' };
}

/** User-facing progress without exposing internal routing details. */
export function webSearchPathLabel(meta) {
  if (!meta) return '';
  if (meta.status === 'searching' || meta.status === 'native_delegated' || meta.status === 'tool_calling') {
    return 'Searching…';
  }
  const sourceCount = meta.source_count ?? (meta.sources?.length ?? 0);
  if (sourceCount > 0) return `${sourceCount} source${sourceCount === 1 ? '' : 's'}`;
  return meta.status === 'error' ? 'Search failed' : '';
}
