/**
 * Normalize SSE JSON chunks from /generate into visible + reasoning text parts.
 */
function normalizeSseError(err) {
  if (err == null) return '';
  if (typeof err === 'string') return err.trim();
  if (typeof err === 'object' && err.message) return String(err.message).trim();
  return String(err).trim();
}

export function extractSseStreamParts(parsed) {
  if (!parsed || typeof parsed !== 'object') {
    return { text: '', reasoning: '', error: '' };
  }
  if (parsed.type === 'error') {
    const msg = normalizeSseError(parsed.error ?? parsed.message ?? parsed.detail);
    if (msg) return { text: '', reasoning: '', error: msg };
  }
  if (parsed.error != null) {
    const msg = normalizeSseError(parsed.error);
    if (msg) return { text: '', reasoning: '', error: msg };
  }
  const topReasoning = parsed.reasoning != null ? String(parsed.reasoning) : '';
  const topText = parsed.text != null ? String(parsed.text) : '';
  if (topReasoning !== '' || topText !== '') {
    return { text: topText, reasoning: topReasoning, error: '' };
  }
  const delta = parsed.choices?.[0]?.delta;
  if (!delta) return { text: '', reasoning: '', error: '' };
  let reasoning = delta.reasoning || delta.reasoning_content || '';
  if (!reasoning && Array.isArray(delta.reasoning_details)) {
    reasoning = delta.reasoning_details
      .map((d) => (d && (d.text || d.content)) || '')
      .join('');
  }
  return {
    text: delta.content || '',
    reasoning: reasoning || '',
    error: '',
  };
}
