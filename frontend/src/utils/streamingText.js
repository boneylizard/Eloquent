/**
 * Splits streaming content into completed text + the final streaming token.
 * The streaming token gets the typewriter fade-in animation.
 *
 * @param {string} content - Full message content
 * @param {boolean} isStreaming - Whether this message is actively streaming
 * @returns {{ completed: string, streaming: string }}
 */
export function splitStreamingContent(content, isStreaming) {
  if (!isStreaming || !content || content.length === 0) {
    return { completed: content || '', streaming: '' };
  }

  // Find last whitespace boundary to split on a word
  const lastSpace = content.lastIndexOf(' ');
  if (lastSpace <= 0) {
    return { completed: '', streaming: content };
  }

  return {
    completed: content.slice(0, lastSpace),
    streaming: content.slice(lastSpace),
  };
}

/**
 * Returns a unique key for the streaming token to force re-render
 * on each content change (for typewriter animation).
 */
let streamingTokenCounter = 0;
export function nextStreamingTokenKey() {
  return `streaming-token-${++streamingTokenCounter}`;
}
