/**
 * summaryUtils.js
 * Manages conversation summaries via backend API (JSON files in backend static/summaries).
 * No localStorage; active summary is in-memory only (no restore on load).
 */

import { getBackendUrl } from '../config/api';

const getBaseUrl = () => getBackendUrl().replace(/\/$/, '');

/**
 * Get all saved summaries from backend
 * @returns {Promise<Array>} Array of summary objects { id, title, content, date }
 */
export async function getSummaries() {
  try {
    const res = await fetch(`${getBaseUrl()}/summaries`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    return Array.isArray(data) ? data : [];
  } catch (e) {
    console.error('Failed to fetch summaries:', e);
    return [];
  }
}

/**
 * Save a new summary to backend
 * @param {string} title - User-defined title
 * @param {string} content - AI generated summary
 * @returns {Promise<object>} The saved summary object
 */
export async function saveSummary(title, content) {
  const res = await fetch(`${getBaseUrl()}/summaries`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title: title || `Summary ${new Date().toLocaleDateString()}`,
      content: content || '',
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || 'Failed to save summary');
  }
  return res.json();
}

/**
 * Delete a summary by ID
 * @param {string} id
 * @returns {Promise<void>}
 */
export async function deleteSummary(id) {
  const res = await fetch(`${getBaseUrl()}/summaries/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(res.statusText || 'Failed to delete summary');
}

