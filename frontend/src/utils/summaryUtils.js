/**
 * summaryUtils.js
 * Utilities for managing manual conversation summaries in localStorage.
 * Key: 'eloquent-chat-summaries' -> Array of { id, title, content, date }
 */

const STORAGE_KEY = 'eloquent-chat-summaries';

/**
 * Get all saved summaries
 * @returns {Array} Array of summary objects
 */
export const getSummaries = () => {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) {
        console.error('Failed to parse summaries:', e);
        return [];
    }
};

/**
 * Save a new summary
 * @param {string} title - User-defined title
 * @param {string} content - AI generated summary
 * @returns {object} The saved summary object
 */
export const saveSummary = (title, content) => {
    const summaries = getSummaries();
    const newSummary = {
        id: Date.now().toString(),
        title: title || `Summary ${new Date().toLocaleDateString()}`,
        content,
        date: new Date().toISOString()
    };

    const updated = [newSummary, ...summaries];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return newSummary;
};

/**
 * Delete a summary by ID
 * @param {string} id 
 */
export const deleteSummary = (id) => {
    const summaries = getSummaries();
    const updated = summaries.filter(s => s.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
};

/**
 * Update an existing summary
 * @param {string} id 
 * @param {object} updates 
 */
export const updateSummary = (id, updates) => {
    const summaries = getSummaries();
    const updated = summaries.map(s => s.id === id ? { ...s, ...updates } : s);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
};
