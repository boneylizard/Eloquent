export const VOICE_MERGE_QUEUE_STORAGE_KEY = 'LiangLocal:voiceMergeQueue';

export function newQueueJobId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `job-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** @typedef {'pending'|'running'|'done'|'error'|'cancelled'} QueueJobStatus */

/**
 * @typedef {Object} VoiceMergeQueueJob
 * @property {string} id
 * @property {string} source
 * @property {string} outputName
 * @property {number} morphBalance
 * @property {boolean} skipRvc
 * @property {boolean} skipUvr
 * @property {string} accentModel
 * @property {number} pitch
 * @property {number} indexRate
 * @property {number} protect
 * @property {string} voicePrompt
 * @property {QueueJobStatus} [status]
 * @property {string} [statusMessage]
 * @property {string} [error]
 * @property {string} [resultVoiceId]
 * @property {string} [resultPath]
 */

export function createQueueJob(partial = {}) {
  return {
    id: newQueueJobId(),
    source: '',
    outputName: '',
    morphBalance: 50,
    skipRvc: true,
    skipUvr: true,
    accentModel: 'default',
    pitch: 0,
    indexRate: 0.75,
    protect: 0.33,
    voicePrompt: '',
    status: 'pending',
    ...partial,
  };
}

export function loadVoiceMergeQueue() {
  try {
    const raw = localStorage.getItem(VOICE_MERGE_QUEUE_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((j) => j && typeof j === 'object' && j.id)
      .map((j) => ({
        ...createQueueJob(),
        ...j,
        status: j.status === 'running' ? 'pending' : (j.status || 'pending'),
      }));
  } catch {
    return [];
  }
}

export function saveVoiceMergeQueue(jobs) {
  try {
    const serializable = jobs.map(({ id, source, outputName, morphBalance, skipRvc, skipUvr, accentModel, pitch, indexRate, protect, voicePrompt }) => ({
      id,
      source,
      outputName,
      morphBalance,
      skipRvc,
      skipUvr,
      accentModel,
      pitch,
      indexRate,
      protect,
      voicePrompt,
    }));
    localStorage.setItem(VOICE_MERGE_QUEUE_STORAGE_KEY, JSON.stringify(serializable));
  } catch {
    /* quota or private mode */
  }
}

export function basename(path) {
  if (!path) return '';
  const parts = String(path).replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || path;
}

export function summarizeJobSources(job) {
  const lines = (job.source || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return '(no sources)';
  if (lines.length === 1) return basename(lines[0]);
  if (lines.length === 2) {
    return `${basename(lines[0])} + ${basename(lines[1])}`;
  }
  return `${basename(lines[0])} + ${lines.length - 1} more`;
}

export function jobLabel(job) {
  const name = (job.outputName || '').trim();
  const sources = summarizeJobSources(job);
  const bal = job.morphBalance ?? 50;
  const lines = (job.source || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const balanceHint = lines.length === 2 ? ` · ${Number(bal).toFixed(1)}%→clip2` : '';
  return name ? `${name} (${sources}${balanceHint})` : `${sources}${balanceHint}`;
}
