const MODULES = Object.freeze({
  chat: { defaultEnabled: true },
  documents: { defaultEnabled: true },
  characters: { defaultEnabled: true },
  pool: { defaultEnabled: false, lockedOff: true },
  chess: { retired: true, lockedOff: true },
  memory: { defaultEnabled: true },
  transcripts: { defaultEnabled: true },
  voice: { defaultEnabled: true },
  images: { defaultEnabled: true },
  elections: { defaultEnabled: false, optional: true },
  market: { retired: true },
  chatlogCondenser: { defaultEnabled: false },
  codeEditor: { retired: true },
  forensics: { retired: true },
  watch: { retired: true },
});

function configuredModules() {
  const configured = String(import.meta.env.VITE_MIRID_ENABLED_MODULES || '');
  return new Set(configured.split(',').map((value) => value.trim()).filter(Boolean));
}

function savedOverrides() {
  if (typeof window === 'undefined') return {};
  try {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    return settings.modules && typeof settings.modules === 'object' ? settings.modules : {};
  } catch {
    return {};
  }
}

export function isModuleEnabled(moduleId) {
  const policy = MODULES[moduleId];
  if (!policy || policy.retired || policy.lockedOff || !isModuleInstalled(moduleId)) return false;
  const override = savedOverrides()[moduleId];
  if (typeof override === 'boolean') return override;
  if (configuredModules().has(moduleId)) return true;
  return policy.defaultEnabled;
}

export function isModuleInstalled(moduleId) {
  const policy = MODULES[moduleId];
  if (!policy || policy.retired) return false;
  if (!policy.optional) return true;
  if (moduleId === 'elections') return __MIRID_INCLUDE_ELECTIONS__;
  return false;
}

export function modulePolicy(moduleId) {
  return MODULES[moduleId] || null;
}
