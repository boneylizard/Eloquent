import { providerIcon, inferProviderFromModelId } from './providerIcon';
import { findNanoGptModel, readNanoGptModelsCache } from './nanoGptModelsCache';

export const ELOQUENT_SETTINGS_KEY = 'Eloquent-settings';

export function readEloquentSettings() {
  try {
    const raw = localStorage.getItem(ELOQUENT_SETTINGS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function getRotationPool(settings) {
  const endpoints = settings?.customApiEndpoints || [];
  return endpoints.filter(
    (ep) => ep?.enabled !== false && ep?.rotate_enabled !== false,
  );
}

export function resolveEndpointRecord(endpointId, settings) {
  if (!endpointId) return null;
  const endpoints = settings?.customApiEndpoints || [];
  return endpoints.find((ep) => ep.id === endpointId) || null;
}

/**
 * Canonical endpoint-* id for /generate when primary is API.
 * Matches backend resolve_api_endpoint_id (id, display name, or provider model field).
 */
export function resolvePrimaryEndpointIdForRequest(primaryModel, primaryIsAPI, settings) {
  if (!primaryIsAPI || !primaryModel) return primaryModel;
  const raw = String(primaryModel).trim();
  if (!raw) return primaryModel;
  if (raw.startsWith('endpoint-')) return raw;
  const endpoints = settings?.customApiEndpoints || [];
  const ep = endpoints.find(
    (e) => e?.id === raw || e?.name === raw || e?.model === raw,
  );
  return ep?.id || primaryModel;
}

export function inferCapabilitiesFromModelId(modelId) {
  const id = String(modelId || '').toLowerCase();
  if (!id) return {};
  const reasoningHints = [
    'thinking',
    'reason',
    'reasoner',
    'r1',
    'o1',
    'o3',
    'deepseek-r1',
    'sonnet-4-thinking',
    'opus-4-thinking',
  ];
  const reasoning = reasoningHints.some((hint) => id.includes(hint));
  return reasoning ? { reasoning: true } : {};
}

/**
 * Resolve endpoint id → human label + provider for UI.
 * @param {string} endpointId
 * @param {object} [settings]
 * @param {object[]} [catalog] - normalized nanoGpt models from cache
 */
export function resolveEndpointDisplay(endpointId, settings, catalog) {
  const s = settings || readEloquentSettings();
  const models = catalog ?? readNanoGptModelsCache().models;
  const ep = resolveEndpointRecord(endpointId, s);

  if (!ep) {
    if (endpointId?.startsWith?.('endpoint-')) {
      return {
        endpointId,
        displayName: endpointId,
        provider: '',
        icon: '⬜',
        modelId: '',
        endpointName: '',
      };
    }
    return null;
  }

  const modelId = ep.model || '';
  const cached = findNanoGptModel(modelId, models);
  const provider =
    cached?.provider
    || inferProviderFromModelId(modelId)
    || inferProviderFromModelId(ep.name);
  const displayName =
    cached?.name
    || (modelId ? modelId.split('/').pop() : '')
    || ep.name
    || endpointId;

  const inferredCaps = inferCapabilitiesFromModelId(modelId);
  const capabilities = cached?.capabilities || inferredCaps;
  const capabilitySource =
    cached?.capabilities
      ? 'catalog'
      : (inferredCaps && inferredCaps.reasoning ? 'inferred' : 'none');

  return {
    endpointId: ep.id,
    displayName,
    provider,
    icon: providerIcon(provider),
    modelId,
    endpointName: ep.name || '',
    enabled: ep.enabled !== false,
    rotateEnabled: ep.rotate_enabled !== false,
    endpoint: ep,
    capabilities,
    capabilitySource,
  };
}

/** Display for primary model chip (API endpoint, local path, or autorouting). */
export function resolvePrimaryModelDisplay({
  primaryModel,
  primaryIsAPI,
  settings,
  catalog,
}) {
  const s = settings || readEloquentSettings();
  const autoOn = s.apiEndpointRoundRobinEnabled === true;
  const pool = getRotationPool(s);

  if (primaryIsAPI && autoOn && pool.length >= 2) {
    const cursorMap = s.apiEndpointRoundRobinCursor || {};
    const cursorIdx = Number(cursorMap.__manual_rotation__ ?? 0) % pool.length;
    const poolDisplays = pool.map((ep) => resolveEndpointDisplay(ep.id, s, catalog));
    return {
      label: '⟳ Auto',
      shortLabel: '⟳ Auto',
      icon: '⟳',
      isAutoRouting: true,
      pool: poolDisplays,
      cursorIndex: cursorIdx,
      primaryModel,
    };
  }

  if (primaryIsAPI && primaryModel?.startsWith?.('endpoint-')) {
    const resolved = resolveEndpointDisplay(primaryModel, s, catalog);
    if (resolved) {
      return {
        label: resolved.displayName,
        shortLabel: resolved.displayName,
        icon: resolved.icon,
        provider: resolved.provider,
        isAutoRouting: false,
        endpointId: primaryModel,
        ...resolved,
      };
    }
  }

  if (!primaryModel) {
    return { label: 'Select model', shortLabel: 'Select model', icon: '⬜', isAutoRouting: false };
  }

  // Local GGUF / path
  let displayName = primaryModel.split('/').pop().split('\\').pop();
  if (displayName.endsWith('.bin') || displayName.endsWith('.gguf')) {
    displayName = displayName.substring(0, displayName.lastIndexOf('.'));
  }
  if (primaryModel.includes('openai')) displayName = 'OpenAI API';

  return {
    label: displayName,
    shortLabel: displayName,
    icon: primaryIsAPI ? providerIcon(inferProviderFromModelId(primaryModel)) : '💻',
    isAutoRouting: false,
    primaryModel,
  };
}

/** Group custom endpoints by resolved nano model name for selector tab A. */
export function groupEndpointsByModel(settings, catalog) {
  const endpoints = (settings?.customApiEndpoints || []).slice();
  const byKey = new Map();

  for (const ep of endpoints) {
    const resolved = resolveEndpointDisplay(ep.id, settings, catalog);
    const key = resolved?.modelId || resolved?.displayName || ep.id;
    if (!byKey.has(key)) byKey.set(key, { key, resolved, endpoints: [] });
    byKey.get(key).endpoints.push({ ep, resolved });
  }

  return Array.from(byKey.values()).sort((a, b) =>
    (a.resolved?.displayName || '').localeCompare(b.resolved?.displayName || ''),
  );
}

/** Bot UI label + provider badge when no character system prompt / persona is active. */
export function resolveApiBotSpeakerDisplay({
  primaryModel,
  primaryIsAPI,
  settings,
  catalog,
}) {
  if (!primaryIsAPI || !primaryModel) return null;

  const s = settings || readEloquentSettings();
  const models = catalog ?? readNanoGptModelsCache().models;
  const primary = resolvePrimaryModelDisplay({
    primaryModel,
    primaryIsAPI: true,
    settings: s,
    catalog: models,
  });

  if (primary?.isAutoRouting) {
    return {
      displayName: primary.shortLabel || '⟳ Auto',
      icon: primary.icon || '⟳',
      modelName: primaryModel,
    };
  }

  if (primaryModel.startsWith?.('endpoint-')) {
    const d = resolveEndpointDisplay(primaryModel, s, models);
    if (d) {
      return {
        displayName: d.displayName,
        icon: d.icon,
        modelName: primaryModel,
        provider: d.provider,
      };
    }
  }

  const cached = findNanoGptModel(primaryModel, models);
  if (cached) {
    return {
      displayName: cached.name,
      icon: providerIcon(cached.provider),
      modelName: primaryModel,
      provider: cached.provider,
    };
  }

  return null;
}

export function messageHasCharacterSpeaker(message, characters) {
  if (!message || message.role !== 'bot') return false;
  if (message.isCharacterIntro) return true;
  if (message.characterId) {
    const live = characters?.find((c) => c.id === message.characterId);
    if (live?.name) return true;
  }
  if (message.characterName) {
    const byName = characters?.find(
      (c) => (c.name || '').toLowerCase() === String(message.characterName).toLowerCase(),
    );
    if (byName) return true;
    if (message.avatar) return true;
  }
  return false;
}

/** Persist API model identity on bot placeholders when no character is speaking. */
export function attachApiBotSpeakerMeta(
  botMsg,
  { speakerCharacter, primaryModel, primaryIsAPI, settings, catalog, characters } = {},
) {
  if (!botMsg || botMsg.role !== 'bot') return botMsg;
  if (speakerCharacter?.id) return botMsg;
  if (messageHasCharacterSpeaker(botMsg, characters)) return botMsg;

  const display = resolveApiBotSpeakerDisplay({
    primaryModel,
    primaryIsAPI,
    settings,
    catalog,
  });
  if (!display) return botMsg;

  return {
    ...botMsg,
    modelName: display.modelName || primaryModel,
    speakerIcon: display.icon,
  };
}

/**
 * Resolve bot message header + avatar for chat UI.
 * Character/name/avatar wins over API model branding.
 */
export function resolveBotMessageSpeaker(message, ctx = {}) {
  const {
    characters,
    activeCharacter,
    primaryCharacter,
    secondaryCharacter,
    primaryModel,
    primaryIsAPI,
    settings,
    catalog,
    getActiveCharacterAvatar,
  } = ctx;

  if (!message || message.role !== 'bot') {
    return { displayName: 'Assistant', icon: null, avatarUrl: null };
  }

  if (messageHasCharacterSpeaker(message, characters)) {
    const name =
      message.characterName
      || characters?.find((c) => c.id === message.characterId)?.name
      || 'Character';
    let avatarUrl = message.avatar;
    if (!avatarUrl && message.characterId && getActiveCharacterAvatar) {
      const live = characters?.find((c) => c.id === message.characterId);
      if (live) avatarUrl = getActiveCharacterAvatar(live);
    }
    return { displayName: name, icon: null, avatarUrl };
  }

  const charForModel =
    message.modelId === 'secondary' ? secondaryCharacter : primaryCharacter;
  if (charForModel?.id) {
    return {
      displayName: charForModel.name,
      icon: null,
      avatarUrl: getActiveCharacterAvatar?.(charForModel) ?? null,
    };
  }

  if (!message.characterId && activeCharacter?.id && activeCharacter?.name) {
    return {
      displayName: activeCharacter.name,
      icon: null,
      avatarUrl: getActiveCharacterAvatar?.(activeCharacter) ?? null,
    };
  }

  const modelKey = message.modelName || primaryModel;
  const useApi = Boolean(message.modelName) || primaryIsAPI;
  const apiDisplay = useApi
    ? resolveApiBotSpeakerDisplay({
      primaryModel: modelKey,
      primaryIsAPI: true,
      settings,
      catalog,
    })
    : null;

  if (apiDisplay) {
    return {
      displayName: apiDisplay.displayName,
      icon: message.speakerIcon || apiDisplay.icon,
      avatarUrl: null,
    };
  }

  if (modelKey && !primaryIsAPI) {
    const local = resolvePrimaryModelDisplay({
      primaryModel: modelKey,
      primaryIsAPI: false,
      settings,
      catalog,
    });
    if (local?.label) {
      return {
        displayName: local.label,
        icon: local.icon || '💻',
        avatarUrl: null,
      };
    }
  }

  return {
    displayName: message.characterName || 'Assistant',
    icon: message.speakerIcon || null,
    avatarUrl: message.avatar || null,
  };
}
