import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Cpu, RefreshCw, Search, Star, Monitor } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { useApp } from '../contexts/AppContext';
import { getContextLength, saveContextLength } from '../utils/apiCall';
import {
  NANO_GPT_MODELS_CACHE_TTL_MS,
  refreshNanoGptModelsCache,
  readNanoGptModelsCache,
  subscribeNanoGptModelsCache,
} from '../utils/nanoGptModelsCache';
import {
  refreshOpenRouterModelsCache,
  readOpenRouterModelsCache,
  subscribeOpenRouterModelsCache,
} from '../utils/openRouterModelsCache';
import { FRONTIER_PROVIDERS } from '../config/frontierProviders';
import {
  annotateHostedModel,
  findHostedModelEndpoint,
  getConnectedHostedProviderIds,
  getHostedModelProvider,
  getHostedProviderKey,
  upsertHostedModelEndpoint,
} from '../utils/hostedModelProviders';
import {
  readFrontierProviderCatalog,
  refreshFrontierProviderCatalog,
  subscribeFrontierProviderCatalogs,
} from '../utils/frontierProviderCatalogCache';
import {
  groupEndpointsByModel,
  resolveEndpointDisplay,
  resolvePrimaryModelDisplay,
} from '../utils/resolveEndpointDisplay';
import { providerIcon } from '../utils/providerIcon';

const FAV_KEY = 'nanoGpt-model-favorites-v1';

function safeJsonParse(str) {
  try {
    return JSON.parse(str);
  } catch {
    return null;
  }
}

function formatOpenRouterPricing(model) {
  if (model?.hostProvider !== 'openrouter') return '';
  if (model.free) return 'Free';
  const promptPrice = Number(model.pricing?.prompt);
  const completionPrice = Number(model.pricing?.completion);
  if (!Number.isFinite(promptPrice) || !Number.isFinite(completionPrice)) return 'Pricing unavailable';
  return `$${(promptPrice * 1_000_000).toFixed(2)} in · $${(completionPrice * 1_000_000).toFixed(2)} out / 1M`;
}

function ModelDisplayPill({ display, className, title }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-[rgba(120,170,220,0.4)] bg-muted px-2.5 py-1 text-xs text-foreground max-w-[240px]',
        className,
      )}
      title={title}
    >
      <span className="flex-shrink-0 text-sm leading-none" aria-hidden>
        {display?.icon || '⬜'}
      </span>
      <span className="truncate">{display?.shortLabel || display?.label || 'Select model'}</span>
      {display?.providerLabel && (
        <span className="flex-shrink-0 text-[9px] uppercase tracking-wide text-muted-foreground">
          {display.providerLabel}
        </span>
      )}
    </span>
  );
}

export { ModelDisplayPill };

export default function NanoGptModelSelectorPopover({
  currentModelId,
  onSelectModelId,
  onCapabilities,
  className,
  primaryApiUrl,
  trigger,
  open: controlledOpen,
  onOpenChange,
  compact = false,
  showAutoRoutingToggle = true,
  /** When false, selection only invokes onSelectModelId (e.g. one-off regen) without changing global primary. */
  updatePrimaryOnSelect = true,
}) {
  const {
    settings,
    updateSettings,
    primaryModel,
    primaryIsAPI,
    setPrimaryModel,
    setPrimaryIsAPI,
    availableModels,
    loadedModels,
    loadModel,
    unloadModel,
    setActiveModel,
    isModelLoading,
  } = useApp();

  const [internalOpen, setInternalOpen] = useState(false);
  const open = controlledOpen ?? internalOpen;
  const setOpen = onOpenChange ?? setInternalOpen;

  const [tab, setTab] = useState('all');
  const [localCtx, setLocalCtx] = useState(() => getContextLength());
  const [query, setQuery] = useState('');
  const [catalog, setCatalog] = useState(() => readNanoGptModelsCache().models);
  const [openRouterCatalog, setOpenRouterCatalog] = useState(() => readOpenRouterModelsCache().models);
  const [frontierCatalogues, setFrontierCatalogues] = useState(() => Object.fromEntries(
    FRONTIER_PROVIDERS.map((provider) => [
      provider.id,
      readFrontierProviderCatalog(provider.id, settings[provider.keySetting] || ''),
    ]),
  ));
  const [status, setStatus] = useState('idle');
  const [catalogError, setCatalogError] = useState(null);
  const [favorites, setFavorites] = useState(() => {
    const parsed = safeJsonParse(localStorage.getItem(FAV_KEY) || '[]');
    return new Set(Array.isArray(parsed) ? parsed : []);
  });
  const anchorRef = useRef(null);
  const panelRef = useRef(null);
  const [panelStyle, setPanelStyle] = useState({ top: 0, left: 0 });

  const effectiveId = currentModelId ?? primaryModel;

  const updatePanelPosition = useCallback(() => {
    const el = anchorRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const width = 440;
    const maxLeft = Math.max(8, window.innerWidth - width - 8);
    setPanelStyle({
      top: rect.bottom + 6,
      left: Math.min(rect.left, maxLeft),
    });
  }, []);

  const primaryDisplay = useMemo(
    () => resolvePrimaryModelDisplay({
      primaryModel: effectiveId,
      primaryIsAPI,
      settings,
      catalog,
    }),
    [effectiveId, primaryIsAPI, settings, catalog],
  );

  const loadCatalog = useCallback(async (force = false) => {
    setCatalogError(null);
    setStatus('loading');
    const errors = [];
    const nanoPromise = refreshNanoGptModelsCache({ forceRefresh: force })
      .then((result) => {
        setCatalog(result.models || []);
        if (result.error) errors.push(`NanoGPT: ${result.error?.message || String(result.error)}`);
        return result;
      });
    const openRouterPromise = refreshOpenRouterModelsCache({
      forceRefresh: force,
      apiKey: settings.openRouterApiKey || '',
    }).then((result) => {
      setOpenRouterCatalog(result.models || []);
      if (result.error) errors.push(`OpenRouter: ${result.error?.message || String(result.error)}`);
      return result;
    });
    const frontierPromises = FRONTIER_PROVIDERS
      .filter((provider) => String(settings[provider.keySetting] || '').trim())
      .map(async (provider) => {
        try {
          const result = await refreshFrontierProviderCatalog({
            providerId: provider.id,
            apiKey: settings[provider.keySetting],
            primaryApiUrl: primaryApiUrl,
            forceRefresh: force,
          });
          setFrontierCatalogues((current) => ({ ...current, [provider.id]: result }));
          return result;
        } catch (error) {
          errors.push(`${provider.label}: ${error.message}`);
          return null;
        }
      });
    const [nanoResult] = await Promise.all([nanoPromise, openRouterPromise, ...frontierPromises]);
    setCatalogError(errors.length > 0 ? errors.join(' ') : null);
    setStatus(errors.length > 0
      ? (catalog.length > 0 || nanoResult?.models?.length > 0 ? 'fallback' : 'error')
      : 'ok');
  }, [catalog.length, primaryApiUrl, settings]);

  useEffect(() => {
    const unsub = subscribeNanoGptModelsCache(({ models }) => setCatalog(models));
    const unsubscribeOpenRouter = subscribeOpenRouterModelsCache(({ models }) => setOpenRouterCatalog(models));
    const unsubscribeFrontier = subscribeFrontierProviderCatalogs(() => {
      setFrontierCatalogues(Object.fromEntries(
        FRONTIER_PROVIDERS.map((provider) => [
          provider.id,
          readFrontierProviderCatalog(provider.id, settings[provider.keySetting] || ''),
        ]),
      ));
    });
    return () => {
      unsub();
      unsubscribeOpenRouter();
      unsubscribeFrontier();
    };
  }, [settings]);

  useEffect(() => {
    if (!open) return;
    loadCatalog(false);
  }, [open, loadCatalog, primaryApiUrl]);

  useEffect(() => {
    if (!open) return;
    updatePanelPosition();
    window.addEventListener('resize', updatePanelPosition);
    window.addEventListener('scroll', updatePanelPosition, true);
    return () => {
      window.removeEventListener('resize', updatePanelPosition);
      window.removeEventListener('scroll', updatePanelPosition, true);
    };
  }, [open, updatePanelPosition]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => {
      // Treat any click within the anchored trigger or panel as "inside" the popover.
      // Using a shared data attribute makes this robust to composed events and
      // nested interactive elements.
      if (e.target?.closest?.('[data-nanogpt-model-popover-root="true"]')) {
        return;
      }
      const anchor = anchorRef.current;
      const panel = panelRef.current;
      if (anchor?.contains(e.target) || panel?.contains(e.target)) return;
      setOpen(false);
    };
    const frame = requestAnimationFrame(() => {
      document.addEventListener('mousedown', onDoc);
    });
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('mousedown', onDoc);
    };
  }, [open, setOpen]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        setOpen(false);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open, setOpen]);

  const endpointGroups = useMemo(
    () => groupEndpointsByModel(settings, catalog),
    [settings, catalog],
  );

  const hostedCatalogModels = useMemo(() => {
    const nanoModels = (catalog || [])
      .filter((model) => model.visible !== false && String(model.api || '').toLowerCase() === 'chat')
      .map((model) => annotateHostedModel(model, 'nanogpt'))
      .filter(Boolean);
    const routerModels = (openRouterCatalog || [])
      .map((model) => annotateHostedModel(model, 'openrouter'))
      .filter(Boolean);
    const directModels = FRONTIER_PROVIDERS.flatMap((provider) => {
      const providerCatalogue = frontierCatalogues[provider.id];
      return (providerCatalogue?.models || [])
        .map((model) => annotateHostedModel(model, provider.id, {
          baseUrl: providerCatalogue.baseUrl,
        }))
        .filter(Boolean);
    });
    return [...nanoModels, ...routerModels, ...directModels];
  }, [catalog, frontierCatalogues, openRouterCatalog]);

  const connectedProviderIds = useMemo(
    () => new Set(getConnectedHostedProviderIds(settings)),
    [settings],
  );

  const filteredCatalog = useMemo(() => {
    const term = query.trim().toLowerCase();
    return hostedCatalogModels.filter((model) => (
      !term
      || `${model.name} ${model.id} ${model.hostProviderLabel} ${model.modelProvider} ${model.category || ''}`
        .toLowerCase()
        .includes(term)
    ));
  }, [hostedCatalogModels, query]);

  const groupHostedModels = useCallback((models) => {
    const groups = new Map();
    for (const model of models) {
      if (!groups.has(model.hostProvider)) {
        groups.set(model.hostProvider, {
          providerId: model.hostProvider,
          label: model.hostProviderLabel,
          models: [],
        });
      }
      groups.get(model.hostProvider).models.push(model);
    }
    return Array.from(groups.values()).map((group) => [group.label, group.models, group.providerId]);
  }, []);

  const connectedCatalogGroups = useMemo(
    () => groupHostedModels(filteredCatalog.filter((model) => connectedProviderIds.has(model.hostProvider))),
    [connectedProviderIds, filteredCatalog, groupHostedModels],
  );

  const currentHostedModel = useMemo(() => {
    const ep = (settings?.customApiEndpoints || []).find((e) => e.id === effectiveId);
    if (ep?.model) {
      return hostedCatalogModels.find((model) => (
        model.hostProvider === ep.provider && model.id === ep.model
      )) || null;
    }
    return hostedCatalogModels.find((model) => model.id === effectiveId) || null;
  }, [effectiveId, hostedCatalogModels, settings?.customApiEndpoints]);

  useEffect(() => {
    const caps = currentHostedModel?.capabilities
      || resolveEndpointDisplay(effectiveId, settings, catalog)?.capabilities;
    if (caps && typeof onCapabilities === 'function') onCapabilities(caps);
  }, [currentHostedModel, effectiveId, settings, catalog, onCapabilities]);

  const selectEndpoint = useCallback(
    (endpointId) => {
      if (updatePrimaryOnSelect) {
        setPrimaryIsAPI(true);
        setPrimaryModel(endpointId);
      }
      if (typeof onSelectModelId === 'function') onSelectModelId(endpointId);
      setOpen(false);
    },
    [onSelectModelId, setOpen, setPrimaryIsAPI, setPrimaryModel, updatePrimaryOnSelect],
  );

  const toggleEndpointEnabled = useCallback(
    (endpointId, enabled) => {
      const endpoints = [...(settings.customApiEndpoints || [])];
      const idx = endpoints.findIndex((e) => e.id === endpointId);
      if (idx < 0) return;
      endpoints[idx] = { ...endpoints[idx], enabled };
      updateSettings({ customApiEndpoints: endpoints });
    },
    [settings.customApiEndpoints, updateSettings],
  );

  const toggleRotateEnabled = useCallback(
    (endpointId, rotate_enabled) => {
      const endpoints = [...(settings.customApiEndpoints || [])];
      const idx = endpoints.findIndex((e) => e.id === endpointId);
      if (idx < 0) return;
      endpoints[idx] = { ...endpoints[idx], rotate_enabled };
      updateSettings({ customApiEndpoints: endpoints });
    },
    [settings.customApiEndpoints, updateSettings],
  );

  const findEndpointForModel = useCallback(
    (model) => findHostedModelEndpoint(
      settings.customApiEndpoints || [],
      model.hostProvider,
      model.id,
    ),
    [settings.customApiEndpoints],
  );

  const selectCatalogModel = useCallback(
    (model) => {
      const apiKey = getHostedProviderKey(settings, model.hostProvider);
      if (!apiKey) {
        window.alert(`Add your ${model.hostProviderLabel} API key in Model Library before using this model.`);
        return;
      }
      const existing = findEndpointForModel(model);
      if (existing) {
        selectEndpoint(existing.id);
        return;
      }
      const result = upsertHostedModelEndpoint({
        endpoints: settings.customApiEndpoints || [],
        model,
        providerId: model.hostProvider,
        apiKey,
        baseUrl: model.baseUrl || getHostedModelProvider(model.hostProvider).baseUrl,
        billingMode: model.hostProvider === 'nanogpt' ? settings.nanoGptBillingMode : undefined,
      });
      updateSettings({ customApiEndpoints: result.endpoints });
      selectEndpoint(result.endpointId);
    },
    [findEndpointForModel, selectEndpoint, settings, updateSettings],
  );

  const autoOn = settings.apiEndpointRoundRobinEnabled === true;
  const rotationPool = useMemo(
    () => (settings.customApiEndpoints || []).filter(
      (ep) => ep.enabled !== false && ep.rotate_enabled !== false,
    ),
    [settings.customApiEndpoints],
  );

  const cacheAgeMin = Math.round(
    (Date.now() - (readNanoGptModelsCache().savedAt || 0)) / 60000,
  );

  const defaultTrigger = (
    <button
      type="button"
      onClick={() => setOpen(true)}
      className={cn(
        'inline-flex items-center gap-2 rounded-full border border-[rgba(120,170,220,0.45)] bg-muted/40 hover:bg-muted/70 transition-colors',
        compact ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
      )}
      title="Select model"
    >
      {!compact && <Cpu size={14} className="text-[#78aadc] flex-shrink-0" />}
      <ModelDisplayPill
        display={primaryDisplay}
        className="border-0 bg-transparent px-0 py-0 max-w-[200px]"
      />
      {status === 'loading' && (
        <span className="text-[10px] text-muted-foreground">…</span>
      )}
    </button>
  );

  const panel = open ? (
    <div
      ref={panelRef}
      data-nanogpt-model-popover-root="true"
      role="dialog"
      aria-label="Model selector"
      className="fixed z-[70] w-[440px] max-w-[92vw] rounded-2xl border border-[rgba(120,170,220,0.35)] bg-background shadow-[0_22px_60px_rgba(3,4,10,0.95)]"
      style={{ top: panelStyle.top, left: panelStyle.left }}
    >
          <div className="p-3 border-b border-[rgba(120,170,220,0.18)] space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="inline-flex rounded-full bg-muted p-0.5 border border-[rgba(120,170,220,0.28)]">
                <button
                  type="button"
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    e.nativeEvent?.stopImmediatePropagation?.();
                  }}
                  className={cn(
                    'px-3 py-1 text-[11px] rounded-full',
                    tab === 'endpoints' ? 'bg-[#78aadc] text-[#050608]' : 'text-foreground/80',
                  )}
                  onClick={() => setTab('endpoints')}
                >
                  Saved
                </button>
                <button
                  type="button"
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    e.nativeEvent?.stopImmediatePropagation?.();
                  }}
                  className={cn(
                    'px-3 py-1 text-[11px] rounded-full',
                    tab === 'all' ? 'bg-[#78aadc] text-[#050608]' : 'text-foreground/80',
                  )}
                  onClick={() => setTab('all')}
                >
                  My Models
                </button>
                <button
                  type="button"
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    e.nativeEvent?.stopImmediatePropagation?.();
                  }}
                  className={cn(
                    'px-3 py-1 text-[11px] rounded-full',
                    tab === 'local' ? 'bg-[#78aadc] text-[#050608]' : 'text-foreground/80',
                  )}
                  onClick={() => setTab('local')}
                >
                  Local
                </button>
              </div>
              <button
                type="button"
                className="p-1.5 rounded-full text-muted-foreground hover:text-foreground"
                title="Refresh model catalog"
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  if (status !== 'loading') loadCatalog(true);
                }}
              >
                <RefreshCw size={14} className={status === 'loading' ? 'animate-spin' : ''} />
              </button>
            </div>

            {showAutoRoutingToggle && (
              <div className="flex items-center justify-between gap-2 rounded-xl border border-[rgba(120,170,220,0.22)] bg-muted/50 px-3 py-2">
                <div className="min-w-0">
                  <div className="text-xs font-medium text-foreground">⟳ Auto-routing</div>
                  <div className="text-[10px] text-[rgba(148,163,184,0.8)] truncate">
                    {autoOn && rotationPool.length > 0
                      ? `Pool: ${rotationPool.length} endpoints`
                      : autoOn
                        ? 'Paused · selected model will be used'
                        : 'Rotate included endpoints each prompt'}
                  </div>
                </div>
                <Switch
                  checked={autoOn}
                  onCheckedChange={(enabled) => {
                    updateSettings({ apiEndpointRoundRobinEnabled: enabled });
                  }}
                  onClick={(e) => {
                    // Prevent outside-click handler from seeing this as an "outside" click
                    // and keep the popover open while toggling auto-routing.
                    e.stopPropagation();
                    e.nativeEvent?.stopImmediatePropagation?.();
                  }}
                />
              </div>
            )}

            {autoOn && rotationPool.length > 0 && (
              <div className="text-[10px] text-[rgba(148,163,184,0.85)] px-1">
                {rotationPool.map((ep, i) => {
                  const d = resolveEndpointDisplay(ep.id, settings, catalog);
                  const cursor = (settings.apiEndpointRoundRobinCursor || {}).__manual_rotation__ ?? 0;
                  const isCursor = Number(cursor) % rotationPool.length === i;
                  return (
                    <span key={ep.id} className={cn('mr-2', isCursor && 'text-[#78aadc] font-medium')}>
                      {d?.icon}
                      {' '}
                      {d?.displayName || ep.name}
                      {isCursor ? ' ◀' : ''}
                    </span>
                  );
                })}
              </div>
            )}

            {tab === 'all' && (
              <div className="flex items-center gap-2 rounded-full border border-[rgba(120,170,220,0.28)] bg-muted/50 px-3 py-2">
                <Search size={14} className="text-[rgba(148,163,184,0.85)]" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onMouseDown={(e) => {
                    // Keep focus/type interactions from bubbling up as outside clicks.
                    e.stopPropagation();
                    e.nativeEvent?.stopImmediatePropagation?.();
                  }}
                  placeholder="Search models…"
                  className="w-full bg-transparent text-sm text-foreground outline-none placeholder:text-[rgba(148,163,184,0.6)]"
                  autoFocus
                />
              </div>
            )}

            {tab !== 'local' && (
            <div className="text-[10px] text-[rgba(148,163,184,0.65)]">
              Cache
              {' '}
              {cacheAgeMin < 60 ? `${cacheAgeMin}m` : `${Math.round(cacheAgeMin / 60)}h`}
              {' '}
              old · TTL
              {' '}
              {Math.round(NANO_GPT_MODELS_CACHE_TTL_MS / 3600000)}
              h
              {status === 'fallback' && ' · offline/cached'}
            </div>
            )}

            {tab !== 'local' && hostedCatalogModels.length === 0 && status !== 'loading' && (
              <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 px-3 py-2 text-xs text-amber-100 flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <span>
                    Provider catalogues could not be loaded. Saved models remain available.
                  </span>
                  {catalogError && (
                    <div className="mt-1 text-[10px] text-amber-200 break-words">
                      {catalogError}
                    </div>
                  )}
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 text-[11px] flex-shrink-0"
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (status !== 'loading') loadCatalog(true);
                  }}
                  disabled={status === 'loading'}
                >
                  <RefreshCw size={12} className={cn('mr-1', status === 'loading' && 'animate-spin')} />
                  {status === 'loading' ? 'Refreshing…' : 'Refresh'}
                </Button>
              </div>
            )}
          </div>

          <div className="max-h-[420px] overflow-y-auto p-2">
            {tab === 'local' ? (
              <div className="space-y-2">
                <div className="px-2 py-1 flex items-center gap-2 text-[10px] uppercase tracking-[0.14em] text-[rgba(148,163,184,0.75)]">
                  <Monitor size={12} />
                  <span>Local GGUF Models</span>
                </div>
                {availableModels.length === 0 ? (
                  <div className="p-3 text-sm text-muted-foreground">
                    No local GGUF models found. Add a model directory in Settings → LLM Settings.
                  </div>
                ) : (
                  availableModels.map((name) => {
                    const loaded = loadedModels.find((m) => m.name === name);
                    const isActive = name === primaryModel && !primaryIsAPI;
                    return (
                      <div
                        key={name}
                        className={cn(
                          'flex items-center justify-between gap-2 rounded-xl border px-3 py-2',
                          isActive
                            ? 'border-[rgba(120,170,220,0.75)] bg-[rgba(120,170,220,0.12)]'
                            : 'border-[rgba(120,170,220,0.18)] bg-muted/50',
                        )}
                      >
                        <div className="min-w-0 flex-1">
                          <div className="text-sm text-foreground truncate">
                            {name.split('/').pop().replace(/\.(bin|gguf)$/i, '')}
                          </div>
                          <div className="text-[10px] text-[rgba(148,163,184,0.75)]">
                            {loaded ? `Loaded on GPU ${loaded.gpu_id}` : 'Not loaded'}
                          </div>
                        </div>
                        {isActive ? (
                          <span className="text-[10px] text-[#78aadc] font-medium whitespace-nowrap">Active</span>
                        ) : loaded ? (
                          <div className="flex gap-1">
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-7 text-[11px]"
                              onClick={() => { setPrimaryIsAPI(false); setPrimaryModel(name); setActiveModel(name); setOpen(false); }}
                            >
                              Select
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-7 text-[11px] text-red-400 border-red-400/30 hover:bg-red-950/30"
                              onClick={() => unloadModel(name)}
                            >
                              Unload
                            </Button>
                          </div>
                        ) : (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="h-7 text-[11px]"
                            disabled={isModelLoading}
                            onClick={async () => { setPrimaryIsAPI(false); await loadModel(name, 0, localCtx); setOpen(false); }}
                          >
                            {isModelLoading ? 'Loading…' : 'Load'}
                          </Button>
                        )}
                      </div>
                    );
                  })
                )}
                <div className="border-t border-[rgba(120,170,220,0.18)] pt-3 mt-3 px-2">
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-[11px] text-muted-foreground">Context Length</label>
                    <Input
                      type="number"
                      min={1}
                      value={localCtx}
                      onChange={(e) => {
                        const parsed = parseInt(e.target.value, 10);
                        if (!isNaN(parsed) && parsed > 0) {
                          setLocalCtx(parsed);
                          saveContextLength(parsed);
                        }
                      }}
                      className="w-28 h-7 text-xs text-right"
                    />
                  </div>
                </div>
              </div>
            ) : tab === 'endpoints' ? (
              endpointGroups.length === 0 ? (
                <div className="p-4 text-sm text-muted-foreground">
                  No saved API models yet. Choose one under My Models, or add a custom endpoint in Settings.
                </div>
              ) : (
                endpointGroups.map((group) => (
                  <div key={group.key} className="mb-3">
                    <div className="px-2 py-1 flex items-center gap-2 text-[10px] uppercase tracking-[0.14em] text-[rgba(148,163,184,0.75)]">
                      <span>{group.resolved?.icon}</span>
                      <span className="truncate">{group.resolved?.displayName || group.key}</span>
                    </div>
                    <div className="space-y-1">
                      {group.endpoints.map(({ ep, resolved }, idx) => {
                        const active = ep.id === effectiveId;
                        return (
                          <div
                            key={ep.id}
                            className={cn(
                              'flex items-center gap-2 rounded-xl border px-2 py-2',
                              active
                                ? 'border-[rgba(120,170,220,0.75)] bg-[rgba(120,170,220,0.12)]'
                                : 'border-[rgba(120,170,220,0.18)] bg-muted/50',
                            )}
                          >
                            <button
                              type="button"
                              className="min-w-0 flex-1 text-left"
                              onClick={() => selectEndpoint(ep.id)}
                            >
                              <div className="text-sm text-foreground truncate">
                                {resolved?.endpointName || `Endpoint #${idx + 1}`}
                              </div>
                              <div className="text-[10px] text-[rgba(148,163,184,0.75)]">
                                {resolved?.providerLabel || 'Custom API'}
                                {' · '}
                                #
                                {idx + 1}
                                {' '}
                                ·
                                {' '}
                                {ep.enabled === false ? 'disabled' : 'enabled'}
                              </div>
                            </button>
                            <label className="flex items-center gap-1 text-[10px] text-muted-foreground" title="Enabled">
                              <input
                                type="checkbox"
                                checked={ep.enabled !== false}
                                onChange={(e) => {
                                  e.stopPropagation();
                                  e.nativeEvent?.stopImmediatePropagation?.();
                                  toggleEndpointEnabled(ep.id, e.target.checked);
                                }}
                              />
                            </label>
                            <label className="flex items-center gap-1 text-[10px] text-muted-foreground" title="Include in rotation">
                              <span className="hidden sm:inline">⟳</span>
                              <input
                                type="checkbox"
                                checked={ep.rotate_enabled !== false}
                                onChange={(e) => {
                                  e.stopPropagation();
                                  e.nativeEvent?.stopImmediatePropagation?.();
                                  toggleRotateEnabled(ep.id, e.target.checked);
                                }}
                              />
                            </label>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))
              )
            ) : connectedCatalogGroups.length === 0 ? (
              <div className="p-4 text-sm text-muted-foreground">
                {query.trim()
                  ? 'No models match your search.'
                  : 'Add a provider key in Model Library to make its models available here.'}
              </div>
            ) : (
              connectedCatalogGroups.map(([providerLabel, list, providerId]) => (
                <div key={providerId} className="mb-2">
                  <div className="flex items-center gap-1.5 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-[rgba(148,163,184,0.75)]">
                    <span>{providerIcon(providerId)}</span>
                    <span>{providerLabel}</span>
                    <span className="normal-case tracking-normal">· available with your key</span>
                  </div>
                  <div className="space-y-1">
                    {list.map((m) => {
                      const favoriteId = `${m.hostProvider}:${m.id}`;
                      const isFav = favorites.has(favoriteId) || favorites.has(m.id);
                      const hasEp = !!findEndpointForModel(m);
                      const pricingLabel = formatOpenRouterPricing(m);
                      return (
                        <div
                          key={favoriteId}
                          className="flex items-center justify-between gap-2 rounded-xl border border-[rgba(120,170,220,0.18)] bg-muted/50 px-3 py-2 hover:border-[rgba(120,170,220,0.45)]"
                        >
                          <button
                            type="button"
                            className="min-w-0 flex-1 text-left"
                            onClick={() => selectCatalogModel(m)}
                          >
                            <div className="flex items-center gap-2">
                              <span>{providerIcon(m.hostProvider)}</span>
                              <span className="truncate text-sm text-foreground">{m.name}</span>
                              <span className="flex-shrink-0 rounded-full border border-[rgba(120,170,220,0.3)] px-1.5 py-0.5 text-[9px] text-muted-foreground">
                                {m.hostProviderLabel}
                              </span>
                              {hasEp && (
                                <span className="text-[9px] uppercase text-[#78aadc]">saved</span>
                              )}
                            </div>
                            <div className="truncate text-[11px] text-[rgba(148,163,184,0.75)] pl-6">
                              {m.id}
                              {m.modelProvider && m.modelProvider !== m.hostProvider ? ` · model by ${m.modelProvider}` : ''}
                              {pricingLabel ? ` · ${pricingLabel}` : ''}
                            </div>
                          </button>
                          <button
                            type="button"
                            className={cn(
                              'rounded-full p-2',
                              isFav ? 'text-yellow-400' : 'text-[rgba(148,163,184,0.75)] hover:text-foreground',
                            )}
                            onMouseDown={(e) => {
                              e.stopPropagation();
                              e.nativeEvent?.stopImmediatePropagation?.();
                            }}
                            onClick={() => {
                              setFavorites((prev) => {
                                const next = new Set(prev);
                                next.delete(m.id);
                                if (next.has(favoriteId)) next.delete(favoriteId);
                                else next.add(favoriteId);
                                localStorage.setItem(FAV_KEY, JSON.stringify(Array.from(next)));
                                return next;
                              });
                            }}
                          >
                            <Star size={16} fill={isFav ? 'currentColor' : 'none'} />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))
            )}
          </div>
    </div>
  ) : null;

  return (
    <div
      ref={anchorRef}
      data-nanogpt-model-popover-root="true"
      className={cn('relative', className)}
    >
      {typeof trigger === 'function'
        ? trigger({ open, setOpen, display: primaryDisplay })
        : (trigger ?? defaultTrigger)}

      {typeof document !== 'undefined' && panel
        ? createPortal(panel, document.body)
        : panel}
    </div>
  );
}
