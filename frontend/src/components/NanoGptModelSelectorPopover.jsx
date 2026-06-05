import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Cpu, RefreshCw, Search, Star } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/utils';
import { useApp } from '../contexts/AppContext';
import {
  NANO_GPT_MODELS_CACHE_TTL_MS,
  refreshNanoGptModelsCache,
  readNanoGptModelsCache,
  subscribeNanoGptModelsCache,
} from '../utils/nanoGptModelsCache';
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

function ModelDisplayPill({ display, className, title }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-[rgba(120,170,220,0.4)] bg-[#181a1f] px-2.5 py-1 text-xs text-slate-100 max-w-[240px]',
        className,
      )}
      title={title}
    >
      <span className="flex-shrink-0 text-sm leading-none" aria-hidden>
        {display?.icon || '⬜'}
      </span>
      <span className="truncate">{display?.shortLabel || display?.label || 'Select model'}</span>
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
  } = useApp();

  const [internalOpen, setInternalOpen] = useState(false);
  const open = controlledOpen ?? internalOpen;
  const setOpen = onOpenChange ?? setInternalOpen;

  const [tab, setTab] = useState('endpoints');
  const [query, setQuery] = useState('');
  const [catalog, setCatalog] = useState(() => readNanoGptModelsCache().models);
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
    const result = await refreshNanoGptModelsCache({ forceRefresh: force });
    setCatalog(result.models);
    setCatalogError(result.error ? (result.error?.message || String(result.error)) : null);
    setStatus(result.status === 'ok' ? 'ok' : result.status === 'fallback' ? 'fallback' : result.status);
  }, []);

  useEffect(() => {
    const unsub = subscribeNanoGptModelsCache(({ models }) => setCatalog(models));
    return unsub;
  }, []);

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

  const filteredCatalog = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (catalog || [])
      .filter((m) => m.visible !== false && String(m.api || '').toLowerCase() === 'chat')
      .filter(
        (m) => !q
          || m.name.toLowerCase().includes(q)
          || m.id.toLowerCase().includes(q)
          || m.category.toLowerCase().includes(q)
          || m.provider.toLowerCase().includes(q),
      );
  }, [catalog, query]);

  const groupedCatalog = useMemo(() => {
    const byCat = new Map();
    for (const m of filteredCatalog) {
      const cat = m.category || 'Models';
      if (!byCat.has(cat)) byCat.set(cat, []);
      byCat.get(cat).push(m);
    }
    const favs = filteredCatalog.filter((m) => favorites.has(m.id));
    const entries = [];
    if (favs.length) entries.push(['Favorites', favs]);
    for (const [cat, list] of byCat.entries()) {
      if (cat === 'Favorites') continue;
      entries.push([cat, list]);
    }
    return entries;
  }, [filteredCatalog, favorites]);

  const currentNano = useMemo(() => {
    const ep = (settings?.customApiEndpoints || []).find((e) => e.id === effectiveId);
    if (ep?.model) {
      return catalog.find((m) => m.id === ep.model) || null;
    }
    return catalog.find((m) => m.id === effectiveId) || null;
  }, [catalog, effectiveId, settings?.customApiEndpoints]);

  useEffect(() => {
    const caps = currentNano?.capabilities
      || resolveEndpointDisplay(effectiveId, settings, catalog)?.capabilities;
    if (caps && typeof onCapabilities === 'function') onCapabilities(caps);
  }, [currentNano, effectiveId, settings, catalog, onCapabilities]);

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
    (modelId) => (settings.customApiEndpoints || []).find(
      (ep) => ep.model === modelId || ep.model === modelId?.replace(/^models\//, ''),
    ),
    [settings.customApiEndpoints],
  );

  const createEndpointForModel = useCallback(
    (modelMeta) => {
      const modelId = modelMeta.id;
      const newEp = {
        id: `endpoint-${Date.now()}`,
        name: modelMeta.name || modelId.split('/').pop(),
        url: 'https://nano-gpt.com/api/v1',
        apiKey: '',
        model: modelId,
        enabled: true,
        rotate_enabled: true,
        context_window: null,
        supports_native_search: null,
      };
      const endpoints = [...(settings.customApiEndpoints || []), newEp];
      updateSettings({ customApiEndpoints: endpoints });
      return newEp.id;
    },
    [settings.customApiEndpoints, updateSettings],
  );

  const selectCatalogModel = useCallback(
    (modelMeta) => {
      const existing = findEndpointForModel(modelMeta.id);
      if (existing) {
        selectEndpoint(existing.id);
        return;
      }
      const ok = window.confirm(
        `No endpoint configured for "${modelMeta.name}". Create a new NanoGPT API endpoint?`,
      );
      if (!ok) return;
      const newId = createEndpointForModel(modelMeta);
      selectEndpoint(newId);
    },
    [createEndpointForModel, findEndpointForModel, selectEndpoint],
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
      className="fixed z-[70] w-[440px] max-w-[92vw] rounded-2xl border border-[rgba(120,170,220,0.35)] bg-[#101113] shadow-[0_22px_60px_rgba(3,4,10,0.95)]"
      style={{ top: panelStyle.top, left: panelStyle.left }}
    >
          <div className="p-3 border-b border-[rgba(120,170,220,0.18)] space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="inline-flex rounded-full bg-[#181a1f] p-0.5 border border-[rgba(120,170,220,0.28)]">
                <button
                  type="button"
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    e.nativeEvent?.stopImmediatePropagation?.();
                  }}
                  className={cn(
                    'px-3 py-1 text-[11px] rounded-full',
                    tab === 'endpoints' ? 'bg-[#78aadc] text-[#050608]' : 'text-slate-300',
                  )}
                  onClick={() => setTab('endpoints')}
                >
                  My Endpoints
                </button>
                <button
                  type="button"
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    e.nativeEvent?.stopImmediatePropagation?.();
                  }}
                  className={cn(
                    'px-3 py-1 text-[11px] rounded-full',
                    tab === 'all' ? 'bg-[#78aadc] text-[#050608]' : 'text-slate-300',
                  )}
                  onClick={() => setTab('all')}
                >
                  All Models
                </button>
              </div>
              <button
                type="button"
                className="p-1.5 rounded-full text-slate-400 hover:text-slate-100"
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
              <div className="flex items-center justify-between gap-2 rounded-xl border border-[rgba(120,170,220,0.22)] bg-[#0b0c10] px-3 py-2">
                <div className="min-w-0">
                  <div className="text-xs font-medium text-slate-100">⟳ Auto-routing</div>
                  <div className="text-[10px] text-[rgba(148,163,184,0.8)] truncate">
                    {autoOn && rotationPool.length >= 2
                      ? `Pool: ${rotationPool.length} endpoints`
                      : 'Rotate enabled endpoints each prompt'}
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

            {autoOn && rotationPool.length >= 2 && (
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
              <div className="flex items-center gap-2 rounded-full border border-[rgba(120,170,220,0.28)] bg-[#0b0c10] px-3 py-2">
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
                  className="w-full bg-transparent text-sm text-slate-100 outline-none placeholder:text-[rgba(148,163,184,0.6)]"
                  autoFocus
                />
              </div>
            )}

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

            {catalog.length === 0 && status !== 'loading' && (
              <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 px-3 py-2 text-xs text-amber-100 flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <span>
                    Couldn&apos;t load the NanoGPT model catalog from the server. You can still use My Endpoints, or retry loading the catalog below.
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
            {tab === 'endpoints' ? (
              endpointGroups.length === 0 ? (
                <div className="p-4 text-sm text-muted-foreground">
                  No custom API endpoints. Add one in Settings → LLM → API Endpoints, or pick a model under All Models.
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
                                : 'border-[rgba(120,170,220,0.18)] bg-[#0b0c10]',
                            )}
                          >
                            <button
                              type="button"
                              className="min-w-0 flex-1 text-left"
                              onClick={() => selectEndpoint(ep.id)}
                            >
                              <div className="text-sm text-slate-100 truncate">
                                {resolved?.endpointName || `Endpoint #${idx + 1}`}
                              </div>
                              <div className="text-[10px] text-[rgba(148,163,184,0.75)]">
                                #
                                {idx + 1}
                                {' '}
                                ·
                                {' '}
                                {ep.enabled === false ? 'disabled' : 'enabled'}
                              </div>
                            </button>
                            <label className="flex items-center gap-1 text-[10px] text-slate-400" title="Enabled">
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
                            <label className="flex items-center gap-1 text-[10px] text-slate-400" title="Include in rotation">
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
            ) : groupedCatalog.length === 0 ? (
              <div className="p-4 text-sm text-muted-foreground">
                {query.trim()
                  ? 'No models match your search.'
                  : 'Optional: refresh NanoGPT catalog. My Endpoints remains available without catalog data.'}
              </div>
            ) : (
              groupedCatalog.map(([cat, list]) => (
                <div key={cat} className="mb-2">
                  <div className="px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-[rgba(148,163,184,0.75)]">
                    {cat}
                  </div>
                  <div className="space-y-1">
                    {list.map((m) => {
                      const isFav = favorites.has(m.id);
                      const hasEp = !!findEndpointForModel(m.id);
                      return (
                        <div
                          key={m.id}
                          className="flex items-center justify-between gap-2 rounded-xl border border-[rgba(120,170,220,0.18)] bg-[#0b0c10] px-3 py-2 hover:border-[rgba(120,170,220,0.45)]"
                        >
                          <button
                            type="button"
                            className="min-w-0 flex-1 text-left"
                            onClick={() => selectCatalogModel(m)}
                          >
                            <div className="flex items-center gap-2">
                              <span>{providerIcon(m.provider)}</span>
                              <span className="truncate text-sm text-slate-100">{m.name}</span>
                              {hasEp && (
                                <span className="text-[9px] uppercase text-[#78aadc]">linked</span>
                              )}
                            </div>
                            <div className="truncate text-[11px] text-[rgba(148,163,184,0.75)] pl-6">
                              {m.id}
                            </div>
                          </button>
                          <button
                            type="button"
                            className={cn(
                              'rounded-full p-2',
                              isFav ? 'text-yellow-400' : 'text-[rgba(148,163,184,0.75)] hover:text-slate-100',
                            )}
                            onMouseDown={(e) => {
                              e.stopPropagation();
                              e.nativeEvent?.stopImmediatePropagation?.();
                            }}
                            onClick={() => {
                              setFavorites((prev) => {
                                const next = new Set(prev);
                                if (next.has(m.id)) next.delete(m.id);
                                else next.add(m.id);
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
