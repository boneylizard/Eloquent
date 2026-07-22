import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ExternalLink, Loader2, RefreshCw, Sparkles } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { FRONTIER_PROVIDERS, getFrontierProvider } from '../config/frontierProviders';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import {
  syncHostedProviderEndpointKey,
  upsertHostedModelEndpoint,
} from '../utils/hostedModelProviders';
import { refreshFrontierProviderCatalog } from '../utils/frontierProviderCatalogCache';

const FrontierModelLibrary = ({ onSettingChange }) => {
  const {
    PRIMARY_API_URL,
    settings,
    updateSettings,
    setPrimaryIsAPI,
    setPrimaryModel,
    setActiveTab,
  } = useApp();
  const [providerId, setProviderId] = useState(
    () => FRONTIER_PROVIDERS.find((item) => settings[item.keySetting])?.id || 'openai',
  );
  const [catalogues, setCatalogues] = useState({});
  const [loadingProvider, setLoadingProvider] = useState('');
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const provider = getFrontierProvider(providerId);
  const apiKey = settings[provider.keySetting] || '';

  const loadCatalogue = useCallback(async (targetProvider = providerId, forceRefresh = false) => {
    const target = getFrontierProvider(targetProvider);
    const targetKey = (settings[target.keySetting] || '').trim();
    if (!targetKey) return;
    setLoadingProvider(targetProvider);
    setError('');
    try {
      const data = await refreshFrontierProviderCatalog({
        providerId: targetProvider,
        apiKey: targetKey,
        primaryApiUrl: PRIMARY_API_URL,
        forceRefresh,
      });
      setCatalogues((current) => ({
        ...current,
        [targetProvider]: { ...data, base_url: data.baseUrl },
      }));
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoadingProvider('');
    }
  }, [PRIMARY_API_URL, providerId, settings]);

  useEffect(() => {
    if (apiKey && !catalogues[providerId] && loadingProvider !== providerId) {
      loadCatalogue(providerId);
    }
  }, [apiKey, catalogues, loadCatalogue, loadingProvider, providerId]);

  const visibleModels = useMemo(() => {
    const term = query.trim().toLowerCase();
    return (catalogues[providerId]?.models || []).filter((model) => (
      !term || `${model.name} ${model.id} ${model.description}`.toLowerCase().includes(term)
    )).slice(0, 160);
  }, [catalogues, providerId, query]);

  const saveKey = (value) => {
    onSettingChange(provider.keySetting, value);
    updateSettings({
      [provider.keySetting]: value,
      customApiEndpoints: syncHostedProviderEndpointKey(
        settings.customApiEndpoints || [],
        provider.id,
        value,
      ),
    });
    setCatalogues((current) => {
      const next = { ...current };
      delete next[providerId];
      return next;
    });
  };

  const useModel = (model) => {
    const baseUrl = catalogues[providerId]?.base_url;
    const result = upsertHostedModelEndpoint({
      endpoints: settings.customApiEndpoints || [],
      model,
      providerId,
      apiKey,
      baseUrl,
    });
    updateSettings({ customApiEndpoints: result.endpoints, modelSetupRequired: false });
    setPrimaryIsAPI(true);
    setPrimaryModel(result.endpointId);
    setActiveTab('chat');
  };

  return (
    <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h4 className="font-semibold">Frontier provider catalogues</h4>
          <p className="mt-1 text-xs text-muted-foreground">Mirid reads the models available to your own API key. Nothing is pinned to an ageing list.</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => loadCatalogue(providerId, true)} disabled={!apiKey || loadingProvider === providerId}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loadingProvider === providerId ? 'animate-spin' : ''}`} />Refresh models
        </Button>
      </div>

      <Alert>
        <Sparkles className="h-4 w-4" />
        <AlertTitle>API access is separate from chat subscriptions</AlertTitle>
        <AlertDescription>A ChatGPT, Claude, Gemini, Grok or Meta AI subscription does not automatically fund developer API use. The provider may charge your API account when you select and use a model.</AlertDescription>
      </Alert>

      <div className="grid gap-3 md:grid-cols-[240px,1fr]">
        <Select value={providerId} onValueChange={(value) => { setProviderId(value); setQuery(''); setError(''); }}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            {FRONTIER_PROVIDERS.map((item) => (
              <SelectItem key={item.id} value={item.id}>
                {item.label}{settings[item.keySetting] ? ' · connected' : ''}{item.preview ? ' · preview' : ''}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={`Search ${provider.label} models`} />
      </div>

      <div className="space-y-2 rounded-xl border border-border/60 bg-background/40 p-4">
        <div className="flex items-center justify-between gap-3">
          <label className="text-sm font-medium">{provider.label} API key</label>
          <div className="flex flex-wrap justify-end gap-3">
            <a className="inline-flex items-center gap-1 text-xs text-primary hover:underline" href={provider.keyUrl} target="_blank" rel="noreferrer">
              Get a key <ExternalLink className="h-3 w-3" />
            </a>
            <a className="inline-flex items-center gap-1 text-xs text-primary hover:underline" href={provider.billingUrl} target="_blank" rel="noreferrer">
              Billing and credit <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">{provider.guidance}</p>
        <Input type="password" className="font-mono" autoComplete="off" value={apiKey} onChange={(event) => saveKey(event.target.value)} placeholder={provider.placeholder} />
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>{provider.label} could not be connected</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {!apiKey ? (
        <p className="py-8 text-center text-sm text-muted-foreground">Add a key to reveal the models this provider makes available to your account.</p>
      ) : loadingProvider === providerId && !catalogues[providerId] ? (
        <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" />Reading {provider.label}…</div>
      ) : (
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {visibleModels.map((model) => (
            <div key={model.id} className="flex flex-col rounded-lg border border-border/60 bg-background/40 p-3">
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm font-medium">{model.name}</p>
                <Badge variant="outline">{provider.label}</Badge>
              </div>
              <p className="mt-0.5 break-all font-mono text-[10px] text-muted-foreground">{model.id}</p>
              {model.description && <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-muted-foreground">{model.description}</p>}
              <div className="mt-3 flex flex-wrap gap-1">
                {model.capabilities?.reasoning && <Badge variant="secondary">Reasoning</Badge>}
                {model.capabilities?.vision && <Badge variant="secondary">Vision</Badge>}
                {model.capabilities?.tools && <Badge variant="secondary">Tools</Badge>}
                {model.context_length && <Badge variant="secondary">{Math.round(model.context_length / 1024)}K context</Badge>}
              </div>
              <Button className="mt-4" size="sm" variant="outline" onClick={() => useModel(model)}>Use in Mirid</Button>
            </div>
          ))}
          {catalogues[providerId] && visibleModels.length === 0 && (
            <p className="col-span-full py-8 text-center text-sm text-muted-foreground">No matching chat models were returned for this key.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default FrontierModelLibrary;
