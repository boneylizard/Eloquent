import React, { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { ArrowLeft, Cloud, ExternalLink, FolderOpen, HardDrive, KeyRound } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { FRONTIER_PROVIDERS } from '../config/frontierProviders';
import { syncHostedProviderEndpointKey, upsertHostedModelEndpoint } from '../utils/hostedModelProviders';
import { readFirstRunIntent } from '../utils/firstRunIntent';
import {
  INTERFACE_ZOOM_DEFAULT,
  INTERFACE_ZOOM_EVENT,
  INTERFACE_ZOOM_MAX,
  INTERFACE_ZOOM_MIN,
  readInterfaceZoom,
  setInterfaceZoom,
} from '../utils/interfaceZoom';
import { Button } from './ui/button';
import { Input } from './ui/input';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';

function formatRuntimeSize(value, fallback) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return fallback;
  return `${(bytes / (1024 ** 3)).toFixed(1)} GB`;
}

const ProviderSetupDialog = () => {
  const {
    settings,
    storageHydrated,
    updateSettings,
    openSettingsTab,
  } = useApp();
  const [step, setStep] = useState('choice');
  const [nanoGptKey, setNanoGptKey] = useState('');
  const [openRouterKey, setOpenRouterKey] = useState('');
  const [frontierKeys, setFrontierKeys] = useState({});
  const [keysInitialised, setKeysInitialised] = useState(false);
  const [runtimeInfo, setRuntimeInfo] = useState(null);
  const [interfaceZoom, setInterfaceZoomValue] = useState(readInterfaceZoom);
  const [firstRunIntent] = useState(readFirstRunIntent);
  const roleplayOnboarding = firstRunIntent?.purpose === 'roleplay';

  useEffect(() => {
    if (!storageHydrated || !firstRunIntent?.purpose || settings.primaryUse === firstRunIntent.purpose) return;
    updateSettings({
      primaryUse: firstRunIntent.purpose,
      roleplayIntroCompleted: firstRunIntent.purpose === 'roleplay' ? false : settings.roleplayIntroCompleted,
    });
  }, [firstRunIntent, settings.primaryUse, settings.roleplayIntroCompleted, storageHydrated, updateSettings]);

  useEffect(() => {
    if (!storageHydrated || keysInitialised) return;
    setNanoGptKey(settings.nanoGptApiKey || '');
    setOpenRouterKey(settings.openRouterApiKey || '');
    setFrontierKeys(Object.fromEntries(
      FRONTIER_PROVIDERS.map((provider) => [provider.keySetting, settings[provider.keySetting] || '']),
    ));
    setKeysInitialised(true);
  }, [keysInitialised, settings, storageHydrated]);

  useEffect(() => {
    const handleZoomChange = (event) => setInterfaceZoomValue(event.detail?.scale || readInterfaceZoom());
    window.addEventListener(INTERFACE_ZOOM_EVENT, handleZoomChange);
    return () => window.removeEventListener(INTERFACE_ZOOM_EVENT, handleZoomChange);
  }, []);

  useEffect(() => {
    if (!window.__TAURI_INTERNALS__) return;
    invoke('get_app_info').then(setRuntimeInfo).catch(() => {});
  }, []);

  const open = storageHydrated && settings.providerSetupCompleted !== true;

  const openModelSetup = useCallback((source = 'huggingface') => {
    updateSettings({
      providerSetupCompleted: true,
      modelSetupRequired: true,
      modelSetupSource: source,
      primaryUse: firstRunIntent?.purpose || settings.primaryUse || 'everything',
      roleplayIntroCompleted: roleplayOnboarding ? false : settings.roleplayIntroCompleted,
    });
    openSettingsTab('models', { forceWindow: false });
  }, [firstRunIntent?.purpose, openSettingsTab, roleplayOnboarding, settings.primaryUse, settings.roleplayIntroCompleted, updateSettings]);

  const connect = () => {
    const trimmedNanoKey = nanoGptKey.trim();
    const trimmedOpenRouterKey = openRouterKey.trim();
    const patch = {
      nanoGptApiKey: trimmedNanoKey,
      openRouterApiKey: trimmedOpenRouterKey,
      providerSetupCompleted: true,
      primaryUse: firstRunIntent?.purpose || settings.primaryUse || 'everything',
      roleplayIntroCompleted: roleplayOnboarding ? false : settings.roleplayIntroCompleted,
    };
    for (const provider of FRONTIER_PROVIDERS) {
      patch[provider.keySetting] = (frontierKeys[provider.keySetting] || '').trim();
    }

    let endpoints = settings.customApiEndpoints || [];
    endpoints = syncHostedProviderEndpointKey(endpoints, 'nanogpt', trimmedNanoKey);
    endpoints = syncHostedProviderEndpointKey(endpoints, 'openrouter', trimmedOpenRouterKey);
    if (trimmedOpenRouterKey) {
      endpoints = upsertHostedModelEndpoint({
        endpoints,
        providerId: 'openrouter',
        apiKey: trimmedOpenRouterKey,
        billingMode: 'free',
        model: {
          id: 'openrouter/free',
          name: 'Free Models Router',
          provider: 'openrouter',
        },
      }).endpoints;
    }
    for (const provider of FRONTIER_PROVIDERS) {
      endpoints = syncHostedProviderEndpointKey(endpoints, provider.id, patch[provider.keySetting]);
    }

    const source = trimmedNanoKey
      ? 'nanogpt'
      : trimmedOpenRouterKey
        ? 'openrouter'
        : 'frontier';
    updateSettings({
      ...patch,
      customApiEndpoints: endpoints,
      modelSetupRequired: true,
      modelSetupSource: source,
    });
    openSettingsTab('models', { forceWindow: false });
  };

  const hasRemoteKey = nanoGptKey.trim()
    || openRouterKey.trim()
    || FRONTIER_PROVIDERS.some((provider) => frontierKeys[provider.keySetting]?.trim());

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => { if (!nextOpen) openModelSetup('huggingface'); }}>
      <DialogContent className="max-h-[90vh] max-w-xl overflow-y-auto">
        {step === 'choice' ? (
          <>
            <DialogHeader>
              <DialogTitle>{roleplayOnboarding ? 'Choose the mind behind your characters' : 'You need a model to start'}</DialogTitle>
              <DialogDescription className="leading-relaxed">
                {roleplayOnboarding
                  ? 'A character supplies the identity; a model supplies the voice. Run one on this computer, or connect a provider online.'
                  : 'Mirid is a Windows app for AI chat and roleplay. Run a model on this computer, or connect a model provider online.'}
              </DialogDescription>
            </DialogHeader>

            {roleplayOnboarding && (
              <div className="rounded-xl border border-primary/35 bg-primary/5 p-3 text-xs leading-relaxed text-muted-foreground">
                <strong className="text-foreground">Your character room is waiting.</strong> Once a model is selected, Mirid will introduce the library, character cards and first conversation in the Character Room theme.
              </div>
            )}

            <div className="grid gap-3 py-2 sm:grid-cols-2">
              <button
                type="button"
                className="rounded-xl border border-border bg-card p-4 text-left transition-colors hover:border-primary/60 hover:bg-accent"
                onClick={() => openModelSetup('huggingface')}
              >
                <HardDrive className="h-5 w-5 text-primary" />
                <p className="mt-3 font-semibold">Download a local model</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">Free to use. The model file is a separate download and runs on this computer.</p>
              </button>
              <button
                type="button"
                className="rounded-xl border border-border bg-card p-4 text-left transition-colors hover:border-primary/60 hover:bg-accent"
                onClick={() => setStep('remote')}
              >
                <Cloud className="h-5 w-5 text-primary" />
                <p className="mt-3 font-semibold">Connect a remote model</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">NanoGPT, OpenRouter, OpenAI, Anthropic, Google Gemini, Mistral or xAI.</p>
              </button>
            </div>

            <div className="rounded-lg border border-border/70 bg-muted/30 p-3 text-xs leading-relaxed text-muted-foreground">
              <p><strong className="text-foreground">What Mirid already downloaded:</strong> the local engine and optional voice and image components ({formatRuntimeSize(runtimeInfo?.runtime_download_size, 'about 3.3 GB')} downloaded; {formatRuntimeSize(runtimeInfo?.runtime_installed_size, 'about 9 GB')} installed). It did not include a chat model.</p>
              <p className="mt-2">Those files live in Windows app data, not beside the app.</p>
              <div className="mt-2 flex flex-wrap gap-3">
                {window.__TAURI_INTERNALS__ && (
                  <button type="button" className="inline-flex items-center gap-1 text-primary hover:underline" onClick={() => invoke('open_runtime_folder')}>
                    Open installed files <FolderOpen className="h-3 w-3" />
                  </button>
                )}
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://github.com/boneylizard/Eloquent" target="_blank" rel="noreferrer">
                  View source <ExternalLink className="h-3 w-3" />
                </a>
              </div>
              {runtimeInfo?.runtime_dir && <p className="mt-2 break-all font-mono text-[10px]">{runtimeInfo.runtime_dir}</p>}
            </div>

            <DialogFooter className="items-center justify-between gap-3 sm:justify-between">
              <div className="flex items-center gap-1" aria-label="Interface size">
                <Button type="button" variant="ghost" size="sm" aria-label="Make interface smaller" onClick={() => void setInterfaceZoom(interfaceZoom - 0.1)} disabled={interfaceZoom <= INTERFACE_ZOOM_MIN}>A−</Button>
                <Button type="button" variant="ghost" size="sm" onClick={() => void setInterfaceZoom(INTERFACE_ZOOM_DEFAULT)}>{Math.round(interfaceZoom * 100)}%</Button>
                <Button type="button" variant="ghost" size="sm" aria-label="Make interface larger" onClick={() => void setInterfaceZoom(interfaceZoom + 0.1)} disabled={interfaceZoom >= INTERFACE_ZOOM_MAX}>A+</Button>
              </div>
              <span className="text-[11px] text-muted-foreground">Ctrl + and Ctrl - also work.</span>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <button type="button" className="mb-2 inline-flex w-fit items-center gap-1 text-xs text-muted-foreground hover:text-foreground" onClick={() => setStep('choice')}>
                <ArrowLeft className="h-3.5 w-3.5" />Back
              </button>
              <DialogTitle>Connect a model provider</DialogTitle>
              <DialogDescription>Choose a provider, paste its API key, then select a model.</DialogDescription>
            </DialogHeader>

            <div className="space-y-3 py-2">
              <div className="rounded-xl border border-primary/40 bg-primary/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <label htmlFor="first-run-nanogpt-key" className="flex items-center gap-2 text-sm font-semibold"><KeyRound className="h-4 w-4" />NanoGPT</label>
                  <div className="flex gap-3 text-xs"><a className="text-primary hover:underline" href="https://nano-gpt.com/subscription" target="_blank" rel="noreferrer">Subscribe</a><a className="text-primary hover:underline" href="https://nano-gpt.com/api" target="_blank" rel="noreferrer">Get key</a></div>
                </div>
                <Input id="first-run-nanogpt-key" className="mt-3 font-mono" type="password" autoComplete="off" value={nanoGptKey} onChange={(event) => setNanoGptKey(event.target.value)} placeholder="Paste NanoGPT API key" />
              </div>

              <div className="rounded-xl border border-border/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <label htmlFor="first-run-openrouter-key" className="flex items-center gap-2 text-sm font-semibold"><KeyRound className="h-4 w-4" />OpenRouter</label>
                  <a className="text-xs text-primary hover:underline" href="https://openrouter.ai/settings/keys" target="_blank" rel="noreferrer">Get key</a>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">The Free Models Router appears first. Mirid does not select a paid model automatically.</p>
                <Input id="first-run-openrouter-key" className="mt-3 font-mono" type="password" autoComplete="off" value={openRouterKey} onChange={(event) => setOpenRouterKey(event.target.value)} placeholder="sk-or-v1-…" />
              </div>

              <details className="group rounded-xl border border-border/70">
                <summary className="cursor-pointer list-none p-4 text-sm font-semibold">Other providers</summary>
                <div className="grid gap-3 border-t p-4 sm:grid-cols-2">
                  {FRONTIER_PROVIDERS.map((provider) => (
                    <div key={provider.id} className="space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <label htmlFor={`first-run-${provider.id}-key`} className="text-xs font-medium">{provider.label}</label>
                        <a className="text-[10px] text-primary hover:underline" href={provider.keyUrl} target="_blank" rel="noreferrer">Get key</a>
                      </div>
                      <Input id={`first-run-${provider.id}-key`} className="h-9 font-mono text-xs" type="password" autoComplete="off" value={frontierKeys[provider.keySetting] || ''} onChange={(event) => setFrontierKeys((current) => ({ ...current, [provider.keySetting]: event.target.value }))} placeholder={provider.placeholder} />
                    </div>
                  ))}
                </div>
              </details>
            </div>

            <DialogFooter className="gap-2 sm:space-x-0">
              <Button variant="ghost" onClick={() => openModelSetup('huggingface')}>Download locally instead</Button>
              <Button onClick={connect} disabled={!hasRemoteKey}>Connect</Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default ProviderSetupDialog;
