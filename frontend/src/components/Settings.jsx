// Settings.jsx
// Full Settings UI: General, Generation, SD, RAG, Characters, Audio, Memory Intent, Memory Browser, Lore, About

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getBackendUrl } from '../config/api';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { Slider } from './ui/slider';
import { Save, Sun, Moon, DownloadCloud, Trash2, ExternalLink, Loader2, RefreshCw, X, Power, RotateCw, FolderOpen } from 'lucide-react';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import CharacterEditor from './CharacterEditor';
import LoreDebugger from '../components/LoreDebugger';
import MemoryIntentDetector from './MemoryIntentDetector';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import RAGSettings from './RAGSettings';
import ProfileSelector from './ProfileSelector';
import SimpleUserProfileEditor from './SimpleUserProfileEditor';

const SettingsSection = ({ title, description, children, actions }) => (
  <div className="rounded-2xl border border-border/70 bg-card/60 shadow-sm">
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2 border-b border-border/60 px-5 py-4">
      <div>
        <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">{title}</p>
        {description ? (
          <p className="text-sm text-foreground/80 mt-1">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
    <div className="p-5 space-y-3">{children}</div>
  </div>
);

const SettingRow = ({ label, description, htmlFor, layout = 'row', children }) => {
  const isStack = layout === 'stack';
  const labelNode = typeof label === 'string'
    ? <Label htmlFor={htmlFor} className="text-sm font-semibold text-foreground">{label}</Label>
    : label;

  return (
    <div
      className={[
        'rounded-lg border border-border/60 bg-background/40 px-4 py-3',
        isStack ? 'space-y-2' : 'grid md:grid-cols-[minmax(200px,1fr),minmax(220px,360px)] items-center gap-4'
      ].join(' ')}
    >
      <div className="space-y-1">
        {labelNode}
        {description ? <p className="text-xs text-muted-foreground">{description}</p> : null}
      </div>
      <div className={isStack ? '' : 'w-full md:max-w-sm'}>
        {children}
      </div>
    </div>
  );
};


const Settings = ({ darkMode, toggleDarkMode, initialTab = 'general' }) => {
  const {
    settings: contextSettings,
    updateSettings,
    userAvatarSize,
    setUserAvatarSize,
    characterAvatarSize,
    setCharacterAvatarSize,
    sttEnabled,
    setSttEnabled,
    ttsEnabled,
    checkSdStatus,
    sdStatus,
    PRIMARY_API_URL,
    SECONDARY_API_URL,
    TTS_API_URL,
    apiError,
    clearError,
    fetchAvailableSTTEngines,
    sttEnginesAvailable,
    playTTS,
    stopTTS,
    isPlayingAudio,
  } = useApp();

  // Local editable copy of context settings
  const [localSettings, setLocalSettings] = useState({
    ...contextSettings,
    directProfileInjection: contextSettings.directProfileInjection ?? false,
    temperature: contextSettings.temperature ?? 0.7,
    max_tokens: contextSettings.max_tokens ?? -1,
    top_p: contextSettings.top_p ?? 0.9,
    top_k: contextSettings.top_k ?? 40,
    repetition_penalty: contextSettings.repetition_penalty ?? 1.1,
    frequencyPenalty: contextSettings.frequencyPenalty ?? 0.0,
    presencePenalty: contextSettings.presencePenalty ?? 0.0,
    antiRepetitionMode: contextSettings.antiRepetitionMode ?? false,
    detectRepeatedPhrases: contextSettings.detectRepeatedPhrases ?? false,
    streamResponses: contextSettings.streamResponses ?? true,
    ttsSpeed: contextSettings.ttsSpeed ?? 1.0,
    ttsPitch: contextSettings.ttsPitch ?? 0,
    ttsAutoPlay: contextSettings.ttsAutoPlay ?? false,
    ttsEngine: contextSettings.ttsEngine ?? 'kokoro',
    ttsVoice: contextSettings.ttsVoice ?? 'af_heart',
    ttsExaggeration: contextSettings.ttsExaggeration ?? 0.5,
    ttsCfg: contextSettings.ttsCfg ?? 0.5,
    ttsSpeedMode: contextSettings.ttsSpeedMode ?? 'standard',
    sdModelDirectory: contextSettings.sdModelDirectory ?? '',
    upscalerModelDirectory: contextSettings.upscalerModelDirectory ?? '',
    sdSteps: contextSettings.sdSteps ?? 20,
    sdSampler: contextSettings.sdSampler ?? 'Euler a',
    sdCfgScale: contextSettings.sdCfgScale ?? 7.0,
    imageEngine: contextSettings.imageEngine ?? 'auto1111',
    enableSdStatus: contextSettings.enableSdStatus ?? true,
    adetailerModelDirectory: contextSettings.adetailerModelDirectory ?? '',
    useOpenAIAPI: contextSettings.useOpenAIAPI ?? false,
    customApiEndpoints: contextSettings.customApiEndpoints ?? [],
    admin_password: contextSettings.admin_password ?? "",
    main_gpu_id: contextSettings.main_gpu_id ?? 0,
    auto_launch_browser: contextSettings.auto_launch_browser ?? true,
  });
  const [hasChanges, setHasChanges] = useState(false);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);
  const [sdModels, setSdModels] = useState([]);
  const [isInstallingEngine, setIsInstallingEngine] = useState(false);
  const [isUploadingVoice, setIsUploadingVoice] = useState(false);
  const [availableVoices, setAvailableVoices] = useState(null);
  const [isUnloadingChatterbox, setIsUnloadingChatterbox] = useState(false);
  const [isReloadingChatterbox, setIsReloadingChatterbox] = useState(false);
  const [currentTensorSplit, setCurrentTensorSplit] = useState([0.5, 0.5]);
  const [gpuCount, setGpuCount] = useState(2);
  const [isUnloadingForensicModels, setIsUnloadingForensicModels] = useState(false);
  const [isShuttingDownTTS, setIsShuttingDownTTS] = useState(false);
  const [isRestartingTTS, setIsRestartingTTS] = useState(false);
  const [directoryPickerKey, setDirectoryPickerKey] = useState(null);
  const [updateStatus, setUpdateStatus] = useState(null);
  const [updateProgress, setUpdateProgress] = useState(null);
  const [updateError, setUpdateError] = useState(null);
  const [isCheckingUpdate, setIsCheckingUpdate] = useState(false);
  const [isUpdateRunning, setIsUpdateRunning] = useState(false);
  const pendingSettingsRef = useRef({});
  const settingsSaveTimerRef = useRef(null);

  // Memory intent input and detected result
  const [memoryIntentInput, setMemoryIntentInput] = useState('');
  const [memoryIntentDetected, setMemoryIntentDetected] = useState(
    contextSettings.memoryIntentText ?? ''
  );

  const handleMemoryIntent = useCallback(intent => {
    setMemoryIntentDetected(intent.content);
  }, []);

  const fetchAvailableVoices = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/tts/voices`);
      if (response.ok) {
        const data = await response.json();
        setAvailableVoices(data);
      }
    } catch (error) {
      console.error("Error fetching available voices:", error);
    }
  }, [PRIMARY_API_URL]);

  useEffect(() => {
    fetchAvailableVoices();
  }, [fetchAvailableVoices]);

  useEffect(() => {
    const fetchGpuInfo = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/system/gpu_info`);
        if (response.ok) {
          const data = await response.json();
          if (data.gpu_count) {
            setGpuCount(data.gpu_count);
          }
        }
      } catch (error) {
        console.error("Error fetching GPU info:", error);
      }
    };

    const fetchTensorSplit = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/models/get-tensor-split`);
        if (response.ok) {
          const data = await response.json();
          if (data.tensor_split) {
            setCurrentTensorSplit(data.tensor_split);
            const input = document.getElementById('tensor-split-input');
            if (input) {
              input.value = data.tensor_split.join(',');
            }
          }
        }
      } catch (error) {
        console.error("Error fetching tensor split:", error);
      }
    };

    fetchGpuInfo();
    fetchTensorSplit();
  }, [PRIMARY_API_URL]);

  useEffect(() => {
    updateSettings({ memoryIntentText: memoryIntentDetected });
  }, [memoryIntentDetected, updateSettings]);

  const handleChange = useCallback((key, value) => {
    setLocalSettings(prev => {
      const updated = { ...prev, [key]: value };
      setHasChanges(JSON.stringify(updated) !== JSON.stringify(contextSettings));
      return updated;
    });
  }, [contextSettings]);

  const handleSave = useCallback(async () => {
    updateSettings(localSettings);
    try {
      await fetch(`${PRIMARY_API_URL}/models/update-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(localSettings)
      });
    } catch (e) {
      console.error("Failed to save settings to backend:", e);
    }
    setHasChanges(false);
  }, [localSettings, updateSettings, PRIMARY_API_URL]);

  const queueSettingsSave = useCallback((patch) => {
    updateSettings(patch);
    pendingSettingsRef.current = { ...pendingSettingsRef.current, ...patch };
    if (settingsSaveTimerRef.current) {
      clearTimeout(settingsSaveTimerRef.current);
    }
    settingsSaveTimerRef.current = setTimeout(async () => {
      const payload = pendingSettingsRef.current;
      pendingSettingsRef.current = {};
      try {
        await fetch(`${PRIMARY_API_URL}/models/update-settings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      } catch (e) {
        console.error("Failed to auto-save settings to backend:", e);
      }
    }, 300);
  }, [PRIMARY_API_URL, updateSettings]);

  useEffect(() => () => {
    if (settingsSaveTimerRef.current) {
      clearTimeout(settingsSaveTimerRef.current);
    }
  }, []);

  const handleReset = useCallback(() => {
    setLocalSettings({ ...contextSettings });
    setHasChanges(false);
  }, [contextSettings]);

  const handleCheckSdStatus = useCallback(async () => {
    setIsCheckingStatus(true);
    try {
      await checkSdStatus();
      if (sdStatus?.automatic1111 && Array.isArray(sdStatus.models)) {
        setSdModels(sdStatus.models.map(model => ({
          name: model.model_name,
          path: model.filename
        })));
      }
    } catch (error) {
      console.error('Error checking SD status:', error);
    } finally {
      setIsCheckingStatus(false);
    }
  }, [checkSdStatus, sdStatus]);

  const handleExportBackendLogs = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/export-logs`);
      if (!response.ok) {
        throw new Error(`Export failed: ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `backend-logs-${new Date().toISOString().split('T')[0]}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to export backend logs:", error);
      alert("Failed to export backend logs.");
    }
  }, [PRIMARY_API_URL]);
  const handleClearBackendLogs = useCallback(async () => {
    if (!confirm("Delete all backend logs? This cannot be undone.")) {
      return;
    }
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/clear-logs`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(`Clear failed: ${response.status}`);
      }
      const data = await response.json();
      alert(`Deleted ${data.deleted ?? 0} log file(s).`);
    } catch (error) {
      console.error("Failed to clear backend logs:", error);
      alert("Failed to clear backend logs.");
    }
  }, [PRIMARY_API_URL]);

  const handleCheckUpdates = useCallback(async () => {
    setIsCheckingUpdate(true);
    setUpdateError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/update-status?fetch=1`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || `Status ${response.status}`);
      }
      setUpdateStatus(data);
    } catch (error) {
      setUpdateError(error.message || 'Failed to check updates.');
    } finally {
      setIsCheckingUpdate(false);
    }
  }, [PRIMARY_API_URL]);

  const fetchUpdateProgress = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/update-progress`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || `Update failed (${response.status})`);
      }
      setUpdateProgress(data);
      if (data.status !== 'running') {
        setIsUpdateRunning(false);
        if (data.status === 'failed') {
          setUpdateError(data.error || 'Update failed.');
        }
        if (data.status === 'success' && data.restart_recommended) {
          alert('Update complete. Please restart the app to apply changes.');
        }
      }
    } catch (error) {
      setUpdateError(error.message || 'Failed to fetch update progress.');
      setIsUpdateRunning(false);
    }
  }, [PRIMARY_API_URL]);

  const handleRunUpdate = useCallback(async () => {
    if (!confirm("Update to the latest git version? This will discard local changes in the app folder. A restart may be required.")) {
      return;
    }
    setUpdateError(null);
    setUpdateProgress(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok && response.status !== 409) {
        throw new Error(data.detail || data.message || `Update failed (${response.status})`);
      }
      setIsUpdateRunning(true);
      fetchUpdateProgress();
    } catch (error) {
      setUpdateError(error.message || 'Failed to start update.');
      setIsUpdateRunning(false);
    }
  }, [PRIMARY_API_URL, fetchUpdateProgress]);

  useEffect(() => {
    if (!isUpdateRunning) return;
    const timer = setInterval(fetchUpdateProgress, 1000);
    return () => clearInterval(timer);
  }, [isUpdateRunning, fetchUpdateProgress]);

  const handleDirectoryBrowse = useCallback(async (settingKey, title) => {
    const baseUrl = PRIMARY_API_URL || getBackendUrl();
    setDirectoryPickerKey(settingKey);
    try {
      const response = await fetch(`${baseUrl}/system/select-directory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_directory: localSettings[settingKey] || null,
          title
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to open directory picker.');
      }
      if (data.status === 'cancelled' || !data.directory) {
        return;
      }
      handleChange(settingKey, data.directory);
    } catch (error) {
      console.error('Directory picker failed:', error);
      alert(`Directory picker failed: ${error.message}`);
    } finally {
      setDirectoryPickerKey(null);
    }
  }, [PRIMARY_API_URL, handleChange, localSettings]);

  const updateLogLines = updateProgress?.logs || [];
  const updateLogText = updateLogLines
    .map((entry) => `[${entry.ts}] ${String(entry.level).toUpperCase()}: ${entry.message}`)
    .join('\n');

  return (
    <div className="w-full min-h-screen p-2 md:p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <h2 className="text-2xl font-bold mb-4">Settings</h2>
        <Tabs defaultValue={initialTab} className="space-y-6">
          <div className="border rounded-lg bg-card p-1 overflow-x-auto">
            <TabsList className="flex w-full flex-wrap justify-start gap-1 h-auto min-h-[40px]">
            <TabsTrigger value="general" className="flex-shrink-0">General</TabsTrigger>
            <TabsTrigger value="generation" className="flex-shrink-0">LLM Settings</TabsTrigger>
            <TabsTrigger value="image-generation" className="flex-shrink-0">Image Generation</TabsTrigger>
            <TabsTrigger value="rag" className="flex-shrink-0">Document Context</TabsTrigger>
            <TabsTrigger value="characters" className="flex-shrink-0">Characters</TabsTrigger>
            <TabsTrigger value="audio" className="flex-shrink-0">Audio</TabsTrigger>
            <TabsTrigger value="memory-intent" className="flex-shrink-0">Memory Intent</TabsTrigger>
            <TabsTrigger value="memory" className="flex-shrink-0">Memory Browser</TabsTrigger>
            <TabsTrigger value="lore" className="flex-shrink-0">Lore Debugger</TabsTrigger>
            <TabsTrigger value="about" className="flex-shrink-0">About</TabsTrigger>
            <TabsTrigger value="profiles" className="flex-shrink-0">User Profiles</TabsTrigger>
            </TabsList>
          </div>

        {/* General */}
        <TabsContent value="general">
          <div className="space-y-6">
            <SettingsSection
              title="Interface"
              description="Theme and layout controls for the main UI."
            >
              <SettingRow label="Dark Mode" htmlFor="dark-mode" description="Toggle the dark theme.">
                <div className="flex items-center justify-end gap-2">
                  <Sun className="h-4 w-4" />
                  <Switch id="dark-mode" checked={darkMode} onCheckedChange={toggleDarkMode} />
                  <Moon className="h-4 w-4" />
                </div>
              </SettingRow>
              <SettingRow
                label={`User Avatar Size (${userAvatarSize}px)`}
                htmlFor="user-avatar-size"
                layout="stack"
                description="Controls the avatar size for your messages."
              >
                <Slider
                  id="user-avatar-size"
                  min={64}
                  max={512}
                  step={16}
                  value={[userAvatarSize]}
                  onValueChange={([v]) => setUserAvatarSize(v)}
                />
              </SettingRow>
              <SettingRow
                label={`Character Avatar Size (${characterAvatarSize}px)`}
                htmlFor="char-avatar-size"
                layout="stack"
                description="Controls the display size of characters' avatars in chat."
              >
                <Slider
                  id="char-avatar-size"
                  min={64}
                  max={512}
                  step={16}
                  value={[characterAvatarSize]}
                  onValueChange={([v]) => setCharacterAvatarSize(v)}
                />
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Access and Endpoints"
              description="Security and API endpoints for the running services."
            >
              <SettingRow
                label="Remote Access Password"
                htmlFor="admin-password"
                description="Set a password to protect your instance."
              >
                <Input
                  id="admin-password"
                  type="password"
                  value={localSettings.admin_password || ''}
                  className="w-full md:max-w-xs"
                  onChange={(e) => handleChange('admin_password', e.target.value)}
                  placeholder="No password set"
                />
              </SettingRow>
              <SettingRow label="Primary API URL" htmlFor="primary-api-url" description="Main backend address (read-only).">
                <Input id="primary-api-url" value={PRIMARY_API_URL} readOnly className="w-full md:max-w-xs" />
              </SettingRow>
              <SettingRow label="Secondary API URL" htmlFor="secondary-api-url" description="Secondary backend address (read-only).">
                <Input id="secondary-api-url" value={SECONDARY_API_URL} readOnly className="w-full md:max-w-xs" />
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Backup and Logs"
              description="Export, import, and manage local logs."
            >
              <SettingRow label="Backup and Restore" layout="stack">
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={() => {
                      try {
                        const config = {};
                        for (let i = 0; i < localStorage.length; i++) {
                          const key = localStorage.key(i);
                          config[key] = localStorage.getItem(key);
                        }
                        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `eloquent-config-${new Date().toISOString().split('T')[0]}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      } catch (err) { alert('Failed to export'); }
                    }}
                    className="flex-1"
                  >
                    <DownloadCloud className="mr-2 h-4 w-4" />
                    Export Config
                  </Button>

                  <div className="relative flex-1">
                    <input
                      type="file"
                      accept=".json"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          const reader = new FileReader();
                          reader.onload = (ev) => {
                            try {
                              const config = JSON.parse(ev.target.result);
                              if (confirm('Import settings?')) {
                                Object.entries(config).forEach(([k, v]) => localStorage.setItem(k, v));
                                window.location.reload();
                              }
                            } catch (err) { alert('Failed to parse'); }
                          };
                          reader.readAsText(file);
                          e.target.value = '';
                        }
                      }}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    />
                    <Button variant="outline" className="w-full">
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Import Config
                    </Button>
                  </div>
                </div>
              </SettingRow>
              <SettingRow
                label="Backend Logs"
                layout="stack"
                description="Export backend logs for bug reports."
              >
                <div className="flex flex-col md:flex-row gap-2">
                  <Button variant="outline" onClick={handleExportBackendLogs}>
                    Export Backend Logs
                  </Button>
                  <Button variant="outline" onClick={handleClearBackendLogs}>
                    Delete Old Logs
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="App Updates"
              description="Force update to the latest git version. Local changes will be discarded."
            >
              <SettingRow label="Update Controls" layout="stack" description="Live progress and logs are shown while updating.">
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={handleCheckUpdates}
                    disabled={isCheckingUpdate || isUpdateRunning}
                  >
                    {isCheckingUpdate ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="mr-2 h-4 w-4" />
                    )}
                    Check for Updates
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleRunUpdate}
                    disabled={isUpdateRunning || isCheckingUpdate}
                  >
                    {isUpdateRunning ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <RotateCw className="mr-2 h-4 w-4" />
                    )}
                    Update Now
                  </Button>
                </div>

                {updateStatus && (
                  <div className="text-xs text-muted-foreground space-y-1">
                    <div>
                      Branch: {updateStatus.branch || 'unknown'} (
                      {updateStatus.current_commit ? updateStatus.current_commit.slice(0, 7) : 'unknown'})
                    </div>
                    {updateStatus.upstream ? (
                      <div>
                        Tracking: {updateStatus.upstream} - Ahead {updateStatus.ahead ?? 'n/a'} - Behind {updateStatus.behind ?? 'n/a'}
                      </div>
                    ) : (
                      <div>No upstream configured for this branch.</div>
                    )}
                    <div>Working tree: {updateStatus.dirty ? `dirty (${updateStatus.dirty_count})` : 'clean'}</div>
                  </div>
                )}

                {updateProgress && (
                  <div className="rounded-lg border border-border/60 bg-muted/30 p-3 space-y-2">
                    <div className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Update Status</div>
                    <div className="text-sm text-foreground">
                      {updateProgress.status === 'running' ? 'Running' : updateProgress.status}
                      {updateProgress.step ? ` - ${updateProgress.step}` : ''}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {updateProgress.before ? `Before: ${updateProgress.before.slice(0, 7)}` : ''}
                      {updateProgress.after ? ` | After: ${updateProgress.after.slice(0, 7)}` : ''}
                    </div>
                    {updateProgress.error && (
                      <Alert variant="destructive">
                        <AlertTitle>Update failed</AlertTitle>
                        <AlertDescription>{updateProgress.error}</AlertDescription>
                      </Alert>
                    )}
                    {updateLogLines.length > 0 && (
                      <div className="space-y-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Update Log</div>
                        <pre className="max-h-56 overflow-y-auto whitespace-pre-wrap rounded-md bg-background/80 p-3 text-xs text-foreground">
{updateLogText}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {typeof updateStatus?.behind === 'number' && updateStatus.behind > 0 && (
                  <Alert>
                    <AlertTitle>Update available</AlertTitle>
                    <AlertDescription>Behind by {updateStatus.behind} commit(s).</AlertDescription>
                  </Alert>
                )}

                {updateError && (
                  <Alert variant="destructive">
                    <AlertTitle>Update failed</AlertTitle>
                    <AlertDescription>{updateError}</AlertDescription>
                  </Alert>
                )}
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Models and Storage"
              description="Directories for local model files."
            >
              <SettingRow label="Models Directory" htmlFor="model-directory" description="Location for local GGUF models.">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="model-directory"
                    value={localSettings.modelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('modelDirectory', e.target.value)}
                    placeholder="C:\models\gguf"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('modelDirectory', 'Select Models Directory')}
                    disabled={directoryPickerKey === 'modelDirectory'}
                  >
                    {directoryPickerKey === 'modelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (localSettings.modelDirectory) {
                        fetch(`${PRIMARY_API_URL}/models/refresh-directory`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ directory: localSettings.modelDirectory })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated! Restart required.' : d.message));
                      }
                    }}
                  >
                    Save
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Startup and GPU"
              description="Launch behavior and GPU distribution."
            >
              <SettingRow
                label="Auto-Launch Browser"
                htmlFor="auto-launch-browser"
                description="Automatically open the browser window on startup."
              >
                <Switch
                  id="auto-launch-browser"
                  checked={localSettings.auto_launch_browser}
                  onCheckedChange={(value) => handleChange('auto_launch_browser', value)}
                />
              </SettingRow>

              <SettingRow
                label="Single GPU Mode"
                htmlFor="single-gpu-mode"
                description={gpuCount <= 1 ? 'Automatically enabled (single GPU detected).' : 'Enable for single GPU setup.'}
              >
                <Switch
                  id="single-gpu-mode"
                  checked={localSettings.singleGpuMode || gpuCount <= 1}
                  disabled={gpuCount <= 1}
                  onCheckedChange={(value) => handleChange('singleGpuMode', value)}
                />
              </SettingRow>

              {gpuCount > 1 && (
                <SettingRow
                  label="Main Model GPU"
                  htmlFor="main-gpu-id"
                  description="Select which GPU runs the heavy LLM model service."
                >
                  <Select
                    value={localSettings.main_gpu_id?.toString() || "0"}
                    onValueChange={(value) => handleChange('main_gpu_id', parseInt(value, 10))}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select Main GPU" />
                    </SelectTrigger>
                    <SelectContent>
                      {Array.from({ length: gpuCount }, (_, i) => (
                        <SelectItem key={i} value={i.toString()}>
                          GPU {i}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </SettingRow>
              )}

              {!localSettings.singleGpuMode && (
                <>
                  <SettingRow label="GPU Usage Mode (Dual GPU)" htmlFor="gpu-usage-mode">
                    <Select
                      value={localSettings.gpuUsageMode || 'split_services'}
                      onValueChange={(value) => {
                        handleChange('gpuUsageMode', value);
                        fetch(`${PRIMARY_API_URL}/models/update-gpu-mode`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ gpuUsageMode: value })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated! Restart required.' : d.message));
                      }}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select GPU usage mode" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="split_services">Split Services</SelectItem>
                        <SelectItem value="unified_model">Unified Model</SelectItem>
                      </SelectContent>
                    </Select>
                  </SettingRow>

                  {localSettings.gpuUsageMode === 'unified_model' && (
                    <SettingRow label="Tensor Split Ratio" htmlFor="tensor-split-input" layout="stack">
                      <div className="flex flex-col md:flex-row gap-3">
                        <Input
                          id="tensor-split-input"
                          type="text"
                          placeholder="1,1"
                          defaultValue={currentTensorSplit.join(',')}
                          key={currentTensorSplit.join(',')}
                          className="flex-1"
                        />
                        <Button
                          variant="outline"
                          className="w-full md:w-auto"
                          onClick={(e) => {
                            const input = document.getElementById('tensor-split-input');
                            const value = input.value.trim();
                            const parts = value.split(',').map(s => parseFloat(s.trim()));
                            const total = parts.reduce((a, b) => a + b, 0);
                            const normalized = parts.map(v => v / total);

                            fetch(`${PRIMARY_API_URL}/models/update-tensor-split`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ tensor_split: normalized })
                            }).then(r => r.json()).then(data => {
                              if (data.status === 'success') {
                                alert('Updated!');
                                setCurrentTensorSplit(data.tensor_split);
                              } else alert(data.message);
                            });
                          }}
                        >
                          <Save className="mr-2 h-4 w-4" /> Apply
                        </Button>
                      </div>
                    </SettingRow>
                  )}
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Services"
              description="Manage optional background services and VRAM usage."
            >
              <SettingRow label="Forensic Models Management" layout="stack">
                <Button
                  variant="outline"
                  onClick={async () => {
                    setIsUnloadingForensicModels(true);
                    try {
                      await fetch(`${PRIMARY_API_URL}/forensic/unload-models`, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' }
                      });
                      alert('Unloaded!');
                    } finally { setIsUnloadingForensicModels(false); }
                  }}
                  disabled={isUnloadingForensicModels}
                  className="w-full"
                >
                  {isUnloadingForensicModels ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Power className="mr-2 h-4 w-4" />}
                  Unload RoBERTa/STAR Models
                </Button>
              </SettingRow>

              <SettingRow label="TTS Service Management (Port 8002)" layout="stack">
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={async () => {
                      setIsShuttingDownTTS(true);
                      try {
                        await fetch(`${PRIMARY_API_URL}/tts/shutdown`, { method: 'POST' });
                        alert('Shutdown initiated.');
                      } finally { setIsShuttingDownTTS(false); }
                    }}
                    disabled={isShuttingDownTTS}
                    className="flex-1"
                  >
                    <Power className="mr-2 h-4 w-4" /> Shutdown TTS
                  </Button>
                  <Button
                    variant="outline"
                    onClick={async () => {
                      setIsRestartingTTS(true);
                      try {
                        await fetch(`${PRIMARY_API_URL}/tts/restart`, { method: 'POST' });
                        alert('Restarting...');
                      } finally { setIsRestartingTTS(false); }
                    }}
                    disabled={isRestartingTTS}
                    className="flex-1"
                  >
                    <RotateCw className="mr-2 h-4 w-4" /> Restart TTS
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>

            <div className="flex justify-end pt-2">
              <Button onClick={handleSave} disabled={!hasChanges} className="w-full md:w-auto">
                <Save className="mr-2 h-4 w-4" />
                Save Changes
              </Button>
            </div>
          </div>
        </TabsContent>

        {/* User Profiles */}
        <TabsContent value="profiles">
          <div className="space-y-6">
            <SettingsSection
              title="User Profiles"
              description="Manage your user identities and preferences."
            >
              <div className="space-y-6">
                <ProfileSelector />
                <div className="border-t border-border/60" />
                <SimpleUserProfileEditor />
              </div>
            </SettingsSection>
          </div>
        </TabsContent>

        <TabsContent value="rag">
          <RAGSettings />
        </TabsContent>

        {/* Generation - RESTORED FULL CONTENT */}
        <TabsContent value="generation">
          <div className="space-y-6">
            <SettingsSection
              title="Sampling"
              description="Core sampling controls for text generation."
            >
              <SettingRow label={`Temperature (${localSettings.temperature.toFixed(2)})`} layout="stack">
                <Slider
                  value={[localSettings.temperature]}
                  min={0}
                  max={2}
                  step={0.05}
                  onValueChange={([v]) => handleChange('temperature', v)}
                />
              </SettingRow>
              <SettingRow label={`Top-P (${localSettings.top_p.toFixed(2)})`} layout="stack">
                <Slider
                  value={[localSettings.top_p]}
                  min={0}
                  max={1}
                  step={0.05}
                  onValueChange={([v]) => handleChange('top_p', v)}
                />
              </SettingRow>
              <SettingRow label={`Top-K (${localSettings.top_k})`} layout="stack">
                <Slider
                  value={[localSettings.top_k]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={([v]) => handleChange('top_k', v)}
                />
              </SettingRow>
              <SettingRow label={`Repetition Penalty (${localSettings.repetition_penalty.toFixed(2)})`} layout="stack">
                <Slider
                  value={[localSettings.repetition_penalty]}
                  min={1}
                  max={2}
                  step={0.01}
                  onValueChange={([v]) => handleChange('repetition_penalty', v)}
                />
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Anti-Repetition"
              description="Reduce loops and repeated phrases."
            >
              <SettingRow label="Anti-Repetition Mode" htmlFor="anti-repetition" description="Enable extra controls to reduce repetition.">
                <Switch
                  id="anti-repetition"
                  checked={localSettings.antiRepetitionMode}
                  onCheckedChange={(checked) => handleChange('antiRepetitionMode', checked)}
                />
              </SettingRow>

              {localSettings.antiRepetitionMode && (
                <>
                  <SettingRow label={`Frequency Penalty (${localSettings.frequencyPenalty.toFixed(2)})`} layout="stack">
                    <Slider
                      value={[localSettings.frequencyPenalty]}
                      min={0}
                      max={2}
                      step={0.1}
                      onValueChange={([v]) => handleChange('frequencyPenalty', v)}
                    />
                  </SettingRow>
                  <SettingRow label={`Presence Penalty (${localSettings.presencePenalty.toFixed(2)})`} layout="stack">
                    <Slider
                      value={[localSettings.presencePenalty]}
                      min={0}
                      max={2}
                      step={0.1}
                      onValueChange={([v]) => handleChange('presencePenalty', v)}
                    />
                  </SettingRow>
                  <SettingRow label="Detect and Remove Repeated Phrases" htmlFor="detect-phrases">
                    <Switch
                      id="detect-phrases"
                      checked={localSettings.detectRepeatedPhrases}
                      onCheckedChange={(checked) => handleChange('detectRepeatedPhrases', checked)}
                    />
                  </SettingRow>
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Limits and Streaming"
              description="Control output size and streaming behavior."
            >
              <SettingRow label="Max Tokens" htmlFor="max_tokens" description="Use -1 to allow auto selection.">
                <Input
                  id="max_tokens"
                  type="number"
                  min={-1}
                  step={1}
                  value={localSettings.max_tokens}
                  onChange={e => handleChange('max_tokens', parseInt(e.target.value, 10))}
                  className="w-full md:max-w-xs"
                />
              </SettingRow>
              <SettingRow label="Stream Responses" htmlFor="stream-responses">
                <Switch
                  id="stream-responses"
                  checked={localSettings.streamResponses}
                  onCheckedChange={(value) => handleChange('streamResponses', value)}
                />
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Profile and API Behavior"
              description="Advanced behavior and API compatibility."
            >
              <SettingRow label="Direct Profile Injection" htmlFor="direct-profile-injection">
                <Switch
                  id="direct-profile-injection"
                  checked={localSettings.directProfileInjection || false}
                  onCheckedChange={(value) => {
                    handleChange('directProfileInjection', value);
                    updateSettings({ directProfileInjection: value });
                    fetch(`${PRIMARY_API_URL}/models/set-direct-profile-injection`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ directProfileInjection: value })
                    }).catch(err => console.error(err));
                  }}
                />
              </SettingRow>
              <SettingRow label="Use OpenAI API Format" htmlFor="use-openai-api">
                <Switch
                  id="use-openai-api"
                  checked={localSettings.useOpenAIAPI || false}
                  onCheckedChange={(value) => {
                    handleChange('useOpenAIAPI', value);
                    fetch(`${PRIMARY_API_URL}/models/set-openai-api-mode`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ useOpenAIAPI: value })
                    }).catch(err => console.error(err));
                  }}
                />
              </SettingRow>
              {localSettings.useOpenAIAPI && (
                <Alert>
                  <AlertTitle>OpenAI API Mode Active</AlertTitle>
                  <AlertDescription className="text-xs">
                    Requests will use the OpenAI compatible format.
                  </AlertDescription>
                </Alert>
              )}
            </SettingsSection>

            <SettingsSection
              title="Custom API Endpoints"
              description="Add external API targets for model selection."
              actions={(
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    const newEndpoints = [...(localSettings.customApiEndpoints || []), {
                      id: `endpoint-${Date.now()}`,
                      name: 'New Endpoint',
                      url: getBackendUrl(),
                      apiKey: '',
                      enabled: true,
                      context_window: null
                    }];
                    handleChange('customApiEndpoints', newEndpoints);
                  }}
                >
                  Add Endpoint
                </Button>
              )}
            >
              {(localSettings.customApiEndpoints || []).map((endpoint, index) => (
                <div key={endpoint.id} className="rounded-lg border border-border/60 bg-muted/20 p-4 space-y-3">
                  <div className="flex flex-row items-center justify-between gap-2">
                    <div className="flex items-center gap-2 flex-1">
                      <Switch
                        checked={endpoint.enabled}
                        onCheckedChange={(enabled) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, enabled };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                      <Input
                        placeholder="Endpoint Name"
                        value={endpoint.name}
                        onChange={(e) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, name: e.target.value };
                          handleChange('customApiEndpoints', updated);
                        }}
                        className="w-full"
                      />
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        const updated = localSettings.customApiEndpoints.filter((_, i) => i !== index);
                        handleChange('customApiEndpoints', updated);
                      }}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div>
                      <Label className="text-xs">API URL</Label>
                      <Input
                        value={endpoint.url}
                        onChange={(e) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, url: e.target.value };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Model Name</Label>
                      <Input
                        value={endpoint.model || ''}
                        onChange={(e) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, model: e.target.value };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">API Key</Label>
                      <Input
                        type="password"
                        value={endpoint.apiKey}
                        onChange={(e) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, apiKey: e.target.value };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Context Window</Label>
                      <Input
                        type="number"
                        min="1024"
                        step="256"
                        placeholder="8192"
                        value={endpoint.context_window ?? ''}
                        onChange={(e) => {
                          const value = e.target.value;
                          const contextWindow = value === '' ? null : parseInt(value, 10);
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, context_window: contextWindow };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}

              <div className="flex justify-end">
                <Button
                  variant="outline"
                  onClick={() => {
                    fetch(`${PRIMARY_API_URL}/models/save-custom-endpoints`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ customApiEndpoints: localSettings.customApiEndpoints || [] })
                    }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Saved!' : d.message));
                  }}
                  className="w-full md:w-auto"
                >
                  <Save className="mr-1 h-4 w-4" />
                  Save Endpoints
                </Button>
              </div>
            </SettingsSection>

            <div className="flex justify-end pt-2">
              <Button onClick={handleSave} disabled={!hasChanges} className="w-full md:w-auto">
                <Save className="mr-1 h-4 w-4" />Save
              </Button>
            </div>
          </div>
        </TabsContent>

        {/* Image Generation */}
        <TabsContent value="image-generation">
          <div className="space-y-6">
            <SettingsSection
              title="Engine Priority"
              description="Select which image engine is preferred."
            >
              <SettingRow
                label="Image Engine Priority"
                layout="stack"
                description="Local SD uses the built-in stable-diffusion.cpp engine. External engines require their own servers."
              >
                <Select
                  value={localSettings.imageEngine || 'auto1111'}
                  onValueChange={(value) => handleChange('imageEngine', value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select image engine" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="EloDiffusion">Local SD (Built-in)</SelectItem>
                    <SelectItem value="auto1111">AUTOMATIC1111 (External)</SelectItem>
                    <SelectItem value="comfyui">ComfyUI (External)</SelectItem>
                    <SelectItem value="nanogpt">NanoGPT (Cloud)</SelectItem>
                    <SelectItem value="both">Show Both Options</SelectItem>
                  </SelectContent>
                </Select>
              </SettingRow>
            </SettingsSection>

            {localSettings.imageEngine === 'nanogpt' && (
              <SettingsSection
                title="NanoGPT (Cloud)"
                description="Generate images and videos using NanoGPT model APIs."
              >
                <SettingRow label="NanoGPT API Key (Shared)" htmlFor="nanogpt-api-key">
                  <Input
                    id="nanogpt-api-key"
                    type="password"
                    value={localSettings.nanoGptApiKey || ''}
                    onChange={(e) => handleChange('nanoGptApiKey', e.target.value)}
                    placeholder="sk-..."
                  />
                </SettingRow>
                <SettingRow label="Image Model Name" htmlFor="nanogpt-model">
                  <Input
                    id="nanogpt-model"
                    value={localSettings.nanoGptModel || 'dall-e-3'}
                    onChange={(e) => handleChange('nanoGptModel', e.target.value)}
                    placeholder="dall-e-3"
                  />
                </SettingRow>
                <SettingRow
                  label="Video Model Name"
                  htmlFor="nanogpt-video-model"
                  description="Default: svd (stable-video-diffusion)."
                >
                  <Input
                    id="nanogpt-video-model"
                    value={localSettings.nanoGptVideoModel || 'svd'}
                    onChange={(e) => handleChange('nanoGptVideoModel', e.target.value)}
                    placeholder="svd"
                  />
                </SettingRow>
              </SettingsSection>
            )}

            <SettingsSection
              title="Local SD (Built-in)"
              description="Built-in image generation using stable-diffusion.cpp."
            >
              <div className="rounded-md border border-border/60 bg-muted/20 p-3">
                <div className="text-xs text-muted-foreground">
                  <div className="font-medium text-foreground">Supported local models</div>
                  <ul className="mt-2 list-disc pl-5 space-y-1">
                    <li>Files: .safetensors, .ckpt, .gguf</li>
                    <li>Families: SD 1.x, SDXL (filename contains sdxl/xl), FLUX (filename contains flux)</li>
                    <li>FLUX needs extra files in the same folder: clip_l.safetensors, t5xxl_fp16.safetensors, ae.safetensors</li>
                    <li>SDXL is heavier; reduce resolution or steps if you hit VRAM limits</li>
                  </ul>
                </div>
              </div>

              <SettingRow label="Local SD Models Directory" htmlFor="sd-model-directory">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="sd-model-directory"
                    value={localSettings.sdModelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('sdModelDirectory', e.target.value)}
                    placeholder="C:\path\to\sd-models"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('sdModelDirectory', 'Select Local SD Models Directory')}
                    disabled={directoryPickerKey === 'sdModelDirectory'}
                  >
                    {directoryPickerKey === 'sdModelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (localSettings.sdModelDirectory) {
                        fetch(`${PRIMARY_API_URL}/sd-local/refresh-directory`, {
                          method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ directory: localSettings.sdModelDirectory })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated!' : d.message));
                      }
                    }}
                  >
                    Save
                  </Button>
                </div>
              </SettingRow>

              <SettingRow label="ADetailer Models Directory" htmlFor="adetailer-model-directory">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="adetailer-model-directory"
                    value={localSettings.adetailerModelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('adetailerModelDirectory', e.target.value)}
                    placeholder="C:\path\to\adetailer-models"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('adetailerModelDirectory', 'Select ADetailer Models Directory')}
                    disabled={directoryPickerKey === 'adetailerModelDirectory'}
                  >
                    {directoryPickerKey === 'adetailerModelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (localSettings.adetailerModelDirectory) {
                        fetch(`${PRIMARY_API_URL}/sd-local/set-adetailer-directory`, {
                          method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ directory: localSettings.adetailerModelDirectory })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated!' : d.message));
                      }
                    }}
                  >
                    Save
                  </Button>
                </div>
              </SettingRow>

              <SettingRow label="Upscaler Models Directory" htmlFor="upscaler-directory">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="upscaler-directory"
                    value={localSettings.upscalerModelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('upscalerModelDirectory', e.target.value)}
                    placeholder="C:\path\to\upscalers"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('upscalerModelDirectory', 'Select Upscaler Models Directory')}
                    disabled={directoryPickerKey === 'upscalerModelDirectory'}
                  >
                    {directoryPickerKey === 'upscalerModelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      fetch(`${PRIMARY_API_URL}/models/update-upscaler-dir`, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ directory: localSettings.upscalerModelDirectory })
                      }).then(r => r.json()).then(d => alert('Updated!'));
                    }}
                  >
                    <Save className="mr-1 h-4 w-4" />Save
                  </Button>
                </div>
              </SettingRow>

              <SettingRow label={`Default Steps (${localSettings.sdSteps || 20})`} layout="stack">
                <Slider
                  min={10}
                  max={50}
                  step={1}
                  value={[localSettings.sdSteps || 20]}
                  onValueChange={([v]) => handleChange('sdSteps', v)}
                />
              </SettingRow>
              <SettingRow label={`Default CFG Scale (${(localSettings.sdCfgScale || 7.0).toFixed(1)})`} layout="stack">
                <Slider
                  min={1.0}
                  max={20.0}
                  step={0.5}
                  value={[localSettings.sdCfgScale || 7.0]}
                  onValueChange={([v]) => handleChange('sdCfgScale', v)}
                />
              </SettingRow>
            </SettingsSection>

            {(localSettings.imageEngine === 'auto1111' || localSettings.imageEngine === 'automatic1111') && (
              <SettingsSection
                title="AUTOMATIC1111 (External)"
                description="Configure connection to Automatic1111 WebUI."
              >
                <SettingRow label="Enable AUTOMATIC1111 Status Checks" htmlFor="enable-sd-status">
                  <Switch
                    id="enable-sd-status"
                    checked={localSettings.enableSdStatus ?? true}
                    onCheckedChange={(value) => handleChange('enableSdStatus', value)}
                  />
                </SettingRow>

                {localSettings.enableSdStatus && (
                  <SettingRow label="Connection Status" layout="stack">
                    <div className="space-y-3">
                      {apiError && (
                        <Alert variant="destructive">
                          <AlertTitle>Error</AlertTitle>
                          <AlertDescription className="flex justify-between items-center">
                            <span>{apiError}</span>
                            <Button variant="ghost" size="sm" onClick={clearError}>Dismiss</Button>
                          </AlertDescription>
                        </Alert>
                      )}

                      {sdStatus?.automatic1111 ? (
                        <Alert><AlertTitle className="text-green-600">Connected</AlertTitle></Alert>
                      ) : (
                        <Alert variant="destructive"><AlertTitle>Not Connected</AlertTitle></Alert>
                      )}

                      <Button
                        variant="outline"
                        className="w-full md:w-auto"
                        onClick={handleCheckSdStatus}
                        disabled={isCheckingStatus}
                      >
                        {isCheckingStatus ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                        Check Connection
                      </Button>
                    </div>
                  </SettingRow>
                )}

                <SettingRow label="Available Models" layout="stack">
                  <div className="max-h-48 overflow-y-auto border rounded p-2">
                    {sdStatus?.models?.map((model, index) => (
                      <div key={index} className="text-sm">{model.model_name || 'Unnamed'}</div>
                    ))}
                  </div>
                </SettingRow>

                <SettingRow label="Default Generation Settings" layout="stack">
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="default-width">Default Width</Label>
                        <Select defaultValue="512">
                          <SelectTrigger id="default-width"><SelectValue /></SelectTrigger>
                          <SelectContent><SelectItem value="512">512px</SelectItem></SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="default-height">Default Height</Label>
                        <Select defaultValue="512">
                          <SelectTrigger id="default-height"><SelectValue /></SelectTrigger>
                          <SelectContent><SelectItem value="512">512px</SelectItem></SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>Default Sampler</Label>
                      <Select defaultValue="Euler a">
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent><SelectItem value="Euler a">Euler a</SelectItem></SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Default Steps: 30</Label>
                      <Slider min={10} max={150} step={1} defaultValue={[30]} />
                    </div>
                  </div>
                </SettingRow>
              </SettingsSection>
            )}

            <div className="flex justify-end space-x-2 pt-2">
              <Button onClick={handleSave} disabled={!hasChanges} className="w-full md:w-auto">
                <Save className="mr-1 h-4 w-4" />Save
              </Button>
              <Button variant="outline" onClick={handleReset} className="w-full md:w-auto">
                Reset
              </Button>
            </div>
          </div>
        </TabsContent>

        {/* Characters */}
        <TabsContent value="characters" className="w-full max-w-none">
          <CharacterEditor />
        </TabsContent>

        {/* Audio - RESTORED FULL CONTENT */}
        <TabsContent value="audio">
          <div className="space-y-6">
            <SettingsSection
              title="Speech to Text"
              description="Configure speech recognition and engine tools."
            >
              <SettingRow label="Enable Speech-to-Text" htmlFor="stt-enabled">
                <Switch
                  id="stt-enabled"
                  checked={sttEnabled}
                  onCheckedChange={setSttEnabled}
                />
              </SettingRow>

              {sttEnabled && (
                <>
                  <SettingRow label="Speech Recognition Engine" htmlFor="stt-engine" layout="stack">
                    <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                      <Select
                        id="stt-engine"
                        value={localSettings.sttEngine || 'whisper'}
                        onValueChange={async (value) => {
                          if (value === 'parakeet') {
                            handleChange('sttEngine', value);
                            updateSettings({ sttEngine: value });
                          } else {
                            handleChange('sttEngine', value);
                            updateSettings({ sttEngine: value });
                          }
                        }}
                      >
                        <SelectTrigger className="w-full md:max-w-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="whisper">OpenAI Whisper</SelectItem>
                          <SelectItem value="whisper3">Whisper 3 Turbo</SelectItem>
                          <SelectItem value="parakeet">NVIDIA Parakeet</SelectItem>
                        </SelectContent>
                      </Select>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={fetchAvailableSTTEngines}
                        disabled={isInstallingEngine}
                        className="mt-2 md:mt-0"
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    </div>
                  </SettingRow>

                  <SettingRow label="Engine Management" layout="stack">
                    <div className="flex flex-col md:flex-row gap-2 flex-wrap">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={async () => {
                          setIsInstallingEngine(true);
                          try {
                            await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet&force=true`, { method: 'POST' });
                            alert('Parakeet engine installed successfully!');
                          } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                        }}
                      >
                        Force Install Parakeet
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={async () => {
                          setIsInstallingEngine(true);
                          try {
                            await fetch(`${PRIMARY_API_URL}/stt/fix-parakeet-numpy`, { method: 'POST' });
                            alert('Fixed! Restart Backend.');
                          } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                        }}
                      >
                        Fix Dependencies
                      </Button>
                    </div>
                  </SettingRow>

                  <SettingRow label="Quick GPU Actions" layout="stack">
                    <div className="flex flex-col md:flex-row gap-2">
                      <Button variant="outline" size="sm" onClick={() => { }}>Load Whisper on GPU1</Button>
                      <Button variant="outline" size="sm" onClick={() => { }}>Load Kokoro on GPU1</Button>
                    </div>
                  </SettingRow>
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Text to Speech"
              description="Configure voices, playback, and TTS engines."
            >
              <SettingRow label="Enable Text-to-Speech" htmlFor="tts-enabled">
                <Switch
                  id="tts-enabled"
                  checked={ttsEnabled}
                  onCheckedChange={value => queueSettingsSave({ ttsEnabled: value })}
                />
              </SettingRow>

              {ttsEnabled && (
                <>
                  <SettingRow label="Text-to-Speech Engine" htmlFor="tts-engine">
                    <Select
                      id="tts-engine"
                      value={localSettings.ttsEngine || 'kokoro'}
                      onValueChange={value => {
                        handleChange('ttsEngine', value);
                        queueSettingsSave({ ttsEngine: value });
                        if (value === 'kokoro') {
                          handleChange('ttsVoice', 'af_heart');
                          queueSettingsSave({ ttsVoice: 'af_heart' });
                        }
                      }}
                    >
                      <SelectTrigger className="w-full md:w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="kokoro">Kokoro TTS</SelectItem>
                        <SelectItem value="chatterbox">Chatterbox (Faster)</SelectItem>
                        <SelectItem value="chatterbox_turbo">Chatterbox Turbo</SelectItem>
                      </SelectContent>
                    </Select>
                  </SettingRow>

                  {(localSettings.ttsEngine || 'kokoro') === 'kokoro' && (
                    <SettingRow label="Kokoro Voice" htmlFor="tts-voice">
                      <Select
                        id="tts-voice"
                        value={localSettings.ttsVoice || 'af_heart'}
                        onValueChange={value => {
                          handleChange('ttsVoice', value);
                          queueSettingsSave({ ttsVoice: value });
                        }}
                      >
                        <SelectTrigger className="w-full md:w-64">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="max-h-64 overflow-y-auto">
                          <SelectItem value="af_heart">Am. English Female (Heart)</SelectItem>
                          <SelectItem value="af_alloy">Am. English Female (Alloy)</SelectItem>
                          <SelectItem value="af_aoede">Am. English Female (Aoede)</SelectItem>
                          <SelectItem value="af_bella">Am. English Female (Bella)</SelectItem>
                          <SelectItem value="af_jessica">Am. English Female (Jessica)</SelectItem>
                          <SelectItem value="af_kore">Am. English Female (Kore)</SelectItem>
                          <SelectItem value="af_nicole">Am. English Female (Nicole)</SelectItem>
                          <SelectItem value="af_nova">Am. English Female (Nova)</SelectItem>
                          <SelectItem value="af_river">Am. English Female (River)</SelectItem>
                          <SelectItem value="af_sarah">Am. English Female (Sarah)</SelectItem>
                          <SelectItem value="af_sky">Am. English Female (Sky)</SelectItem>
                          <SelectItem value="am_adam">Am. English Male (Adam)</SelectItem>
                          <SelectItem value="am_echo">Am. English Male (Echo)</SelectItem>
                        </SelectContent>
                      </Select>
                    </SettingRow>
                  )}

                  {(localSettings.ttsEngine === 'chatterbox' || localSettings.ttsEngine === 'chatterbox_turbo') && (
                    <>
                      <SettingRow label="Upload Voice Reference" htmlFor="voice-upload" layout="stack">
                        <Input
                          id="voice-upload"
                          type="file"
                          accept=".wav,.mp3,.flac,.m4a"
                          onChange={async (e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;
                            try {
                              setIsUploadingVoice(true);
                              const formData = new FormData();
                              formData.append('file', file);
                              const response = await fetch(`${PRIMARY_API_URL}/tts/upload-voice`, {
                                method: 'POST', body: formData,
                              });
                              const result = await response.json();
                              handleChange('ttsVoice', result.voice_id);
                              queueSettingsSave({ ttsVoice: result.voice_id });
                              await fetchAvailableVoices();
                              alert(`Voice "${file.name}" uploaded successfully!`);
                            } catch (error) {
                              alert(`Failed to upload voice: ${error.message}`);
                            } finally {
                              setIsUploadingVoice(false);
                            }
                          }}
                          disabled={!ttsEnabled || isUploadingVoice}
                        />
                      </SettingRow>

                      <SettingRow label="Active Voice" htmlFor="chatterbox-voice">
                        <Select
                          id="chatterbox-voice"
                          value={localSettings.ttsVoice || 'default'}
                          onValueChange={value => {
                            handleChange('ttsVoice', value);
                            queueSettingsSave({ ttsVoice: value });
                            fetch(`${PRIMARY_API_URL}/tts/save-voice-preference`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ voice_id: value, engine: localSettings.ttsEngine })
                            });
                          }}
                        >
                          <SelectTrigger className="w-full md:w-64">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="default">Default Voice</SelectItem>
                            {availableVoices?.chatterbox_voices?.map(voice => (
                              <SelectItem key={voice.id} value={voice.id}>{voice.name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </SettingRow>

                      <SettingRow label={`Emotion Exaggeration (${(localSettings.ttsExaggeration || 0.5).toFixed(1)})`} layout="stack">
                        <Slider
                          id="tts-exaggeration"
                          min={0.0} max={1.0} step={0.1}
                          value={[localSettings.ttsExaggeration || 0.5]}
                          onValueChange={([v]) => { handleChange('ttsExaggeration', v); queueSettingsSave({ ttsExaggeration: v }); }}
                        />
                      </SettingRow>

                      <SettingRow label={`Guidance Scale (${(localSettings.ttsCfg || 0.5).toFixed(1)})`} layout="stack">
                        <Slider
                          id="tts-cfg"
                          min={0.1} max={1.0} step={0.1}
                          value={[localSettings.ttsCfg || 0.5]}
                          onValueChange={([v]) => { handleChange('ttsCfg', v); queueSettingsSave({ ttsCfg: v }); }}
                        />
                      </SettingRow>

                      <SettingRow label="Generation Speed Mode" htmlFor="tts-speed-mode">
                        <Select
                          value={localSettings.ttsSpeedMode || 'standard'}
                          onValueChange={(value) => {
                            handleChange('ttsSpeedMode', value);
                            queueSettingsSave({ ttsSpeedMode: value });
                          }}
                        >
                          <SelectTrigger className="w-full md:w-64">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="standard">Standard</SelectItem>
                            <SelectItem value="quality">Quality</SelectItem>
                          </SelectContent>
                        </Select>
                      </SettingRow>

                      <SettingRow label="VRAM Management" layout="stack">
                        <div className="flex flex-col md:flex-row gap-3">
                          <Button
                            variant="outline"
                            onClick={async () => {
                              try {
                                setIsUnloadingChatterbox(true);
                                await fetch(`${TTS_API_URL}/tts/unload-chatterbox`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                alert('Unloaded!');
                              } finally { setIsUnloadingChatterbox(false); }
                            }}
                            disabled={isUnloadingChatterbox}
                            className="flex-1"
                          >
                            Unload Chatterbox
                          </Button>
                          <Button
                            variant="outline"
                            onClick={async () => {
                              try {
                                setIsReloadingChatterbox(true);
                                await fetch(`${TTS_API_URL}/tts/reload-chatterbox`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                alert('Reloaded!');
                              } finally { setIsReloadingChatterbox(false); }
                            }}
                            disabled={isReloadingChatterbox}
                            className="flex-1"
                          >
                            Reload Chatterbox
                          </Button>
                        </div>
                      </SettingRow>
                    </>
                  )}

                  <SettingRow label="Auto-Play TTS" htmlFor="tts-autoplay">
                    <Switch
                      id="tts-autoplay"
                      checked={localSettings.ttsAutoPlay}
                      onCheckedChange={value => {
                        handleChange('ttsAutoPlay', value);
                        queueSettingsSave({ ttsAutoPlay: value });
                      }}
                    />
                  </SettingRow>

                  <SettingRow label={`Speech Speed (${(localSettings.ttsSpeed || 1.0).toFixed(1)}x)`} layout="stack">
                    <Slider
                      id="tts-speed"
                      min={0.5} max={3.0} step={0.1}
                      value={[localSettings.ttsSpeed || 1.0]}
                      onValueChange={([v]) => {
                        handleChange('ttsSpeed', v);
                        queueSettingsSave({ ttsSpeed: v });
                      }}
                    />
                  </SettingRow>

                  {(localSettings.ttsEngine || 'kokoro') === 'kokoro' && (
                    <SettingRow label={`Pitch (${localSettings.ttsPitch} semitones)`} htmlFor="tts-pitch" layout="stack">
                      <Slider
                        id="tts-pitch"
                        min={-12} max={12} step={1}
                        value={[localSettings.ttsPitch || 0]}
                        onValueChange={([v]) => {
                          handleChange('ttsPitch', v);
                          queueSettingsSave({ ttsPitch: v });
                        }}
                      />
                    </SettingRow>
                  )}

                  <SettingRow label="Test Voice" layout="stack">
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={!ttsEnabled || isPlayingAudio === 'test-tts'}
                        onClick={() => {
                          playTTS('test-tts', 'This is a test of the text-to-speech system.', localSettings);
                        }}
                      >
                        {isPlayingAudio === 'test-tts' ? 'Playing...' : 'Test Voice Settings'}
                      </Button>
                      {isPlayingAudio === 'test-tts' && (
                        <Button variant="ghost" size="sm" onClick={() => stopTTS()}>Stop</Button>
                      )}
                    </div>
                  </SettingRow>
                </>
              )}
            </SettingsSection>
          </div>
        </TabsContent>

        {/* Memory Intent Detection */}
        <TabsContent value="memory-intent">
          <div className="space-y-6">
            <SettingsSection
              title="Memory Intent Detector"
              description="Type text below to detect memory intent patterns."
            >
              <SettingRow label="Input Text" htmlFor="memory-intent-input" layout="stack">
                <textarea
                  id="memory-intent-input"
                  className="w-full p-3 border rounded text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800"
                  rows={3}
                  value={memoryIntentInput}
                  onChange={e => setMemoryIntentInput(e.target.value)}
                  placeholder="e.g. Remember that my favorite color is blue."
                />
              </SettingRow>
              <SettingRow label="Detection" layout="stack">
                <MemoryIntentDetector
                  text={memoryIntentInput}
                  onDetected={handleMemoryIntent}
                  allowExplicitCreation={true}
                />
              </SettingRow>
            </SettingsSection>
          </div>
        </TabsContent>

        {/* Memory Browser */}
        <TabsContent value="memory">
          <MemoryEditorTab />
        </TabsContent>

        {/* Lore Debugger */}
        <TabsContent value="lore">
          <Card>
            <CardHeader><CardTitle>Lore Debugger</CardTitle></CardHeader>
            <CardContent><LoreDebugger /></CardContent>
          </Card>
        </TabsContent>
        {/*local sd*/}
        {/* About */}
        <TabsContent value="about">
          <div className="space-y-6">
            <SettingsSection
              title="About Eloquent"
              description="Local-first AI platform built for power users."
            >
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Eloquent is a local-first AI platform that combines LLM chat, image generation, voice,
                  evaluation, and tooling in one interface. It is designed to run on your hardware
                  (Windows + NVIDIA GPUs) with optional OpenAI-compatible API endpoints.
                </p>
                <p className="text-sm text-muted-foreground">
                  The stack pairs a React frontend with a FastAPI backend and includes multi-GPU orchestration,
                  a built-in Stable Diffusion pipeline, streaming TTS, a tool-calling code editor, and
                  a deep roleplay toolkit (character creator, multi-character chat, and lore).
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <h4 className="font-semibold mb-2">Core Systems</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground list-disc pl-4">
                    <li>Local LLM inference with multi-GPU support and OpenAI-compatible APIs</li>
                    <li>Built-in Stable Diffusion (SD, SDXL, FLUX) plus optional external engines</li>
                    <li>Streaming TTS with Kokoro and Chatterbox voice cloning</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Creator Tools</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground list-disc pl-4">
                    <li>Character creator and library with persona management</li>
                    <li>Multi-character chat with roster control, roles, and narrator support</li>
                    <li>Mobile-friendly UI and call-mode voice interface</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Evaluation and Analysis</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground list-disc pl-4">
                    <li>Model ELO testing, A/B comparisons, and judge workflows</li>
                    <li>Forensic linguistics analysis with embedding models</li>
                    <li>Memory, RAG, and document ingestion tools</li>
                  </ul>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-2">Local-First Philosophy</h4>
                  <p className="text-sm text-muted-foreground">
                    Runs offline by default. Your data stays on your machine unless you enable external API
                    endpoints or web search.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Credits and License</h4>
                  <p className="text-sm text-muted-foreground">
                    Built with FastAPI, React, llama.cpp, stable-diffusion.cpp, Kokoro, Chatterbox,
                    and ultralytics YOLO. Licensed under AGPL-3.0. Created by Bernard Peter Fitzgerald.
                  </p>
                </div>
              </div>
            </SettingsSection>
          </div>
        </TabsContent>

        </Tabs>
      </div>
    </div>
  );
};

const MemoryEditorTab = () => {
  const { activeProfileId } = useMemory();
  const { SECONDARY_API_URL } = useApp(); // Use SECONDARY_API_URL for memory operations
  const [memories, setMemories] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [deletingIndex, setDeletingIndex] = useState(null);
  const API_URL = SECONDARY_API_URL; // Memory operations use secondary URL

  // Helper function to format dates
  const formatDate = (s) => {
    const d = new Date(s);
    return isNaN(d.getTime()) ? 'Invalid Date' : d.toLocaleDateString();
  };

  // Helper function for category colors
  const getCategoryColor = (category) => {
    const colors = {
      'expertise': 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
      'personal_interest': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      'preferences': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
      'personal_info': 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
    };
    return colors[category] || colors.personal_info;
  };

  // Fetch memories from backend
  const fetchMemories = useCallback(async () => {
    if (!activeProfileId) {
      setMemories([]);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/memory/get_all?user_id=${activeProfileId}`);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      if (data.status === 'success' && Array.isArray(data.memories)) {
        const sorted = data.memories.sort((a, b) => new Date(b.created) - new Date(a.created));
        setMemories(sorted);
      } else {
        throw new Error(data.error || 'Unexpected response');
      }
    } catch (err) {
      setError(err.message);
      setMemories([]);
    } finally {
      setIsLoading(false);
    }
  }, [activeProfileId, API_URL]);

  const handleDelete = useCallback(async (memory, index) => {
    if (!activeProfileId) return;
    if (!window.confirm(`Delete this memory?`)) return;

    setDeletingIndex(index);

    try {
      const freshResponse = await fetch(`${API_URL}/memory/get_all?user_id=${activeProfileId}`);
      if (!freshResponse.ok) throw new Error(`Failed to fetch fresh data`);

      const freshData = await freshResponse.json();
      const freshMemories = Array.isArray(freshData.memories) ? freshData.memories : [];
      const targetMemory = freshMemories.find(m => m.content === memory.content && m.created === memory.created);

      if (!targetMemory) throw new Error('Memory not found');

      const memoriesToKeep = freshMemories.filter(m => !(m.content === targetMemory.content && m.created === targetMemory.created));

      const clearResponse = await fetch(`${API_URL}/memory/clear?user_id=${activeProfileId}`, {
        method: 'DELETE', headers: { 'Content-Type': 'application/json' }
      });

      if (!clearResponse.ok) throw new Error(`Clear failed`);

      for (const memoryToSave of memoriesToKeep) {
        const memoryWithUserId = { ...memoryToSave, user_id: activeProfileId };
        await fetch(`${API_URL}/memory/memory/create`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(memoryWithUserId)
        });
      }
      setMemories(prev => prev.filter((_, i) => i !== index));
      setTimeout(fetchMemories, 1000);

    } catch (err) {
      alert(`Failed to delete memory: ${err.message}`);
      await fetchMemories();
    } finally {
      setDeletingIndex(null);
    }
  }, [activeProfileId, API_URL, fetchMemories]);

  // Debug function to check available endpoints
  const debugEndpoints = useCallback(async () => {
    const endpointsToCheck = [`${API_URL}/docs`, `${API_URL}/openapi.json`, `${API_URL}/memory/`];
    for (const url of endpointsToCheck) {
      try { await fetch(url, { method: 'GET' }); } catch (err) { console.log(err); }
    }
  }, [API_URL]);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <CardTitle>Memory Browser</CardTitle>
            <CardDescription>
              Profile: {activeProfileId ?? 'None'} · {memories.length} memories
            </CardDescription>
          </div>
          <div className="flex gap-2 w-full md:w-auto">
            <Button size="sm" variant="outline" onClick={debugEndpoints}>
              Debug API
            </Button>
            <Button size="sm" variant="outline" onClick={fetchMemories} disabled={isLoading}>
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            </Button>
            <Button size="sm" variant="outline" onClick={() => window.open(`${API_URL}/docs`, '_blank')}>
              <ExternalLink className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin mr-2 h-4 w-4" />
            Loading memories...
          </div>
        )}

        {error && (
          <div className="text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded">Error: {error}</div>
        )}

        {!isLoading && !error && memories.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">No memories found for this profile.</div>
        )}

        {!isLoading && !error && memories.length > 0 && (
          <div className="space-y-2">
            {memories.map((memory, index) => (
              <div
                key={`${memory.content}-${index}`}
                className="flex flex-col md:flex-row items-start justify-between p-3 border rounded-lg hover:bg-muted/30 transition-colors gap-3"
              >
                <div className="flex-1 min-w-0 w-full">
                  <p className="text-sm mb-2 text-wrap break-words">{memory.content}</p>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getCategoryColor(memory.category)}`}>
                      {memory.category?.replace('_', ' ') || 'unknown'}
                    </span>
                    <span>★ {memory.importance?.toFixed(1) || '0.0'}</span>
                    <span>{formatDate(memory.created)}</span>
                    {memory.accessed && <span>accessed {memory.accessed}x</span>}
                  </div>
                </div>

                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => handleDelete(memory, index)}
                  disabled={deletingIndex === index}
                  className="self-end md:self-start flex-shrink-0 ml-3 text-muted-foreground hover:text-destructive"
                >
                  {deletingIndex === index ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                </Button>
              </div>
            ))}
          </div>
        )}

        {!activeProfileId && (
          <div className="text-center py-8 text-amber-600 bg-amber-50 dark:bg-amber-950/20 rounded">
            No active profile selected. Please select a profile to view memories.
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default Settings;

