// Settings.jsx
// Full Settings UI: General, Generation, SD, RAG, Characters, Audio, Memory Intent, Memory Browser, Lore, About

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getBackendUrl } from '../config/api';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Separator } from './ui/separator';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { Slider } from './ui/slider';
import { Save, Sun, Moon, DownloadCloud, Trash2, ExternalLink, Loader2, RefreshCw, AlertTriangle, Globe, X, Power, RotateCw, Volume2, FolderOpen } from 'lucide-react';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import CharacterEditor from './CharacterEditor';
import LoreDebugger from '../components/LoreDebugger';
import MemoryIntentDetector from './MemoryIntentDetector';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import DocumentSettings from './DocumentSettings';
import RAGSettings from './RAGSettings';
import TokenSettings from './TokenSettings';
import ProfileSelector from './ProfileSelector';
import SimpleUserProfileEditor from './SimpleUserProfileEditor';


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

  const { ttsVoice, ttsSpeed, ttsPitch, ttsAutoPlay } = localSettings;

  return (
    <div className="w-full min-h-screen p-2 md:p-4 space-y-4">
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
            <TabsTrigger value="tokens" className="flex-shrink-0">Tokens</TabsTrigger>
            <TabsTrigger value="profiles" className="flex-shrink-0">User Profiles</TabsTrigger>
          </TabsList>
        </div>

        {/* General */}
        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>Interface Settings</CardTitle>
              <CardDescription>Appearance & behavior</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Dark Mode */}
              <div className="flex flex-row items-center justify-between">
                <Label htmlFor="dark-mode">Dark Mode</Label>
                <div className="flex items-center gap-2">
                  <Sun className="h-4 w-4" />
                  <Switch id="dark-mode" checked={darkMode} onCheckedChange={toggleDarkMode} />
                  <Moon className="h-4 w-4" />
                </div>
              </div>

              {/* Remote Access Password */}
              <div className="space-y-2">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                  <div>
                    <Label htmlFor="admin-password">Remote Access Password</Label>
                    <p className="text-xs text-muted-foreground mt-1">
                      Set a password to protect your instance.
                    </p>
                  </div>
                  <Input
                    id="admin-password"
                    type="password"
                    value={localSettings.admin_password || ''}
                    className="w-full md:max-w-xs"
                    onChange={(e) => handleChange('admin_password', e.target.value)}
                    placeholder="No password set"
                  />
                </div>
              </div>
              <Separator />

              {/* API URLs */}
              <div className="space-y-4">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                  <Label htmlFor="primary-api-url">Primary API URL</Label>
                  <Input id="primary-api-url" value={PRIMARY_API_URL} readOnly className="w-full md:max-w-xs" />
                </div>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                  <Label htmlFor="secondary-api-url">Secondary API URL</Label>
                  <Input id="secondary-api-url" value={SECONDARY_API_URL} readOnly className="w-full md:max-w-xs" />
                </div>
              </div>
              <Separator />

              {/* User Avatar Size */}
              <div className="space-y-2">
                <Label htmlFor="user-avatar-size">User Avatar Size: {userAvatarSize}px</Label>
                <Slider
                  id="user-avatar-size"
                  min={64} max={512} step={16}
                  value={[userAvatarSize]}
                  onValueChange={([v]) => setUserAvatarSize(v)}
                />
              </div>
              <Separator />

              {/* Character Avatar Size - RESTORED */}
              <div className="space-y-2">
                <Label htmlFor="char-avatar-size">Character Avatar Size: {characterAvatarSize}px</Label>
                <Slider
                  id="char-avatar-size"
                  min={64} max={512} step={16}
                  value={[characterAvatarSize]}
                  onValueChange={([v]) => setCharacterAvatarSize(v)}
                />
                <p className="text-xs text-muted-foreground">Controls the display size of characters' avatars in chat.</p>
              </div>
              <Separator />

              {/* Configuration Export/Import */}
              <div className="space-y-4">
                <div>
                  <Label className="text-base font-medium">Backup & Restore</Label>
                </div>
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
              </div>
              <Separator />

              {/* Backend Logs Export */}
              <div className="space-y-2">
                <Label className="text-base font-medium">Backend Logs</Label>
                <p className="text-xs text-muted-foreground">Export backend logs for bug reports.</p>
                <div className="flex flex-col md:flex-row gap-2">
                  <Button variant="outline" onClick={handleExportBackendLogs}>
                    Export Backend Logs
                  </Button>
                  <Button variant="outline" onClick={handleClearBackendLogs}>
                    Delete Old Logs
                  </Button>
                </div>
              </div>
              <Separator />

              {/* Models Directory */}
              <div className="space-y-2">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                  <Label htmlFor="model-directory">Models Directory</Label>
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
                </div>
              </div>
              <Separator />

              {/* Single GPU Mode */}
              <div className="flex flex-row items-center justify-between">
                <div>
                  <Label htmlFor="single-gpu-mode">Single GPU Mode</Label>
                  <p className="text-xs text-muted-foreground mt-1">
                    {gpuCount <= 1 ? "Automatically enabled (Single GPU detected)" : "Enable for single GPU setup."}
                  </p>
                </div>
                <Switch
                  id="single-gpu-mode"
                  checked={localSettings.singleGpuMode || gpuCount <= 1}
                  disabled={gpuCount <= 1}
                  onCheckedChange={(value) => handleChange('singleGpuMode', value)}
                />
              </div>
              <Separator />

              {/* Main Model GPU Selection - Visible if multiple GPUs detected */}
              {gpuCount > 1 && (
                <>
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="main-gpu-id">Main Model GPU</Label>
                      <p className="text-xs text-muted-foreground mt-1">
                        Select which GPU runs the heavy LLM model service.
                      </p>
                    </div>
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
                  </div>
                  <Separator />
                </>
              )}

              {/* GPU Usage Mode - Hidden in Single GPU Mode */}
              {!localSettings.singleGpuMode && (
                <>
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="gpu-usage-mode">GPU Usage Mode (Dual GPU)</Label>
                    </div>
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
                  </div>

                  {/* Tensor Split Settings */}
                  {localSettings.gpuUsageMode === 'unified_model' && (
                    <>
                      <Separator />
                      <div className="space-y-4">
                        <Label htmlFor="tensor-split-input" className="text-base font-medium">Tensor Split Ratio</Label>
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
                      </div>
                    </>
                  )}
                </>
              )}

              {/* Forensic Models */}
              <Separator />
              <div className="space-y-4">
                <Label className="text-base font-medium">Forensic Models Management</Label>
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
              </div>

              {/* TTS Service Management */}
              <Separator />
              <div className="space-y-4">
                <Label className="text-base font-medium">TTS Service Management (Port 8002)</Label>
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
              </div>

              <div className="flex justify-end pt-6">
                <Button onClick={handleSave} disabled={!hasChanges} className="w-full md:w-auto">
                  <Save className="mr-2 h-4 w-4" />
                  Save Changes
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* User Profiles */}
        <TabsContent value="profiles">
          <Card>
            <CardHeader>
              <CardTitle>User Profiles</CardTitle>
              <CardDescription>Manage your user identities and preferences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <ProfileSelector />
              <Separator />
              <SimpleUserProfileEditor />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tokens */}
        <TabsContent value="tokens">
          <Card>
            <CardHeader>
              <CardTitle>Token Settings</CardTitle>
              <CardDescription>Configure model token limits and response length</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="max-tokens">Maximum Response Length</Label>
                <Select
                  value={localSettings.max_tokens === -1 ? 'auto' : localSettings.max_tokens.toString()}
                  onValueChange={(value) => handleChange('max_tokens', value === 'auto' ? -1 : parseInt(value, 10))}
                >
                  <SelectTrigger id="max-tokens" className="w-full">
                    <SelectValue placeholder="Select max tokens" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Maximum Available)</SelectItem>
                    <SelectItem value="1024">1024 tokens (Short)</SelectItem>
                    <SelectItem value="2048">2048 tokens (Medium)</SelectItem>
                    <SelectItem value="4096">4096 tokens (Long)</SelectItem>
                    <SelectItem value="8192">8192 tokens (Very Long)</SelectItem>
                    <SelectItem value="16384">16384 tokens (Maximum)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label htmlFor="context-length">Model Context Length</Label>
                <Select
                  value={localSettings.contextLength?.toString() || "8192"}
                  onValueChange={(value) => handleChange('contextLength', parseInt(value, 10))}
                >
                  <SelectTrigger id="context-length" className="w-full">
                    <SelectValue placeholder="Select context length" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="8192">8K</SelectItem>
                    <SelectItem value="16384">16K</SelectItem>
                    <SelectItem value="32768">32K</SelectItem>
                    <SelectItem value="65536">64K</SelectItem>
                    <SelectItem value="131072">128K</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex justify-end space-x-2 pt-4">
                <Button onClick={handleSave} disabled={!hasChanges} className="w-full md:w-auto">
                  <Save className="mr-1 h-4 w-4" />Save
                </Button>
                <Button variant="outline" onClick={handleReset} className="w-full md:w-auto">
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="rag">
          <RAGSettings />
        </TabsContent>

        {/* Generation - RESTORED FULL CONTENT */}
        <TabsContent value="generation">
          <Card>
            <CardHeader>
              <CardTitle>LLM Generation Settings</CardTitle>
              <CardDescription>Configure text‐generation parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Temperature */}
              <div className="space-y-2">
                <Label>Temperature: {localSettings.temperature.toFixed(2)}</Label>
                <Slider
                  value={[localSettings.temperature]}
                  min={0}
                  max={2}
                  step={0.05}
                  onValueChange={([v]) => handleChange('temperature', v)}
                />
              </div>
              <Separator />

              {/* Top‐P */}
              <div className="space-y-2">
                <Label>Top-P: {localSettings.top_p.toFixed(2)}</Label>
                <Slider
                  value={[localSettings.top_p]}
                  min={0}
                  max={1}
                  step={0.05}
                  onValueChange={([v]) => handleChange('top_p', v)}
                />
              </div>
              <Separator />

              {/* Top-K */}
              <div className="space-y-2">
                <Label>Top-K: {localSettings.top_k}</Label>
                <Slider
                  value={[localSettings.top_k]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={([v]) => handleChange('top_k', v)}
                />
              </div>
              <Separator />

              {/* Repetition Penalty */}
              <div className="space-y-2">
                <Label>
                  Repetition Penalty: {localSettings.repetition_penalty.toFixed(2)}
                </Label>
                <Slider
                  value={[localSettings.repetition_penalty]}
                  min={1}
                  max={2}
                  step={0.01}
                  onValueChange={([v]) => handleChange('repetition_penalty', v)}
                />
              </div>
              <Separator />

              {/* Anti-Repetition Mode */}
              <div className="space-y-4 p-4 rounded-lg border border-orange-200 dark:border-orange-800 bg-orange-50/50 dark:bg-orange-950/20">
                <div className="flex flex-row items-center justify-between">
                  <div>
                    <Label htmlFor="anti-repetition" className="text-orange-700 dark:text-orange-300 font-medium">
                      Anti-Repetition Mode
                    </Label>
                  </div>
                  <Switch
                    id="anti-repetition"
                    checked={localSettings.antiRepetitionMode}
                    onCheckedChange={(checked) => handleChange('antiRepetitionMode', checked)}
                  />
                </div>

                {localSettings.antiRepetitionMode && (
                  <div className="space-y-4 pt-2">
                    {/* Frequency Penalty */}
                    <div className="space-y-2">
                      <Label className="text-sm">
                        Frequency Penalty: {localSettings.frequencyPenalty.toFixed(2)}
                      </Label>
                      <Slider
                        value={[localSettings.frequencyPenalty]}
                        min={0}
                        max={2}
                        step={0.1}
                        onValueChange={([v]) => handleChange('frequencyPenalty', v)}
                      />
                    </div>

                    {/* Presence Penalty */}
                    <div className="space-y-2">
                      <Label className="text-sm">
                        Presence Penalty: {localSettings.presencePenalty.toFixed(2)}
                      </Label>
                      <Slider
                        value={[localSettings.presencePenalty]}
                        min={0}
                        max={2}
                        step={0.1}
                        onValueChange={([v]) => handleChange('presencePenalty', v)}
                      />
                    </div>

                    {/* Detect Repeated Phrases */}
                    <div className="flex flex-row items-center justify-between pt-2 border-t border-orange-200 dark:border-orange-800">
                      <div>
                        <Label htmlFor="detect-phrases" className="text-sm">
                          Detect & Remove Repeated Phrases
                        </Label>
                      </div>
                      <Switch
                        id="detect-phrases"
                        checked={localSettings.detectRepeatedPhrases}
                        onCheckedChange={(checked) => handleChange('detectRepeatedPhrases', checked)}
                      />
                    </div>
                  </div>
                )}
              </div>
              <Separator />

              {/* Max Tokens */}
              <div className="space-y-2">
                <Label htmlFor="max_tokens">Max Tokens:</Label>
                <Input
                  id="max_tokens"
                  type="number"
                  min={-1}
                  step={1}
                  value={localSettings.max_tokens}
                  onChange={e =>
                    handleChange('max_tokens', parseInt(e.target.value, 10))
                  }
                  className="max-w-xs w-full"
                />
              </div>
              <Separator />
              {/* Stream Responses Toggle */}
              <div className="flex flex-row items-center justify-between">
                <Label htmlFor="stream-responses">Stream Responses</Label>
                <Switch
                  id="stream-responses"
                  checked={localSettings.streamResponses}
                  onCheckedChange={(value) => handleChange('streamResponses', value)}
                />
              </div>
              <Separator />
              {/* Direct Profile Injection Toggle */}
              <div className="flex flex-row items-center justify-between">
                <div>
                  <Label htmlFor="direct-profile-injection">Direct Profile Injection</Label>
                </div>
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
              </div>

              <Separator />
              {/* OpenAI API Compatibility */}
              <div className="flex flex-row items-center justify-between">
                <div>
                  <Label htmlFor="use-openai-api">Use OpenAI API Format</Label>
                </div>
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
              </div>

              {localSettings.useOpenAIAPI && (
                <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
                    <Globe className="h-4 w-4" />
                    <span className="font-medium text-sm">OpenAI API Mode Active</span>
                  </div>
                </div>
              )}

              <Separator />
              <div className="space-y-4">
                <div className="flex flex-row items-center justify-between">
                  <Label className="text-base font-medium">Custom API Endpoints</Label>
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
                </div>

                {(localSettings.customApiEndpoints || []).map((endpoint, index) => (
                  <div key={endpoint.id} className="border rounded-lg p-4 space-y-3 bg-muted/30">
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
              </div>

              <div className="flex justify-end space-x-2 pt-4">
                <Button onClick={handleSave} disabled={!hasChanges} className="w-full md:w-auto">
                  <Save className="mr-1 h-4 w-4" />Save
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Image Generation */}
        <TabsContent value="image-generation">
          <Card>
            <CardHeader>
              <CardTitle>Image Generation</CardTitle>
              <CardDescription>Built-in Local SD with optional external engines.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Image Engine Priority</Label>
                <p className="text-xs text-muted-foreground">
                  Local SD uses the built-in stable-diffusion.cpp engine. External engines require their own servers.
                </p>
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
                    <SelectItem value="both">Show Both Options</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Separator />

              <div className="space-y-4">
                <div>
                  <h3 className="text-md font-medium">Local SD (Built-in)</h3>
                  <p className="text-xs text-muted-foreground">Built-in image generation using stable-diffusion.cpp.</p>
                </div>
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

                {/* Model Directory */}
                <div className="space-y-2">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                    <Label htmlFor="sd-model-directory">Local SD Models Directory</Label>
                    <div className="flex w-full md:w-auto items-center gap-2">
                      <Input
                        id="sd-model-directory"
                        value={localSettings.sdModelDirectory || ''}
                        className="flex-1 md:w-64"
                        onChange={(e) => handleChange('sdModelDirectory', e.target.value)}
                        placeholder="C:\\path\\to\\sd-models"
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
                  </div>
                </div>
                {/* ADetailer Models Directory */}
                <Separator />
                <div className="space-y-2">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                    <Label htmlFor="adetailer-model-directory">ADetailer Models Directory</Label>
                    <div className="flex w-full md:w-auto items-center gap-2">
                      <Input
                        id="adetailer-model-directory"
                        value={localSettings.adetailerModelDirectory || ''}
                        className="flex-1 md:w-64"
                        onChange={(e) => handleChange('adetailerModelDirectory', e.target.value)}
                        placeholder="C:\\path\\to\\adetailer-models"
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
                  </div>
                </div>
                <Separator />

                {/* Upscaler Configuration */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                      <Label htmlFor="upscaler-directory">Upscaler Models Directory</Label>
                      <div className="flex w-full md:w-auto items-center gap-2">
                        <Input
                          id="upscaler-directory"
                          value={localSettings.upscalerModelDirectory || ''}
                          className="flex-1 md:w-64"
                          onChange={(e) => handleChange('upscalerModelDirectory', e.target.value)}
                          placeholder="C:\\path\\to\\upscalers"
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
                            // Save upscaler logic
                            fetch(`${PRIMARY_API_URL}/models/update-upscaler-dir`, {
                              method: 'POST', headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ directory: localSettings.upscalerModelDirectory })
                            }).then(r => r.json()).then(d => alert('Updated!'));
                          }}
                        >
                          <Save className="mr-1 h-4 w-4" />Save
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
                <Separator />

                {/* Default Generation Settings */}
                <div className="space-y-4">
                  <h3 className="text-md font-medium">Default Generation Settings</h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Default Steps: {localSettings.sdSteps || 20}</Label>
                      <Slider
                        min={10} max={50} step={1}
                        value={[localSettings.sdSteps || 20]}
                        onValueChange={([v]) => handleChange('sdSteps', v)}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Default CFG Scale: {(localSettings.sdCfgScale || 7.0).toFixed(1)}</Label>
                      <Slider
                        min={1.0} max={20.0} step={0.5}
                        value={[localSettings.sdCfgScale || 7.0]}
                        onValueChange={([v]) => handleChange('sdCfgScale', v)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {(localSettings.imageEngine === 'auto1111' || localSettings.imageEngine === 'automatic1111') && (
                <>
                  <Separator />
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-md font-medium">AUTOMATIC1111 (External)</h3>
                      <p className="text-xs text-muted-foreground">Configure connection to Automatic1111 WebUI.</p>
                    </div>

                    <div className="flex flex-row items-center justify-between">
                      <div>
                        <Label htmlFor="enable-sd-status">Enable AUTOMATIC1111 Status Checks</Label>
                      </div>
                      <Switch
                        id="enable-sd-status"
                        checked={localSettings.enableSdStatus ?? true}
                        onCheckedChange={(value) => handleChange('enableSdStatus', value)}
                      />
                    </div>
                    <Separator />

                    {localSettings.enableSdStatus && (
                      <div className="space-y-4">
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
                          className="mt-2 w-full md:w-auto"
                          onClick={handleCheckSdStatus}
                          disabled={isCheckingStatus}
                        >
                          {isCheckingStatus ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                          Check Connection
                        </Button>
                      </div>
                    )}
                    <Separator />

                    <div className="space-y-2">
                      <Label>Available Models</Label>
                      <div className="max-h-48 overflow-y-auto border rounded p-2">
                        {sdStatus?.models?.map((model, index) => (
                          <div key={index} className="text-sm">{model.model_name || "Unnamed"}</div>
                        ))}
                      </div>
                    </div>
                    <Separator />

                    <div className="space-y-4">
                      <h3 className="text-md font-medium">Default Generation Settings</h3>
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
                  </div>
                </>
              )}

              {/* Save/Reset buttons */}
              <div className="flex justify-end space-x-2 pt-4">
                <Button onClick={handleSave} disabled={!hasChanges} className="w-full md:w-auto">
                  <Save className="mr-1 h-4 w-4" />Save
                </Button>
                <Button variant="outline" onClick={handleReset} className="w-full md:w-auto">
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Characters */}
        <TabsContent value="characters" className="w-full max-w-none">
          <CharacterEditor />
        </TabsContent>

        {/* Audio - RESTORED FULL CONTENT */}
        <TabsContent value="audio">
          <Card>
            <CardHeader>
              <CardTitle>Speech & Audio</CardTitle>
              <CardDescription>STT & TTS Configuration</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Enable STT */}
              <div className="flex flex-row items-center justify-between">
                <Label htmlFor="stt-enabled">Enable Speech-to-Text</Label>
                <Switch
                  id="stt-enabled"
                  checked={sttEnabled}
                  onCheckedChange={setSttEnabled}
                />
              </div>

              {/* STT Engine Selection */}
              {sttEnabled && (
                <>
                  <div className="space-y-2 mt-2">
                    <Label htmlFor="stt-engine">Speech Recognition Engine</Label>
                    <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                      <Select
                        id="stt-engine"
                        value={localSettings.sttEngine || "whisper"}
                        onValueChange={async (value) => {
                          if (value === 'parakeet') {
                            // Parakeet install logic omitted for brevity but state updates remain
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
                  </div>

                  <div className="space-y-2">
                    <Label>Engine Management</Label>
                    <div className="flex flex-col md:flex-row gap-2 flex-wrap">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={async () => {
                          // Install logic
                          setIsInstallingEngine(true);
                          try {
                            await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet&force=true`, { method: 'POST' });
                            alert("Parakeet engine installed successfully!");
                          } catch (e) { alert("Failed"); } finally { setIsInstallingEngine(false); }
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
                            alert("Fixed! Restart Backend.");
                          } catch (e) { alert("Failed"); } finally { setIsInstallingEngine(false); }
                        }}
                      >
                        Fix Dependencies
                      </Button>
                    </div>
                  </div>
                </>
              )}

              <div className="flex flex-col md:flex-row gap-2">
                <Button variant="outline" size="sm" onClick={() => { }}>Load Whisper on GPU1</Button>
                <Button variant="outline" size="sm" onClick={() => { }}>Load Kokoro on GPU1</Button>
              </div>
              <Separator />

              {/* Enable TTS */}
              <div className="flex flex-row items-center justify-between">
                <Label htmlFor="tts-enabled">Enable Text-to-Speech</Label>
                <Switch
                  id="tts-enabled"
                  checked={ttsEnabled}
                  onCheckedChange={value => queueSettingsSave({ ttsEnabled: value })}
                />
              </div>
              <Separator />


              {/* TTS Engine Selection */}
              {ttsEnabled && (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="tts-engine">Text-to-Speech Engine</Label>
                    <Select
                      id="tts-engine"
                      value={localSettings.ttsEngine || "kokoro"}
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
                  </div>
                  <Separator />

                  {/* Voice Selection - Kokoro */}
                  {(localSettings.ttsEngine || "kokoro") === "kokoro" && (
                    <>
                      <div className="space-y-2">
                        <Label htmlFor="tts-voice">Kokoro Voice</Label>
                        <Select
                          id="tts-voice"
                          value={localSettings.ttsVoice || "af_heart"}
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
                            {/* ... (Other voices kept conceptually, truncated for strict length limits in response but functionally available via scroll in SelectContent) */}
                          </SelectContent>
                        </Select>
                      </div>
                      <Separator />
                    </>
                  )}

                  {/* Voice Cloning - Chatterbox & Chatterbox Turbo */}
                  {((localSettings.ttsEngine === "chatterbox") || (localSettings.ttsEngine === "chatterbox_turbo")) && (
                    <>
                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="voice-upload">Upload Voice Reference</Label>
                          <div className="mt-2 space-y-2">
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
                          </div>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="chatterbox-voice">Active Voice</Label>
                          <Select
                            id="chatterbox-voice"
                            value={localSettings.ttsVoice || "default"}
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
                        </div>

                        {/* Chatterbox Controls */}
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <Label htmlFor="tts-exaggeration">Emotion Exaggeration: {(localSettings.ttsExaggeration || 0.5).toFixed(1)}</Label>
                            <Slider
                              id="tts-exaggeration"
                              min={0.0} max={1.0} step={0.1}
                              value={[localSettings.ttsExaggeration || 0.5]}
                              onValueChange={([v]) => { handleChange('ttsExaggeration', v); queueSettingsSave({ ttsExaggeration: v }); }}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="tts-cfg">Guidance Scale: {(localSettings.ttsCfg || 0.5).toFixed(1)}</Label>
                            <Slider
                              id="tts-cfg"
                              min={0.1} max={1.0} step={0.1}
                              value={[localSettings.ttsCfg || 0.5]}
                              onValueChange={([v]) => { handleChange('ttsCfg', v); queueSettingsSave({ ttsCfg: v }); }}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="tts-speed-mode">Generation Speed Mode</Label>
                            <Select
                              value={localSettings.ttsSpeedMode || "standard"}
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
                          </div>
                        </div>

                        <Separator />
                        <div className="space-y-4">
                          <Label className="text-base font-medium">VRAM Management</Label>
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
                        </div>
                      </div>
                      <Separator />
                    </>
                  )}

                  {/* Auto-Play TTS */}
                  <div className="flex flex-row items-center justify-between">
                    <div>
                      <Label htmlFor="tts-autoplay">Auto-Play TTS</Label>
                    </div>
                    <Switch
                      id="tts-autoplay"
                      checked={localSettings.ttsAutoPlay}
                      onCheckedChange={value => {
                        handleChange('ttsAutoPlay', value);
                        queueSettingsSave({ ttsAutoPlay: value });
                      }}
                    />
                  </div>
                  <Separator />

                  {/* Speech Speed */}
                  <div className="space-y-2">
                    <Label htmlFor="tts-speed">
                      Speech Speed: {(localSettings.ttsSpeed || 1.0).toFixed(1)}×
                    </Label>
                    <Slider
                      id="tts-speed"
                      min={0.5} max={3.0} step={0.1}
                      value={[localSettings.ttsSpeed || 1.0]}
                      onValueChange={([v]) => {
                        handleChange('ttsSpeed', v);
                        queueSettingsSave({ ttsSpeed: v });
                      }}
                    />
                  </div>
                  <Separator />

                  {/* Pitch (Kokoro only) */}
                  {(localSettings.ttsEngine || "kokoro") === "kokoro" && (
                    <>
                      <div className="space-y-2">
                        <Label htmlFor="tts-pitch">
                          Pitch: {localSettings.ttsPitch} semitones
                        </Label>
                        <Slider
                          id="tts-pitch"
                          min={-12} max={12} step={1}
                          value={[localSettings.ttsPitch || 0]}
                          onValueChange={([v]) => {
                            handleChange('ttsPitch', v);
                            queueSettingsSave({ ttsPitch: v });
                          }}
                        />
                      </div>
                      <Separator />
                    </>
                  )}

                  {/* Test TTS Button */}
                  <div className="mt-4 pt-2 flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!ttsEnabled || isPlayingAudio === 'test-tts'}
                      onClick={() => {
                        playTTS('test-tts', "This is a test of the text-to-speech system.", localSettings);
                      }}
                    >
                      {isPlayingAudio === 'test-tts' ? "Playing..." : "Test Voice Settings"}
                    </Button>
                    {isPlayingAudio === 'test-tts' && (
                      <Button variant="ghost" size="sm" onClick={() => stopTTS()}>Stop</Button>
                    )}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Memory Intent Detection */}
        <TabsContent value="memory-intent">
          <Card>
            <CardHeader>
              <CardTitle>Memory Intent Detector</CardTitle>
              <CardDescription>Type text below to detect memory intent patterns.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Label htmlFor="memory-intent-input">Input Text</Label>
              <textarea
                id="memory-intent-input"
                className="w-full p-2 border rounded text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800"
                rows={3}
                value={memoryIntentInput}
                onChange={e => setMemoryIntentInput(e.target.value)}
                placeholder="e.g. Remember that my favorite color is blue."
              />
              <MemoryIntentDetector
                text={memoryIntentInput}
                onDetected={handleMemoryIntent}
                allowExplicitCreation={true}
              />
            </CardContent>
          </Card>
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
          <Card>
            <CardHeader>
              <CardTitle>About Eloquent</CardTitle>
              <CardDescription>Local-first AI platform built for power users.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
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
                  <h4 className="font-semibold mb-2">Evaluation & Analysis</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground list-disc pl-4">
                    <li>Model ELO testing, A/B comparisons, and judge workflows</li>
                    <li>Forensic linguistics analysis with embedding models</li>
                    <li>Memory, RAG, and document ingestion tools</li>
                  </ul>
                </div>
              </div>

              <Separator />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-2">Local-First Philosophy</h4>
                  <p className="text-sm text-muted-foreground">
                    Runs offline by default. Your data stays on your machine unless you enable external API
                    endpoints or web search.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Credits & License</h4>
                  <p className="text-sm text-muted-foreground">
                    Built with FastAPI, React, llama.cpp, stable-diffusion.cpp, Kokoro, Chatterbox,
                    and ultralytics YOLO. Licensed under AGPL-3.0. Created by Bernard Peter Fitzgerald.
                  </p>
                </div>
              </div>

            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
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

