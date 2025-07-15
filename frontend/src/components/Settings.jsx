// Settings.jsx
// Full Settings UI: General, Generation, SD, RAG, Characters, Audio, Memory Intent, Memory Browser, Lore, About

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Separator } from './ui/separator';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { Slider } from './ui/slider';
import { Save, Sun, Moon, DownloadCloud, Trash2, ExternalLink, Loader2, RefreshCw, AlertTriangle, Globe, X } from 'lucide-react';
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
    setTtsEnabled,
    checkSdStatus,
    sdStatus,
    PRIMARY_API_URL,
    SECONDARY_API_URL,
    apiError,
    clearError,
    fetchAvailableSTTEngines,
    sttEnginesAvailable,
  
  } = useApp();

  // Local editable copy of context settings
  const [localSettings, setLocalSettings] = useState({
    ...contextSettings,
    temperature: contextSettings.temperature ?? 0.7,
    max_tokens: contextSettings.max_tokens ?? -1,
    top_p: contextSettings.top_p ?? 0.9,
    top_k: contextSettings.top_k ?? 40,
    repetition_penalty: contextSettings.repetition_penalty ?? 1.1,
    streamResponses: contextSettings.streamResponses ?? true,
    ttsSpeed: contextSettings.ttsSpeed ?? 1.0,
    ttsPitch: contextSettings.ttsPitch ?? 0,
    ttsAutoPlay: contextSettings.ttsAutoPlay ?? false,
    ttsEngine: contextSettings.ttsEngine ?? 'kokoro',
    ttsVoice: contextSettings.ttsVoice ?? 'af_heart', 
    ttsExaggeration: contextSettings.ttsExaggeration ?? 0.5,
    ttsCfg: contextSettings.ttsCfg ?? 0.5,
    sdModelDirectory: contextSettings.sdModelDirectory ?? '',
    sdSteps: contextSettings.sdSteps ?? 20,
    sdSampler: contextSettings.sdSampler ?? 'Euler a',
    sdCfgScale: contextSettings.sdCfgScale ?? 7.0,
    imageEngine: contextSettings.imageEngine ?? 'automatic1111',
    enableSdStatus: contextSettings.enableSdStatus ?? true,
    adetailerModelDirectory: contextSettings.adetailerModelDirectory ?? '',
    useOpenAIAPI: contextSettings.useOpenAIAPI ?? false,
    customApiEndpoints: contextSettings.customApiEndpoints ?? [], // Add this line
  });
  const [hasChanges, setHasChanges] = useState(false);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);
  const [sdModels, setSdModels] = useState([]);
  const [isInstallingEngine, setIsInstallingEngine] = useState(false);
  const [isUploadingVoice, setIsUploadingVoice] = useState(false);
  const [availableVoices, setAvailableVoices] = useState(null);

  // Memory intent input and detected result
  const [memoryIntentInput, setMemoryIntentInput] = useState('');
  const [memoryIntentDetected, setMemoryIntentDetected] = useState(
    contextSettings.memoryIntentText ?? ''
  );

  const handleMemoryIntent = useCallback(intent => {
    setMemoryIntentDetected(intent.content);
  }, []);

// Add this function to fetch available voices
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

  // Persist only detected
  useEffect(() => {
    updateSettings({ memoryIntentText: memoryIntentDetected });
  }, [memoryIntentDetected, updateSettings]);

  // Update localSettings and track changes
  const handleChange = useCallback((key, value) => {
    setLocalSettings(prev => {
      const updated = { ...prev, [key]: value };
      setHasChanges(JSON.stringify(updated) !== JSON.stringify(contextSettings));
      return updated;
    });
  }, [contextSettings]);

  // Then in your Settings component's JSX
<DocumentSettings />

  // Save & reset handlers
  const handleSave = useCallback(() => {
    updateSettings(localSettings);
    setHasChanges(false);
  }, [localSettings, updateSettings]);

  const handleReset = useCallback(() => {
    setLocalSettings({ ...contextSettings });
    setHasChanges(false);
  }, [contextSettings]);

  // Check Stable Diffusion status manually
  const handleCheckSdStatus = useCallback(async () => {
    setIsCheckingStatus(true);
    try {
      await checkSdStatus();
      // If successful and models exist, update SD models list
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

  // Fetch SD status on mount
  useEffect(() => {
    handleCheckSdStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps: run only once
  
// full list of Kokoro voices by language
const voices = [
  // 🇺🇸 American English
  { id: 'af_heart',   name: 'Am. English Female (Heart)' },
  { id: 'af_alloy',   name: 'Am. English Female (Alloy)' },
  { id: 'af_aoede',   name: 'Am. English Female (Aoede)' },
  { id: 'af_bella',   name: 'Am. English Female (Bella)' },
  { id: 'af_jessica', name: 'Am. English Female (Jessica)' },
  { id: 'af_kore',    name: 'Am. English Female (Kore)' },
  { id: 'af_nicole',  name: 'Am. English Female (Nicole)' },
  { id: 'af_nova',    name: 'Am. English Female (Nova)' },
  { id: 'af_river',   name: 'Am. English Female (River)' },
  { id: 'af_sarah',   name: 'Am. English Female (Sarah)' },
  { id: 'af_sky',     name: 'Am. English Female (Sky)' },

  { id: 'am_adam',    name: 'Am. English Male (Adam)' },
  { id: 'am_echo',    name: 'Am. English Male (Echo)' },
  { id: 'am_eric',    name: 'Am. English Male (Eric)' },
  { id: 'am_fenrir',  name: 'Am. English Male (Fenrir)' },
  { id: 'am_liam',    name: 'Am. English Male (Liam)' },
  { id: 'am_michael', name: 'Am. English Male (Michael)' },
  { id: 'am_onyx',    name: 'Am. English Male (Onyx)' },
  { id: 'am_puck',    name: 'Am. English Male (Puck)' },
  { id: 'am_santa',   name: 'Am. English Male (Santa)' },

  // 🇬🇧 British English (misaki[en], fallback via espeak-ng en-gb)
  { id: 'bf_alice',   name: 'Br. English Female (Alice)' },
  { id: 'bf_emma',    name: 'Br. English Female (Emma)' },
  { id: 'bf_isabella',name: 'Br. English Female (Isabella)' },
  { id: 'bf_lily',    name: 'Br. English Female (Lily)' },

  { id: 'bm_daniel',  name: 'Br. English Male (Daniel)' },
  { id: 'bm_fable',   name: 'Br. English Male (Fable)' },
  { id: 'bm_george',  name: 'Br. English Male (George)' },
  { id: 'bm_lewis',   name: 'Br. English Male (Lewis)' },

  // 🇯🇵 Japanese (misaki[ja])
  { id: 'jf_alpha',      name: 'Japanese Female (Alpha)' },
  { id: 'jf_gongitsune', name: 'Japanese Female (Gongitsune)' },
  { id: 'jf_nezumi',     name: 'Japanese Female (Nezumi)' },
  { id: 'jf_tebukuro',   name: 'Japanese Female (Tebukuro)' },

  { id: 'jm_kumo',       name: 'Japanese Male (Kumo)' },

  // 🇨🇳 Mandarin Chinese (misaki[zh])
  { id: 'zf_xiaobei',  name: 'Mandarin Female (Xiaobei)' },
  { id: 'zf_xiaoni',   name: 'Mandarin Female (Xiaoni)' },
  { id: 'zf_xiaoxiao', name: 'Mandarin Female (Xiaoxiao)' },
  { id: 'zf_xiaoyi',   name: 'Mandarin Female (Xiaoyi)' },

  { id: 'zm_yunjian',  name: 'Mandarin Male (Yunjian)' },
  { id: 'zm_yunxi',    name: 'Mandarin Male (Yunxi)' },
  { id: 'zm_yunxia',   name: 'Mandarin Male (Yunxia)' },
  { id: 'zm_yunyang',  name: 'Mandarin Male (Yunyang)' },

  // 🇪🇸 Spanish (misaki[en] + espeak-ng es)
  { id: 'ef_dora',     name: 'Spanish Female (Dora)' },
  { id: 'em_alex',     name: 'Spanish Male (Alex)' },
  { id: 'em_santa',    name: 'Spanish Male (Santa)' },

  // 🇫🇷 French (misaki[en] + espeak-ng fr-fr)
  { id: 'ff_siwis',    name: 'French Female (Siwis)' },

  // 🇮🇳 Hindi (misaki[en] + espeak-ng hi)
  { id: 'hf_alpha',    name: 'Hindi Female (Alpha)' },
  { id: 'hf_beta',     name: 'Hindi Female (Beta)' },
  { id: 'hm_omega',    name: 'Hindi Male (Omega)' },
  { id: 'hm_psi',      name: 'Hindi Male (Psi)' },

  // 🇮🇹 Italian (misaki[en] + espeak-ng it)
  { id: 'if_sara',     name: 'Italian Female (Sara)' },
  { id: 'im_nicola',   name: 'Italian Male (Nicola)' },

  // 🇧🇷 Brazilian Portuguese (misaki[en] + espeak-ng pt-br)
  { id: 'pf_dora',     name: 'Br. Portuguese Female (Dora)' },
  { id: 'pm_alex',     name: 'Br. Portuguese Male (Alex)' },
  { id: 'pm_santa',    name: 'Br. Portuguese Male (Santa)' },
];
const { ttsVoice, ttsSpeed, ttsPitch, ttsAutoPlay } = localSettings;

  return (
<div className="w-full min-h-screen p-4 space-y-4">
  <h2 className="text-2xl font-bold mb-4">Settings</h2>
  <Tabs defaultValue={initialTab} className="space-y-6">
    <div className="border rounded-lg bg-card p-1">
      <TabsList className="flex w-full flex-wrap justify-start gap-1">
        <TabsTrigger value="general" className="flex-shrink-0">General</TabsTrigger>
        <TabsTrigger value="generation" className="flex-shrink-0">LLM Settings</TabsTrigger>
        <TabsTrigger value="sd" className="flex-shrink-0">Stable Diffusion</TabsTrigger>
        <TabsTrigger value="rag" className="flex-shrink-0">Document Context</TabsTrigger>
        <TabsTrigger value="characters" className="flex-shrink-0">Characters</TabsTrigger>
        <TabsTrigger value="audio" className="flex-shrink-0">Audio</TabsTrigger>
        <TabsTrigger value="memory-intent" className="flex-shrink-0">Memory Intent</TabsTrigger>
        <TabsTrigger value="memory" className="flex-shrink-0">Memory Browser</TabsTrigger>
        <TabsTrigger value="lore" className="flex-shrink-0">Lore Debugger</TabsTrigger>
        <TabsTrigger value="about" className="flex-shrink-0">About</TabsTrigger>
        <TabsTrigger value="tokens" className="flex-shrink-0">Tokens</TabsTrigger>
        <TabsTrigger value="profiles" className="flex-shrink-0">User Profiles</TabsTrigger>
        <TabsTrigger value="EloDiffusion" className="flex-shrink-0">Local SD</TabsTrigger>
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
              <div className="flex items-center justify-between">
                <Label htmlFor="dark-mode">Dark Mode</Label>
                <div className="flex items-center gap-2">
                  <Sun className="h-4 w-4"/>
                  <Switch id="dark-mode" checked={darkMode} onCheckedChange={toggleDarkMode}/>
                  <Moon className="h-4 w-4"/>
                </div>
              </div>
              <Separator/>
              

              {/* API URLs */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="primary-api-url">Primary API URL</Label>
                  <Input id="primary-api-url" value={PRIMARY_API_URL} readOnly className="max-w-xs"/>
                </div>
                <div className="flex items-center justify-between">
                  <Label htmlFor="secondary-api-url">Secondary API URL</Label>
                  <Input id="secondary-api-url" value={SECONDARY_API_URL} readOnly className="max-w-xs"/>
                </div>
              </div>
              <Separator/>

              {/* User Avatar Size */}
              <div className="space-y-2">
                <Label htmlFor="user-avatar-size">User Avatar Size: {userAvatarSize}px</Label>
                <Slider
                  id="user-avatar-size"
                  min={64} max={512} step={16}
                  value={[userAvatarSize]}
                  onValueChange={([v]) => setUserAvatarSize(v)}
                />
                <p className="text-xs text-muted-foreground">Controls the display size of your own avatar in chat.</p>
              </div>
              <Separator/>

              {/* Character Avatar Size */}
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
              {/* Add this to the "General" tab in Settings.jsx */}
              <Separator/>

<div className="space-y-2">
  <div className="flex items-center justify-between">
    <Label htmlFor="model-directory">Models Directory</Label>
    <div className="flex items-center gap-2">
      <Input 
        id="model-directory" 
        value={localSettings.modelDirectory || ''}
        className="max-w-xs" 
        onChange={(e) => handleChange('modelDirectory', e.target.value)}
        placeholder="C:\models\gguf or /home/user/models/gguf"
      />
      <Button 
        variant="outline" 
        onClick={() => {
          if (localSettings.modelDirectory) {
            fetch(`${PRIMARY_API_URL}/models/refresh-directory`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ directory: localSettings.modelDirectory })
            })
            .then(response => response.json())
            .then(data => {
              if (data.status === 'success') {
                alert(`Model directory updated to ${localSettings.modelDirectory}\n\nPlease restart the backend for changes to take effect.`);
              } else {
                alert(`Error: ${data.message || 'Failed to update directory'}`);
              }
            })
            .catch(err => {
              console.error("Error updating directory:", err);
              alert("Failed to update directory.");
            });
          } else {
            alert("Please enter a directory path first.");
          }
        }}
      >
        Save
      </Button>
    </div>
  </div>
  <p className="text-xs text-muted-foreground">
    Enter the full path to your GGUF models directory. You will need to restart the backend for changes to take effect. Note: After restart, the path may not display here again, but your models directory will remain correctly set until you change it.
  </p>
</div>
{/* SD Models Directory */}
<Separator/>
<div className="space-y-2">
  <div className="flex items-center justify-between">
    <Label htmlFor="sd-model-directory">Stable Diffusion Models Directory</Label>
    <div className="flex items-center gap-2">
      <Input 
        id="sd-model-directory" 
        value={localSettings.sdModelDirectory || ''}
        className="max-w-xs" 
        onChange={(e) => handleChange('sdModelDirectory', e.target.value)}
        placeholder="C:\models\sd or /home/user/models/stable-diffusion"
      />
      <Button 
        variant="outline" 
        onClick={() => {
          if (localSettings.sdModelDirectory) {
            fetch(`${PRIMARY_API_URL}/sd-local/refresh-directory`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ directory: localSettings.sdModelDirectory })
            })
            .then(response => response.json())
            .then(data => {
              if (data.status === 'success') {
                alert(`SD model directory updated to ${localSettings.sdModelDirectory}\n\nRestart backend for changes to take effect.`);
              } else {
                alert(`Error: ${data.message || 'Failed to update directory'}`);
              }
            })
            .catch(err => {
              console.error("Error updating SD directory:", err);
              alert("Failed to update SD directory.");
            });
          } else {
            alert("Please enter a directory path first.");
          }
        }}
      >
        Save
      </Button>
    </div>
  </div>
  <p className="text-xs text-muted-foreground">
    Enter the full path to your Stable Diffusion models directory (.safetensors, .ckpt files). 
    Restart backend for changes to take effect.
  </p>
</div>
              <Separator/>
              <div className="flex items-center justify-between">
  <div>
    <Label htmlFor="single-gpu-mode">Single GPU Mode</Label>
    <p className="text-xs text-muted-foreground mt-1">
      Use only GPU 0 for all operations. Enable this if you only have one GPU.
    </p>
  </div>
  <Switch
    id="single-gpu-mode"
    checked={localSettings.singleGpuMode}
    onCheckedChange={(value) => handleChange('singleGpuMode', value)}
  />
</div>
<Separator/>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="gpu-usage-mode">GPU Usage Mode (Dual GPU)</Label>
                  <p className="text-xs text-muted-foreground mt-1 mb-2">
                    Choose how to use your GPUs when you have multiple available.
                  </p>
                </div>
                <Select
                  value={localSettings.gpuUsageMode || 'split_services'}
                  onValueChange={(value) => {
                    // Update local state for immediate UI feedback
                    handleChange('gpuUsageMode', value);
                    
                    // Save to backend settings.json file
                    fetch(`${PRIMARY_API_URL}/models/update-gpu-mode`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ gpuUsageMode: value })
                    })
                    .then(response => response.json())
                    .then(data => {
                      if (data.status === 'success') {
                        alert(`GPU usage mode updated to ${value === 'split_services' ? 'Split Services' : 'Unified Model'}\n\nPlease restart the backend for changes to take effect.`);
                      } else {
                        alert(`Error: ${data.message || 'Failed to update GPU mode'}`);
                      }
                    })
                    .catch(err => {
                      console.error("Error updating GPU mode:", err);
                      alert("Failed to update GPU usage mode.");
                    });
                  }}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select GPU usage mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="split_services">
                      Split Services (GPU 0: Inference, GPU 1: Memory/Audio/SD)
                    </SelectItem>
                    <SelectItem value="unified_model">
                      Unified Model (Both GPUs: Large model split across GPUs)
                    </SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Split Services: Current behavior, optimal for running multiple services simultaneously.<br/>
                  Unified Model: Use both GPUs together for larger models (40GB+ models), memory/audio services may have reduced GPU resources.<br/>
                  <strong>Note: Backend restart required for changes to take effect.</strong>
                </p>
              </div>

            </CardContent>
          </Card>
        </TabsContent>
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
        <p className="text-xs text-muted-foreground">
          'Auto' uses maximum available length based on model's context window.
        </p>
      </div>
      
      <Separator />
      
      <div className="space-y-2">
        <Label htmlFor="context-length">Model Context Length</Label>
        <Select
          value={localSettings.contextLength.toString()}
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
        <p className="text-xs text-muted-foreground">
          Set context length when loading models. Higher values allow longer conversations but use more memory.
        </p>
      </div>
      
      {/* Save / Reset */}
      <div className="flex justify-end space-x-2 pt-4">
        <Button onClick={handleSave} disabled={!hasChanges}>
          <Save className="mr-1 h-4 w-4"/>Save
        </Button>
        <Button variant="outline" onClick={handleReset}>
          Reset
        </Button>
      </div>
    </CardContent>
  </Card>
</TabsContent>
        
        <TabsContent value="rag">
          <RAGSettings />
        </TabsContent>

        {/* Generation */}
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
                  className="max-w-xs"
                />
                <p className="text-xs text-muted-foreground">
                  Use –1 for model default.
                </p>
              </div>
              <Separator />
              {/* Stream Responses Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="stream-responses">Stream Responses</Label>
                  <p className="text-xs text-muted-foreground mt-1">
                    Show responses as they're being generated.
                  </p>
                </div>
                <Switch
                  id="stream-responses"
                  checked={localSettings.streamResponses}
                  onCheckedChange={(value) => handleChange('streamResponses', value)}
                />
              </div>
            <Separator />
{/* OpenAI API Compatibility */}
<div className="flex items-center justify-between">
  <div>
    <Label htmlFor="use-openai-api">Use OpenAI API Format</Label>
    <p className="text-xs text-muted-foreground mt-1">
      Use OpenAI-compatible endpoint instead of custom Eloquent API. Useful for testing API compatibility.
    </p>
  </div>
<Switch
  id="use-openai-api"
  checked={localSettings.useOpenAIAPI || false}
  onCheckedChange={(value) => {
    handleChange('useOpenAIAPI', value);
    
    // Also save to settings.json
    fetch(`${PRIMARY_API_URL}/models/set-openai-api-mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ useOpenAIAPI: value })
    })
    .then(response => response.json())
    .catch(err => console.error("Error saving OpenAI API setting:", err));
  }}
/>
</div>

{/* Show OpenAI API status when enabled */}
{localSettings.useOpenAIAPI && (
  <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
    <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
      <Globe className="h-4 w-4" />
      <span className="font-medium text-sm">OpenAI API Mode Active</span>
    </div>
    <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
      Using /v1/chat/completions endpoint. Memory and some advanced features may be limited.
    </p>
  </div>
)}

<Separator />
<div className="space-y-4">
  <div className="flex items-center justify-between">
    <Label className="text-base font-medium">Custom API Endpoints</Label>
    <Button 
      size="sm" 
      variant="outline"
      onClick={() => {
        const newEndpoints = [...(localSettings.customApiEndpoints || []), {
          id: `endpoint-${Date.now()}`,
          name: 'New Endpoint',
          url: 'http://localhost:8000',
          apiKey: '',
          enabled: true
        }];
        handleChange('customApiEndpoints', newEndpoints);
      }}
    >
      Add Endpoint
    </Button>
  </div>
  
  <div className="text-sm text-muted-foreground">
    Configure custom OpenAI-compatible API endpoints (actual OpenAI, local servers, etc.)
  </div>
  
  {(localSettings.customApiEndpoints || []).map((endpoint, index) => (
    <div key={endpoint.id} className="border rounded-lg p-4 space-y-3 bg-muted/30">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
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
            className="w-48"
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
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <Label className="text-xs">API URL</Label>
          <Input
            placeholder="https://api.openai.com"
            value={endpoint.url}
            onChange={(e) => {
              const updated = [...localSettings.customApiEndpoints];
              updated[index] = { ...endpoint, url: e.target.value };
              handleChange('customApiEndpoints', updated);
            }}
          />
        </div>
        <div>
          <Label className="text-xs">API Key (optional)</Label>
          <Input
            type="password"
            placeholder="sk-..."
            value={endpoint.apiKey}
            onChange={(e) => {
              const updated = [...localSettings.customApiEndpoints];
              updated[index] = { ...endpoint, apiKey: e.target.value };
              handleChange('customApiEndpoints', updated);
            }}
          />
        </div>
      </div>
      
      {endpoint.url.includes('openai.com') && (
        <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/20 p-2 rounded">
          ⚠️ This will send data to OpenAI's servers. Make sure you comply with their usage policies.
        </div>
      )}
    </div>
  ))}
  
  {(!localSettings.customApiEndpoints || localSettings.customApiEndpoints.length === 0) && (
    <div className="text-sm text-muted-foreground text-center py-4 border-2 border-dashed rounded-lg">
      No custom API endpoints configured. Click "Add Endpoint" to add one.
    </div>
  )}

  {/* Save Endpoints Button */}
  <div className="flex justify-end">
    <Button 
      variant="outline"
      onClick={() => {
        fetch(`${PRIMARY_API_URL}/models/save-custom-endpoints`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ customApiEndpoints: localSettings.customApiEndpoints || [] })
        })
        .then(response => response.json())
        .then(data => {
          if (data.status === 'success') {
            alert('Custom API endpoints saved to backend!');
          } else {
            alert(`Error: ${data.message || 'Failed to save endpoints'}`);
          }
        })
        .catch(err => {
          console.error("Error saving endpoints:", err);
          alert("Failed to save endpoints.");
        });
      }}
    >
      <Save className="mr-1 h-4 w-4"/>
      Save Endpoints
    </Button>
  </div>
</div>

<Separator />
              {/* Save / Reset */}
              <div className="flex justify-end space-x-2 pt-4">
                <Button onClick={handleSave} disabled={!hasChanges}>
                  <Save className="mr-1 h-4 w-4"/>Save
                </Button>
                <Button variant="outline" onClick={handleReset}>
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Stable Diffusion Settings */}
        <TabsContent value="sd">
          <Card>
            <CardHeader>
              <CardTitle>Stable Diffusion Settings</CardTitle>
              <CardDescription>Configure connection to Automatic1111 WebUI</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
{/* Enable/Disable SD Status Checks */}
<div className="flex items-center justify-between">
  <div>
    <Label htmlFor="enable-sd-status">Enable AUTOMATIC1111 Status Checks</Label>
    <p className="text-xs text-muted-foreground mt-1">
      Automatically check if AUTOMATIC1111 WebUI is running (may cause backend spam)
    </p>
  </div>
  <Switch
    id="enable-sd-status"
    checked={localSettings.enableSdStatus ?? true}
    onCheckedChange={(value) => handleChange('enableSdStatus', value)}
  />
</div>
<Separator />

{/* Only show status if enabled */}
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
      <Alert>
        <AlertTitle className="flex items-center text-green-600">
          <div className="h-2 w-2 rounded-full bg-green-600 mr-2"></div>
          Connected
        </AlertTitle>
        <AlertDescription>
          Stable Diffusion WebUI is running and available at http://127.0.0.1:7860/
        </AlertDescription>
      </Alert>
    ) : (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Not Connected</AlertTitle>
        <AlertDescription>
          Stable Diffusion WebUI is not running or inaccessible. Make sure Automatic1111 WebUI is running on http://127.0.0.1:7860/
        </AlertDescription>
      </Alert>
    )}

    <Button 
      variant="outline" 
      className="mt-2" 
      onClick={handleCheckSdStatus}
      disabled={isCheckingStatus}
    >
      {isCheckingStatus ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Checking...
        </>
      ) : (
        <>
          <RefreshCw className="mr-2 h-4 w-4" />
          Check Connection
        </>
      )}
    </Button>
  </div>
)}
<Separator />

              {/* Available Models */}
              <div className="space-y-2">
                <Label>Available Models</Label>
                <div className="max-h-48 overflow-y-auto border rounded p-2">
                  {sdStatus?.automatic1111 ? (
                    sdStatus?.models && sdStatus.models.length > 0 ? (
                      <ul className="space-y-1">
                        {sdStatus.models.map((model, index) => (
                          <li key={index} className="text-sm">
                            {model.model_name || model.title || model.name || "Unnamed Model"}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-muted-foreground">No models found.</p>
                    )
                  ) : (
                    <p className="text-sm text-muted-foreground">Connect to Stable Diffusion to view models.</p>
                  )}
                </div>
              </div>
              <Separator />

              {/* Default Settings */}
              <div className="space-y-4">
                <h3 className="text-md font-medium">Default Generation Settings</h3>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="default-width">Default Width</Label>
                    <Select defaultValue="512">
                      <SelectTrigger id="default-width">
                        <SelectValue placeholder="Select width" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="512">512px</SelectItem>
                        <SelectItem value="768">768px</SelectItem>
                        <SelectItem value="1024">1024px</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="default-height">Default Height</Label>
                    <Select defaultValue="512">
                      <SelectTrigger id="default-height">
                        <SelectValue placeholder="Select height" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="512">512px</SelectItem>
                        <SelectItem value="768">768px</SelectItem>
                        <SelectItem value="1024">1024px</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label>Default Sampler</Label>
                  <Select defaultValue="Euler a">
                    <SelectTrigger>
                      <SelectValue placeholder="Select sampler" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Euler a">Euler a</SelectItem>
                      <SelectItem value="Euler">Euler</SelectItem>
                      <SelectItem value="LMS">LMS</SelectItem>
                      <SelectItem value="Heun">Heun</SelectItem>
                      <SelectItem value="DPM2">DPM2</SelectItem>
                      <SelectItem value="DPM2 a">DPM2 a</SelectItem>
                      <SelectItem value="DPM++ 2S a">DPM++ 2S a</SelectItem>
                      <SelectItem value="DPM++ 2M">DPM++ 2M</SelectItem>
                      <SelectItem value="DPM++ SDE">DPM++ SDE</SelectItem>
                      <SelectItem value="DDIM">DDIM</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Default Steps: 30</Label>
                  <Slider
                    min={10}
                    max={150}
                    step={1}
                    defaultValue={[30]}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Default Guidance Scale: 7.0</Label>
                  <Slider
                    min={1.0}
                    max={30.0}
                    step={0.5}
                    defaultValue={[7.0]}
                  />
                </div>
              </div>
              
              <div className="flex justify-end space-x-2 pt-4">
                <Button>
                  <Save className="mr-1 h-4 w-4"/>Save Defaults
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>



        {/* Characters */}
<TabsContent value="characters" className="w-full max-w-none">
  <CharacterEditor />
</TabsContent>


<TabsContent value="audio">
  <Card>
    <CardHeader>
      <CardTitle>Speech & Audio</CardTitle>
      <CardDescription>STT & TTS Configuration</CardDescription>
    </CardHeader>
    <CardContent className="space-y-6">
      {/* Enable STT */}
      <div className="flex items-center justify-between">
        <Label htmlFor="stt-enabled">Enable Speech-to-Text</Label>
        <Switch
          id="stt-enabled"
          checked={sttEnabled}
          onCheckedChange={setSttEnabled}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Enable microphone input for voice commands and dictation.
      </p>
      
      {/* STT Engine Selection */}
      {sttEnabled && (
        <>
          <div className="space-y-2 mt-2">
            <Label htmlFor="stt-engine">Speech Recognition Engine</Label>
            <div className="flex items-center gap-2">
              <Select
                id="stt-engine"
                value={localSettings.sttEngine || "whisper"}
                onValueChange={async (value) => {
                  if (value === 'parakeet') {
                    try {
                      setIsInstallingEngine(true);
                      console.log("Installing Parakeet engine...");
                      
                      const response = await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet`, {
                        method: 'POST'
                      });
                      
                      if (!response.ok) {
                        throw new Error(`Failed to install Parakeet: ${response.status}`);
                      }
                      
                      await fetchAvailableSTTEngines();
                      handleChange('sttEngine', value);
                      updateSettings({ sttEngine: value });
                      
                    } catch (error) {
                      console.error("Error installing Parakeet:", error);
                    } finally {
                      setIsInstallingEngine(false);
                    }
                  } else {
                    handleChange('sttEngine', value);
                    updateSettings({ sttEngine: value });
                  }
                }}
                disabled={isInstallingEngine}
              >
                <SelectTrigger className="w-full max-w-xs">
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
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
            
            {isInstallingEngine && (
              <div className="flex items-center text-amber-500 text-sm">
                <Loader2 className="animate-spin mr-2 h-4 w-4" />
                Installing engine... This may take a few minutes.
              </div>
            )}
          </div>
          
          {/* STT Engine Management Buttons */}
          <div className="space-y-2">
            <Label>Engine Management</Label>
            <div className="flex gap-2 flex-wrap">
              <Button 
                variant="outline" 
                size="sm"
                onClick={async () => {
                  try {
                    setIsInstallingEngine(true);
                    const response = await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet&force=true`, {
                      method: 'POST'
                    });
                    if (response.ok) {
                      await fetchAvailableSTTEngines();
                      alert("Parakeet engine installed successfully!");
                    } else {
                      throw new Error(`Install failed: ${response.status}`);
                    }
                  } catch (error) {
                    console.error("Error installing Parakeet:", error);
                    alert(`Failed to install Parakeet: ${error.message}`);
                  } finally {
                    setIsInstallingEngine(false);
                  }
                }}
                disabled={isInstallingEngine}
              >
                Force Install Parakeet
              </Button>
              
              <Button 
                variant="outline" 
                size="sm"
                onClick={async () => {
                  try {
                    setIsInstallingEngine(true);
                    const response = await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=whisper3&force=true`, {
                      method: 'POST'
                    });
                    if (response.ok) {
                      await fetchAvailableSTTEngines();
                      alert("Whisper 3 Turbo installed successfully!");
                    } else {
                      throw new Error(`Install failed: ${response.status}`);
                    }
                  } catch (error) {
                    console.error("Error installing Whisper 3:", error);
                    alert(`Failed to install Whisper 3: ${error.message}`);
                  } finally {
                    setIsInstallingEngine(false);
                  }
                }}
                disabled={isInstallingEngine}
              >
                Force Install Whisper 3 Turbo
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Use force install if engines are not working properly or dependencies are missing.
            </p>
          </div>
        </>
      )}
      <Separator />

      {/* Enable TTS */}
      <div className="flex items-center justify-between">
        <Label htmlFor="tts-enabled">Enable Text-to-Speech</Label>
        <Switch
          id="tts-enabled"
          checked={ttsEnabled}
          onCheckedChange={setTtsEnabled}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Enable voice output for assistant responses.
      </p>
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
                    // Only allow kokoro for now
                    if (value === 'kokoro') {
                      handleChange('ttsEngine', value);
                      updateSettings({ ttsEngine: value });
                      handleChange('ttsVoice', 'af_heart');
                      updateSettings({ ttsVoice: 'af_heart' });
                    }
                  }}
                  disabled={!ttsEnabled}
                >
                  <SelectTrigger className={`w-48 ${!ttsEnabled ? "opacity-50" : ""}`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="kokoro">Kokoro TTS</SelectItem>
                    <SelectItem value="kyutai" disabled className="opacity-50 cursor-not-allowed">
                      Kyutai TTS (coming one day maybe I dunno)
                    </SelectItem>
                    <SelectItem value="chatterbox" disabled className="opacity-50 cursor-not-allowed">
                      Chatterbox TTS (coming one day maybe I dunno)
                    </SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Kokoro TTS is currently the only working engine. Other engines are kept for future implementation.
                </p>
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
                    updateSettings({ ttsVoice: value });
                  }}
                  disabled={!ttsEnabled}
                >
                  <SelectTrigger className={`w-64 ${!ttsEnabled ? "opacity-50" : ""}`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="max-h-64 overflow-y-auto">
                    {/* American English */}
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
                    <SelectItem value="am_eric">Am. English Male (Eric)</SelectItem>
                    <SelectItem value="am_fenrir">Am. English Male (Fenrir)</SelectItem>
                    <SelectItem value="am_liam">Am. English Male (Liam)</SelectItem>
                    <SelectItem value="am_michael">Am. English Male (Michael)</SelectItem>
                    <SelectItem value="am_onyx">Am. English Male (Onyx)</SelectItem>
                    <SelectItem value="am_puck">Am. English Male (Puck)</SelectItem>
                    <SelectItem value="am_santa">Am. English Male (Santa)</SelectItem>
                    
                    {/* British English */}
                    <SelectItem value="bf_alice">Br. English Female (Alice)</SelectItem>
                    <SelectItem value="bf_emma">Br. English Female (Emma)</SelectItem>
                    <SelectItem value="bf_isabella">Br. English Female (Isabella)</SelectItem>
                    <SelectItem value="bf_lily">Br. English Female (Lily)</SelectItem>
                    
                    <SelectItem value="bm_daniel">Br. English Male (Daniel)</SelectItem>
                    <SelectItem value="bm_fable">Br. English Male (Fable)</SelectItem>
                    <SelectItem value="bm_george">Br. English Male (George)</SelectItem>
                    <SelectItem value="bm_lewis">Br. English Male (Lewis)</SelectItem>
                    
                    {/* Japanese */}
                    <SelectItem value="jf_alpha">Japanese Female (Alpha)</SelectItem>
                    <SelectItem value="jf_gongitsune">Japanese Female (Gongitsune)</SelectItem>
                    <SelectItem value="jf_nezumi">Japanese Female (Nezumi)</SelectItem>
                    <SelectItem value="jf_tebukuro">Japanese Female (Tebukuro)</SelectItem>
                    <SelectItem value="jm_kumo">Japanese Male (Kumo)</SelectItem>
                    
                    {/* Mandarin Chinese */}
                    <SelectItem value="zf_xiaobei">Mandarin Female (Xiaobei)</SelectItem>
                    <SelectItem value="zf_xiaoni">Mandarin Female (Xiaoni)</SelectItem>
                    <SelectItem value="zf_xiaoxiao">Mandarin Female (Xiaoxiao)</SelectItem>
                    <SelectItem value="zf_xiaoyi">Mandarin Female (Xiaoyi)</SelectItem>
                    <SelectItem value="zm_yunjian">Mandarin Male (Yunjian)</SelectItem>
                    <SelectItem value="zm_yunxi">Mandarin Male (Yunxi)</SelectItem>
                    <SelectItem value="zm_yunxia">Mandarin Male (Yunxia)</SelectItem>
                    <SelectItem value="zm_yunyang">Mandarin Male (Yunyang)</SelectItem>
                    
                    {/* Spanish */}
                    <SelectItem value="ef_dora">Spanish Female (Dora)</SelectItem>
                    <SelectItem value="em_alex">Spanish Male (Alex)</SelectItem>
                    <SelectItem value="em_santa">Spanish Male (Santa)</SelectItem>
                    
                    {/* French */}
                    <SelectItem value="ff_siwis">French Female (Siwis)</SelectItem>
                    
                    {/* Hindi */}
                    <SelectItem value="hf_alpha">Hindi Female (Alpha)</SelectItem>
                    <SelectItem value="hf_beta">Hindi Female (Beta)</SelectItem>
                    <SelectItem value="hm_omega">Hindi Male (Omega)</SelectItem>
                    <SelectItem value="hm_psi">Hindi Male (Psi)</SelectItem>
                    
                    {/* Italian */}
                    <SelectItem value="if_sara">Italian Female (Sara)</SelectItem>
                    <SelectItem value="im_nicola">Italian Male (Nicola)</SelectItem>
                    
                    {/* Brazilian Portuguese */}
                    <SelectItem value="pf_dora">Br. Portuguese Female (Dora)</SelectItem>
                    <SelectItem value="pm_alex">Br. Portuguese Male (Alex)</SelectItem>
                    <SelectItem value="pm_santa">Br. Portuguese Male (Santa)</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Select from Kokoro's preset voices in multiple languages.
                </p>
              </div>
              <Separator />
            </>
          )}

          {/* Voice Cloning - Chatterbox */}
          {localSettings.ttsEngine === "chatterbox" && (
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
                            method: 'POST',
                            body: formData,
                          });
                          
                          if (!response.ok) {
                            throw new Error(`Upload failed: ${response.status}`);
                          }
                          
                          const result = await response.json();
                          
                          // Update the selected voice to the uploaded one
                          handleChange('ttsVoice', result.voice_id);
                          updateSettings({ ttsVoice: result.voice_id });
                          
                          // Refresh available voices
                          await fetchAvailableVoices();
                          
                          alert(`Voice "${file.name}" uploaded successfully!`);
                        } catch (error) {
                          console.error('Voice upload error:', error);
                          alert(`Failed to upload voice: ${error.message}`);
                        } finally {
                          setIsUploadingVoice(false);
                        }
                      }}
                      disabled={!ttsEnabled || isUploadingVoice}
                      className={!ttsEnabled ? "opacity-50" : ""}
                    />
                    {isUploadingVoice && (
                      <div className="flex items-center text-blue-500 text-sm">
                        <Loader2 className="animate-spin mr-2 h-4 w-4" />
                        Uploading voice reference...
                      </div>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Upload a clear audio sample (WAV, MP3, FLAC, M4A) to clone this voice.
                  </p>
                </div>
                
                {/* Chatterbox Voice Selection */}
                <div className="space-y-2">
                  <Label htmlFor="chatterbox-voice">Active Voice</Label>
                  <Select
                    id="chatterbox-voice"
                    value={localSettings.ttsVoice || "default"}
                    onValueChange={value => {
                      handleChange('ttsVoice', value);
                      updateSettings({ ttsVoice: value });
                    }}
                    disabled={!ttsEnabled}
                  >
                    <SelectTrigger className={`w-64 ${!ttsEnabled ? "opacity-50" : ""}`}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">Default Voice</SelectItem>
                      {availableVoices?.chatterbox_voices?.map(voice => (
                        <SelectItem key={voice.id} value={voice.id}>
                          {voice.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Chatterbox Controls */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="tts-exaggeration">
                      Emotion Exaggeration: {(localSettings.ttsExaggeration || 0.5).toFixed(1)}
                    </Label>
                    <Slider
                      id="tts-exaggeration"
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      value={[localSettings.ttsExaggeration || 0.5]}
                      onValueChange={([v]) => {
                        handleChange('ttsExaggeration', v);
                        updateSettings({ ttsExaggeration: v });
                      }}
                      disabled={!ttsEnabled}
                      className={!ttsEnabled ? "opacity-50" : ""}
                    />
                    <p className="text-xs text-muted-foreground">
                      Control emotional intensity. Higher values = more dramatic speech.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="tts-cfg">
                      Guidance Scale: {(localSettings.ttsCfg || 0.5).toFixed(1)}
                    </Label>
                    <Slider
                      id="tts-cfg"
                      min={0.1}
                      max={1.0}
                      step={0.1}
                      value={[localSettings.ttsCfg || 0.5]}
                      onValueChange={([v]) => {
                        handleChange('ttsCfg', v);
                        updateSettings({ ttsCfg: v });
                      }}
                      disabled={!ttsEnabled}
                      className={!ttsEnabled ? "opacity-50" : ""}
                    />
                    <p className="text-xs text-muted-foreground">
                      Lower values = slower, more deliberate pacing. Higher = faster speech.
                    </p>
                  </div>
                </div>
              </div>
              <Separator />
            </>
          )}

          {/* Auto-Play TTS */}
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="tts-autoplay">Auto-Play TTS</Label>
              <p className="text-xs text-muted-foreground mt-1">
                Automatically play new bot messages using text-to-speech.
              </p>
            </div>
            <Switch
              id="tts-autoplay"
              checked={localSettings.ttsAutoPlay}
              onCheckedChange={value => {
                handleChange('ttsAutoPlay', value);
                updateSettings({ ttsAutoPlay: value });
              }}
              disabled={!ttsEnabled}
              className={!ttsEnabled ? "opacity-50" : ""}
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
              min={0.5}
              max={3.0}
              step={0.1}
              value={[localSettings.ttsSpeed || 1.0]}
              onValueChange={([v]) => {
                handleChange('ttsSpeed', v);
                updateSettings({ ttsSpeed: v });
              }}
              disabled={!ttsEnabled}
              className={!ttsEnabled ? "opacity-50" : ""}
            />
            <p className="text-xs text-muted-foreground">
              Adjust playback speed from 0.5× (slower) to 3.0× (faster).
            </p>
          </div>
          <Separator />
{/* Voice Selection - Kyutai */}
{localSettings.ttsEngine === "kyutai" && (
  <>
    <div className="space-y-2">
      <Label htmlFor="tts-voice">Kyutai Voice</Label>
      <Select
        id="tts-voice"
        value={localSettings.ttsVoice || "default"}
        onValueChange={value => {
          handleChange('ttsVoice', value);
          updateSettings({ ttsVoice: value });
        }}
        disabled={!ttsEnabled}
      >
        <SelectTrigger className={`w-64 ${!ttsEnabled ? "opacity-50" : ""}`}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent className="max-h-64 overflow-y-auto">
          <SelectItem value="default">Default Voice</SelectItem>
          
          {/* EARS Dataset - High Quality Emotions */}
          <SelectItem value="ears/p003/emo_desire_freeform.wav">EARS Female Desire (p003)</SelectItem>
          <SelectItem value="ears/p003/emo_neutral_freeform.wav">EARS Female Neutral (p003)</SelectItem>
          <SelectItem value="ears/p003/emo_contentment_freeform.wav">EARS Female Happy (p003)</SelectItem>
          <SelectItem value="ears/p003/emo_anger_freeform.wav">EARS Female Angry (p003)</SelectItem>
          <SelectItem value="ears/p003/emo_sadness_freeform.wav">EARS Female Sad (p003)</SelectItem>
          
          <SelectItem value="ears/p031/emo_desire_freeform.wav">EARS Male Desire (p031)</SelectItem>
          <SelectItem value="ears/p031/emo_neutral_freeform.wav">EARS Male Neutral (p031)</SelectItem>
          <SelectItem value="ears/p031/emo_anger_freeform.wav">EARS Male Angry (p031)</SelectItem>
          
          {/* Expresso Dataset - Clear Emotions */}
          <SelectItem value="expresso/ex03-ex01_happy_001_channel1_334s.wav">Expresso Female Happy (ex03)</SelectItem>
          <SelectItem value="expresso/ex03-ex01_desire_004_channel1_545s.wav">Expresso Female Desire (ex03)</SelectItem>
          <SelectItem value="expresso/ex03-ex01_angry_001_channel1_201s.wav">Expresso Female Angry (ex03)</SelectItem>
          <SelectItem value="expresso/ex03-ex01_calm_001_channel1_1143s.wav">Expresso Female Calm (ex03)</SelectItem>
          <SelectItem value="expresso/ex03-ex01_sarcastic_001_channel1_435s.wav">Expresso Female Sarcastic (ex03)</SelectItem>
          <SelectItem value="expresso/ex03-ex01_laughing_001_channel1_188s.wav">Expresso Female Laughing (ex03)</SelectItem>
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        Choose from curated Kyutai voices. EARS voices offer high-quality emotions, Expresso provides clear expressions.
      </p>
    </div>
    <Separator />
  </>
)}
          {/* Pitch (Kokoro only) */}
          {(localSettings.ttsEngine || "kokoro") === "kokoro" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="tts-pitch">
                  Pitch: {localSettings.ttsPitch >= 0 ? `+${localSettings.ttsPitch}` : localSettings.ttsPitch} semitone{Math.abs(localSettings.ttsPitch || 0) === 1 ? '' : 's'}
                </Label>
                <Slider
                  id="tts-pitch"
                  min={-12}
                  max={12}
                  step={1}
                  value={[localSettings.ttsPitch || 0]}
                  onValueChange={([v]) => {
                    handleChange('ttsPitch', v);
                    updateSettings({ ttsPitch: v });
                  }}
                  disabled={!ttsEnabled}
                  className={!ttsEnabled ? "opacity-50" : ""}
                />
                <p className="text-xs text-muted-foreground">
                  Shift voice pitch up or down by up to 12 semitones (one octave).
                </p>
              </div>
              <Separator />
            </>
          )}

          {/* Test TTS Button */}
          <div className="mt-4 pt-2">
            <Button 
              variant="outline" 
              disabled={!ttsEnabled}
              onClick={() => {
                const testText = "This is a test of the text-to-speech system with the current settings.";
                
                // Save current settings so the test uses them
                updateSettings({
                  ttsVoice: localSettings.ttsVoice || 'af_heart',
                  ttsSpeed: localSettings.ttsSpeed || 1.0,
                  ttsPitch: localSettings.ttsPitch || 0,
                  ttsEngine: localSettings.ttsEngine || 'kokoro',
                  ttsExaggeration: localSettings.ttsExaggeration || 0.5,
                  ttsCfg: localSettings.ttsCfg || 0.5
                });
                
                // Use your existing playTTS function - don't change AppContext
                playTTS('test-tts', testText);
              }}
            >
              Test Voice Settings
            </Button>
            {/* Call Mode Settings */}
{sttEnabled && ttsEnabled && (
  <>
    <Separator />
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-base font-medium">Call Mode Settings</Label>
          <p className="text-sm text-muted-foreground">
            Configure voice activity detection for immersive conversations
          </p>
        </div>
      </div>
      
      {/* Silence Threshold */}
      <div className="space-y-2">
        <Label htmlFor="call-mode-silence-threshold">
          Silence Threshold (milliseconds)
        </Label>
        <div className="flex items-center gap-4">
          <input
            id="call-mode-silence-threshold"
            type="range"
            min="500"
            max="5000"
            step="250"
            value={localSettings.callModeSilenceThreshold || 2000}
            onChange={(e) => {
              const value = parseInt(e.target.value);
              handleChange('callModeSilenceThreshold', value);
              updateSettings({ callModeSilenceThreshold: value });
            }}
            className="flex-1"
          />
          <div className="w-20 text-sm text-center bg-muted rounded px-2 py-1">
            {(localSettings.callModeSilenceThreshold || 2000)}ms
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          How long to wait after you stop speaking before processing your message.
          Lower values = more responsive, but may cut off if you pause while thinking.
        </p>
        
        {/* Preset buttons */}
        <div className="flex gap-2 mt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              handleChange('callModeSilenceThreshold', 1000);
              updateSettings({ callModeSilenceThreshold: 1000 });
            }}
          >
            Fast (1s)
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              handleChange('callModeSilenceThreshold', 2000);
              updateSettings({ callModeSilenceThreshold: 2000 });
            }}
          >
            Normal (2s)
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              handleChange('callModeSilenceThreshold', 3500);
              updateSettings({ callModeSilenceThreshold: 3500 });
            }}
          >
            Patient (3.5s)
          </Button>
        </div>
      </div>
      
      {/* Auto-start TTS */}
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="call-mode-auto-tts">Auto-play Response</Label>
          <p className="text-xs text-muted-foreground">
            Automatically play TTS for responses in Call Mode
          </p>
        </div>
        <Switch
          id="call-mode-auto-tts"
          checked={localSettings.callModeAutoTTS ?? true}
          onCheckedChange={(checked) => {
            handleChange('callModeAutoTTS', checked);
            updateSettings({ callModeAutoTTS: checked });
          }}
        />
      </div>
    </div>
  </>
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
              {memoryIntentDetected && (
                <div className="mt-2 text-sm">
                  <strong>Detected Content:</strong> {memoryIntentDetected}
                </div>
              )}
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
<TabsContent value="EloDiffusion">
  <Card>
    <CardHeader>
      <CardTitle>EloDiffusion</CardTitle>
      <CardDescription>Built-in image generation using stable-diffusion.cpp</CardDescription>
    </CardHeader>
    <CardContent className="space-y-6">
      {/* Model Directory */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="sd-model-directory">SD Models Directory</Label>
          <div className="flex items-center gap-2">
            <Input 
              id="sd-model-directory" 
              value={localSettings.sdModelDirectory || ''}
              className="max-w-xs" 
              onChange={(e) => handleChange('sdModelDirectory', e.target.value)}
              placeholder="C:\models\sd or /path/to/sd/models"
            />
            <Button 
              variant="outline" 
              onClick={() => {
                if (localSettings.sdModelDirectory) {
                  fetch(`${PRIMARY_API_URL}/sd-local/refresh-directory`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ directory: localSettings.sdModelDirectory })
                  })
                  .then(response => response.json())
                  .then(data => {
                    if (data.status === 'success') {
                      alert(`SD model directory updated. Restart backend for changes to take effect.`);
                    } else {
                      alert(`Error: ${data.message || 'Failed to update directory'}`);
                    }
                  })
                  .catch(err => {
                    console.error("Error updating SD directory:", err);
                    alert("Failed to update SD directory.");
                  });
                } else {
                  alert("Please enter a directory path first.");
                }
              }}
            >
              Save
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Directory containing .safetensors or .ckpt Stable Diffusion models
        </p>
      </div>
{/* ADetailer Models Directory */}
<Separator/>
<div className="space-y-2">
  <div className="flex items-center justify-between">
    <Label htmlFor="adetailer-model-directory">ADetailer Models Directory</Label>
    <div className="flex items-center gap-2">
      <Input 
        id="adetailer-model-directory" 
        value={localSettings.adetailerModelDirectory || ''}
        className="max-w-xs" 
        onChange={(e) => handleChange('adetailerModelDirectory', e.target.value)}
        placeholder="C:\models\adetailer or /path/to/adetailer/models"
      />
      <Button 
        variant="outline" 
        onClick={() => {
          if (localSettings.adetailerModelDirectory) {
            fetch(`${PRIMARY_API_URL}/sd-local/set-adetailer-directory`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ directory: localSettings.adetailerModelDirectory })
            })
            .then(response => response.json())
            .then(data => {
              if (data.status === 'success') {
                alert(`ADetailer model directory updated. Restart backend for changes to take effect.`);
              } else {
                alert(`Error: ${data.message || 'Failed to update directory'}`);
              }
            })
            .catch(err => {
              console.error("Error updating ADetailer directory:", err);
              alert("Failed to update ADetailer directory.");
            });
          } else {
            alert("Please enter a directory path first.");
          }
        }}
      >
        Save
      </Button>
    </div>
  </div>
  <p className="text-xs text-muted-foreground">
    Directory containing ADetailer .pt model files. Restart backend for changes to take effect.
  </p>
</div>
      <Separator />

      {/* Default Generation Settings */}
      <div className="space-y-4">
        <h3 className="text-md font-medium">Default Generation Settings</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Default Steps: {localSettings.sdSteps || 20}</Label>
            <Slider
              min={10}
              max={50}
              step={1}
              value={[localSettings.sdSteps || 20]}
              onValueChange={([v]) => handleChange('sdSteps', v)}
            />
          </div>
          
          <div className="space-y-2">
            <Label>Default CFG Scale: {(localSettings.sdCfgScale || 7.0).toFixed(1)}</Label>
            <Slider
              min={1.0}
              max={20.0}
              step={0.5}
              value={[localSettings.sdCfgScale || 7.0]}
              onValueChange={([v]) => handleChange('sdCfgScale', v)}
            />
          </div>
        </div>
        
        <div className="space-y-2">
          <Label>Image Engine Priority</Label>
          <Select 
            value={localSettings.imageEngine || 'auto1111'}
            onValueChange={(value) => handleChange('imageEngine', value)}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select image engine" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto1111">AUTOMATIC1111 (External)</SelectItem>
              <SelectItem value="EloDiffusion">Local SD (Built-in)</SelectItem>
              <SelectItem value="both">Show Both Options</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Choose default image generation engine or show both options
          </p>
        </div>
      </div>

      {/* Save/Reset buttons */}
      <div className="flex justify-end space-x-2 pt-4">
        <Button onClick={handleSave} disabled={!hasChanges}>
          <Save className="mr-1 h-4 w-4"/>Save
        </Button>
        <Button variant="outline" onClick={handleReset}>
          Reset
        </Button>
      </div>
    </CardContent>
  </Card>
</TabsContent>

        {/* About */}
<TabsContent value="about">
  <Card>
    <CardHeader>
      <CardTitle>About Eloquent</CardTitle>
      <CardDescription>Advanced Local AI Interface with Iterative Alignment</CardDescription>
    </CardHeader>
    <CardContent className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="font-semibold mb-2">🧠 Core Architecture</h4>
          <ul className="text-sm space-y-1 text-muted-foreground">
            <li>• Dual-GPU intelligent memory system</li>
            <li>• Local-first privacy (100% offline)</li>
            <li>• Custom LLaMa.cpp Python backend</li>
            <li>• Iterative Alignment Theory implementation</li>
          </ul>
        </div>
        
        <div>
          <h4 className="font-semibold mb-2">🎯 Advanced Features</h4>
          <ul className="text-sm space-y-1 text-muted-foreground">
            <li>• Model ELO testing & analysis chat</li>
            <li>• Perspective-driven AI questioning</li>
            <li>• RAG document intelligence</li>
            <li>• Multi-modal chat (text, audio, images)</li>
          </ul>
        </div>
        
        <div>
          <h4 className="font-semibold mb-2">🎨 Creative Tools</h4>
          <ul className="text-sm space-y-1 text-muted-foreground">
            <li>• Custom character personas & lore</li>
            <li>• Stable Diffusion integration</li>
            <li>• 60+ voice TTS (Kokoro & Chatterbox)</li>
            <li>• NVIDIA Parakeet & Whisper STT</li>
          </ul>
        </div>
        
        <div>
          <h4 className="font-semibold mb-2">🔬 Research Grade</h4>
          <ul className="text-sm space-y-1 text-muted-foreground">
            <li>• Automated model evaluation</li>
            <li>• Satirical analysis perspectives</li>
            <li>• Conversation memory persistence</li>
            <li>• Dynamic alignment feedback loops</li>
          </ul>
        </div>
      </div>
      
      <div className="border-t pt-4">
        <p className="text-sm text-muted-foreground mb-2">
          <strong>Version 1.0.0</strong> • Developed over 4 months through AI-assisted collaborative development
        </p>
        <p className="text-sm text-muted-foreground">
          Eloquent represents a complete rethinking of human-AI interaction, combining cutting-edge research 
          with practical UX design. Built as a proof-of-concept for Iterative Alignment Theory, it demonstrates 
          how sophisticated AI alignment concepts can be implemented in accessible, privacy-focused interfaces.
        </p>
      </div>
      
      <Button variant="outline" className="w-full" disabled>
        <DownloadCloud className="mr-1 h-4 w-4"/>
        Check for Updates
      </Button>
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

  // Delete memory function - uses clear+recreate since no delete endpoint exists
  const handleDelete = useCallback(async (memory, index) => {
    if (!activeProfileId) return;
    
    setDeletingIndex(index);
    
    try {
      // Clear all memories for this user
      const clearResponse = await fetch(`${API_URL}/memory/clear?user_id=${activeProfileId}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!clearResponse.ok) {
        throw new Error(`Clear failed: ${clearResponse.status}`);
      }

      // Save back all memories except the one being deleted
      const memoriesToKeep = memories.filter((_, i) => i !== index);
      let saveCount = 0;
      
      for (const memoryToSave of memoriesToKeep) {
        // Ensure the memory has user_id field (not user)
        const memoryWithUserId = {
          ...memoryToSave,
          user_id: activeProfileId
        };
        
        const saveResponse = await fetch(`${API_URL}/memory/memory/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(memoryWithUserId)
        });
        
        if (saveResponse.ok) {
          saveCount++;
        } else {
          const errorData = await saveResponse.json().catch(() => ({}));
          console.error(`Failed to save memory: ${errorData.detail || saveResponse.status}`);
        }
      }

      if (saveCount === memoriesToKeep.length) {
        // Update UI immediately
        setMemories(prev => prev.filter((_, i) => i !== index));
        // Refresh to confirm
        setTimeout(fetchMemories, 500);
      } else {
        throw new Error(`Only saved ${saveCount}/${memoriesToKeep.length} memories`);
      }

    } catch (err) {
      alert(`Failed to delete memory: ${err.message}`);
    } finally {
      setDeletingIndex(null);
    }
  }, [activeProfileId, memories, fetchMemories, API_URL]);

  // Debug function to check available endpoints
  const debugEndpoints = useCallback(async () => {
    console.log('=== DEBUGGING AVAILABLE ENDPOINTS ===');
    console.log(`Using API_URL: ${API_URL}`);
    
    const endpointsToCheck = [
      `${API_URL}/docs`,
      `${API_URL}/openapi.json`,
      `${API_URL}/memory/`,
      `${API_URL}/memory/delete`,
      `${API_URL}/memory/remove`, 
      `${API_URL}/memory/clear`,
      `${API_URL}/memory/save`
    ];
    
    for (const url of endpointsToCheck) {
      try {
        const response = await fetch(url, { method: 'GET' });
        console.log(`${url}: ${response.status} ${response.statusText}`);
      } catch (err) {
        console.log(`${url}: ERROR - ${err.message}`);
      }
    }
    
    // Try to get OpenAPI spec
    try {
      const response = await fetch(`${API_URL}/openapi.json`);
      if (response.ok) {
        const spec = await response.json();
        console.log('Available paths:', Object.keys(spec.paths || {}));
      }
    } catch (err) {
      console.log('Could not fetch OpenAPI spec');
    }
  }, [API_URL]);

  // Load memories on component mount and when activeProfileId changes
  useEffect(() => { 
    fetchMemories(); 
  }, [fetchMemories]);

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Memory Browser</CardTitle>
            <CardDescription>
              Profile: {activeProfileId ?? 'None'} · {memories.length} memories
            </CardDescription>
          </div>
          <div className="flex gap-2">
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
            <Loader2 className="animate-spin mr-2 h-4 w-4"/>
            Loading memories...
          </div>
        )}
        
        {error && (
          <div className="text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded">
            Error: {error}
          </div>
        )}
        
        {!isLoading && !error && memories.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            No memories found for this profile.
          </div>
        )}

        {!isLoading && !error && memories.length > 0 && (
          <div className="space-y-2">
            {memories.map((memory, index) => (
              <div 
                key={`${memory.content}-${index}`}
                className="flex items-start justify-between p-3 border rounded-lg hover:bg-muted/30 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm mb-2 text-wrap break-words">
                    {memory.content}
                  </p>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
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
                  className="flex-shrink-0 ml-3 text-muted-foreground hover:text-destructive"
                  title="Delete this memory"
                >
                  {deletingIndex === index ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
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