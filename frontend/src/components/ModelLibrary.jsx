import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { BookOpenCheck, CreditCard, Download, ExternalLink, FolderOpen, HardDrive, Image, KeyRound, Loader2, RefreshCw, Search, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Progress } from './ui/progress';
import FrontierModelLibrary from './FrontierModelLibrary';
import {
  readNanoGptModelsCache,
  refreshNanoGptModelsCache,
  normalizeNanoGptModel,
  subscribeNanoGptModelsCache,
} from '../utils/nanoGptModelsCache';
import {
  readOpenRouterModelsCache,
  refreshOpenRouterModelsCache,
  subscribeOpenRouterModelsCache,
} from '../utils/openRouterModelsCache';
import {
  HOSTED_MODEL_PROVIDERS,
  syncHostedProviderEndpointKey,
  upsertHostedModelEndpoint,
} from '../utils/hostedModelProviders';


const formatBytes = (bytes) => {
  if (!Number.isFinite(bytes)) return 'Size unavailable';
  if (bytes >= 1024 ** 3) return `${(bytes / (1024 ** 3)).toFixed(1)} GB`;
  return `${(bytes / (1024 ** 2)).toFixed(0)} MB`;
};

const readModelLibraryIntent = () => {
  try {
    const raw = sessionStorage.getItem('mirid-model-library-intent');
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (_) {
    return {};
  }
};

const formatDownloadRate = (bytesPerSecond) => {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) return '';
  if (bytesPerSecond >= 1024 ** 2) return `${(bytesPerSecond / (1024 ** 2)).toFixed(1)} MB/s`;
  return `${Math.round(bytesPerSecond / 1024)} KB/s`;
};

const isImageCheckpoint = (filename) => /\.(safetensors|ckpt|gguf)$/i.test(String(filename || ''));

const getVramFit = (model, totalVramGb) => {
  if (!Number.isFinite(totalVramGb)) return { label: 'Hardware not detected', tone: 'outline' };
  const fileSizeGb = Number(model.size) / (1024 ** 3);
  const estimatedWorkingVramGb = (fileSizeGb * 1.08) + 1.5;
  if (estimatedWorkingVramGb <= totalVramGb * 0.85) return { label: 'Comfortable fit', tone: 'default', estimatedWorkingVramGb };
  if (estimatedWorkingVramGb <= totalVramGb * 0.95) return { label: 'Close fit', tone: 'secondary', estimatedWorkingVramGb };
  return { label: 'RAM / partial offload', tone: 'outline', estimatedWorkingVramGb };
};

const ModelLibrary = ({ onSettingChange }) => {
  const [entryIntent] = useState(readModelLibraryIntent);
  const imageSetup = entryIntent.mode === 'image';
  const navigate = useNavigate();
  const {
    PRIMARY_API_URL,
    settings,
    updateSettings,
    setPrimaryIsAPI,
    setPrimaryModel,
    setActiveTab,
    primaryModel,
    fetchModels,
    loadModel,
    isModelLoading,
  } = useApp();
  const [source, setSource] = useState(() => entryIntent.source || settings.modelSetupSource || 'huggingface');
  const [recommendationsOpen, setRecommendationsOpen] = useState(() => !imageSetup && settings.modelSetupRequired === true);
  const [destinations, setDestinations] = useState([]);
  const [reference, setReference] = useState('');
  const [searchQuery, setSearchQuery] = useState(() => entryIntent.searchQuery || '');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [inspection, setInspection] = useState(null);
  const [selectedFile, setSelectedFile] = useState('');
  const [destinationType, setDestinationType] = useState(() => imageSetup ? 'image' : 'text');
  const [downloadJob, setDownloadJob] = useState(null);
  const [error, setError] = useState(null);
  const [nanoModels, setNanoModels] = useState(() => readNanoGptModelsCache().models);
  const [nanoQuery, setNanoQuery] = useState('');
  const [nanoProvider, setNanoProvider] = useState('all');
  const [nanoLoading, setNanoLoading] = useState(false);
  const [nanoSubscriptionModels, setNanoSubscriptionModels] = useState([]);
  const [nanoSubscriptionLoading, setNanoSubscriptionLoading] = useState(false);
  const [nanoSubscriptionError, setNanoSubscriptionError] = useState('');
  const [openRouterModels, setOpenRouterModels] = useState(() => readOpenRouterModelsCache().models);
  const [openRouterQuery, setOpenRouterQuery] = useState('');
  const [openRouterFreeOnly, setOpenRouterFreeOnly] = useState(false);
  const [openRouterLoading, setOpenRouterLoading] = useState(false);
  const [gpuInfo, setGpuInfo] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const [cardSummary, setCardSummary] = useState('');
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [destinationDrafts, setDestinationDrafts] = useState({});
  const [destinationBusy, setDestinationBusy] = useState('');
  const [destinationMessage, setDestinationMessage] = useState({});
  const [imageModelLoading, setImageModelLoading] = useState(false);
  const [civitaiQuery, setCivitaiQuery] = useState('');
  const [civitaiModels, setCivitaiModels] = useState([]);
  const [civitaiSearching, setCivitaiSearching] = useState(false);
  const [civitaiInspecting, setCivitaiInspecting] = useState(false);
  const [civitaiModel, setCivitaiModel] = useState(null);
  const [civitaiVersionId, setCivitaiVersionId] = useState('');
  const [initialImageSearchStarted, setInitialImageSearchStarted] = useState(false);

  useEffect(() => {
    try { sessionStorage.removeItem('mirid-model-library-intent'); } catch (_) {}
  }, []);

  const loadDestinations = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/destinations`);
      if (!response.ok) throw new Error('Could not read model folders.');
      const data = await response.json();
      const nextDestinations = data.destinations || [];
      setDestinations(nextDestinations);
      setDestinationDrafts(Object.fromEntries(
        nextDestinations.map((destination) => [destination.type, destination.path]),
      ));
      return nextDestinations;
    } catch (loadError) {
      setError(loadError.message);
      return [];
    }
  }, [PRIMARY_API_URL]);

  useEffect(() => {
    loadDestinations();
    const unsubscribeNano = subscribeNanoGptModelsCache(({ models }) => setNanoModels(models));
    const unsubscribeOpenRouter = subscribeOpenRouterModelsCache(({ models }) => setOpenRouterModels(models));
    return () => {
      unsubscribeNano();
      unsubscribeOpenRouter();
    };
  }, [loadDestinations]);

  useEffect(() => {
    if (entryIntent.focus !== 'folders' || destinations.length === 0) return;
    window.requestAnimationFrame(() => {
      document.getElementById('model-folders')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }, [destinations.length, entryIntent.focus]);

  const loadRecommendations = useCallback(async () => {
    setRecommendationsLoading(true);
    try {
      const [gpuResponse, picksResponse] = await Promise.all([
        fetch(`${PRIMARY_API_URL}/system/gpu_info`),
        fetch(`${PRIMARY_API_URL}/model-library/recommendations`),
      ]);
      if (gpuResponse.ok) setGpuInfo(await gpuResponse.json());
      const picksData = await picksResponse.json();
      if (!picksResponse.ok) throw new Error(picksData.detail || 'Could not refresh Mirid’s picks.');
      setRecommendations(picksData.models || []);
    } catch (recommendationError) {
      setError(recommendationError.message);
    } finally {
      setRecommendationsLoading(false);
    }
  }, [PRIMARY_API_URL]);

  useEffect(() => {
    loadRecommendations();
  }, [loadRecommendations]);

  useEffect(() => {
    if (!downloadJob?.id || !['queued', 'downloading'].includes(downloadJob.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/model-library/downloads/${downloadJob.id}`);
        if (!response.ok) return;
        const next = await response.json();
        setDownloadJob(next);
      } catch (_) {
        // The next poll may recover if the backend was briefly busy.
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [PRIMARY_API_URL, downloadJob?.id, downloadJob?.status]);

  useEffect(() => {
    if (downloadJob?.status === 'complete' && downloadJob.destination_type === 'text') {
      fetchModels();
    }
  }, [downloadJob?.destination_type, downloadJob?.status, fetchModels]);

  const loadDownloadedModel = useCallback(async () => {
    if (!downloadJob?.filename) return;
    setPrimaryIsAPI(false);
    const loaded = await loadModel(downloadJob.filename, 0);
    if (!loaded) return;
    updateSettings({ modelSetupRequired: false });
    setActiveTab('chat');
  }, [downloadJob?.filename, loadModel, setActiveTab, setPrimaryIsAPI, updateSettings]);

  const loadDownloadedImageModel = useCallback(async () => {
    if (!downloadJob?.filename) return;
    const imageDestination = destinations.find((destination) => destination.type === 'image');
    if (!imageDestination?.path) {
      setError('Mirid could not find the image model folder.');
      return;
    }
    setImageModelLoading(true);
    setError(null);
    try {
      const refreshResponse = await fetch(`${PRIMARY_API_URL}/sd-local/refresh-directory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory: imageDestination.path }),
      });
      const refreshData = await refreshResponse.json().catch(() => ({}));
      if (!refreshResponse.ok) {
        throw new Error(refreshData.detail || refreshData.message || 'The image model folder could not be refreshed.');
      }
      const loadResponse = await fetch(`${PRIMARY_API_URL}/sd-local/load-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_filename: downloadJob.filename, gpu_id: 0 }),
      });
      const loadData = await loadResponse.json().catch(() => ({}));
      if (!loadResponse.ok) {
        throw new Error(loadData.detail || loadData.message || 'The downloaded image model could not be loaded.');
      }
      onSettingChange?.('sdModelDirectory', imageDestination.path);
      onSettingChange?.('imageEngine', 'EloDiffusion');
      updateSettings({
        sdModelDirectory: imageDestination.path,
        imageEngine: 'EloDiffusion',
      });
      try { sessionStorage.setItem('mirid-open-image-generator', 'true'); } catch (_) {}
      setActiveTab('chat');
    } catch (loadError) {
      setError(`The model was downloaded, but Mirid could not load it. ${loadError.message}`);
    } finally {
      setImageModelLoading(false);
    }
  }, [PRIMARY_API_URL, destinations, downloadJob?.filename, onSettingChange, setActiveTab, updateSettings]);

  const searchHuggingFace = useCallback(async (queryOverride = null) => {
    const query = typeof queryOverride === 'string' ? queryOverride.trim() : searchQuery.trim();
    if (query.length < 2) return;
    setSearchQuery(query);
    setSearching(true);
    setError(null);
    try {
      const response = await fetch(
        `${PRIMARY_API_URL}/model-library/huggingface/search?q=${encodeURIComponent(query)}${imageSetup ? '&kind=image' : ''}`,
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Hugging Face search failed.');
      setSearchResults(data.models || []);
    } catch (searchError) {
      setError(searchError.message);
    } finally {
      setSearching(false);
    }
  }, [PRIMARY_API_URL, imageSetup, searchQuery]);

  useEffect(() => {
    if (!imageSetup || entryIntent.focus !== 'search' || initialImageSearchStarted || !entryIntent.searchQuery) return;
    setInitialImageSearchStarted(true);
    searchHuggingFace(entryIntent.searchQuery);
  }, [entryIntent.focus, entryIntent.searchQuery, imageSetup, initialImageSearchStarted, searchHuggingFace]);

  const inspectReference = useCallback(async (nextReference = reference) => {
    const value = nextReference.trim();
    if (!value) return;
    setReference(value);
    setInspecting(true);
    setInspection(null);
    setError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/huggingface/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference: value }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not inspect that repository.');
      const availableFiles = imageSetup
        ? (data.files || []).filter((file) => isImageCheckpoint(file.filename))
        : (data.files || []);
      setInspection({ ...data, files: availableFiles });
      setCardSummary('');
      const preferred = availableFiles.find((file) => file.quant_match && file.role === 'model' && /\.(safetensors|gguf)$/i.test(file.filename))
        || availableFiles.find((file) => file.quant_match && file.role === 'model')
        || availableFiles.find((file) => file.role === 'model')
        || availableFiles[0];
      setSelectedFile(preferred?.filename || '');
      setDestinationType(imageSetup ? 'image' : (preferred?.suggested_destination || 'text'));
      setDestinations(data.destinations || []);
    } catch (inspectError) {
      setError(inspectError.message);
    } finally {
      setInspecting(false);
    }
  }, [PRIMARY_API_URL, imageSetup, reference]);

  const selectedFileInfo = useMemo(
    () => inspection?.files?.find((file) => file.filename === selectedFile),
    [inspection, selectedFile],
  );

  useEffect(() => {
    if (selectedFileInfo?.suggested_destination) {
      setDestinationType(selectedFileInfo.suggested_destination);
    }
  }, [selectedFileInfo]);

  const startDownload = useCallback(async () => {
    if (!inspection?.repository?.id || !selectedFile) return;
    setError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/huggingface/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_id: inspection.repository.id,
          revision: inspection.revision,
          filename: selectedFile,
          filenames: selectedFileInfo?.companion_files || [selectedFile],
          destination_type: destinationType,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not start the download.');
      setDownloadJob(data);
    } catch (downloadError) {
      setError(downloadError.message);
    }
  }, [PRIMARY_API_URL, destinationType, inspection, selectedFile, selectedFileInfo]);

  const searchCivitai = useCallback(async () => {
    const query = civitaiQuery.trim();
    if (query.length < 2) return;
    setCivitaiSearching(true);
    setCivitaiModel(null);
    setError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/civitai/search?q=${encodeURIComponent(query)}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Civitai search failed.');
      setCivitaiModels(data.models || []);
    } catch (searchError) {
      setError(searchError.message);
    } finally {
      setCivitaiSearching(false);
    }
  }, [PRIMARY_API_URL, civitaiQuery]);

  const inspectCivitaiModel = useCallback(async (modelId) => {
    setCivitaiInspecting(true);
    setError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/civitai/models/${modelId}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Civitai model details could not be read.');
      setCivitaiModel(data.model);
      setCivitaiVersionId(String(data.model?.versions?.[0]?.id || ''));
    } catch (inspectError) {
      setError(inspectError.message);
    } finally {
      setCivitaiInspecting(false);
    }
  }, [PRIMARY_API_URL]);

  const selectedCivitaiVersion = useMemo(
    () => civitaiModel?.versions?.find((version) => String(version.id) === civitaiVersionId),
    [civitaiModel, civitaiVersionId],
  );

  const startCivitaiDownload = useCallback(async () => {
    if (!selectedCivitaiVersion?.file?.id) return;
    setError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/civitai/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          version_id: selectedCivitaiVersion.id,
          file_id: selectedCivitaiVersion.file.id,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Could not start the Civitai download.');
      setDownloadJob(data);
    } catch (downloadError) {
      setError(downloadError.message);
    }
  }, [PRIMARY_API_URL, selectedCivitaiVersion]);

  const startRecommendedDownload = useCallback(async (model) => {
    setError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/huggingface/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_id: model.repo_id,
          revision: 'main',
          filename: model.filename,
          filenames: model.filenames,
          destination_type: 'text',
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not start the download.');
      setDownloadJob(data);
    } catch (downloadError) {
      setError(downloadError.message);
    }
  }, [PRIMARY_API_URL]);

  const summariseModelCard = useCallback(async () => {
    if (!inspection?.repository?.id) return;
    if (!primaryModel) {
      setError('Select a working local model or API endpoint before asking for a model-card summary.');
      return;
    }
    setSummaryLoading(true);
    setCardSummary('');
    setError(null);
    try {
      const cardResponse = await fetch(
        `${PRIMARY_API_URL}/model-library/huggingface/model-card?repo_id=${encodeURIComponent(inspection.repository.id)}&revision=${encodeURIComponent(inspection.revision || 'main')}`,
      );
      const cardData = await cardResponse.json();
      if (!cardResponse.ok) throw new Error(cardData.detail || 'Could not read the current model card.');
      const response = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `Summarise this Hugging Face model card for a newcomer in at most 90 words. State what the model is for, notable capabilities, and any important limitation stated by the author. Use only the supplied text; do not infer benchmark results or compatibility.\n\nRepository: ${inspection.repository.id}\n\n${cardData.excerpt}`,
          model_name: primaryModel,
          max_tokens: 220,
          temperature: 0.2,
          stream: false,
          gpu_id: 0,
          request_purpose: 'model_library_summary',
          memoryEnabled: false,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The selected model could not summarise this card.');
      setCardSummary((data.text || '').trim());
    } catch (summaryError) {
      setError(summaryError.message);
    } finally {
      setSummaryLoading(false);
    }
  }, [PRIMARY_API_URL, inspection, primaryModel]);

  const primaryGpu = useMemo(() => {
    const gpus = gpuInfo?.gpus || [];
    return gpus.reduce((largest, gpu) => (!largest || gpu.total_mb > largest.total_mb ? gpu : largest), null);
  }, [gpuInfo]);
  const totalVramGb = primaryGpu ? primaryGpu.total_mb / 1024 : Number.NaN;
  const nanoBillingMode = settings.nanoGptBillingMode === 'subscription' ? 'subscription' : 'payg';
  const bestPickRepo = useMemo(() => (
    recommendations
      .filter((model) => {
        const modelSizeGb = Number(model.size) / (1024 ** 3);
        return Number.isFinite(totalVramGb) && ((modelSizeGb * 1.08) + 1.5) <= totalVramGb * 0.95;
      })
      .sort((left, right) => Number(right.size) - Number(left.size))[0]?.repo_id
  ), [recommendations, totalVramGb]);

  const refreshNano = useCallback(async (force = false) => {
    setNanoLoading(true);
    setError(null);
    const result = await refreshNanoGptModelsCache({ forceRefresh: force });
    setNanoModels(result.models || []);
    if (result.status === 'error') setError('NanoGPT’s model catalogue could not be reached.');
    setNanoLoading(false);
  }, []);

  const refreshNanoSubscription = useCallback(async () => {
    const apiKey = String(settings.nanoGptApiKey || '').trim();
    if (!apiKey) return;
    setNanoSubscriptionLoading(true);
    setNanoSubscriptionError('');
    try {
      const response = await fetch(`${PRIMARY_API_URL}/model-library/nanogpt/subscription-models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'NanoGPT’s subscription catalogue could not be read.');
      setNanoSubscriptionModels((data.models || []).map(normalizeNanoGptModel).filter(Boolean));
    } catch (subscriptionError) {
      setNanoSubscriptionModels([]);
      setNanoSubscriptionError(subscriptionError.message);
    } finally {
      setNanoSubscriptionLoading(false);
    }
  }, [PRIMARY_API_URL, settings.nanoGptApiKey]);

  useEffect(() => {
    if (source === 'nanogpt' && nanoModels.length === 0) refreshNano(false);
  }, [nanoModels.length, refreshNano, source]);

  useEffect(() => {
    if (source === 'nanogpt' && nanoBillingMode === 'subscription' && settings.nanoGptApiKey) {
      refreshNanoSubscription();
    }
  }, [nanoBillingMode, refreshNanoSubscription, settings.nanoGptApiKey, source]);

  const activeNanoModels = nanoBillingMode === 'subscription' ? nanoSubscriptionModels : nanoModels;

  const nanoProviders = useMemo(
    () => [...new Set(activeNanoModels.map((model) => model.provider).filter(Boolean))].sort(),
    [activeNanoModels],
  );
  const visibleNanoModels = useMemo(() => {
    const query = nanoQuery.trim().toLowerCase();
    return activeNanoModels.filter((model) => {
      if (nanoProvider !== 'all' && model.provider !== nanoProvider) return false;
      if (!query) return model.visible !== false;
      return `${model.name} ${model.id} ${model.provider} ${model.category}`.toLowerCase().includes(query);
    }).slice(0, 120);
  }, [activeNanoModels, nanoProvider, nanoQuery]);

  const useNanoModel = useCallback((model) => {
    const result = upsertHostedModelEndpoint({
      endpoints: settings.customApiEndpoints || [],
      model,
      providerId: 'nanogpt',
      apiKey: settings.nanoGptApiKey || '',
      billingMode: nanoBillingMode,
    });
    updateSettings({ customApiEndpoints: result.endpoints, modelSetupRequired: false });
    setPrimaryIsAPI(true);
    setPrimaryModel(result.endpointId);
    setActiveTab('chat');
  }, [nanoBillingMode, setActiveTab, setPrimaryIsAPI, setPrimaryModel, settings, updateSettings]);

  const selectNanoBillingMode = useCallback((mode) => {
    setNanoProvider('all');
    setNanoQuery('');
    updateSettings({ nanoGptBillingMode: mode });
    onSettingChange('nanoGptBillingMode', mode);
  }, [onSettingChange, updateSettings]);

  const refreshOpenRouter = useCallback(async (force = false) => {
    setOpenRouterLoading(true);
    setError(null);
    const result = await refreshOpenRouterModelsCache({
      forceRefresh: force,
      apiKey: settings.openRouterApiKey || '',
    });
    setOpenRouterModels(result.models || []);
    if (result.status === 'error') setError('OpenRouter’s model catalogue could not be reached.');
    setOpenRouterLoading(false);
  }, [settings.openRouterApiKey]);

  useEffect(() => {
    if (source === 'openrouter' && openRouterModels.length === 0) refreshOpenRouter(false);
  }, [openRouterModels.length, refreshOpenRouter, source]);

  const visibleOpenRouterModels = useMemo(() => {
    const query = openRouterQuery.trim().toLowerCase();
    return openRouterModels.filter((model) => {
      if (openRouterFreeOnly && !model.free) return false;
      if (!query) return true;
      return `${model.name} ${model.id} ${model.provider} ${model.description}`.toLowerCase().includes(query);
    }).slice(0, 120);
  }, [openRouterFreeOnly, openRouterModels, openRouterQuery]);

  const useOpenRouterModel = useCallback((model) => {
    const result = upsertHostedModelEndpoint({
      endpoints: settings.customApiEndpoints || [],
      model,
      providerId: 'openrouter',
      apiKey: settings.openRouterApiKey || '',
    });
    updateSettings({ customApiEndpoints: result.endpoints, modelSetupRequired: false });
    setPrimaryIsAPI(true);
    setPrimaryModel(result.endpointId);
    setActiveTab('chat');
  }, [setActiveTab, setPrimaryIsAPI, setPrimaryModel, settings, updateSettings]);

  const saveProviderKey = useCallback((key, value) => {
    onSettingChange(key, value);
    const provider = HOSTED_MODEL_PROVIDERS.find((item) => item.keySetting === key);
    updateSettings({
      [key]: value,
      ...(provider ? {
        customApiEndpoints: syncHostedProviderEndpointKey(
          settings.customApiEndpoints || [],
          provider.id,
          value,
        ),
      } : {}),
    });
  }, [onSettingChange, settings.customApiEndpoints, updateSettings]);

  const browseDestination = useCallback(async (destination) => {
    setDestinationBusy(destination.type);
    setDestinationMessage((current) => ({ ...current, [destination.type]: '' }));
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/select-directory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_directory: destinationDrafts[destination.type] || destination.path,
          title: `Choose ${destination.label} folder`,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.message || 'The folder picker could not be opened.');
      if (data.status !== 'cancelled' && data.directory) {
        setDestinationDrafts((current) => ({ ...current, [destination.type]: data.directory }));
      }
    } catch (browseError) {
      setDestinationMessage((current) => ({ ...current, [destination.type]: browseError.message }));
    } finally {
      setDestinationBusy('');
    }
  }, [PRIMARY_API_URL, destinationDrafts]);

  const saveDestination = useCallback(async (destination) => {
    const path = String(destinationDrafts[destination.type] || '').trim();
    if (!path) {
      setDestinationMessage((current) => ({ ...current, [destination.type]: 'Choose or enter a folder first.' }));
      return;
    }

    setDestinationBusy(destination.type);
    setDestinationMessage((current) => ({ ...current, [destination.type]: '' }));
    try {
      const patch = { [destination.setting_key]: path };
      if (destination.type === 'upscaler') patch.upscaler_model_directory = path;
      const response = await fetch(`${PRIMARY_API_URL}/models/update-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.message || 'Mirid could not save this folder.');

      onSettingChange(destination.setting_key, path);
      updateSettings(patch);

      const refreshRoutes = {
        text: '/models/refresh-directory',
        image: '/sd-local/refresh-directory',
        adetailer: '/sd-local/set-adetailer-directory',
        upscaler: '/models/update-upscaler-dir',
      };
      const refreshRoute = refreshRoutes[destination.type];
      let refreshWarning = '';
      if (refreshRoute) {
        const refreshResponse = await fetch(`${PRIMARY_API_URL}${refreshRoute}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ directory: path }),
        });
        const refreshData = await refreshResponse.json().catch(() => ({}));
        if (!refreshResponse.ok) {
          refreshWarning = refreshData.detail || refreshData.message || 'Its model list could not be refreshed yet.';
        }
      }

      if (destination.type === 'text' && !refreshWarning) await fetchModels();
      await loadDestinations();
      setDestinationMessage((current) => ({
        ...current,
        [destination.type]: refreshWarning ? `Folder saved. ${refreshWarning}` : 'Folder saved.',
      }));
    } catch (saveError) {
      setDestinationMessage((current) => ({ ...current, [destination.type]: saveError.message }));
    } finally {
      setDestinationBusy('');
    }
  }, [PRIMARY_API_URL, destinationDrafts, fetchModels, loadDestinations, onSettingChange, updateSettings]);

  return (
    <div className="space-y-5">
      {settings.modelSetupRequired === true && (
        <Alert>
          <HardDrive className="h-4 w-4" />
          <AlertTitle>Choose one model to start</AlertTitle>
          <AlertDescription>Download and load a local GGUF model, or connect a provider and choose one of its models. Chat opens as soon as the model is ready.</AlertDescription>
        </Alert>
      )}
      {imageSetup && (
        <Alert>
          <Image className="h-4 w-4" />
          <AlertTitle>Set up local image generation</AlertTitle>
          <AlertDescription>
            Find a compatible checkpoint, download it into Mirid's image folder, then load it. This is separate from the text model used for chat.
          </AlertDescription>
        </Alert>
      )}
      <div className="rounded-2xl border border-border/70 bg-card/60 p-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Model Library</p>
            <h3 className="mt-1 text-xl font-semibold">{imageSetup ? 'Find and install an image model.' : 'Find a model. Understand it. Put it to work.'}</h3>
            <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
              {imageSetup
                ? 'Use Hugging Face wherever it is available, or connect Civitai through its official API. Mirid keeps image models separate from chat models.'
                : 'Search Hugging Face for local files, or browse live NanoGPT and OpenRouter catalogues. Mirid keeps the technical choices visible without making you memorise the plumbing.'}
            </p>
          </div>
          <div className="flex flex-wrap rounded-lg border bg-background/50 p-1">
            <Button size="sm" variant={source === 'huggingface' ? 'default' : 'ghost'} onClick={() => setSource('huggingface')}>Hugging Face</Button>
            <Button size="sm" variant={source === 'civitai' ? 'default' : 'ghost'} onClick={() => setSource('civitai')}>Civitai</Button>
            {!imageSetup && <Button size="sm" variant={source === 'nanogpt' ? 'default' : 'ghost'} onClick={() => setSource('nanogpt')}>NanoGPT</Button>}
            {!imageSetup && <Button size="sm" variant={source === 'openrouter' ? 'default' : 'ghost'} onClick={() => setSource('openrouter')}>OpenRouter</Button>}
            {!imageSetup && <Button size="sm" variant={source === 'frontier' ? 'default' : 'ghost'} onClick={() => setSource('frontier')}>Frontier APIs</Button>}
          </div>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Model library could not complete that action</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {source === 'huggingface' ? (
        <>
          <Alert>
            <HardDrive className="h-4 w-4" />
            <AlertTitle>{imageSetup ? 'Hugging Face supplies the checkpoint; Mirid runs it' : 'Hugging Face supplies files; your computer runs them'}</AlertTitle>
            <AlertDescription>
              {imageSetup
                ? 'Start with a self-contained Safetensors or GGUF checkpoint. Some newer model families need separate text encoders or a VAE; read the model card before downloading.'
                : 'Public repositories usually need no account or token. Gated repositories need a read token and may require accepting the model author’s terms first.'}
              <span className="mt-2 flex flex-wrap gap-3">
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href={imageSetup ? 'https://huggingface.co/models?pipeline_tag=text-to-image' : 'https://huggingface.co/models'} target="_blank" rel="noreferrer">Browse models <ExternalLink className="h-3 w-3" /></a>
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">Create a read token <ExternalLink className="h-3 w-3" /></a>
              </span>
            </AlertDescription>
          </Alert>
          {!imageSetup && <details
            open={recommendationsOpen}
            onToggle={(event) => setRecommendationsOpen(event.currentTarget.open)}
            className="group rounded-2xl border border-border/70 bg-card/60"
          >
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-5">
              <div>
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  <h4 className="font-semibold">Mirid’s Picks</h4>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Current GGUF files chosen to make a sensible start on this machine.
                </p>
              </div>
              <span className="text-xs text-muted-foreground group-open:hidden">Show recommendations</span>
              <span className="hidden text-xs text-muted-foreground group-open:inline">Hide recommendations</span>
            </summary>
            <div className="space-y-4 border-t p-5">
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <p className="text-xs text-muted-foreground">
                  {primaryGpu
                    ? `${primaryGpu.name}: ${(primaryGpu.total_mb / 1024).toFixed(0)} GB VRAM, ${(primaryGpu.free_mb / 1024).toFixed(1)} GB currently free.`
                    : 'Mirid could not read VRAM, so it is showing the full shortlist.'}
                  {' '}Fit uses model size and total VRAM; long context and other services still need room.
                </p>
                <Button variant="ghost" size="sm" onClick={loadRecommendations} disabled={recommendationsLoading}>
                  <RefreshCw className={`mr-2 h-4 w-4 ${recommendationsLoading ? 'animate-spin' : ''}`} />Refresh picks
                </Button>
              </div>
              {recommendationsLoading && recommendations.length === 0 ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />Reading current Hugging Face repositories…
                </div>
              ) : (
                <div className="grid gap-3 lg:grid-cols-3">
                  {recommendations.map((model) => {
                    const fit = getVramFit(model, totalVramGb);
                    const isBestPick = model.repo_id === bestPickRepo;
                    return (
                      <div key={model.repo_id} className={`flex flex-col rounded-xl border p-4 ${isBestPick ? 'border-primary/60 bg-primary/5' : 'border-border/60 bg-background/40'}`}>
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <p className="font-medium">{model.title}</p>
                            <p className="mt-0.5 text-[11px] text-muted-foreground">{model.quantisation} · {formatBytes(model.size)}</p>
                          </div>
                          <div className="flex flex-wrap justify-end gap-1">
                            <Badge variant="outline">Hugging Face</Badge>
                            <Badge variant={fit.tone}>{isBestPick ? 'Best fit' : fit.label}</Badge>
                          </div>
                        </div>
                        {isBestPick && <p className="mt-2 text-xs font-medium text-primary">Best use of the VRAM Mirid detected.</p>}
                        {!isBestPick && <p className="mt-2 text-xs text-muted-foreground">{fit.label}</p>}
                        <p className="mt-2 flex-1 text-xs leading-relaxed text-muted-foreground">{model.reason}</p>
                        <p className="mt-3 break-all font-mono text-[10px] text-muted-foreground">{model.repo_id}</p>
                        <div className="mt-3 grid grid-cols-2 gap-2">
                          <Button size="sm" variant="outline" onClick={() => inspectReference(model.reference)}>Review files</Button>
                          <Button
                            size="sm"
                            onClick={() => startRecommendedDownload(model)}
                            disabled={['queued', 'downloading'].includes(downloadJob?.status)}
                          >
                            <Download className="mr-2 h-3.5 w-3.5" />Download
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              <p className="text-[11px] text-muted-foreground">
                Repository details and file sizes are refreshed from Hugging Face. Mirid never invokes a paid model while checking recommendations.
              </p>
            </div>
          </details>}

          <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
            <div>
              <h4 className="font-semibold">{imageSetup ? 'Search image checkpoints or paste a repository' : 'Paste a repository or search by name'}</h4>
              <p className="mt-1 text-xs text-muted-foreground">
                {imageSetup
                  ? <>Search by model name, or paste <code>owner/repository</code> from a Hugging Face model page. Check the model card for hardware needs and companion files.</>
                  : <>Examples: <code>bartowski/Qwen-GGUF</code>, a full Hugging Face URL, or <code>bartowski/Qwen-GGUF:Q4_K_M</code> to bring matching quantisations to the top.</>}
              </p>
            </div>
            <div className="flex flex-col gap-2 md:flex-row">
              <Input
                value={reference}
                onChange={(event) => setReference(event.target.value)}
                onKeyDown={(event) => { if (event.key === 'Enter') inspectReference(); }}
                placeholder={imageSetup ? 'owner/image-model' : 'owner/repository:Q4_K_M'}
                className="font-mono text-sm"
              />
              <Button onClick={() => inspectReference()} disabled={!reference.trim() || inspecting}>
                {inspecting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <HardDrive className="mr-2 h-4 w-4" />}
                Inspect repository
              </Button>
            </div>
            <div className="flex flex-col gap-2 border-t pt-4 md:flex-row">
              <Input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                onKeyDown={(event) => { if (event.key === 'Enter') searchHuggingFace(); }}
                placeholder={imageSetup ? 'Search image models — e.g. SDXL checkpoint' : 'Search Hugging Face — e.g. Qwen GGUF'}
              />
              <Button variant="outline" onClick={() => searchHuggingFace()} disabled={searchQuery.trim().length < 2 || searching}>
                {searching ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                Search
              </Button>
            </div>
            {searchResults.length > 0 && (
              <div className="grid gap-2 md:grid-cols-2">
                {searchResults.map((model) => (
                  <button
                    type="button"
                    key={model.id}
                    onClick={() => inspectReference(model.id)}
                    className="rounded-lg border border-border/60 bg-background/40 p-3 text-left transition-colors hover:bg-muted/50"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="break-all text-sm font-medium">{model.id}</span>
                      <span className="flex flex-wrap justify-end gap-1">
                        <Badge variant="outline">Hugging Face</Badge>
                        {model.gated && <Badge variant="outline">Access required</Badge>}
                      </span>
                    </div>
                    <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
                      <span>{model.downloads.toLocaleString()} downloads</span>
                      <span>{model.likes.toLocaleString()} likes</span>
                      {model.pipeline_tag && <span>{model.pipeline_tag}</span>}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {inspection && (
            <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Repository</p>
                    <Badge variant="outline">Hugging Face</Badge>
                  </div>
                  <h4 className="mt-1 break-all font-semibold">{inspection.repository.id}</h4>
                  {inspection.requested_quantisation && (
                    <p className="mt-1 text-xs text-muted-foreground">Showing <strong>{inspection.requested_quantisation}</strong> matches first. Mirid still lists the other files so nothing is hidden.</p>
                  )}
                </div>
                <div className="text-right">
                  <Button variant="outline" size="sm" onClick={summariseModelCard} disabled={summaryLoading || !primaryModel}>
                    {summaryLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
                    Summarise model card
                  </Button>
                  <p className="mt-1 max-w-xs text-[10px] text-muted-foreground">Uses your selected model once. A hosted endpoint may charge for the generation.</p>
                </div>
              </div>
              {cardSummary && (
                <Alert>
                  <Sparkles className="h-4 w-4" />
                  <AlertTitle>From the current model card</AlertTitle>
                  <AlertDescription className="leading-relaxed">{cardSummary}</AlertDescription>
                </Alert>
              )}
              {inspection.files.length === 0 ? (
                <Alert>
                  <AlertTitle>No directly installable files found</AlertTitle>
                  <AlertDescription>This repository may contain source code or a model format Mirid does not run directly.</AlertDescription>
                </Alert>
              ) : (
                <>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Model file</label>
                      <Select value={selectedFile} onValueChange={setSelectedFile}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {inspection.files.map((file) => (
                            <SelectItem key={file.filename} value={file.filename}>
                              {file.quant_match ? '★ ' : ''}{file.role === 'vision_companion' ? '[Vision companion] ' : ''}{file.filename} · {formatBytes(file.size)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {selectedFileInfo?.quantisation && <p className="text-xs text-muted-foreground">Detected quantisation: {selectedFileInfo.quantisation}</p>}
                      {selectedFileInfo?.role === 'vision_companion' && <p className="text-xs text-muted-foreground">This is a projector used alongside a compatible vision model; it is not a complete text model by itself.</p>}
                      {selectedFileInfo?.companion_files?.length > 1 && (
                        <p className="text-xs text-muted-foreground">This model is split across {selectedFileInfo.companion_files.length} shards. Mirid will install the complete set.</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Install as</label>
                      <Select value={destinationType} onValueChange={setDestinationType} disabled={imageSetup}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {destinations.map((destination) => (
                            <SelectItem key={destination.type} value={destination.type}>{destination.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="break-all text-xs text-muted-foreground">
                        {destinations.find((destination) => destination.type === destinationType)?.path}
                      </p>
                      {imageSetup && <p className="text-xs text-muted-foreground">Image setup keeps this download in the folder scanned by the local image engine.</p>}
                    </div>
                  </div>
                  <div className="flex justify-end">
                    <Button onClick={startDownload} disabled={!selectedFile || ['queued', 'downloading'].includes(downloadJob?.status)}>
                      <Download className="mr-2 h-4 w-4" />Download selected file
                    </Button>
                  </div>
                </>
              )}
            </div>
          )}

          <div className="space-y-2 rounded-2xl border border-border/70 bg-card/60 p-5">
            <div className="flex items-center justify-between gap-3">
              <label className="text-sm font-medium">Hugging Face read token <span className="font-normal text-muted-foreground">(optional)</span></label>
              <a className="text-xs text-primary hover:underline" href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">Create or manage tokens</a>
            </div>
            <Input type="password" value={settings.huggingFaceToken || ''} onChange={(event) => saveProviderKey('huggingFaceToken', event.target.value)} placeholder="hf_…" />
            <p className="text-xs text-muted-foreground">Public repositories need no token. Gated models may also require accepting the author's terms on Hugging Face.</p>
          </div>

          {downloadJob && downloadJob.provider !== 'civitai' && (
            <Alert variant={downloadJob.status === 'failed' ? 'destructive' : 'default'}>
              {['queued', 'downloading'].includes(downloadJob.status) ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              <AlertTitle>{downloadJob.status === 'complete' ? 'Model installed' : downloadJob.status === 'failed' ? 'Download failed' : 'Downloading model'}</AlertTitle>
              <AlertDescription>
                <span>{downloadJob.error || downloadJob.message}</span>
                {Number.isFinite(downloadJob.progress) && ['queued', 'downloading'].includes(downloadJob.status) && (
                  <span className="mt-2 block space-y-1">
                    <Progress value={downloadJob.progress} />
                    <span className="flex justify-between text-[11px] text-muted-foreground">
                      <span>{downloadJob.progress.toFixed(1)}%</span>
                      <span>{formatDownloadRate(downloadJob.bytes_per_second)}</span>
                    </span>
                  </span>
                )}
                {downloadJob.path && <span className="mt-1 block break-all font-mono text-xs">{downloadJob.path}</span>}
                {downloadJob.status === 'complete' && downloadJob.destination_type === 'text' && (
                  <Button className="mt-3" size="sm" onClick={loadDownloadedModel} disabled={isModelLoading}>
                    {isModelLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                    Load and open chat
                  </Button>
                )}
                {downloadJob.status === 'complete' && downloadJob.destination_type === 'image' && (
                  <Button className="mt-3" size="sm" onClick={loadDownloadedImageModel} disabled={imageModelLoading}>
                    {imageModelLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Image className="mr-2 h-4 w-4" />}
                    {imageModelLoading ? 'Loading image model…' : 'Load for image generation'}
                  </Button>
                )}
              </AlertDescription>
            </Alert>
          )}
        </>
      ) : source === 'civitai' ? (
        <div className="space-y-4 rounded-2xl border border-border/70 bg-card/60 p-5">
          <Alert>
            <Image className="h-4 w-4" />
            <AlertTitle>Civitai access depends on your region</AlertTitle>
            <AlertDescription>
              Mirid uses Civitai's official API and cannot change where the service is available. If it cannot be reached, use Hugging Face. Automatic downloads are limited to primary Safetensors or GGUF checkpoints that pass Civitai's scans.
              <span className="mt-2 flex flex-wrap gap-3">
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://civitai.com/models?types=Checkpoint" target="_blank" rel="noreferrer">Browse Civitai <ExternalLink className="h-3 w-3" /></a>
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://civitai.com/user/account" target="_blank" rel="noreferrer">Create an API key <ExternalLink className="h-3 w-3" /></a>
              </span>
            </AlertDescription>
          </Alert>

          <div className="grid gap-3 md:grid-cols-[1fr,auto]">
            <Input
              value={civitaiQuery}
              onChange={(event) => setCivitaiQuery(event.target.value)}
              onKeyDown={(event) => { if (event.key === 'Enter') searchCivitai(); }}
              placeholder="Search Civitai checkpoints"
            />
            <Button variant="outline" onClick={searchCivitai} disabled={civitaiQuery.trim().length < 2 || civitaiSearching}>
              {civitaiSearching ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
              Search
            </Button>
          </div>

          {civitaiModels.length > 0 && (
            <div className="grid gap-2 md:grid-cols-2">
              {civitaiModels.map((model) => {
                const latest = model.versions?.[0];
                return (
                  <button
                    type="button"
                    key={model.id}
                    onClick={() => inspectCivitaiModel(model.id)}
                    className="rounded-lg border border-border/60 bg-background/40 p-3 text-left transition-colors hover:bg-muted/50"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium">{model.name}</p>
                        <p className="text-xs text-muted-foreground">{model.creator ? `by ${model.creator}` : 'Creator unavailable'}</p>
                      </div>
                      <span className="flex flex-wrap justify-end gap-1">
                        <Badge variant="outline">Civitai</Badge>
                        {model.nsfw ? <Badge variant="secondary">Mature</Badge> : null}
                      </span>
                    </div>
                    {latest && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        {latest.base_model || 'Base model not listed'} · {latest.file.filename} · {formatBytes(latest.file.size)}
                      </p>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          {civitaiInspecting && (
            <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />Reading current model versions…
            </div>
          )}

          {civitaiModel && !civitaiInspecting && (
            <div className="space-y-4 rounded-xl border border-border/60 bg-background/40 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Civitai checkpoint</p>
                  <h4 className="mt-1 font-semibold">{civitaiModel.name}</h4>
                  <p className="text-xs text-muted-foreground">{civitaiModel.creator ? `by ${civitaiModel.creator}` : 'Creator unavailable'}</p>
                  <a className="mt-1 inline-flex items-center gap-1 text-xs text-primary hover:underline" href={civitaiModel.url} target="_blank" rel="noreferrer">
                    Read the model page and licence <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
                <Badge variant="outline">Scan-passed files only</Badge>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Version</label>
                <Select value={civitaiVersionId} onValueChange={setCivitaiVersionId}>
                  <SelectTrigger><SelectValue placeholder="Choose a version" /></SelectTrigger>
                  <SelectContent>
                    {civitaiModel.versions.map((version) => (
                      <SelectItem key={version.id} value={String(version.id)}>
                        {version.name} · {version.base_model || 'Base unknown'} · {formatBytes(version.file.size)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {selectedCivitaiVersion && (
                <div className="rounded-lg border border-border/50 p-3 text-xs text-muted-foreground">
                  <p className="break-all font-medium text-foreground">{selectedCivitaiVersion.file.filename}</p>
                  <p className="mt-1">{selectedCivitaiVersion.file.format}{selectedCivitaiVersion.file.precision ? ` · ${selectedCivitaiVersion.file.precision}` : ''} · {formatBytes(selectedCivitaiVersion.file.size)}</p>
                  <p className="mt-1">Virus scan: {selectedCivitaiVersion.file.virus_scan}. Pickle scan: {selectedCivitaiVersion.file.pickle_scan}.</p>
                  {selectedCivitaiVersion.trained_words?.length > 0 && <p className="mt-1">Trigger words: {selectedCivitaiVersion.trained_words.join(', ')}</p>}
                </div>
              )}
              <div className="flex justify-end">
                <Button onClick={startCivitaiDownload} disabled={!selectedCivitaiVersion || ['queued', 'downloading'].includes(downloadJob?.status)}>
                  <Download className="mr-2 h-4 w-4" />Download checkpoint
                </Button>
              </div>
            </div>
          )}

          {downloadJob && downloadJob.provider === 'civitai' && (
            <Alert variant={downloadJob.status === 'failed' ? 'destructive' : 'default'}>
              {['queued', 'downloading'].includes(downloadJob.status) ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              <AlertTitle>{downloadJob.status === 'complete' ? 'Image model installed' : downloadJob.status === 'failed' ? 'Download failed' : 'Downloading image model'}</AlertTitle>
              <AlertDescription>
                <span>{downloadJob.error || downloadJob.message}</span>
                {Number.isFinite(downloadJob.progress) && ['queued', 'downloading'].includes(downloadJob.status) && (
                  <span className="mt-2 block space-y-1">
                    <Progress value={downloadJob.progress} />
                    <span className="flex justify-between text-[11px] text-muted-foreground">
                      <span>{downloadJob.progress.toFixed(1)}%</span>
                      <span>{formatDownloadRate(downloadJob.bytes_per_second)}</span>
                    </span>
                  </span>
                )}
                {downloadJob.path && <span className="mt-1 block break-all font-mono text-xs">{downloadJob.path}</span>}
                {downloadJob.status === 'complete' && (
                  <Button className="mt-3" size="sm" onClick={loadDownloadedImageModel} disabled={imageModelLoading}>
                    {imageModelLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Image className="mr-2 h-4 w-4" />}
                    {imageModelLoading ? 'Loading image model…' : 'Load for image generation'}
                  </Button>
                )}
              </AlertDescription>
            </Alert>
          )}

          <div className="space-y-2 border-t pt-4">
            <div className="flex items-center justify-between gap-3">
              <label className="text-sm font-medium">Civitai API key <span className="font-normal text-muted-foreground">(optional)</span></label>
              <a className="text-xs text-primary hover:underline" href="https://civitai.com/user/account" target="_blank" rel="noreferrer">Create or manage keys</a>
            </div>
            <Input type="password" value={settings.civitaiApiKey || ''} onChange={(event) => saveProviderKey('civitaiApiKey', event.target.value)} placeholder="Paste your Civitai API key" />
            <p className="text-xs text-muted-foreground">Public checkpoints usually need no key. Add one only when Civitai requires your account for a download.</p>
          </div>
        </div>
      ) : source === 'nanogpt' ? (
        <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
          <div className="rounded-2xl border border-primary/40 bg-primary/5 p-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="max-w-3xl">
                <div className="flex flex-wrap items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  <h4 className="font-semibold">NanoGPT Pro</h4>
                  <Badge>Mirid recommends</Badge>
                  <Badge variant="outline">Frequent personal roleplay</Badge>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  NanoGPT currently advertises 60 million included input-token units each week and 100 included images per day. Most included text models count at 1×; models marked 2× use the allowance twice as quickly.
                </p>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                  Your normal NanoGPT API key carries the account’s subscription. Eligible calls through the standard API appear as free in NanoGPT usage. The live subscription page governs current limits, included models, price, and terms.
                </p>
              </div>
              <div className="flex shrink-0 flex-wrap gap-2 lg:max-w-sm lg:justify-end">
                <a className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline" href="https://nano-gpt.com/subscription" target="_blank" rel="noreferrer">Subscribe or manage <ExternalLink className="h-3 w-3" /></a>
                <a className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline" href="https://nano-gpt.com/api" target="_blank" rel="noreferrer">Create API key <ExternalLink className="h-3 w-3" /></a>
                <a className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline" href="https://nano-gpt.com/balance" target="_blank" rel="noreferrer">Add credit <ExternalLink className="h-3 w-3" /></a>
                <button type="button" className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline" onClick={() => navigate('/docs')}>Why Mirid recommends it <BookOpenCheck className="h-3 w-3" /></button>
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-2 border-t border-primary/20 pt-4">
              <Button size="sm" variant={nanoBillingMode === 'subscription' ? 'default' : 'outline'} onClick={() => selectNanoBillingMode('subscription')}>
                <Sparkles className="mr-2 h-4 w-4" />Show subscription-included models
              </Button>
              <Button size="sm" variant={nanoBillingMode === 'payg' ? 'default' : 'outline'} onClick={() => selectNanoBillingMode('payg')}>
                <CreditCard className="mr-2 h-4 w-4" />Show full pay-as-you-go catalogue
              </Button>
            </div>
          </div>

          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div>
              <h4 className="font-semibold">{nanoBillingMode === 'subscription' ? 'NanoGPT subscription catalogue' : 'NanoGPT model catalogue'}</h4>
              <p className="mt-1 text-xs text-muted-foreground">
                {nanoBillingMode === 'subscription'
                  ? 'Only models NanoGPT currently reports as included for this API-key account appear here.'
                  : 'Browse every public model. Paid requests draw from your NanoGPT balance.'}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => (nanoBillingMode === 'subscription' ? refreshNanoSubscription() : refreshNano(true))}
              disabled={nanoBillingMode === 'subscription' ? (!settings.nanoGptApiKey || nanoSubscriptionLoading) : nanoLoading}
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${(nanoLoading || nanoSubscriptionLoading) ? 'animate-spin' : ''}`} />Refresh catalogue
            </Button>
          </div>
          <div className="grid gap-3 md:grid-cols-[1fr,220px]">
            <Input value={nanoQuery} onChange={(event) => setNanoQuery(event.target.value)} placeholder="Search models, providers or capabilities" />
            <Select value={nanoProvider} onValueChange={setNanoProvider}>
              <SelectTrigger><SelectValue placeholder="All providers" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All providers</SelectItem>
                {nanoProviders.map((provider) => <SelectItem key={provider} value={provider}>{provider}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          {!settings.nanoGptApiKey && (
            <Alert>
              <KeyRound className="h-4 w-4" />
              <AlertTitle>Add your NanoGPT API key</AlertTitle>
              <AlertDescription>{nanoBillingMode === 'subscription' ? 'Mirid needs your key to ask NanoGPT which models your account currently includes.' : 'Browsing is public. Add a key below before using a model.'}</AlertDescription>
            </Alert>
          )}
          {nanoSubscriptionError && (
            <Alert variant="destructive">
              <AlertTitle>Subscription models could not be confirmed</AlertTitle>
              <AlertDescription>{nanoSubscriptionError} Mirid will not substitute paid models into this filtered view.</AlertDescription>
            </Alert>
          )}
          {nanoBillingMode === 'subscription' && nanoSubscriptionLoading && (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" />Reading the models included for this account…</div>
          )}
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {visibleNanoModels.map((model) => (
              <div key={model.id} className={`flex flex-col rounded-lg border p-3 ${nanoBillingMode === 'subscription' ? 'border-primary/30 bg-primary/5' : 'border-border/60 bg-background/40'}`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium" title={model.name}>{model.name}</p>
                    <p className="truncate text-xs text-muted-foreground" title={model.id}>{model.id}</p>
                  </div>
                  <div className="flex flex-wrap justify-end gap-1">
                    <Badge variant="outline">NanoGPT</Badge>
                    {nanoBillingMode === 'subscription' && <Badge>Included</Badge>}
                  </div>
                </div>
                {model.description && <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-muted-foreground">{model.description}</p>}
                <div className="mt-3 flex flex-wrap gap-1">
                  {model.capabilities?.reasoning && <Badge variant="secondary">Reasoning</Badge>}
                  {model.capabilities?.vision && <Badge variant="secondary">Vision</Badge>}
                  {model.capabilities?.pdf && <Badge variant="secondary">PDF</Badge>}
                  {model.provider && <Badge variant="secondary">Model by {model.provider}</Badge>}
                  {model.category && <Badge variant="secondary">{model.category}</Badge>}
                  {model.contextLength && <Badge variant="secondary">{Math.round(model.contextLength / 1024)}K context</Badge>}
                </div>
                <Button className="mt-4" size="sm" variant={nanoBillingMode === 'subscription' ? 'default' : 'outline'} disabled={!settings.nanoGptApiKey} onClick={() => useNanoModel(model)}>
                  {nanoBillingMode === 'subscription' ? 'Use subscription-covered model' : 'Use in Mirid'}
                </Button>
              </div>
            ))}
          </div>
          {nanoBillingMode === 'subscription' && !nanoSubscriptionLoading && settings.nanoGptApiKey && visibleNanoModels.length === 0 && !nanoSubscriptionError && (
            <p className="py-8 text-center text-sm text-muted-foreground">NanoGPT returned no subscription-included chat models for this key.</p>
          )}
          <div className="grid gap-4 border-t pt-4 md:grid-cols-2">
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-3">
                <label className="text-sm font-medium">NanoGPT API key</label>
                <a className="text-xs text-primary hover:underline" href="https://nano-gpt.com/api" target="_blank" rel="noreferrer">Create or manage keys</a>
              </div>
              <Input type="password" value={settings.nanoGptApiKey || ''} onChange={(event) => saveProviderKey('nanoGptApiKey', event.target.value)} placeholder="Paste your NanoGPT key" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Account and billing</label>
              <div className="flex h-10 items-center gap-4 rounded-md border border-input bg-background px-3 text-xs">
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://nano-gpt.com/subscription" target="_blank" rel="noreferrer">Subscription <ExternalLink className="h-3 w-3" /></a>
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://nano-gpt.com/balance" target="_blank" rel="noreferrer">Pay-as-you-go balance <ExternalLink className="h-3 w-3" /></a>
              </div>
            </div>
          </div>
        </div>
      ) : source === 'openrouter' ? (
        <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
          <div className="rounded-xl border border-primary/30 bg-primary/5 p-4">
            <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
              <div>
                <div className="flex flex-wrap items-center gap-2"><Sparkles className="h-4 w-4 text-primary" /><h4 className="font-semibold">The easiest free starting point</h4><Badge variant="outline">No model download</Badge></div>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">Create one OpenRouter key and choose Free Models Router. OpenRouter selects an available free model without allowing Mirid to silently spend credit. Add credit only when you deliberately choose paid models.</p>
              </div>
              <div className="flex shrink-0 flex-wrap gap-3 text-xs">
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://openrouter.ai/settings/keys" target="_blank" rel="noreferrer">Create key <ExternalLink className="h-3 w-3" /></a>
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://openrouter.ai/settings/credits" target="_blank" rel="noreferrer">Add credit <ExternalLink className="h-3 w-3" /></a>
                <a className="inline-flex items-center gap-1 text-primary hover:underline" href="https://openrouter.ai/models" target="_blank" rel="noreferrer">Compare models <ExternalLink className="h-3 w-3" /></a>
              </div>
            </div>
          </div>
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div>
              <h4 className="font-semibold">OpenRouter model catalogue</h4>
              <p className="mt-1 text-xs text-muted-foreground">Free routes appear first. Paid models show their published token price before you create an endpoint.</p>
            </div>
            <Button variant="outline" size="sm" onClick={() => refreshOpenRouter(true)} disabled={openRouterLoading}>
              <RefreshCw className={`mr-2 h-4 w-4 ${openRouterLoading ? 'animate-spin' : ''}`} />Refresh catalogue
            </Button>
          </div>

          <div className="flex flex-col gap-3 md:flex-row">
            <Input
              value={openRouterQuery}
              onChange={(event) => setOpenRouterQuery(event.target.value)}
              placeholder="Search models, providers or capabilities"
            />
            <Button
              type="button"
              variant={openRouterFreeOnly ? 'default' : 'outline'}
              onClick={() => setOpenRouterFreeOnly((current) => !current)}
            >
              Free models only
            </Button>
          </div>

          {!settings.openRouterApiKey && (
            <Alert>
              <Sparkles className="h-4 w-4" />
              <AlertTitle>Add your OpenRouter API key</AlertTitle>
              <AlertDescription>The catalogue is public, but even free routes require your own key.</AlertDescription>
            </Alert>
          )}

          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {visibleOpenRouterModels.map((model) => {
              const promptPrice = Number(model.pricing?.prompt);
              const completionPrice = Number(model.pricing?.completion);
              const priceLabel = model.free
                ? 'No inference charge'
                : (Number.isFinite(promptPrice) && Number.isFinite(completionPrice)
                  ? `$${(promptPrice * 1_000_000).toFixed(2)} in · $${(completionPrice * 1_000_000).toFixed(2)} out / 1M`
                  : 'Check current pricing');
              return (
                <div key={model.id} className={`flex flex-col rounded-lg border p-3 ${model.isFreeRouter ? 'border-primary/40 bg-primary/5' : 'border-border/60 bg-background/40'}`}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium" title={model.name}>{model.name}</p>
                      <p className="truncate text-xs text-muted-foreground" title={model.id}>{model.id}</p>
                    </div>
                    <div className="flex flex-wrap justify-end gap-1">
                      <Badge variant="outline">OpenRouter</Badge>
                      {model.free && <Badge>Free</Badge>}
                    </div>
                  </div>
                  {model.description && <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-muted-foreground">{model.description}</p>}
                  <div className="mt-3 flex flex-wrap gap-1">
                    {model.isFreeRouter && <Badge variant="secondary">Automatic routing</Badge>}
                    {model.capabilities?.reasoning && <Badge variant="secondary">Reasoning</Badge>}
                    {model.capabilities?.vision && <Badge variant="secondary">Vision</Badge>}
                    {model.capabilities?.tools && <Badge variant="secondary">Tools</Badge>}
                    {model.provider && model.provider !== 'openrouter' && <Badge variant="secondary">Model by {model.provider}</Badge>}
                    {model.contextLength && <Badge variant="secondary">{Math.round(model.contextLength / 1024)}K context</Badge>}
                  </div>
                  <p className="mt-3 text-[11px] text-muted-foreground">{priceLabel}</p>
                  <Button className="mt-4" size="sm" variant={model.isFreeRouter ? 'default' : 'outline'} disabled={!settings.openRouterApiKey} onClick={() => useOpenRouterModel(model)}>
                    {model.isFreeRouter ? 'Start free' : 'Use in Mirid'}
                  </Button>
                </div>
              );
            })}
          </div>

          <div className="space-y-2 border-t pt-4">
            <div className="flex items-center justify-between gap-3">
              <label className="text-sm font-medium">OpenRouter API key</label>
              <a className="text-xs text-primary hover:underline" href="https://openrouter.ai/settings/keys" target="_blank" rel="noreferrer">Create or manage keys</a>
            </div>
            <Input type="password" value={settings.openRouterApiKey || ''} onChange={(event) => saveProviderKey('openRouterApiKey', event.target.value)} placeholder="sk-or-v1-…" />
          </div>
        </div>
      ) : (
        <FrontierModelLibrary onSettingChange={onSettingChange} />
      )}

      <div id="model-folders" className={`scroll-mt-6 rounded-2xl border bg-card/60 p-5 ${imageSetup && entryIntent.focus === 'folders' ? 'border-primary/50' : 'border-border/70'}`}>
        <div className="flex items-center gap-2">
          <FolderOpen className="h-4 w-4 text-muted-foreground" />
          <h4 className="font-semibold">Model folders</h4>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">Mirid starts with sensible folders. Change any location here; new downloads will follow it.</p>
        <div className="mt-4 grid gap-2 md:grid-cols-2">
          {destinations.map((destination) => (
            <div key={destination.type} className="rounded-lg border border-border/60 bg-background/40 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">{destination.label}</span>
                <Badge variant="outline">{destination.custom ? 'Custom' : 'Preset'}</Badge>
              </div>
              <Input
                className="mt-3 font-mono text-xs"
                aria-label={`${destination.label} folder`}
                value={destinationDrafts[destination.type] ?? destination.path}
                onChange={(event) => {
                  const value = event.target.value;
                  setDestinationDrafts((current) => ({ ...current, [destination.type]: value }));
                  setDestinationMessage((current) => ({ ...current, [destination.type]: '' }));
                }}
              />
              <div className="mt-2 flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={destinationBusy === destination.type}
                  onClick={() => browseDestination(destination)}
                >
                  {destinationBusy === destination.type ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" /> : <FolderOpen className="mr-2 h-3.5 w-3.5" />}
                  Browse
                </Button>
                <Button
                  type="button"
                  size="sm"
                  disabled={destinationBusy === destination.type || (destinationDrafts[destination.type] ?? destination.path) === destination.path}
                  onClick={() => saveDestination(destination)}
                >
                  Save folder
                </Button>
              </div>
              {destinationMessage[destination.type] && (
                <p className={`mt-2 text-xs ${destinationMessage[destination.type] === 'Folder saved.' ? 'text-emerald-500' : destinationMessage[destination.type].startsWith('Folder saved.') ? 'text-amber-500' : 'text-destructive'}`}>
                  {destinationMessage[destination.type]}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelLibrary;
