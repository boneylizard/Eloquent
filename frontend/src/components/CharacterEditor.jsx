// CharacterEditor.jsx - A React component for creating and editing characters with lore and dialogue management.
// This component allows users to define character attributes, upload avatars, and manage example dialogues and lore entries. It integrates with a context for state management and provides a user-friendly interface for character creation and editing.
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch'; // <-- Ensure this line is present
import { useApp } from '../contexts/AppContext';
import { getBackendUrl } from '../config/api';
import { Trash2, PlusCircle, Upload, Download, FileJson, Image, Loader2, CheckCircle2, AlertTriangle, ChevronDown, ChevronUp, Sparkles, X } from 'lucide-react';
import { CharacterCardIntegration } from '../utils/CharacterCardUtils';
import { resolveUnifiedRequestRoute } from '../utils/requestRouting';
import {
  MAX_CHARACTER_AVATARS,
  MAX_AVATAR_FOLDER_ITEMS,
  getCharacterAvatarFolderUrls,
  analyzeAvatarFolderPick,
  avatarFolderPickDebugSample,
  prepareLocalAvatarFolderFiles,
  setCharacterAvatarFolder,
  setLocalAvatarFolderBlobs,
  clearCharacterAvatarFolder,
  revokeAvatarFolderBlobUrls,
  omitPersistedLocalAvatarFolder,
  getLocalAvatarFolderBlobUrls,
  getCharacterAvatarList,
  getManualCharacterAvatarList,
  normalizeCharacterAvatars,
  addAvatarToCharacter,
  removeAvatarAtIndex,
  setAvatarIndexOnCharacter,
  resolveAvatarDisplayUrl,
  isAllowedAvatarUpload,
  avatarUploadMaxBytes,
  isAvatarVideoUrl,
} from '../utils/characterAvatars';
import CharacterAvatarMedia from './CharacterAvatarMedia';


// Helper to ensure array type
const ensureArray = (possibleArray) => (Array.isArray(possibleArray) ? possibleArray : []);

// Define default structure reflecting the NEW simplified fields + lore
const DEFAULT_CHARACTER = {
  id: null,
  name: '',
  description: '', // Will be used for "Persona"
  model_instructions: '', // NEW field
  scenario: '',
  first_message: '', // Keep first message / greeting
  alternate_greetings: [],
  example_dialogue: [{ role: 'user', content: '' }, { role: 'character', content: '' }], // Keep example dialogue
  loreEntries: [], // NEW field for lore [{ content: string, keywords: string[] }]
  avatar: null,
  avatars: [],
  activeAvatarIndex: 0,
  created_at: '',
  speech_style: '', // NEW field
  personality: '', // Restored field
  background: '', // Restored field
  creator_notes: '',
  post_history_instructions: '',
  tags: [],
  creator: '',
  character_version: '',
  chat_role: 'npc', // NEW field
  /** Optional; injected into the system prompt when set. */
  ethics_justification: '',
};

const CharacterEditor = ({ initialCharacter = null, onSave, showLibraryList }) => {
  const isEmbedded = typeof onSave === 'function';
  const showSavedLibrary = showLibraryList ?? !isEmbedded;
  const {
    characters = [],
    activeCharacter,
    saveCharacter,
    setActiveCharacter,
    deleteCharacter,
    duplicateCharacter,
    PRIMARY_API_URL,
    storageHydrated,
    primaryModel,
    primaryIsAPI,
    settings,
  } = useApp();

  // Prefer activeCharacter (e.g. from Edit button in Saved Characters list) so that clicking Edit on a card loads that character; otherwise use initialCharacter from parent (CharacterManager)
  const effectiveCharacter = (activeCharacter !== undefined && activeCharacter !== null) ? activeCharacter : initialCharacter;

  const [isImporting, setIsImporting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const importFileRef = useRef(null);
  const avatarFolderInputRef = useRef(null);
  const avatarFolderStatusRef = useRef(null);
  const [avatarFolderUploading, setAvatarFolderUploading] = useState(false);
  const [avatarFolderProgress, setAvatarFolderProgress] = useState(null);
  const [avatarFolderStatus, setAvatarFolderStatus] = useState(null);
  const [avatarFolderErrorsExpanded, setAvatarFolderErrorsExpanded] = useState(false);
  const [avatarFolderNoUpload, setAvatarFolderNoUpload] = useState(false);
  const [character, setCharacter] = useState({ ...DEFAULT_CHARACTER });
  const [isCreatingNew, setIsCreatingNew] = useState(!effectiveCharacter);
  const [newLoreEntries, setNewLoreEntries] = useState('');
  const [batchDeleteMode, setBatchDeleteMode] = useState(false);
  const [selectedCharacterIds, setSelectedCharacterIds] = useState(() => new Set());
  const [sequencingMode, setSequencingMode] = useState(false);
  const [writingHelpEnabled, setWritingHelpEnabled] = useState(false);
  const [assistingField, setAssistingField] = useState(null);
  const [fieldSuggestions, setFieldSuggestions] = useState({});
  const [writingHelpError, setWritingHelpError] = useState('');
  const [tagsInput, setTagsInput] = useState('');

  const writingHelpRoute = useMemo(() => resolveUnifiedRequestRoute({
    primaryModel,
    primaryIsAPI,
    settings,
    requestPurpose: 'refine_character',
  }), [primaryIsAPI, primaryModel, settings]);

  /** Editor form visible when editing/creating (uses effectiveCharacter, not only activeCharacter). */
  const showEditorForm = Boolean(effectiveCharacter) || isCreatingNew;

  const publishAvatarFolderStatus = useCallback((next) => {
    setAvatarFolderStatus(next);
    requestAnimationFrame(() => {
      avatarFolderStatusRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  }, []);

  useEffect(() => {
    const input = avatarFolderInputRef.current;
    if (!input) return;
    input.setAttribute('webkitdirectory', '');
    input.setAttribute('directory', '');
  }, [showEditorForm]);

  // Effect to load character from context or initialCharacter, or reset form for create
  useEffect(() => {
    if (avatarFolderUploading) return;
    const source = effectiveCharacter;
    if (source) {
      setIsCreatingNew(!source.id);
      setCharacter({
        ...DEFAULT_CHARACTER,
        ...source,
        example_dialogue: ensureArray(source.example_dialogue).length > 0
          ? ensureArray(source.example_dialogue)
          : DEFAULT_CHARACTER.example_dialogue,
        loreEntries: ensureArray(source.loreEntries),
        ...normalizeCharacterAvatars(source),
      });
      setTagsInput(ensureArray(source.tags).join(', '));
    } else {
      setIsCreatingNew(true);
      setCharacter({ ...DEFAULT_CHARACTER });
      setTagsInput('');
    }
    setFieldSuggestions({});
    setWritingHelpError('');
  }, [effectiveCharacter, avatarFolderUploading]);

  // Effect to sync with characters list changes (for updates from duplicates, etc.)
  useEffect(() => {
    if (avatarFolderUploading) return;
    if (effectiveCharacter?.id) {
      const updatedCharacter = characters.find((c) => c.id === effectiveCharacter.id);
      if (!updatedCharacter) return;
      const localBlobs = getLocalAvatarFolderBlobUrls(character);
      const merged =
        localBlobs.length > 0
          ? normalizeCharacterAvatars({
              ...updatedCharacter,
              localAvatarFolderBlobUrls: localBlobs,
              avatarFolderLabel:
                character.avatarFolderLabel?.trim() || updatedCharacter.avatarFolderLabel,
            })
          : updatedCharacter;
      if (JSON.stringify(merged) !== JSON.stringify(effectiveCharacter)) {
        console.log('Character updated in library, refreshing editor view');
        setActiveCharacter(merged);
      }
    }
  }, [characters, effectiveCharacter, setActiveCharacter, avatarFolderUploading, character]);

  useEffect(() => {
    setSelectedCharacterIds((prev) => {
      if (!prev.size) return prev;
      const validIds = new Set((characters || []).map((c) => c.id).filter(Boolean));
      const next = new Set([...prev].filter((id) => validIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [characters]);

  // Handle importing character cards
  const handleImportCard = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsImporting(true);
    try {
      const importedCharacter = await CharacterCardIntegration.importCharacterCard(file, PRIMARY_API_URL);

      // Don't save directly to library - just load into editor for user to review and save
      const newCharacter = {
        ...importedCharacter,
        id: null, // Clear the ID so it's treated as new
        created_at: new Date().toISOString().split('T')[0]
      };

      // Clear active character and set editor to "create new" mode
      setActiveCharacter(null);
      setIsCreatingNew(true);

      // Load the imported data into the editor
      setCharacter(newCharacter);

      alert(`Character imported successfully! Please review and click "Create Character" to save to your library.`);

    } catch (error) {
      console.error('Import failed:', error);
      alert(`Import failed: ${error.message}`);
    } finally {
      setIsImporting(false);
      event.target.value = ''; // Reset file input
    }
  }, [setActiveCharacter, setIsCreatingNew, setCharacter]);

  // Handle exporting as TavernAI JSON
  const handleExportTavernJSON = useCallback(() => {
    const charToExport = activeCharacter || character;
    if (!charToExport || !charToExport.name?.trim()) {
      alert('Please select a character to export');
      return;
    }

    setIsExporting(true);
    try {
      CharacterCardIntegration.exportAsJSON(charToExport, 'tavern');
    } catch (error) {
      console.error('Export failed:', error);
      alert(`Export failed: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  }, [activeCharacter, character]);


  // Handle PNG export instructions
  // Replace handleExportPNGInstructions with this:
  const handleExportPNG = useCallback(async () => {
    const charToExport = activeCharacter || character;
    if (!charToExport || !charToExport.name?.trim()) {
      alert('Please select a character to export');
      return;
    }

    setIsExporting(true);
    try {
      await CharacterCardIntegration.exportAsPNG(charToExport, PRIMARY_API_URL);
      alert('PNG character card exported successfully!');
    } catch (error) {
      console.error('PNG export failed:', error);
      alert(`PNG export failed: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  }, [activeCharacter, character, PRIMARY_API_URL]);

  // Handle standard input/textarea changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setCharacter(prev => ({ ...prev, [name]: value }));
  };

  const requestFieldSuggestion = useCallback(async (field, fieldLabel) => {
    if (assistingField) return;
    if (!writingHelpRoute.effectiveModel) {
      setWritingHelpError('Select or load a text model before asking Mirid for writing help.');
      return;
    }

    setAssistingField(field);
    setWritingHelpError('');
    try {
      const response = await fetch(`${PRIMARY_API_URL}/character/refine-generated`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_json: {
            ...character,
            example_dialogue: ensureArray(character.example_dialogue),
            loreEntries: ensureArray(character.loreEntries),
          },
          feedback: `Suggest a stronger ${fieldLabel} for this character. Change only the ${field} field. Preserve the creator's intent, avoid generic filler, and return the complete character JSON.`,
          original_messages: [],
          model_name: writingHelpRoute.effectiveModel,
          selected_model: writingHelpRoute.selectedModel,
          frontend_round_robin_enabled: writingHelpRoute.autoEnabled,
          request_purpose: 'refine_character',
          gpu_id: 0,
        }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || result.status !== 'success' || !result.character_json) {
        throw new Error(result.detail || result.error || 'The selected model could not produce a suggestion.');
      }

      const suggestion = result.character_json[field];
      if (typeof suggestion !== 'string' || !suggestion.trim()) {
        throw new Error(`The model did not return a usable ${fieldLabel} suggestion.`);
      }
      if (suggestion.trim() === String(character[field] || '').trim()) {
        throw new Error(`The model returned the existing ${fieldLabel} unchanged.`);
      }
      setFieldSuggestions((current) => ({ ...current, [field]: suggestion.trim() }));
    } catch (error) {
      setWritingHelpError(error.message || 'Mirid could not suggest text for this field.');
    } finally {
      setAssistingField(null);
    }
  }, [PRIMARY_API_URL, assistingField, character, writingHelpRoute]);

  const acceptFieldSuggestion = useCallback((field) => {
    const suggestion = fieldSuggestions[field];
    if (!suggestion) return;
    setCharacter((draft) => ({ ...draft, [field]: suggestion }));
    setFieldSuggestions((current) => {
      const next = { ...current };
      delete next[field];
      return next;
    });
  }, [fieldSuggestions]);

  const dismissFieldSuggestion = useCallback((field) => {
    setFieldSuggestions((current) => {
      const next = { ...current };
      delete next[field];
      return next;
    });
  }, []);

  const renderWritingHelp = (field, fieldLabel) => {
    if (!writingHelpEnabled) return null;
    const suggestion = fieldSuggestions[field];
    return (
      <div className="mt-2 rounded-md border border-primary/20 bg-primary/5 p-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-muted-foreground">Mirid can suggest an alternative without replacing your text.</p>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7"
            disabled={Boolean(assistingField)}
            onClick={() => requestFieldSuggestion(field, fieldLabel)}
          >
            {assistingField === field ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Sparkles className="mr-1.5 h-3.5 w-3.5" />}
            Suggest
          </Button>
        </div>
        {suggestion ? (
          <div className="mt-2 rounded border bg-background p-3">
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{suggestion}</p>
            <div className="mt-3 flex gap-2">
              <Button type="button" size="sm" onClick={() => acceptFieldSuggestion(field)}>Use suggestion</Button>
              <Button type="button" size="sm" variant="ghost" onClick={() => dismissFieldSuggestion(field)}>Dismiss</Button>
            </div>
          </div>
        ) : null}
      </div>
    );
  };

  // --- LORE MANAGEMENT ---

  // Add a new empty lore entry
  const addLoreEntry = () => {
    setCharacter(prev => ({
      ...prev,
      loreEntries: [...ensureArray(prev.loreEntries), {
        content: '',
        keywords: [],
        keywordsInput: '' // Initialize empty input
      }]
    }));
  };

  // Remove a lore entry by index
  const removeLoreEntry = (index) => {
    setCharacter(prev => ({
      ...prev,
      loreEntries: ensureArray(prev.loreEntries).filter((_, i) => i !== index)
    }));
  };

  // Handle changes within a specific lore entry's content
  const handleLoreContentChange = (index, value) => {
    setCharacter(prev => ({
      ...prev,
      loreEntries: ensureArray(prev.loreEntries).map((entry, i) =>
        i === index ? { ...entry, content: value } : entry
      )
    }));
  };

  // STEP 1: Replace the keywords change handler with this:
  const handleLoreKeywordsChange = (index, value) => {
    // Store the raw input without processing - let user type freely
    setCharacter(prev => ({
      ...prev,
      loreEntries: ensureArray(prev.loreEntries).map((entry, i) =>
        i === index ? {
          ...entry,
          keywordsInput: value // Store raw input for display
        } : entry
      )
    }));
  };

  // STEP 2: Add a new function to process keywords on blur/save:
  const processKeywords = (index, value) => {
    const keywords = value ? value.split(',').map(k => k.trim()).filter(Boolean) : [];
    setCharacter(prev => ({
      ...prev,
      loreEntries: ensureArray(prev.loreEntries).map((entry, i) =>
        i === index ? {
          ...entry,
          keywords: keywords,
          keywordsInput: keywords.join(', ') // Clean up the display
        } : entry
      )
    }));
  };


  // --- EXAMPLE DIALOGUE MANAGEMENT ---

  const handleDialogueChange = (index, field, value) => {
    setCharacter(prev => ({
      ...prev,
      example_dialogue: ensureArray(prev.example_dialogue).map((entry, i) =>
        i === index ? { ...entry, [field]: value } : entry
      )
    }));
  };

  const addDialogueExchange = () => {
    setCharacter((current) => ({
      ...current,
      example_dialogue: [
        ...ensureArray(current.example_dialogue),
        { role: 'user', content: '' },
        { role: 'character', content: '' },
      ],
    }));
  };

  const removeDialogueTurn = (index) => {
    setCharacter((current) => ({
      ...current,
      example_dialogue: ensureArray(current.example_dialogue).filter((_, turnIndex) => turnIndex !== index),
    }));
  };

  const addAlternateGreeting = () => {
    setCharacter((current) => ({
      ...current,
      alternate_greetings: [...ensureArray(current.alternate_greetings), ''],
    }));
  };

  const updateAlternateGreeting = (index, value) => {
    setCharacter((current) => ({
      ...current,
      alternate_greetings: ensureArray(current.alternate_greetings).map((greeting, greetingIndex) => (
        greetingIndex === index ? value : greeting
      )),
    }));
  };

  const removeAlternateGreeting = (index) => {
    setCharacter((current) => ({
      ...current,
      alternate_greetings: ensureArray(current.alternate_greetings).filter((_, greetingIndex) => greetingIndex !== index),
    }));
  };

  const uploadAvatarFile = useCallback(async (file, { fromFolder = false } = {}) => {
    if (!fromFolder && !isAllowedAvatarUpload(file)) {
      throw new Error('Invalid file type. Use images (PNG, JPG, GIF, WebP) or experimental video (MP4, WebM).');
    }
    const maxBytes = avatarUploadMaxBytes(file);
    const maxSizeMB = maxBytes / (1024 * 1024);
    if (file.size > maxBytes) {
      throw new Error(`File is too large. Maximum size is ${maxSizeMB}MB.`);
    }
    const formData = new FormData();
    formData.append("file", file);
    const uploadUrl = `${PRIMARY_API_URL || getBackendUrl()}/upload_avatar`;
    const response = await fetch(uploadUrl, { method: 'POST', body: formData });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown server error' }));
      throw new Error(`Avatar upload failed: ${response.status} - ${errorData.detail || response.statusText}`);
    }
    const result = await response.json();
    if (result.status === 'success' && result.file_url) return result.file_url;
    throw new Error(result.detail || "Backend indicated upload failure.");
  }, [PRIMARY_API_URL]);

  const handleAvatarUpload = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const list = getManualCharacterAvatarList(character);
    if (list.length >= MAX_CHARACTER_AVATARS) {
      alert(`Maximum ${MAX_CHARACTER_AVATARS} individual uploads per character. Remove one or use Import avatar folder.`);
      e.target.value = null;
      return;
    }
    try {
      const fileUrl = await uploadAvatarFile(file);
      setCharacter((prev) => addAvatarToCharacter(prev, fileUrl));
    } catch (error) {
      console.error("Error uploading avatar:", error);
      alert(`Avatar upload failed: ${error.message}`);
    } finally {
      e.target.value = null;
    }
  }, [character, uploadAvatarFile]);

  const handleRemoveAvatar = useCallback((index) => {
    setCharacter((prev) => removeAvatarAtIndex(prev, index));
  }, []);

  const handleSelectAvatar = useCallback((index) => {
    setCharacter((prev) => setAvatarIndexOnCharacter(prev, index));
  }, []);

  const dismissAvatarFolderStatus = useCallback(() => {
    setAvatarFolderStatus(null);
    setAvatarFolderErrorsExpanded(false);
  }, []);

  const handleAvatarFolderSelect = useCallback(
    async (event) => {
      const fileList = event.target.files;
      const pickedCount = fileList?.length ?? 0;
      const pickedFiles = pickedCount ? Array.from(fileList) : [];
      const pickDebugNames = pickedFiles
        .slice(0, 3)
        .map((f) => f.webkitRelativePath || f.name || '(unnamed)');
      console.info(
        '[CharacterEditor] avatar folder fileList.length=',
        pickedCount,
        'first3=',
        pickDebugNames
      );

      event.target.value = '';

      if (!pickedCount) {
        publishAvatarFolderStatus({
          type: 'error',
          title: 'Folder pick empty',
          message: 'Browser returned 0 files - try selecting the folder again',
        });
        return;
      }

      setAvatarFolderErrorsExpanded(false);

      if (avatarFolderNoUpload) {
        const {
          files: localFiles,
          folderLabel,
          truncatedCount,
          junkSkipped,
        } = prepareLocalAvatarFolderFiles(pickedFiles, MAX_AVATAR_FOLDER_ITEMS);
        const startMsg = `${pickedCount} file${pickedCount === 1 ? '' : 's'} in folder — building local call-mode cycle (${localFiles.length} blob URL${localFiles.length === 1 ? '' : 's'}${junkSkipped ? `, ${junkSkipped} OS junk skipped` : ''}).`;
        setAvatarFolderUploading(true);
        publishAvatarFolderStatus({ type: 'progress', message: startMsg });
        try {
          const blobUrls = localFiles.map((file) => URL.createObjectURL(file));
          const localLabel = folderLabel ? `${folderLabel} (local)` : 'folder (local)';
          let persistedNote = ' Blob URLs last until you reload or clear the folder.';
          setCharacter((prev) => {
            revokeAvatarFolderBlobUrls(prev);
            const next = normalizeCharacterAvatars(
              setLocalAvatarFolderBlobs(prev, blobUrls, localLabel)
            );
            if (next.id && storageHydrated) {
              try {
                const saved = saveCharacter(omitPersistedLocalAvatarFolder(next));
                const merged = { ...(saved || next), localAvatarFolderBlobUrls: blobUrls };
                if (merged.id && activeCharacter?.id === merged.id) {
                  setActiveCharacter(merged);
                }
                persistedNote = ' Saved folder label; reload clears local images.';
                return merged;
              } catch (saveErr) {
                console.error('[CharacterEditor] auto-save after local folder failed', saveErr);
              }
            }
            return next;
          });
          const extra =
            truncatedCount > 0
              ? ` (all ${pickedCount} files available for call mode)`
              : '';
          const message = `Using ${pickedCount} local file${pickedCount === 1 ? '' : 's'} for call-mode cycling${extra}.${persistedNote}`;
          publishAvatarFolderStatus({
            type: 'success',
            title: 'Local folder ready',
            message,
            folderLabel: localLabel,
            importedCount: pickedCount,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          publishAvatarFolderStatus({ type: 'error', title: 'Local folder failed', message: msg });
          window.alert(`Avatar folder (local): ${msg}`);
        } finally {
          setAvatarFolderUploading(false);
          setAvatarFolderProgress(null);
        }
        return;
      }

      const analysis = analyzeAvatarFolderPick(pickedFiles, MAX_AVATAR_FOLDER_ITEMS);
      let { folderLabel, files, junkCount, truncatedCount } = analysis;
      if (!files.length) {
        files = pickedFiles.slice(0, MAX_AVATAR_FOLDER_ITEMS);
      }
      const total = files.length;
      const startMsg = `${pickedCount} file${pickedCount === 1 ? '' : 's'} found in folder — starting upload (${total} to process${junkCount ? `, ${junkCount} dotfile/junk skipped` : ''}).`;

      setAvatarFolderUploading(true);
      setAvatarFolderProgress({ current: 0, total });
      publishAvatarFolderStatus({ type: 'progress', message: startMsg });

      const uploaded = [];
      const failures = [];

      try {
        for (let i = 0; i < files.length; i += 1) {
          const file = files[i];
          setAvatarFolderProgress({ current: i + 1, total });
          publishAvatarFolderStatus({
            type: 'progress',
            message: `Uploading ${i + 1} of ${total} (${pickedCount} in folder)…`,
          });
          try {
            const url = await uploadAvatarFile(file, { fromFolder: true });
            uploaded.push(url);
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            const displayName =
              file.webkitRelativePath || file.name || `file ${i + 1}`;
            console.error('[CharacterEditor] folder avatar upload failed:', displayName, err);
            failures.push({ name: displayName, message: msg });
          }
        }

        if (!uploaded.length) {
          const debug = avatarFolderPickDebugSample(pickedFiles);
          const firstErr = failures[0]?.message || 'Upload failed for every file.';
          const looksNetwork = /failed to fetch|network|load failed/i.test(firstErr);
          const message = `${pickedCount} files were in the folder; all ${total} upload attempts failed. ${firstErr}${
            debug.sample?.length ? ` Sample: ${debug.sample.join(', ')}.` : ''
          }`;
          console.error('[CharacterEditor] avatar folder import failed:', message, failures);
          publishAvatarFolderStatus({
            type: 'error',
            title: looksNetwork ? 'Upload failed (network)' : 'Upload failed',
            message,
            failures,
          });
          window.alert(`Avatar folder import: ${message}`);
          return;
        }

        let persistedNote = '';
        setCharacter((prev) => {
          revokeAvatarFolderBlobUrls(prev);
          const next = normalizeCharacterAvatars(setCharacterAvatarFolder(prev, uploaded, folderLabel));
          if (next.id && storageHydrated) {
            try {
              const saved = saveCharacter(next);
              if (saved?.id && activeCharacter?.id === saved.id) {
                setActiveCharacter(saved);
              }
              persistedNote = ' Saved to your character library.';
              return saved || next;
            } catch (saveErr) {
              console.error('[CharacterEditor] auto-save after folder import failed', saveErr);
              persistedNote = ' Click Update Character to save folder avatars.';
            }
          } else {
            persistedNote = ' Click Save to Library to keep these avatars.';
          }
          return next;
        });

        const failedCount = failures.length;

        if (failedCount > 0) {
          const preNote = truncatedCount
            ? ` (all ${total} files processed)`
            : '';
          const message = `Imported ${uploaded.length} of ${total} from ${pickedCount} picked${preNote} — ${failedCount} failed.${persistedNote}`;
          console.warn('[CharacterEditor] avatar folder import partial:', message);
          publishAvatarFolderStatus({
            type: 'partial',
            title: 'Folder import incomplete',
            message,
            folderLabel,
            importedCount: uploaded.length,
            attemptedCount: total,
            failures,
          });
        } else {
          let extra = '';
          if (truncatedCount > 0) {
            extra = ` (${truncatedCount} over the ${MAX_AVATAR_FOLDER_ITEMS} limit were not uploaded)`;
          }
          const message = `Imported ${uploaded.length} avatar${uploaded.length === 1 ? '' : 's'} from "${folderLabel}" (${pickedCount} in folder)${extra}.${persistedNote}`;
          console.info('[CharacterEditor] avatar folder import success:', message);
          publishAvatarFolderStatus({
            type: 'success',
            title: 'Folder imported',
            message,
            folderLabel,
            importedCount: uploaded.length,
          });
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error('[CharacterEditor] avatar folder import error:', err);
        publishAvatarFolderStatus({
          type: 'error',
          title: 'Upload failed',
          message: `${pickedCount} files in folder. ${msg}`,
          failures,
        });
        window.alert(`Avatar folder import failed: ${msg}`);
      } finally {
        setAvatarFolderUploading(false);
        setAvatarFolderProgress(null);
      }
    },
    [
      uploadAvatarFile,
      publishAvatarFolderStatus,
      storageHydrated,
      saveCharacter,
      activeCharacter,
      setActiveCharacter,
      avatarFolderNoUpload,
    ]
  );

  const handleClearAvatarFolder = useCallback(() => {
    if (
      !getCharacterAvatarFolderUrls(character).length &&
      !getLocalAvatarFolderBlobUrls(character).length
    ) {
      return;
    }
    if (!window.confirm('Remove all folder-imported avatars for this character?')) return;
    setCharacter((prev) => clearCharacterAvatarFolder(prev));
    dismissAvatarFolderStatus();
  }, [character, dismissAvatarFolderStatus]);

  // Replace the handleSubmit function in CharacterEditor.jsx:
  const handleSubmit = () => {
    if (!character.name.trim()) {
      alert("Character name is required");
      return;
    }

    // Prepare the character data
    const normalizedChatRole = character.chat_role === 'user' ? 'user' : 'npc';
    const characterToSave = normalizeCharacterAvatars({
      ...character,
      tags: tagsInput.split(',').map((tag) => tag.trim()).filter(Boolean),
      chat_role: normalizedChatRole,
      loreEntries: ensureArray(character.loreEntries),
      example_dialogue: ensureArray(character.example_dialogue).length > 0
        ? ensureArray(character.example_dialogue)
        : DEFAULT_CHARACTER.example_dialogue,
    });

    try {
      // Save the character and get the saved character with its final ID
      const localBlobs = getLocalAvatarFolderBlobUrls(characterToSave);
      const savedCharacter = saveCharacter(omitPersistedLocalAvatarFolder(characterToSave));
      const merged =
        localBlobs.length > 0
          ? normalizeCharacterAvatars({
              ...savedCharacter,
              localAvatarFolderBlobUrls: localBlobs,
              avatarFolderLabel: characterToSave.avatarFolderLabel,
            })
          : savedCharacter;

      setCharacter(merged);
      setActiveCharacter(merged);
      setIsCreatingNew(false);

      console.log("Character saved via context:", merged.name, "with ID:", merged.id);
      alert("Character saved successfully!");
      if (typeof onSave === 'function') {
        onSave(merged);
      }
    } catch (error) {
      console.error("Failed to save character via context:", error);
      alert("Failed to save character.");
    }
  };
  // Action to start creating a new character
  const handleCreateNew = () => {
    setActiveCharacter(null); // Clear active character in context, useEffect will reset form
  };

  // Action to delete the currently edited character
  const handleDelete = () => {
    if (character.id && window.confirm(`Are you sure you want to delete character: ${character.name}?`)) {
      try { deleteCharacter(character.id); console.log("Character deleted:", character.name); }
      catch (error) { console.error("Failed to delete character:", error); alert("Failed to delete character."); }
    }
  };

  const toggleBatchDeleteMode = () => {
    setBatchDeleteMode((prev) => {
      const next = !prev;
      if (!next) setSelectedCharacterIds(new Set());
      return next;
    });
  };

  const toggleCharacterSelection = (characterId) => {
    if (!characterId) return;
    setSelectedCharacterIds((prev) => {
      const next = new Set(prev);
      if (next.has(characterId)) next.delete(characterId);
      else next.add(characterId);
      return next;
    });
  };

  const selectAllCharacters = () => {
    setSelectedCharacterIds(new Set((characters || []).map((c) => c.id).filter(Boolean)));
  };

  const clearSelectedCharacters = () => {
    setSelectedCharacterIds(new Set());
  };

  const handleBatchDeleteCharacters = () => {
    const ids = [...selectedCharacterIds];
    if (!ids.length) return;
    const selectedNames = (characters || [])
      .filter((c) => ids.includes(c.id))
      .map((c) => c.name || 'Unnamed Character');
    const preview = selectedNames.slice(0, 8).join(', ');
    const extra = selectedNames.length > 8 ? `, and ${selectedNames.length - 8} more` : '';
    const ok = window.confirm(
      `Delete ${ids.length} selected character${ids.length === 1 ? '' : 's'}?\n\n${preview}${extra}\n\nThis removes them from the character library.`
    );
    if (!ok) return;
    try {
      ids.forEach((id) => deleteCharacter(id));
      setSelectedCharacterIds(new Set());
      setBatchDeleteMode(false);
    } catch (error) {
      console.error("Failed to batch delete characters:", error);
      alert("Failed to delete one or more characters.");
    }
  };

  // Action to duplicate the currently edited character
  const handleDuplicate = () => {
    if (character.id) {
      try { duplicateCharacter(character.id); console.log("Character duplicated:", character.name); alert(`${character.name} duplicated.`); }
      catch (error) { console.error("Failed to duplicate character:", error); alert("Failed to duplicate character."); }
    }
  };

  return (
    <div
      className={`character-editor-root w-full space-y-4 ${isEmbedded ? 'p-0 pb-28' : 'min-h-0 p-4'}`}
    >
      {/* Header with Import/Export - ALWAYS VISIBLE */}
      <div className="flex flex-col gap-3 mb-4 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-lg font-bold sm:text-2xl">Character Management</h2>

        {/* Import/Export Controls */}
        <div className="grid grid-cols-2 gap-2 sm:flex sm:items-center sm:gap-3">
          {/* Import Section */}
          <input
            ref={importFileRef}
            type="file"
            accept=".json,.png"
            onChange={handleImportCard}
            style={{ display: 'none' }}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => importFileRef.current?.click()}
            disabled={isImporting}
            className="flex w-full items-center justify-center gap-2 sm:w-auto"
          >
            {isImporting ? (
              <>
                <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                Importing...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Import Card
              </>
            )}
          </Button>

          {/* Export Buttons - Only show if there's an active character */}

          <>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExportTavernJSON}
            disabled={isExporting}
            className="flex w-full items-center justify-center gap-2 sm:w-auto"
          >
            <FileJson className="w-4 h-4" />
            Export TavernAI
          </Button>


            <Button
              variant="outline"
              size="sm"
              onClick={handleExportPNG}
              disabled={isExporting}
              className="flex w-full items-center justify-center gap-2 sm:w-auto"
            >
              <Image className="w-4 h-4" />
              PNG Card
            </Button>
          </>

          <Button onClick={handleCreateNew} variant="default" className="w-full sm:w-auto">
            + New Character
          </Button>
        </div>
      </div>

      {/* Current Character Editing Section */}
      {showEditorForm && (
        <Card className="mb-8 w-full">
          <CardHeader>
            <CardTitle>
              {isCreatingNew ? 'Create New Character' : `Edit Character: ${character.name || ''}`}
            </CardTitle>
            <CardDescription>Write the card directly. Only the name is required; every other field should earn its place.</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-6">

              <div className="rounded-lg border bg-muted/20 p-4">
                <p className="text-sm font-medium">TavernAI and SillyTavern compatible</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  Description, personality, scenario, greetings and example messages map to the standard character-card fields. Mirid-specific fields are stored in a namespaced extension and preserved on export.
                </p>
              </div>

              <div className="rounded-lg border border-primary/25 bg-primary/5 p-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <Label htmlFor="character-writing-help" className="text-sm font-medium">Writing help</Label>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                      Optional. When enabled, suggestion buttons appear beside key fields. Your current draft is sent to the selected local model or API only when you press one.
                    </p>
                  </div>
                  <Switch
                    id="character-writing-help"
                    checked={writingHelpEnabled}
                    onCheckedChange={(checked) => {
                      setWritingHelpEnabled(checked === true);
                      if (!checked) {
                        setFieldSuggestions({});
                        setWritingHelpError('');
                      }
                    }}
                  />
                </div>
                {writingHelpError ? (
                  <div className="mt-3 flex items-start justify-between gap-3 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                    <span>{writingHelpError}</span>
                    <Button type="button" variant="ghost" size="icon" className="h-6 w-6" onClick={() => setWritingHelpError('')} aria-label="Dismiss writing help error">
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ) : null}
              </div>

              <div className="border-b pb-2">
                <h3 className="font-semibold">Identity</h3>
                <p className="mt-1 text-xs text-muted-foreground">Establish who this character is before deciding how the model should perform them.</p>
              </div>

              {/* Character Name */}
              <div>
                <Label htmlFor="name" className="block text-sm font-medium mb-1">Character Name *</Label>
                <Input id="name" name="name" value={character.name || ''} onChange={handleChange} placeholder="e.g. Professor Eldrin" required />
              </div>

              {/* Chat Role */}
              <div>
                <Label htmlFor="chat_role" className="block text-sm font-medium mb-1">Chat Role</Label>
                <select
                  id="chat_role"
                  name="chat_role"
                  value={character.chat_role === 'user' ? 'user' : 'npc'}
                  onChange={handleChange}
                  className="w-full h-9 px-3 rounded-md border border-input bg-background text-sm"
                >
                  <option value="user">User</option>
                  <option value="npc">NPC</option>
                </select>
                <p className="text-xs text-muted-foreground mt-1">Used by Multi-Role Mode to prevent AI from speaking for the user character.</p>
              </div>

              {/* Standard card Description field */}
              <div>
                <Label htmlFor="description" className="block text-sm font-medium mb-1">Description</Label>
                <Textarea
                  id="description"
                  name="description"
                  value={character.description || ''}
                  onChange={handleChange}
                  placeholder="Who are they? Include enduring facts such as appearance, role, relationships, abilities and relevant history."
                  className="h-32" // Make text area larger
                />
                <p className="text-xs text-muted-foreground mt-1">The broad character definition. This is the standard card field sometimes called persona or character description.</p>
                {renderWritingHelp('description', 'description')}
              </div>

              {/* Personality - Large Textarea */}
              <div>
                <Label htmlFor="personality" className="block text-sm font-medium mb-1">Personality</Label>
                <Textarea
                  id="personality"
                  name="personality"
                  value={character.personality || ''}
                  onChange={handleChange}
                  placeholder="Temperament, desires, fears, contradictions, boundaries and relationship habits."
                  className="h-24"
                />
                <p className="text-xs text-muted-foreground mt-1">A focused behavioural summary. Keep biography and appearance in Description.</p>
                {renderWritingHelp('personality', 'personality summary')}
              </div>

              {/* Background - Large Textarea */}
              <div>
                <Label htmlFor="background" className="block text-sm font-medium mb-1">Background / Story</Label>
                <Textarea
                  id="background"
                  name="background"
                  value={character.background || ''}
                  onChange={handleChange}
                  placeholder="Character's history, origin story, and past experiences..."
                  className="h-24"
                />
                <p className="text-xs text-muted-foreground mt-1">Optional Mirid detail. Use it for formative events that are too specific for the main Description.</p>
                {renderWritingHelp('background', 'background')}
              </div>

              <div className="border-b pb-2 pt-2">
                <h3 className="font-semibold">Voice and behaviour</h3>
                <p className="mt-1 text-xs text-muted-foreground">Use these fields to direct performance, not to repeat the character biography.</p>
              </div>

              {/* Model Instructions - Large Textarea */}
              <div>
                <Label htmlFor="model_instructions" className="block text-sm font-medium mb-1">Model Instructions</Label>
                <Textarea
                  id="model_instructions"
                  name="model_instructions"
                  value={character.model_instructions || ''}
                  onChange={handleChange}
                  placeholder="Base instructions for the LLM's behavior. E.g., 'Respond in character as [Name]. Use markdown for non-verbal actions like *smiles*. Avoid discussing forbidden topics. Keep responses under 150 words.'"
                  className="h-32" // Make text area larger
                />
                <p className="text-xs text-muted-foreground mt-1">Maps to the V2 system prompt. Leave it blank to use Mirid's normal roleplay prompt.</p>
                {renderWritingHelp('model_instructions', 'roleplay instructions')}
              </div>

              {/* Ethics justification — optional system-prompt block */}
              <div>
                <Label htmlFor="ethics_justification" className="block text-sm font-medium mb-1">
                  Ethics justification
                </Label>
                <Textarea
                  id="ethics_justification"
                  name="ethics_justification"
                  value={character.ethics_justification || ''}
                  onChange={handleChange}
                  placeholder="Ethics or deployment rationale you want included with this character (e.g. purpose, boundaries, oversight)."
                  className="h-28"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  If filled, it is added at the top of the system prompt as read-only context (not a second persona to act out).
                </p>
              </div>

              {/* Speaking Style - Large Textarea */}
              <div>
                <Label htmlFor="speech_style" className="block text-sm font-medium mb-1">Speaking Style</Label>
                <Textarea
                  id="speech_style"
                  name="speech_style"
                  value={character.speech_style || ''}
                  onChange={handleChange}
                  placeholder="Describe the character's speaking style. E.g., 'Formal, uses archaic words', 'Stutters when nervous', 'Uses lots of slang'."
                  className="h-24"
                />
                <p className="text-xs text-muted-foreground mt-1">Concrete vocabulary, rhythm, tone, verbal habits and action prose.</p>
                {renderWritingHelp('speech_style', 'speaking style')}
              </div>

              <div className="border-b pb-2 pt-2">
                <h3 className="font-semibold">Opening the chat</h3>
                <p className="mt-1 text-xs text-muted-foreground">The scenario sets the situation. The greeting demonstrates the prose, pacing and voice you want the model to continue.</p>
              </div>

              {/* Scenario - Large Textarea */}
              <div>
                <Label htmlFor="scenario" className="block text-sm font-medium mb-1">Scenario / Setting</Label>
                <Textarea
                  id="scenario"
                  name="scenario"
                  value={character.scenario || ''}
                  onChange={handleChange}
                  placeholder="Describe the context or situation for the interaction. E.g., 'The scene is a dusty, ancient library. The user is seeking a lost artifact.'"
                  className="h-24" // Make text area larger
                />
                <p className="text-xs text-muted-foreground mt-1">Where and when is this interaction taking place?</p>
                {renderWritingHelp('scenario', 'scenario')}
              </div>

              {/* First Message (Greeting) - Textarea */}
              <div>
                <Label htmlFor="first_message" className="block text-sm font-medium mb-1">Greeting Message</Label>
                <Textarea
                  id="first_message"
                  name="first_message"
                  value={character.first_message || ''}
                  onChange={handleChange}
                  className="h-20"
                  placeholder="The first message the character says when a chat starts. E.g., *You enter the dimly lit library. Professor Eldrin looks up from a large tome.* 'Ah, welcome seeker. What knowledge do you pursue today?'"
                />
                <p className="text-xs text-muted-foreground mt-1">This strongly influences the length and style of later replies. Markdown and <code>{'{{user}}'}</code>/<code>{'{{char}}'}</code> placeholders are supported.</p>
                {renderWritingHelp('first_message', 'opening message')}
              </div>

              <div>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <Label className="block text-sm font-medium">Alternate greetings</Label>
                    <p className="mt-1 text-xs text-muted-foreground">Optional V2 opening-message variants that compatible apps can present as swipes.</p>
                  </div>
                  <Button type="button" variant="outline" size="sm" onClick={addAlternateGreeting}>
                    <PlusCircle className="mr-2 h-4 w-4" /> Add greeting
                  </Button>
                </div>
                <div className="space-y-3">
                  {ensureArray(character.alternate_greetings).map((greeting, index) => (
                    <div key={`alternate-greeting-${index}`} className="relative rounded-md border bg-muted/20 p-3 pr-11">
                      <Label htmlFor={`alternate-greeting-${index}`} className="text-xs font-medium">Greeting {index + 2}</Label>
                      <Textarea
                        id={`alternate-greeting-${index}`}
                        value={greeting || ''}
                        onChange={(event) => updateAlternateGreeting(index, event.target.value)}
                        className="mt-1 min-h-[88px]"
                        placeholder="Another way this chat could begin…"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="absolute right-2 top-2 h-7 w-7 text-muted-foreground hover:text-destructive"
                        onClick={() => removeAlternateGreeting(index)}
                        aria-label={`Remove alternate greeting ${index + 2}`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>

              <div className="border-b pb-2 pt-2">
                <h3 className="font-semibold">Example messages</h3>
                <p className="mt-1 text-xs text-muted-foreground">Teach by demonstration. Add as many short exchanges as needed to establish voice, formatting and conversational boundaries.</p>
              </div>

              {/* Example Dialogue Section */}
              <div>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <Label className="block text-sm font-medium">Example dialogue</Label>
                  <Button type="button" variant="outline" size="sm" onClick={addDialogueExchange}>
                    <PlusCircle className="mr-2 h-4 w-4" /> Add exchange
                  </Button>
                </div>
                <Card className="p-4 bg-muted/30">
                  <div className="space-y-3">
                    {ensureArray(character.example_dialogue).map((turn, index) => (
                      <div key={index} className="relative space-y-2 rounded-md border bg-background/70 p-3 pr-11">
                        <Label htmlFor={`dialogue-${index}-role`} className="text-xs font-semibold">Speaker</Label>
                        <select
                          id={`dialogue-${index}-role`}
                          value={turn.role === 'user' ? 'user' : 'character'}
                          onChange={(event) => handleDialogueChange(index, 'role', event.target.value)}
                          className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                        >
                          <option value="user">User</option>
                          <option value="character">Character</option>
                        </select>
                        <Label htmlFor={`dialogue-${index}-content`} className="text-xs font-semibold">Message</Label>
                        <Textarea
                          id={`dialogue-${index}-content`}
                          value={turn.content || ''}
                          onChange={(e) => handleDialogueChange(index, 'content', e.target.value)}
                          placeholder={turn.role === 'user' ? 'Example user input...' : 'Example character response...'}
                          className="h-16 text-sm" // Smaller text area for examples
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="absolute right-2 top-2 h-7 w-7 text-muted-foreground hover:text-destructive"
                          onClick={() => removeDialogueTurn(index)}
                          aria-label={`Remove example message ${index + 1}`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                    {ensureArray(character.example_dialogue).length === 0 ? (
                      <p className="py-4 text-center text-sm text-muted-foreground">No example messages yet.</p>
                    ) : null}
                  </div>
                </Card>
                <p className="text-xs text-muted-foreground mt-1">Exports to the standard <code>mes_example</code> field using <code>{'{{user}}'}</code> and <code>{'{{char}}'}</code>.</p>
              </div>

              <details className="rounded-lg border bg-muted/10 p-4">
                <summary className="cursor-pointer text-sm font-medium">Advanced character-card fields</summary>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">These V2 fields are preserved on import and export. Most characters do not need to set them.</p>
                <div className="mt-4 space-y-4">
                  <div>
                    <Label htmlFor="post_history_instructions">Post-history instructions</Label>
                    <Textarea
                      id="post_history_instructions"
                      name="post_history_instructions"
                      value={character.post_history_instructions || ''}
                      onChange={handleChange}
                      className="mt-1 min-h-[88px]"
                      placeholder="Instructions applied after chat history by compatible clients."
                    />
                  </div>
                  <div>
                    <Label htmlFor="creator_notes">Creator notes</Label>
                    <Textarea
                      id="creator_notes"
                      name="creator_notes"
                      value={character.creator_notes || ''}
                      onChange={handleChange}
                      className="mt-1 min-h-[88px]"
                      placeholder="Notes for people using the card. This is metadata, not a model prompt."
                    />
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <Label htmlFor="creator">Creator</Label>
                      <Input id="creator" name="creator" value={character.creator || ''} onChange={handleChange} className="mt-1" />
                    </div>
                    <div>
                      <Label htmlFor="character_version">Character version</Label>
                      <Input id="character_version" name="character_version" value={character.character_version || ''} onChange={handleChange} className="mt-1" placeholder="e.g. 1.0" />
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="character_tags">Tags</Label>
                    <Input
                      id="character_tags"
                      value={tagsInput}
                      onChange={(event) => setTagsInput(event.target.value)}
                      onBlur={() => setCharacter((current) => ({
                        ...current,
                        tags: tagsInput.split(',').map((tag) => tag.trim()).filter(Boolean),
                      }))}
                      className="mt-1"
                      placeholder="fantasy, detective, slow burn"
                    />
                  </div>
                </div>
              </details>

              <div className="border-b pb-2 pt-2">
                <h3 className="font-semibold">Presentation</h3>
                <p className="mt-1 text-xs text-muted-foreground">Avatars affect display and call mode, not the written character definition.</p>
              </div>

              {/* Avatars (up to 10 manual + optional folder) */}
              <div>
                <Label htmlFor="avatar" className="block text-sm font-medium mb-1">
                  Avatars ({getCharacterAvatarList(character).length}
                  {getCharacterAvatarFolderUrls(character).length ||
                  getLocalAvatarFolderBlobUrls(character).length
                    ? ` · folder ${
                        getLocalAvatarFolderBlobUrls(character).length ||
                        getCharacterAvatarFolderUrls(character).length
                      }`
                    : ''}
                  )
                </Label>
                <p className="text-xs text-muted-foreground mb-2">
                  Upload up to {MAX_CHARACTER_AVATARS} individual images or videos, or import a whole folder (unlimited) for call-mode cycling. Scroll wheel or [ ] switches looks in call mode.
                </p>
                <Input
                  id="avatar"
                  type="file"
                  accept="image/*,video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov"
                  onChange={handleAvatarUpload}
                  disabled={getManualCharacterAvatarList(character).length >= MAX_CHARACTER_AVATARS}
                  className="mb-2"
                />
                <input
                  ref={avatarFolderInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  disabled={avatarFolderUploading}
                  onChange={handleAvatarFolderSelect}
                  {...{ webkitdirectory: '', directory: '' }}
                />
                <label className="flex items-center gap-2 text-xs text-muted-foreground mb-2 cursor-pointer">
                  <input
                    type="checkbox"
                    className="rounded border-input"
                    checked={avatarFolderNoUpload}
                    disabled={avatarFolderUploading}
                    onChange={(e) => setAvatarFolderNoUpload(e.target.checked)}
                  />
                  Use folder for call mode (no upload) — local blob URLs, this session only
                </label>
                <div className="flex flex-wrap gap-2 mb-2 items-center">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={avatarFolderUploading}
                    onClick={() => avatarFolderInputRef.current?.click()}
                  >
                    {avatarFolderUploading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        {avatarFolderProgress
                          ? `Uploading ${avatarFolderProgress.current}/${avatarFolderProgress.total}…`
                          : 'Uploading…'}
                      </>
                    ) : getCharacterAvatarFolderUrls(character).length ||
                      getLocalAvatarFolderBlobUrls(character).length ? (
                      'Change avatar folder'
                    ) : (
                      'Import avatar folder'
                    )}
                  </Button>
                  {getCharacterAvatarFolderUrls(character).length ||
                  getLocalAvatarFolderBlobUrls(character).length ? (
                    <span className="text-xs text-muted-foreground self-center truncate max-w-[280px]">
                      {character.avatarFolderLabel ? (
                        <>
                          <span className="font-medium text-foreground">{character.avatarFolderLabel}</span>
                          {' · '}
                        </>
                      ) : null}
                      {getLocalAvatarFolderBlobUrls(character).length ||
                        getCharacterAvatarFolderUrls(character).length}{' '}
                      folder avatar
                      {(getLocalAvatarFolderBlobUrls(character).length ||
                        getCharacterAvatarFolderUrls(character).length) === 1
                        ? ''
                        : 's'}{' '}
                      (call-mode cycle)
                    </span>
                  ) : null}
                  {getCharacterAvatarFolderUrls(character).length ||
                  getLocalAvatarFolderBlobUrls(character).length ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      disabled={avatarFolderUploading}
                      onClick={handleClearAvatarFolder}
                    >
                      Clear folder
                    </Button>
                  ) : null}
                </div>
                <div
                  ref={avatarFolderStatusRef}
                  role="status"
                  aria-live="polite"
                  aria-atomic="true"
                  className="mb-3 min-h-[2px]"
                >
                  {avatarFolderUploading || avatarFolderStatus ? (
                    <div
                      className={`rounded-md border-2 px-3 py-2 text-sm shadow-sm ${
                        avatarFolderUploading || avatarFolderStatus?.type === 'progress'
                          ? 'border-primary bg-primary/15 text-foreground'
                          : avatarFolderStatus?.type === 'success'
                            ? 'border-green-600 bg-green-600/15 text-foreground'
                            : avatarFolderStatus?.type === 'partial'
                              ? 'border-amber-600 bg-amber-500/20 text-foreground'
                              : 'border-destructive bg-destructive/15 text-destructive'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {avatarFolderUploading || avatarFolderStatus?.type === 'progress' ? (
                          <Loader2 className="h-4 w-4 shrink-0 animate-spin mt-0.5" />
                        ) : avatarFolderStatus?.type === 'success' ? (
                          <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600 mt-0.5" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                        )}
                        <div className="flex-1 min-w-0 space-y-1">
                          <p className="font-semibold leading-snug">
                            {avatarFolderUploading
                              ? 'Uploading avatar folder…'
                              : avatarFolderStatus?.title || 'Avatar folder'}
                          </p>
                          <p className="text-sm leading-snug break-words">
                            {avatarFolderUploading
                              ? avatarFolderProgress
                                ? `Uploading ${avatarFolderProgress.current} of ${avatarFolderProgress.total}…`
                                : avatarFolderStatus?.message || 'Starting upload…'
                              : avatarFolderStatus?.message}
                          </p>
                          {avatarFolderStatus?.type === 'partial' && avatarFolderStatus.failures?.length ? (
                            <div>
                              <button
                                type="button"
                                className="inline-flex items-center gap-1 text-xs font-medium underline"
                                onClick={() => setAvatarFolderErrorsExpanded((v) => !v)}
                              >
                                {avatarFolderErrorsExpanded ? (
                                  <ChevronUp className="h-3 w-3" />
                                ) : (
                                  <ChevronDown className="h-3 w-3" />
                                )}
                                {avatarFolderErrorsExpanded ? 'Hide' : 'Show'} failed files (
                                {avatarFolderStatus.failures.length})
                              </button>
                              {avatarFolderErrorsExpanded ? (
                                <ul className="mt-2 max-h-32 overflow-y-auto text-xs space-y-1 list-disc pl-4">
                                  {avatarFolderStatus.failures.map((f) => (
                                    <li key={`${f.name}-${f.message}`}>
                                      <span className="font-medium">{f.name}</span>
                                      {f.message ? `: ${f.message}` : null}
                                    </li>
                                  ))}
                                </ul>
                              ) : (
                                <p className="text-xs opacity-90 mt-1">
                                  First error: {avatarFolderStatus.failures[0].message}
                                </p>
                              )}
                            </div>
                          ) : null}
                          {avatarFolderStatus?.type === 'error' && avatarFolderStatus.failures?.length > 1 ? (
                            <ul className="text-xs list-disc pl-4 max-h-24 overflow-y-auto">
                              {avatarFolderStatus.failures.slice(0, 5).map((f) => (
                                <li key={`${f.name}-${f.message}`}>
                                  {f.name}: {f.message}
                                </li>
                              ))}
                            </ul>
                          ) : null}
                        </div>
                        {!avatarFolderUploading && avatarFolderStatus ? (
                          <Button type="button" variant="ghost" size="sm" className="shrink-0" onClick={dismissAvatarFolderStatus}>
                            Dismiss
                          </Button>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </div>
                <div className="flex flex-wrap gap-3">
                  {getCharacterAvatarList(character).slice(0, 50).map((url, index) => {
                    const isActive = (character.activeAvatarIndex ?? 0) === index;
                    const previewUrl = resolveAvatarDisplayUrl(url, PRIMARY_API_URL || getBackendUrl());
                    return (
                      <div key={`${url}-${index}`} className="relative group">
                        <button
                          type="button"
                          onClick={() => handleSelectAvatar(index)}
                          className={`block rounded-full overflow-hidden border-2 transition-all ${isActive ? 'border-primary ring-2 ring-primary/40' : 'border-border hover:border-primary/50'}`}
                          title={isActive ? 'Active avatar' : 'Set as active avatar'}
                        >
                          <CharacterAvatarMedia
                            url={previewUrl}
                            alt={`Avatar ${index + 1}`}
                            className="w-16 h-16 object-cover"
                            videoKey={`${index}-${url}`}
                            onError={(e) => {
                              const el = e?.currentTarget;
                              if (el) el.style.display = 'none';
                            }}
                          />
                          {isAvatarVideoUrl(url) ? (
                            <span className="absolute bottom-0 left-0 right-0 text-[9px] text-center bg-black/55 text-white py-0.5">
                              video
                            </span>
                          ) : null}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleRemoveAvatar(index)}
                          className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-destructive text-destructive-foreground text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                          aria-label="Remove avatar"
                        >
                          ×
                        </button>
                      </div>
                    );
                  })}
                  {getCharacterAvatarList(character).length > 50 && (
                    <div className="text-xs text-muted-foreground self-center">
                      +{getCharacterAvatarList(character).length - 50} more (call mode only)
                    </div>
                  )}
                </div>
              </div>

            </div>
          </CardContent>
        </Card>
      )}

      {/* World Lore / Context Section */}
      {showEditorForm && (
        <Card className="mb-8 w-full">
          <CardHeader>
            <CardTitle>World Lore / Context Entries</CardTitle>
            <CardDescription>Define specific facts or rules triggered by keywords during chat.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {ensureArray(character.loreEntries).map((entry, index) => (
              <Card key={index} className="p-4 relative bg-muted/50">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => removeLoreEntry(index)}
                  className="absolute top-2 right-2 h-6 w-6 text-muted-foreground hover:text-destructive"
                  aria-label="Remove Lore Entry"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
                <div className="space-y-2">
                  <div>
                    <Label htmlFor={`lore-content-${index}`} className="text-xs font-semibold">Lore Content:</Label>
                    <Textarea
                      id={`lore-content-${index}`}
                      value={entry.content || ''}
                      onChange={(e) => handleLoreContentChange(index, e.target.value)}
                      placeholder="Enter the fact, rule, or piece of lore..."
                      className="h-20" // Adjust height as needed
                    />
                  </div>
                  <div>
                    <Label htmlFor={`lore-keywords-${index}`} className="text-xs font-semibold">Trigger Keywords (comma-separated):</Label>
                    <Input
                      id={`lore-keywords-${index}`}
                      value={entry.keywordsInput !== undefined ? entry.keywordsInput : ensureArray(entry.keywords).join(', ')}
                      onChange={(e) => handleLoreKeywordsChange(index, e.target.value)}
                      onBlur={(e) => processKeywords(index, e.target.value)}
                      placeholder="e.g., castle, king, prophecy"
                    />
                  </div>
                </div>
              </Card>
            ))}
            <Button onClick={addLoreEntry} variant="outline" size="sm">
              <PlusCircle className="mr-2 h-4 w-4" /> Add Lore Entry
            </Button>
          </CardContent>
        </Card>
      )}

      {showEditorForm && !isEmbedded && (
        <div className="hidden md:flex flex-wrap gap-2 pt-4 justify-center">
          <Button onClick={handleSubmit} className="bg-green-600 hover:bg-green-700 min-w-[150px]">
            {isCreatingNew ? 'Create Character' : 'Update Character'}
          </Button>
          {!isCreatingNew && character.id && (
            <>
              <Button onClick={handleDelete} variant="destructive" size="sm" className="min-w-[150px]">
                Delete Character
              </Button>
              <Button onClick={handleDuplicate} variant="secondary" size="sm" className="min-w-[150px]">
                Duplicate Character
              </Button>
            </>
          )}
        </div>
      )}

      {showEditorForm && (
        <div
          className="character-editor-sticky-actions fixed inset-x-0 bottom-0 z-40 border-t border-border bg-background/95 backdrop-blur-md px-4 py-3 shadow-[0_-4px_24px_rgba(0,0,0,0.15)] md:static md:z-auto md:border-0 md:bg-transparent md:backdrop-blur-none md:shadow-none md:px-0 md:py-0 md:mt-4"
          style={{ paddingBottom: 'max(0.75rem, env(safe-area-inset-bottom))' }}
        >
          <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-center gap-2">
            <Button
              onClick={handleSubmit}
              disabled={!storageHydrated}
              className="bg-green-600 hover:bg-green-700 min-w-[min(100%,180px)] flex-1 sm:flex-none"
            >
              {!storageHydrated
                ? 'Loading library…'
                : isCreatingNew
                  ? 'Save to Library'
                  : 'Update Character'}
            </Button>
            {!isCreatingNew && character.id && (
              <>
                <Button onClick={handleDuplicate} variant="secondary" size="sm" className="min-w-[120px]">
                  Duplicate
                </Button>
                <Button onClick={handleDelete} variant="destructive" size="sm" className="min-w-[100px]">
                  Delete
                </Button>
              </>
            )}
          </div>
        </div>
      )}

      {showSavedLibrary && (
      <div className="mt-12">
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-xl font-bold">Saved Characters</h3>
            {Array.isArray(characters) && characters.length > 0 ? (
              <p className="text-xs text-muted-foreground">
                {characters.length} saved · {selectedCharacterIds.size} selected
              </p>
            ) : null}
          </div>
          {Array.isArray(characters) && characters.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              <Button type="button" size="sm" variant="outline" onClick={toggleBatchDeleteMode}>
                {batchDeleteMode ? 'Cancel selection' : 'Select to delete'}
              </Button>
              {batchDeleteMode ? (
                <>
                  <Button type="button" size="sm" variant="ghost" onClick={selectAllCharacters}>
                    Select all
                  </Button>
                  <Button type="button" size="sm" variant="ghost" onClick={clearSelectedCharacters} disabled={selectedCharacterIds.size === 0}>
                    Clear
                  </Button>
                  <Button type="button" size="sm" variant="destructive" onClick={handleBatchDeleteCharacters} disabled={selectedCharacterIds.size === 0}>
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete selected ({selectedCharacterIds.size})
                  </Button>
                </>
              ) : null}
            </div>
          ) : null}
        </div>
        {Array.isArray(characters) && characters.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
            {characters.map((char) => {
              const isSelected = selectedCharacterIds.has(char.id);
              return (
              <Card
                key={char.id}
                className={`overflow-hidden flex flex-col hover:shadow-md transition-shadow ${isSelected ? 'ring-2 ring-destructive border-destructive' : ''}`}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-start space-x-3">
                    {batchDeleteMode ? (
                      <input
                        type="checkbox"
                        className="mt-3 h-4 w-4 shrink-0 rounded border-input"
                        checked={isSelected}
                        onChange={() => toggleCharacterSelection(char.id)}
                        aria-label={`Select ${char.name || 'unnamed character'} for deletion`}
                      />
                    ) : null}
                    {char.avatar && (
                      <img
                        src={resolveAvatarDisplayUrl(char.avatar, PRIMARY_API_URL || getBackendUrl())}
                        alt={`${char.name || 'Character'} Avatar`}
                        className="w-10 h-10 rounded-full object-cover flex-shrink-0 border border-border"
                        onError={(e) => {
                          console.error("Failed to load image:", e.target.src);
                          e.target.style.display = 'none';
                        }}
                      />
                    )}
                    <div className="flex-grow overflow-hidden">
                      <CardTitle className="text-lg truncate" title={char.name}>
                        {char.name || 'Unnamed Character'}
                      </CardTitle>
                      <p className="text-sm text-muted-foreground truncate" title={char.description}>
                        {char.description || 'No persona description'}
                      </p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="flex-grow flex flex-col justify-end">
                  <div className="flex space-x-2 mt-4">
                    <Button
                      size="sm"
                      onClick={() => {
                        if (batchDeleteMode) {
                          toggleCharacterSelection(char.id);
                          return;
                        }
                        setActiveCharacter(char);
                      }}
                      variant="outline"
                    >
                      {batchDeleteMode ? (isSelected ? 'Selected' : 'Select') : 'Edit'}
                    </Button>

                    {/* Individual Export Button */}
                    {!batchDeleteMode ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          setIsExporting(true);
                          try {
                            CharacterCardIntegration.exportAsJSON(char, 'tavern');
                          } catch (error) {
                            console.error('Export failed:', error);
                            alert(`Export failed: ${error.message}`);
                          } finally {
                            setIsExporting(false);
                          }
                        }}
                        disabled={isExporting}
                        title="Export this character"
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-muted-foreground mb-4">
              No characters saved yet.
            </p>
            <Button onClick={handleCreateNew} variant="default">
              Create Your First Character
            </Button>
          </div>
        )}
      </div>
      )}
    </div>
  );
};

export default CharacterEditor;
