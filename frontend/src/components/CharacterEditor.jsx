// CharacterEditor.jsx - A React component for creating and editing characters with lore and dialogue management.
// This component allows users to define character attributes, upload avatars, and manage example dialogues and lore entries. It integrates with a context for state management and provides a user-friendly interface for character creation and editing.
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label'; // <-- Ensure this line is present
import { useApp } from '../contexts/AppContext';
import { getBackendUrl } from '../config/api';
import { Trash2, PlusCircle, Upload, Download, FileJson, Image } from 'lucide-react';
import { CharacterCardIntegration } from '../utils/CharacterCardUtils';

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
  example_dialogue: [{ role: 'user', content: '' }, { role: 'character', content: '' }], // Keep example dialogue
  loreEntries: [], // NEW field for lore [{ content: string, keywords: string[] }]
  avatar: null,
  created_at: '',
  speech_style: '', // NEW field
  // Removed: personality, background, tags (unless needed elsewhere)
};

const CharacterEditor = () => {
  const {
    characters = [],
    activeCharacter,
    saveCharacter,
    setActiveCharacter,
    deleteCharacter,
    duplicateCharacter,
    PRIMARY_API_URL,
  } = useApp();

  const [isImporting, setIsImporting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const importFileRef = useRef(null);
  const [character, setCharacter] = useState({ ...DEFAULT_CHARACTER });
  const [isCreatingNew, setIsCreatingNew] = useState(!activeCharacter);
  const [newLoreEntries, setNewLoreEntries] = useState(''); // Temp state for adding keywords

  // Effect to load active character or reset form
  useEffect(() => {
    if (activeCharacter) {
      setIsCreatingNew(false);
      setCharacter({
        ...DEFAULT_CHARACTER,
        ...activeCharacter,
        // Ensure arrays exist even if loaded data is missing them
        example_dialogue: ensureArray(activeCharacter.example_dialogue).length > 0
          ? ensureArray(activeCharacter.example_dialogue)
          : DEFAULT_CHARACTER.example_dialogue,
        loreEntries: ensureArray(activeCharacter.loreEntries),
        avatar: activeCharacter.avatar || null,
      });
    } else {
      setIsCreatingNew(true);
      setCharacter({ ...DEFAULT_CHARACTER });
    }
  }, [activeCharacter]);

  // Effect to sync with characters list changes (for updates from duplicates, etc.)
  useEffect(() => {
    if (activeCharacter && activeCharacter.id) {
      const updatedCharacter = characters.find(c => c.id === activeCharacter.id);
      if (updatedCharacter && JSON.stringify(updatedCharacter) !== JSON.stringify(activeCharacter)) {
        console.log('Character updated in library, refreshing editor view');
        setActiveCharacter(updatedCharacter);
      }
    }
  }, [characters, activeCharacter, setActiveCharacter]);

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


  // --- EXAMPLE DIALOGUE MANAGEMENT --- (Basic for now)

  const handleDialogueChange = (index, field, value) => {
    setCharacter(prev => ({
      ...prev,
      example_dialogue: ensureArray(prev.example_dialogue).map((entry, i) =>
        i === index ? { ...entry, [field]: value } : entry
      )
    }));
  };

  // --- AVATAR UPLOAD ---
  const handleAvatarUpload = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const allowedTypes = ["image/png", "image/jpeg", "image/gif", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
      alert(`Invalid file type. Please select: ${allowedTypes.join(', ')}`);
      return;
    }

    const maxSizeMB = 5;
    if (file.size > maxSizeMB * 1024 * 1024) {
      alert(`File is too large. Maximum size is ${maxSizeMB}MB.`);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("🧠 Uploading avatar to backend...");
      const uploadUrl = `${PRIMARY_API_URL || getBackendUrl()}/upload_avatar`;
      const response = await fetch(uploadUrl, { method: 'POST', body: formData });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown server error' }));
        throw new Error(`Avatar upload failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();
      if (result.status === 'success' && result.file_url) {
        console.log("🧠 Avatar uploaded successfully. URL:", result.file_url);
        setCharacter(prev => ({ ...prev, avatar: result.file_url }));
        alert("Avatar uploaded successfully!");
      } else {
        throw new Error(result.detail || "Backend indicated upload failure.");
      }
    } catch (error) {
      console.error("Error uploading avatar:", error);
      alert(`Avatar upload failed: ${error.message}`);
    } finally {
      e.target.value = null;
    }
  }, [setCharacter, PRIMARY_API_URL]);

  // Replace the handleSubmit function in CharacterEditor.jsx:
  const handleSubmit = () => {
    if (!character.name.trim()) {
      alert("Character name is required");
      return;
    }

    // Prepare the character data
    const characterToSave = {
      ...character,
      loreEntries: ensureArray(character.loreEntries),
      example_dialogue: ensureArray(character.example_dialogue).length > 0
        ? ensureArray(character.example_dialogue)
        : DEFAULT_CHARACTER.example_dialogue,
    };

    try {
      // Save the character and get the saved character with its final ID
      const savedCharacter = saveCharacter(characterToSave);

      // Update our local state with the complete character data (including ID)
      setCharacter(savedCharacter);

      // Set as active character and switch to edit mode
      setActiveCharacter(savedCharacter);
      setIsCreatingNew(false);

      console.log("Character saved via context:", savedCharacter.name, "with ID:", savedCharacter.id);
      alert("Character saved successfully!");
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

  // Action to duplicate the currently edited character
  const handleDuplicate = () => {
    if (character.id) {
      try { duplicateCharacter(character.id); console.log("Character duplicated:", character.name); alert(`${character.name} duplicated.`); }
      catch (error) { console.error("Failed to duplicate character:", error); alert("Failed to duplicate character."); }
    }
  };

  return (
    <div className="w-full min-h-screen p-4 space-y-4">
      {/* Header with Import/Export - ALWAYS VISIBLE */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">Character Management</h2>

        {/* Import/Export Controls */}
        <div className="flex items-center gap-3">
          {/* Import Section */}
          <input
            ref={importFileRef}
            type="file"
            accept=".json,.png,.webp,.jpg,.jpeg"
            onChange={handleImportCard}
            style={{ display: 'none' }}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => importFileRef.current?.click()}
            disabled={isImporting}
            className="flex items-center gap-2"
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
              className="flex items-center gap-2"
            >
              <FileJson className="w-4 h-4" />
              Export TavernAI
            </Button>


            <Button
              variant="outline"
              size="sm"
              onClick={handleExportPNG}
              disabled={isExporting}
              className="flex items-center gap-2"
            >
              <Image className="w-4 h-4" />
              PNG Card
            </Button>
          </>

          <Button onClick={handleCreateNew} variant="default">
            + New Character
          </Button>
        </div>
      </div>

      {/* Current Character Editing Section */}
      {(activeCharacter || isCreatingNew) && (
        <Card className="mb-8 w-full">
          <CardHeader>
            <CardTitle>
              {isCreatingNew ? 'Create New Character' : `Edit Character: ${character.name || ''}`}
            </CardTitle>
            <CardDescription>Define the core attributes and instructions for your character.</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-6">

              {/* Character Name */}
              <div>
                <Label htmlFor="name" className="block text-sm font-medium mb-1">Character Name *</Label>
                <Input id="name" name="name" value={character.name || ''} onChange={handleChange} placeholder="e.g. Professor Eldrin" required />
              </div>

              {/* Persona (Description) - Large Textarea */}
              <div>
                <Label htmlFor="description" className="block text-sm font-medium mb-1">Character Persona</Label>
                <Textarea
                  id="description"
                  name="description"
                  value={character.description || ''}
                  onChange={handleChange}
                  placeholder="Describe the character's personality, background, core traits, visual appearance, quirks..."
                  className="h-32" // Make text area larger
                />
                <p className="text-xs text-muted-foreground mt-1">Combine personality, backstory, visual description, etc., here.</p>
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
                <p className="text-xs text-muted-foreground mt-1">Tell the AI *how* to act (style, format, constraints).</p>
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
                <p className="text-xs text-muted-foreground mt-1">How does the character speak? (Tone, dialect, quirks)</p>
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
              </div>

              {/* Example Dialogue Section */}
              <div>
                <Label className="block text-sm font-medium mb-1">Example Dialogue</Label>
                <Card className="p-4 bg-muted/30">
                  <div className="space-y-3">
                    {ensureArray(character.example_dialogue).map((turn, index) => (
                      <div key={index} className="space-y-1">
                        <Label htmlFor={`dialogue-${index}-role`} className="text-xs font-semibold">{turn.role === 'user' ? 'User says:' : 'Character says:'}</Label>
                        <Textarea
                          id={`dialogue-${index}-content`}
                          value={turn.content || ''}
                          onChange={(e) => handleDialogueChange(index, 'content', e.target.value)}
                          placeholder={turn.role === 'user' ? 'Example user input...' : 'Example character response...'}
                          className="h-16 text-sm" // Smaller text area for examples
                        />
                      </div>
                    ))}
                  </div>
                </Card>
                <p className="text-xs text-muted-foreground mt-1">Provide a short snippet to demonstrate the character's voice and interaction style.</p>
              </div>

              {/* Avatar Upload */}
              <div>
                <Label htmlFor="avatar" className="block text-sm font-medium mb-1">Avatar</Label>
                <div className="flex items-center space-x-4">
                  <Input id="avatar" type="file" accept="image/*" onChange={handleAvatarUpload} className="flex-grow" />
                  {character.avatar && (
                    <div className="mt-2 flex-shrink-0">
                      <img
                        src={character.avatar}
                        alt="Avatar Preview"
                        className="w-16 h-16 rounded-full object-cover border border-border"
                        onError={(e) => {
                          console.error("Failed to load image:", e.target.src);
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  )}
                </div>
              </div>

            </div>
          </CardContent>
        </Card>
      )}

      {/* World Lore / Context Section */}
      {(activeCharacter || isCreatingNew) && (
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

      {/* Action Buttons */}
      {(activeCharacter || isCreatingNew) && (
        <div className="flex flex-wrap gap-2 pt-4 justify-center">
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

      {/* Character List - ALWAYS VISIBLE */}
      <div className="mt-12">
        <h3 className="text-xl font-bold mb-4">Saved Characters</h3>
        {Array.isArray(characters) && characters.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
            {characters.map((char) => (
              <Card key={char.id} className="overflow-hidden flex flex-col hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex items-start space-x-3">
                    {char.avatar && (
                      <img
                        src={char.avatar}
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
                      onClick={() => setActiveCharacter(char)}
                      variant="outline"
                    >
                      Edit
                    </Button>

                    {/* Individual Export Button */}
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
                  </div>
                </CardContent>
              </Card>
            ))}
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
    </div>
  );
};

export default CharacterEditor;