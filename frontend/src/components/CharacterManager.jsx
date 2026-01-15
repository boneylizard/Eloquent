import React, { useState, useRef, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';
import CharacterEditor from './CharacterEditor';
import { CharacterCardIntegration } from '../utils/CharacterCardUtils';
import './CharacterManager.css';

const CharacterManager = ({ onSelectCharacter }) => {
  const {
    characters,
    saveCharacter,
    deleteCharacter,
    duplicateCharacter,
    applyCharacter,
    PRIMARY_API_URL,
    setCharacter,
    setActiveCharacter,
    setIsCreatingNew,
    buildSystemPrompt
  } = useApp();

  const [activeView, setActiveView] = useState('list');
  const [editingCharacter, setEditingCharacter] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterTag, setFilterTag] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const importFileRef = useRef(null);

  // Import character card handler
  // Replace the handleImportCard function in CharacterEditor.jsx with this:
  const handleImportCard = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsImporting(true);
    try {
      const importedCharacter = await CharacterCardIntegration.importCharacterCard(file, PRIMARY_API_URL);

      // Prepare character for editing (don't save directly)
      const newCharacter = {
        ...importedCharacter,
        id: null, // Clear ID so it's treated as new
        created_at: new Date().toISOString().split('T')[0]
      };

      // Set the character in the context so CharacterEditor can see it
      setActiveCharacter(newCharacter);
      setEditingCharacter(newCharacter);
      setActiveView('edit'); // Use edit view so it passes the character data

      alert(`Character imported successfully! Please review and save to add to your library.`);

    } catch (error) {
      console.error('Import failed:', error);
      alert(`Import failed: ${error.message}`);
    } finally {
      setIsImporting(false);
      event.target.value = ''; // Reset file input
    }
  }, [setActiveCharacter, setEditingCharacter, setActiveView]);

  // Export character as TavernAI JSON
  const handleExportTavernJSON = useCallback((character) => {
    if (!character || !character.name?.trim()) {
      alert('Invalid character data');
      return;
    }

    setIsExporting(true);
    try {
      CharacterCardIntegration.exportAsJSON(character, 'tavern');
    } catch (error) {
      console.error('Export failed:', error);
      alert(`Export failed: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  }, []);

  // Export character as GingerGUI JSON
  const handleExportGingerJSON = useCallback((character) => {
    if (!character || !character.name?.trim()) {
      alert('Invalid character data');
      return;
    }

    setIsExporting(true);
    try {
      CharacterCardIntegration.exportAsJSON(character, 'ginger');
    } catch (error) {
      console.error('Export failed:', error);
      alert(`Export failed: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  }, []);

  // Export character as actual PNG file with embedded data (using CharacterCardUtils)
  const handleExportPNG = useCallback(async (character) => {
    if (!character || !character.name?.trim()) {
      alert('Invalid character data');
      return;
    }

    setIsExporting(true);
    try {
      await CharacterCardIntegration.exportAsPNG(character, PRIMARY_API_URL);
      alert('PNG character card exported successfully! The file contains embedded character data and will work with SillyTavern and other compatible tools.');
    } catch (error) {
      console.error('PNG export failed:', error);
      alert(`PNG export failed: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  }, [PRIMARY_API_URL]);

  const handleSaveCharacter = (characterData) => {
    saveCharacter(characterData);
    setActiveView('list');
    setEditingCharacter(null);
  };

  const handleEditCharacter = (character) => {
    setEditingCharacter(character);
    setActiveView('edit');
  };

  const handleDeleteCharacter = (characterId) => {
    if (window.confirm('Are you sure you want to delete this character?')) {
      deleteCharacter(characterId);
    }
  };

  const handleDuplicateCharacter = (character) => {
    duplicateCharacter(character.id);
  };

  const handleSelectCharacter = (character) => {
    applyCharacter(character.id);

    // Also call the parent's onSelectCharacter if provided
    if (onSelectCharacter) {
      const systemPrompt = buildSystemPrompt(character);
      onSelectCharacter(character, systemPrompt);
    }
  };


  // Get all unique tags from all characters
  const allTags = [...new Set((characters || []).flatMap(char => char.tags || []))].sort();

  // Filter characters based on search term and tag filter
  const filteredCharacters = (characters || []).filter(char => {
    const matchesSearch = searchQuery === '' ||
      char.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      char.description.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesTag = filterTag === '' ||
      (char.tags && char.tags.includes(filterTag));

    return matchesSearch && matchesTag;
  });

  // Sort characters by most recently created/updated
  const sortedCharacters = [...filteredCharacters].sort((a, b) => {
    return new Date(b.created_at || 0) - new Date(a.created_at || 0);
  });

  return (
    <div className="character-manager">
      {activeView === 'list' && (
        <div className="character-list-view">
          <div className="character-list-header">
            <h2>Character Library</h2>
            <div className="header-buttons">
              {/* Import/Export Controls */}
              <input
                ref={importFileRef}
                type="file"
                accept=".json,.png"
                onChange={handleImportCard}
                style={{ display: 'none' }}
              />

              <button
                className="import-btn"
                onClick={() => importFileRef.current?.click()}
                disabled={isImporting}
              >
                {isImporting ? 'Importing...' : '📥 Import Card'}
              </button>

              <button
                className="create-btn"
                onClick={() => setActiveView('create')}
              >
                + Create New Character
              </button>
            </div>
          </div>

          <div className="character-filters">
            <div className="search-box">
              <input
                type="text"
                placeholder="Search characters..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="tag-filter">
              <select
                value={filterTag}
                onChange={(e) => setFilterTag(e.target.value)}
              >
                <option value="">All Tags</option>
                {allTags.map(tag => (
                  <option key={tag} value={tag}>{tag}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="character-grid">
            {sortedCharacters.length > 0 ? (
              sortedCharacters.map(character => (
                <div key={character.id} className="character-card">
                  <div className="character-avatar">
                    {character.avatar ? (
                      <img src={character.avatar} alt={character.name} />
                    ) : (
                      <div className="avatar-placeholder">
                        {character.name.charAt(0)}
                      </div>
                    )}
                  </div>

                  <div className="character-info">
                    <h3>{character.name}</h3>
                    <p className="character-desc">{character.description}</p>

                    {character.tags && character.tags.length > 0 && (
                      <div className="character-tags">
                        {character.tags.map(tag => (
                          <span key={tag} className="character-tag">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="character-actions">
                    <div className="main-actions">
                      <button
                        className="action-btn select-btn"
                        onClick={() => handleSelectCharacter(character)}
                        title="Chat with this character"
                      >
                        Chat
                      </button>

                      <button
                        className="action-btn edit-btn"
                        onClick={() => handleEditCharacter(character)}
                        title="Edit character"
                      >
                        Edit
                      </button>

                      <button
                        className="action-btn duplicate-btn"
                        onClick={() => handleDuplicateCharacter(character)}
                        title="Duplicate character"
                      >
                        Copy
                      </button>

                      <button
                        className="action-btn delete-btn"
                        onClick={() => handleDeleteCharacter(character.id)}
                        title="Delete character"
                      >
                        Delete
                      </button>
                    </div>

                    <div className="export-actions">
                      <span className="export-label">Export:</span>
                      <button
                        className="export-btn tavern-export"
                        onClick={() => handleExportTavernJSON(character)}
                        disabled={isExporting}
                        title="Export as TavernAI JSON"
                      >
                        TavernAI
                      </button>

                      <button
                        className="export-btn ginger-export"
                        onClick={() => handleExportGingerJSON(character)}
                        disabled={isExporting}
                        title="Export as GingerGUI JSON"
                      >
                        GingerGUI
                      </button>

                      <button
                        className="export-btn png-export"
                        onClick={() => handleExportPNG(character)}
                        disabled={isExporting}
                        title="Export as PNG character card"
                      >
                        PNG Card
                      </button>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="no-characters">
                <p>No characters found. Create your first character to get started!</p>
              </div>
            )}
          </div>
        </div>
      )}

      {activeView === 'create' && (
        <div className="character-create-view">
          <div className="view-header">
            <button
              className="back-btn"
              onClick={() => setActiveView('list')}
            >
              ← Back to Characters
            </button>
          </div>

          <CharacterEditor onSave={handleSaveCharacter} />
        </div>
      )}

      {activeView === 'edit' && editingCharacter && (
        <div className="character-edit-view">
          <div className="view-header">
            <button
              className="back-btn"
              onClick={() => {
                setActiveView('list');
                setEditingCharacter(null);
              }}
            >
              ← Back to Characters
            </button>
          </div>

          <CharacterEditor
            initialCharacter={editingCharacter}
            onSave={handleSaveCharacter}
          />
        </div>
      )}
    </div>
  );
};

export default CharacterManager;