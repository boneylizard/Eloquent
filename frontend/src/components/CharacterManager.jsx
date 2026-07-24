import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Bot,
  Copy,
  Database,
  Download,
  FolderOpen,
  LayoutGrid,
  MessageCircle,
  MoreVertical,
  Pencil,
  Plus,
  Search,
  Sparkles,
  Trash2,
  Upload,
  Users,
  X,
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import CharacterEditor from './CharacterEditor';
import CharacterCreatorStudio from './CharacterCreatorStudio';
import CharacterDatasetImporter from './CharacterDatasetImporter';
import { CharacterCardIntegration } from '../utils/CharacterCardUtils';
import {
  createCharacterGroupId,
  loadCharacterGroups,
  saveCharacterGroups,
} from '../utils/characterGroups';
import { getBackendUrl } from '../config/api';
import { resolveAvatarDisplayUrl } from '../utils/characterAvatars';
import './CharacterManager.css';

const CharacterManager = ({ onSelectCharacter }) => {
  const {
    characters,
    saveCharacter,
    saveCharacters,
    deleteCharacter,
    duplicateCharacter,
    applyCharacter,
    PRIMARY_API_URL,
    setActiveCharacter,
    setPrimaryCharacter,
    buildSystemPrompt,
    setActiveTab,
    startCharacterConversation,
    updateSettings,
  } = useApp();

  const [activeView, setActiveView] = useState('list');
  const [editingCharacter, setEditingCharacter] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterTag, setFilterTag] = useState('');
  const [sortMode, setSortMode] = useState('recent');
  const [cardSize, setCardSize] = useState(210);
  const [isImporting, setIsImporting] = useState(false);
  const [importStatus, setImportStatus] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  const [characterGroups, setCharacterGroups] = useState([]);
  const [groupEditor, setGroupEditor] = useState(null);
  const [groupEditorError, setGroupEditorError] = useState('');
  const importFileRef = useRef(null);
  const importFolderRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    loadCharacterGroups()
      .then((groups) => {
        if (!cancelled) setCharacterGroups(groups);
      })
      .catch((error) => {
        console.error('Could not load character groups:', error);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const importSelectedCards = useCallback(async (files, sourceLabel) => {
    const selectedFiles = Array.from(files || []);
    if (selectedFiles.length === 0) return;
    setIsImporting(true);
    setImportStatus({
      type: 'progress',
      title: 'Reading character cards',
      message: `Checking ${selectedFiles.length} selected file${selectedFiles.length === 1 ? '' : 's'}…`,
      failed: [],
    });

    try {
      const result = await CharacterCardIntegration.importCharacterCardFiles(
        selectedFiles,
        PRIMARY_API_URL,
        {
          onProgress: ({ current, total, fileName }) => {
            setImportStatus({
              type: 'progress',
              title: `Importing ${current} of ${total}`,
              message: fileName,
              failed: [],
            });
          },
        },
      );
      const savedCharacters = saveCharacters(
        result.imported.map(({ character }) => ({
          ...character,
          id: null,
          created_at: new Date().toISOString().split('T')[0],
        })),
      );
      const importedCount = savedCharacters.length;
      const failedCount = result.failed.length;
      const skippedCount = result.skipped.length;
      const messageParts = [];

      if (importedCount > 0) {
        messageParts.push(
          `Added ${importedCount} character${importedCount === 1 ? '' : 's'} from the ${sourceLabel}.`,
        );
      } else if (result.supportedCount === 0) {
        messageParts.push('No TavernAI-compatible JSON or PNG cards were found.');
      } else {
        messageParts.push('No characters were added.');
      }
      if (failedCount > 0) {
        messageParts.push(`${failedCount} card${failedCount === 1 ? '' : 's'} could not be read.`);
      }
      if (skippedCount > 0) {
        messageParts.push(`Ignored ${skippedCount} other file${skippedCount === 1 ? '' : 's'}.`);
      }

      setImportStatus({
        type: importedCount > 0 ? (failedCount > 0 ? 'warning' : 'success') : 'error',
        title: importedCount > 0
          ? (failedCount > 0 ? 'Import finished with errors' : 'Import complete')
          : 'Nothing imported',
        message: messageParts.join(' '),
        failed: result.failed,
      });
    } catch (error) {
      console.error('Character import failed:', error);
      setImportStatus({
        type: 'error',
        title: 'Import failed',
        message: error instanceof Error ? error.message : String(error),
        failed: [],
      });
    } finally {
      setIsImporting(false);
    }
  }, [PRIMARY_API_URL, saveCharacters]);

  const handleImportCards = useCallback(async (event) => {
    const input = event.currentTarget;
    await importSelectedCards(input.files, 'file selection');
    input.value = '';
  }, [importSelectedCards]);

  const handleImportFolder = useCallback(async (event) => {
    const input = event.currentTarget;
    await importSelectedCards(input.files, 'selected folder');
    input.value = '';
  }, [importSelectedCards]);

  const handleExportTavernJSON = useCallback((character) => {
    if (!character?.name?.trim()) {
      alert('This character has no name and cannot be exported.');
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

  const handleExportGingerJSON = useCallback((character) => {
    if (!character?.name?.trim()) {
      alert('This character has no name and cannot be exported.');
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

  const handleExportPNG = useCallback(async (character) => {
    if (!character?.name?.trim()) {
      alert('This character has no name and cannot be exported.');
      return;
    }

    setIsExporting(true);
    try {
      await CharacterCardIntegration.exportAsPNG(character, PRIMARY_API_URL);
    } catch (error) {
      console.error('PNG export failed:', error);
      alert(`PNG export failed: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  }, [PRIMARY_API_URL]);

  const handleSaveCharacter = (characterData) => {
    const savedCharacter = saveCharacter(characterData);
    if (!savedCharacter?.id) return;
    setActiveCharacter(savedCharacter);
    setPrimaryCharacter(savedCharacter);
    setActiveTab('chat');
  };

  const handleOpenFullEditor = (characterData) => {
    const nextCharacter = { ...characterData, id: characterData?.id || null };
    setActiveCharacter(nextCharacter);
    setEditingCharacter(nextCharacter);
    setActiveView('edit');
  };

  const handleImportDatasetCharacters = (importedCharacters) => {
    importedCharacters.forEach((character) => saveCharacter(character));
    setActiveView('list');
  };

  const handleEditCharacter = (character) => {
    setActiveCharacter(character);
    setEditingCharacter(character);
    setActiveView('edit');
  };

  const handleDeleteCharacter = (characterId) => {
    if (window.confirm('Delete this character? This cannot be undone.')) {
      deleteCharacter(characterId);
    }
  };

  const handleSelectCharacter = (character) => {
    applyCharacter(character.id);
    setActiveTab('chat');

    if (onSelectCharacter) {
      onSelectCharacter(character, buildSystemPrompt(character));
    }
  };

  const handleSelectAssistant = () => {
    applyCharacter(null);
    setActiveTab('chat');
  };

  const openGroupEditor = (group = null) => {
    setGroupEditor({
      id: group?.id || '',
      name: group?.name || '',
      characterIds: [...(group?.characterIds || [])],
      context: group?.context || '',
      created_at: group?.created_at || '',
    });
    setGroupEditorError('');
  };

  const toggleGroupMember = (characterId) => {
    setGroupEditor((current) => {
      if (!current) return current;
      const selected = new Set(current.characterIds);
      if (selected.has(characterId)) selected.delete(characterId);
      else selected.add(characterId);
      return { ...current, characterIds: [...selected] };
    });
    setGroupEditorError('');
  };

  const handleSaveGroup = async (event) => {
    event.preventDefault();
    if (!groupEditor) return;

    const name = groupEditor.name.trim();
    if (!name) {
      setGroupEditorError('Name the group.');
      return;
    }
    if (groupEditor.characterIds.length < 2) {
      setGroupEditorError('Choose at least two characters.');
      return;
    }

    const now = new Date().toISOString();
    const nextGroup = {
      ...groupEditor,
      id: groupEditor.id || createCharacterGroupId(),
      name,
      context: groupEditor.context.trim(),
      created_at: groupEditor.created_at || now,
      updated_at: now,
    };
    const nextGroups = characterGroups.some((group) => group.id === nextGroup.id)
      ? characterGroups.map((group) => (group.id === nextGroup.id ? nextGroup : group))
      : [...characterGroups, nextGroup];

    try {
      const savedGroups = await saveCharacterGroups(nextGroups);
      setCharacterGroups(savedGroups);
      setGroupEditor(null);
      setGroupEditorError('');
    } catch (error) {
      console.error('Could not save character group:', error);
      setGroupEditorError('Mirid could not save this group.');
    }
  };

  const handleDeleteGroup = async (group) => {
    if (!window.confirm(`Delete the group "${group.name}"? Characters and chats will not be deleted.`)) {
      return;
    }
    try {
      const nextGroups = characterGroups.filter((candidate) => candidate.id !== group.id);
      const savedGroups = await saveCharacterGroups(nextGroups);
      setCharacterGroups(savedGroups);
    } catch (error) {
      console.error('Could not delete character group:', error);
      alert('Mirid could not delete this group.');
    }
  };

  const runMenuAction = (event, action) => {
    const menu = event.currentTarget.closest('details');
    menu?.removeAttribute('open');
    action();
  };

  const characterList = Array.isArray(characters) ? characters : [];
  const groupEligibleCharacters = characterList.filter(
    (character) => character.chat_role !== 'user'
  );
  const charactersById = new Map(characterList.map((character) => [character.id, character]));

  const handleStartGroup = (group) => {
    const members = group.characterIds
      .map((characterId) => charactersById.get(characterId))
      .filter((character) => character && character.chat_role !== 'user');

    if (members.length < 2) {
      alert('This group needs at least two available characters. Edit the group and choose its members again.');
      return;
    }

    updateSettings({ multiRoleMode: true });
    startCharacterConversation(members[0], {
      conversationName: group.name,
      activeCharacterIds: members.map((member) => member.id),
      multiRoleContext: group.context,
    });
  };

  const allTags = [...new Set(characterList.flatMap((character) => character.tags || []))]
    .filter(Boolean)
    .sort((a, b) => String(a).localeCompare(String(b)));

  const query = searchQuery.trim().toLowerCase();
  const filteredCharacters = characterList.filter((character) => {
    const searchableText = [
      character.name,
      character.description,
      ...(Array.isArray(character.tags) ? character.tags : []),
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    const matchesSearch = query === '' || searchableText.includes(query);
    const matchesTag = filterTag === '' || character.tags?.includes(filterTag);
    return matchesSearch && matchesTag;
  });

  const sortedCharacters = [...filteredCharacters].sort((left, right) => {
    if (sortMode === 'name') {
      return String(left.name || '').localeCompare(String(right.name || ''));
    }
    return new Date(right.updated_at || right.created_at || 0)
      - new Date(left.updated_at || left.created_at || 0);
  });

  const sortedGroups = characterGroups
    .map((group) => ({
      ...group,
      members: group.characterIds
        .map((characterId) => charactersById.get(characterId))
        .filter(Boolean),
    }))
    .filter((group) => {
      const searchableText = [
        group.name,
        group.context,
        ...group.members.map((member) => member.name),
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      const matchesSearch = query === '' || searchableText.includes(query);
      const matchesTag =
        filterTag === ''
        || group.members.some((member) => member.tags?.includes(filterTag));
      return matchesSearch && matchesTag;
    })
    .sort((left, right) => {
      if (sortMode === 'name') {
        return left.name.localeCompare(right.name);
      }
      return new Date(right.updated_at || right.created_at || 0)
        - new Date(left.updated_at || left.created_at || 0);
    });

  return (
    <div className="character-manager">
      {activeView === 'list' && (
        <div className="character-list-view">
          <header className="character-list-header">
            <div className="character-library-heading">
              <h2>Character Library</h2>
              <span>
                {characterList.length} character{characterList.length === 1 ? '' : 's'}
                {' · '}
                {characterGroups.length} group{characterGroups.length === 1 ? '' : 's'}
              </span>
            </div>

            <div className="header-buttons">
              <input
                ref={importFileRef}
                type="file"
                accept=".json,.png"
                multiple
                onChange={handleImportCards}
                hidden
              />
              <input
                ref={importFolderRef}
                type="file"
                accept=".json,.png"
                multiple
                webkitdirectory=""
                directory=""
                onChange={handleImportFolder}
                hidden
              />

              <button
                type="button"
                className="create-btn"
                onClick={() => {
                  setActiveCharacter(null);
                  setEditingCharacter(null);
                  setActiveView('create');
                }}
              >
                <Plus size={16} aria-hidden="true" />
                New character
              </button>

              <button
                type="button"
                className="import-btn"
                onClick={() => importFileRef.current?.click()}
                disabled={isImporting}
                title="Import TavernAI or SillyTavern V1/V2 JSON and PNG cards"
              >
                <Upload size={16} aria-hidden="true" />
                {isImporting ? 'Importing…' : 'Import cards'}
              </button>

              <button
                type="button"
                className="import-btn"
                onClick={() => importFolderRef.current?.click()}
                disabled={isImporting}
                title="Import every TavernAI or SillyTavern JSON and PNG card in a folder"
              >
                <FolderOpen size={16} aria-hidden="true" />
                Import folder
              </button>

              <button
                type="button"
                className="import-btn"
                onClick={() => openGroupEditor()}
              >
                <Users size={16} aria-hidden="true" />
                New group
              </button>

              <button
                type="button"
                className="import-btn mirid-builder-btn"
                onClick={() => {
                  setActiveCharacter(null);
                  setActiveView('builder');
                }}
              >
                <Sparkles size={16} aria-hidden="true" />
                Build with Mirid
              </button>

              <details className="library-more-menu">
                <summary aria-label="More library actions" title="More">
                  <MoreVertical size={18} aria-hidden="true" />
                </summary>
                <div className="library-menu-panel">
                  <button type="button" onClick={() => setActiveView('dataset')}>
                    <Database size={15} aria-hidden="true" />
                    Import dataset
                  </button>
                </div>
              </details>
            </div>
          </header>

          {importStatus ? (
            <section
              className={`character-import-status ${importStatus.type}`}
              role={importStatus.type === 'error' ? 'alert' : 'status'}
              aria-live="polite"
            >
              <div className="character-import-status-copy">
                <strong>{importStatus.title}</strong>
                <span>{importStatus.message}</span>
                {importStatus.failed.length > 0 ? (
                  <details>
                    <summary>Show files that could not be imported</summary>
                    <ul>
                      {importStatus.failed.slice(0, 50).map((failure) => (
                        <li key={`${failure.fileName}-${failure.message}`}>
                          <span>{failure.fileName}</span>
                          <small>{failure.message}</small>
                        </li>
                      ))}
                    </ul>
                    {importStatus.failed.length > 50 ? (
                      <p>{importStatus.failed.length - 50} more failures are not shown.</p>
                    ) : null}
                  </details>
                ) : null}
              </div>
              {importStatus.type !== 'progress' ? (
                <button
                  type="button"
                  className="character-import-status-dismiss"
                  onClick={() => setImportStatus(null)}
                  aria-label="Dismiss import result"
                >
                  <X size={16} aria-hidden="true" />
                </button>
              ) : null}
            </section>
          ) : null}

          <div className="character-filters">
            <label className="search-box">
              <Search size={18} aria-hidden="true" />
              <input
                type="search"
                placeholder="Search characters and tags"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                aria-label="Search characters and tags"
              />
            </label>

            <select
              className="tag-filter"
              value={filterTag}
              onChange={(event) => setFilterTag(event.target.value)}
              aria-label="Filter character cards by tag"
            >
              <option value="">All tags</option>
              {allTags.map((tag) => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>

            <select
              className="character-sort"
              value={sortMode}
              onChange={(event) => setSortMode(event.target.value)}
              aria-label="Sort character cards"
            >
              <option value="recent">Recently added</option>
              <option value="name">Name</option>
            </select>

            <label className="card-size-control" title="Card size">
              <LayoutGrid size={17} aria-hidden="true" />
              <input
                type="range"
                min="170"
                max="280"
                step="10"
                value={cardSize}
                onChange={(event) => setCardSize(Number(event.target.value))}
                aria-label="Character card size"
              />
            </label>
          </div>

          <div
            className="character-grid"
            style={{ '--character-card-min-width': `${cardSize}px` }}
          >
            <article className="character-card assistant-card">
              <button type="button" className="character-card-main" onClick={handleSelectAssistant}>
                <div className="character-avatar assistant-avatar">
                  <Bot size={52} strokeWidth={1.35} aria-hidden="true" />
                  <span>Plain chat</span>
                </div>
                <div className="character-info">
                  <div className="character-card-title-row">
                    <h3>Assistant</h3>
                    <MessageCircle size={16} aria-hidden="true" />
                  </div>
                  <p className="character-desc">No character card or roleplay instructions.</p>
                </div>
              </button>
            </article>

            {sortedGroups.map((group) => (
              <article key={group.id} className="character-card character-group-card">
                <button
                  type="button"
                  className="character-card-main"
                  onClick={() => handleStartGroup(group)}
                  aria-label={`Start group chat with ${group.name}`}
                >
                  <div className={`character-avatar group-avatar group-count-${Math.min(group.members.length, 4)}`}>
                    {group.members.length > 0 ? (
                      group.members.slice(0, 4).map((member) => (
                        <div key={member.id} className="group-member-portrait">
                          {member.avatar ? (
                            <img
                              src={resolveAvatarDisplayUrl(member.avatar, PRIMARY_API_URL || getBackendUrl())}
                              alt=""
                            />
                          ) : (
                            <span aria-hidden="true">{String(member.name || '?').charAt(0)}</span>
                          )}
                        </div>
                      ))
                    ) : (
                      <Users size={56} strokeWidth={1.25} aria-hidden="true" />
                    )}
                    <span className="group-member-count">
                      {group.members.length} character{group.members.length === 1 ? '' : 's'}
                    </span>
                  </div>

                  <div className="character-info">
                    <div className="character-card-title-row">
                      <h3>{group.name}</h3>
                      <Users size={16} aria-hidden="true" />
                    </div>
                    <p className="character-desc">
                      {group.members.map((member) => member.name).join(', ') || 'No available characters'}
                    </p>
                  </div>
                </button>

                <details className="character-card-menu">
                  <summary aria-label={`Actions for ${group.name}`} title="Actions">
                    <MoreVertical size={18} aria-hidden="true" />
                  </summary>
                  <div className="character-card-menu-panel">
                    <button type="button" onClick={(event) => runMenuAction(event, () => openGroupEditor(group))}>
                      <Pencil size={14} aria-hidden="true" />
                      Edit group
                    </button>
                    <button
                      type="button"
                      className="danger"
                      onClick={(event) => runMenuAction(event, () => handleDeleteGroup(group))}
                    >
                      <Trash2 size={14} aria-hidden="true" />
                      Delete group
                    </button>
                  </div>
                </details>
              </article>
            ))}

            {sortedCharacters.map((character) => (
              <article key={character.id} className="character-card">
                <button
                  type="button"
                  className="character-card-main"
                  onClick={() => handleSelectCharacter(character)}
                  aria-label={`Chat with ${character.name}`}
                >
                  <div className="character-avatar">
                    {character.avatar ? (
                      <img
                        src={resolveAvatarDisplayUrl(character.avatar, PRIMARY_API_URL || getBackendUrl())}
                        alt=""
                      />
                    ) : (
                      <div className="avatar-placeholder" aria-hidden="true">
                        {String(character.name || '?').charAt(0)}
                      </div>
                    )}
                    {character.tags?.[0] && (
                      <span className="character-primary-tag">{character.tags[0]}</span>
                    )}
                  </div>

                  <div className="character-info">
                    <div className="character-card-title-row">
                      <h3>{character.name || 'Unnamed character'}</h3>
                      <MessageCircle size={16} aria-hidden="true" />
                    </div>
                    <p className="character-desc">{character.description || 'No description'}</p>
                  </div>
                </button>

                <details className="character-card-menu">
                  <summary aria-label={`Actions for ${character.name}`} title="Actions">
                    <MoreVertical size={18} aria-hidden="true" />
                  </summary>
                  <div className="character-card-menu-panel">
                    <button type="button" onClick={(event) => runMenuAction(event, () => handleEditCharacter(character))}>
                      <Pencil size={14} aria-hidden="true" />
                      Edit
                    </button>
                    <button type="button" onClick={(event) => runMenuAction(event, () => duplicateCharacter(character.id))}>
                      <Copy size={14} aria-hidden="true" />
                      Duplicate
                    </button>
                    <button
                      type="button"
                      disabled={isExporting}
                      onClick={(event) => runMenuAction(event, () => handleExportPNG(character))}
                    >
                      <Download size={14} aria-hidden="true" />
                      Export PNG
                    </button>
                    <button
                      type="button"
                      disabled={isExporting}
                      onClick={(event) => runMenuAction(event, () => handleExportTavernJSON(character))}
                    >
                      <Download size={14} aria-hidden="true" />
                      Export TavernAI JSON
                    </button>
                    <button
                      type="button"
                      disabled={isExporting}
                      onClick={(event) => runMenuAction(event, () => handleExportGingerJSON(character))}
                    >
                      <Download size={14} aria-hidden="true" />
                      Export GingerGUI JSON
                    </button>
                    <button
                      type="button"
                      className="danger"
                      onClick={(event) => runMenuAction(event, () => handleDeleteCharacter(character.id))}
                    >
                      <Trash2 size={14} aria-hidden="true" />
                      Delete
                    </button>
                  </div>
                </details>
              </article>
            ))}
          </div>

          {sortedCharacters.length === 0 && sortedGroups.length === 0 && (
            <div className="no-characters">
              <p>{query || filterTag ? 'Nothing matches this search.' : 'No character cards saved yet.'}</p>
            </div>
          )}

          {groupEditor && (
            <div
              className="character-group-dialog-backdrop"
              onMouseDown={(event) => {
                if (event.target === event.currentTarget) setGroupEditor(null);
              }}
            >
              <form
                className="character-group-dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby="character-group-dialog-title"
                onSubmit={handleSaveGroup}
              >
                <header>
                  <div>
                    <h3 id="character-group-dialog-title">
                      {groupEditor.id ? 'Edit group' : 'New group'}
                    </h3>
                    <p>A saved group starts a new chat with the same characters and shared instructions.</p>
                  </div>
                  <button
                    type="button"
                    className="group-dialog-close"
                    onClick={() => setGroupEditor(null)}
                    aria-label="Close group editor"
                  >
                    <X size={18} aria-hidden="true" />
                  </button>
                </header>

                <label className="group-dialog-field">
                  <span>Group name</span>
                  <input
                    type="text"
                    value={groupEditor.name}
                    onChange={(event) => {
                      setGroupEditor((current) => ({ ...current, name: event.target.value }));
                      setGroupEditorError('');
                    }}
                    placeholder="Name this group"
                    autoFocus
                  />
                </label>

                <fieldset className="group-member-picker">
                  <legend>Characters</legend>
                  {groupEligibleCharacters.length > 0 ? (
                    <div className="group-member-options">
                      {groupEligibleCharacters.map((character) => {
                        const selected = groupEditor.characterIds.includes(character.id);
                        return (
                          <label key={character.id} className={selected ? 'selected' : ''}>
                            <input
                              type="checkbox"
                              checked={selected}
                              onChange={() => toggleGroupMember(character.id)}
                            />
                            <span className="group-member-option-avatar">
                              {character.avatar ? (
                                <img
                                  src={resolveAvatarDisplayUrl(character.avatar, PRIMARY_API_URL || getBackendUrl())}
                                  alt=""
                                />
                              ) : (
                                String(character.name || '?').charAt(0)
                              )}
                            </span>
                            <span>{character.name || 'Unnamed character'}</span>
                          </label>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="group-member-empty">Create or import characters before making a group.</p>
                  )}
                </fieldset>

                <label className="group-dialog-field">
                  <span>Shared instructions</span>
                  <textarea
                    value={groupEditor.context}
                    onChange={(event) => setGroupEditor((current) => ({ ...current, context: event.target.value }))}
                    placeholder="Optional instructions that apply to the whole group"
                    rows={4}
                  />
                </label>

                {groupEditorError && (
                  <p className="group-dialog-error" role="alert">{groupEditorError}</p>
                )}

                <footer>
                  <button type="button" className="import-btn" onClick={() => setGroupEditor(null)}>
                    Cancel
                  </button>
                  <button type="submit" className="create-btn">
                    Save group
                  </button>
                </footer>
              </form>
            </div>
          )}
        </div>
      )}

      {activeView === 'create' && (
        <div className="character-create-view">
          <div className="view-header">
            <button
              type="button"
              className="back-btn"
              onClick={() => setActiveView('list')}
            >
              ← Back to characters
            </button>
          </div>

          <CharacterEditor onSave={handleSaveCharacter} />
        </div>
      )}

      {activeView === 'builder' && (
        <CharacterCreatorStudio
          onSave={handleSaveCharacter}
          onOpenFullEditor={handleOpenFullEditor}
          onCancel={() => setActiveView('list')}
        />
      )}

      {activeView === 'dataset' && (
        <CharacterDatasetImporter
          onImport={handleImportDatasetCharacters}
          onCancel={() => setActiveView('list')}
        />
      )}

      {activeView === 'edit' && editingCharacter && (
        <div className="character-edit-view">
          <div className="view-header">
            <button
              type="button"
              className="back-btn"
              onClick={() => {
                setActiveView('list');
                setEditingCharacter(null);
              }}
            >
              ← Back to characters
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
