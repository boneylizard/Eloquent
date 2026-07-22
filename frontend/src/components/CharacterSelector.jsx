import React, { useEffect, useMemo } from 'react';
import { useApp } from '../contexts/AppContext';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';

const DEFAULT_ASSISTANT_VALUE = '__mirid_default_assistant__';

const CharacterSelector = () => {
  const {
    characters,
    primaryCharacter,
    applyCharacter,
    loadCharacters,
    settings,
    activeCharacterIds
  } = useApp();

  useEffect(() => { loadCharacters(); }, [loadCharacters]);

  const filteredCharacters = useMemo(() => {
    if (!settings?.multiRoleMode) return characters;
    const nonUser = characters.filter(c => (c?.chat_role === 'user' ? false : true));
    const rosterIds = Array.isArray(activeCharacterIds) ? activeCharacterIds : [];
    const rosterSet = rosterIds.length ? new Set(rosterIds) : null;
    const rosterFiltered = rosterSet ? nonUser.filter(c => rosterSet.has(c.id)) : nonUser;
    return rosterFiltered.length ? rosterFiltered : nonUser;
  }, [characters, settings?.multiRoleMode, activeCharacterIds]);

  useEffect(() => {
    if (!settings?.multiRoleMode) return;
    const isUserRole = primaryCharacter?.chat_role === 'user';
    if (isUserRole) {
      applyCharacter(filteredCharacters[0]?.id || null);
    }
  }, [applyCharacter, filteredCharacters, primaryCharacter, settings?.multiRoleMode]);

  return (
    <Select
      value={primaryCharacter?.id || DEFAULT_ASSISTANT_VALUE}
      onValueChange={(value) => applyCharacter(value === DEFAULT_ASSISTANT_VALUE ? null : value)}
    >
      <SelectTrigger className="w-[180px]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={DEFAULT_ASSISTANT_VALUE}>Assistant · plain chat</SelectItem>
        {filteredCharacters.map(c => (
          <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default CharacterSelector;
