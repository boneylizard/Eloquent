import React, { useEffect, useMemo } from 'react';
import { useApp } from '../contexts/AppContext';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';

const CharacterSelector = () => {
  const {
    characters,
    primaryCharacter,
    setPrimaryCharacter,
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
      setPrimaryCharacter(filteredCharacters[0] || null);
    }
  }, [filteredCharacters, primaryCharacter, setPrimaryCharacter, settings?.multiRoleMode]);

  return (
    <Select
      value={primaryCharacter?.id || ''}
      onValueChange={id =>
        setPrimaryCharacter(filteredCharacters.find(c => c.id === id) || null)
      }
    >
      <SelectTrigger className="w-[180px]">
        <SelectValue placeholder="Select Character" />
      </SelectTrigger>
      <SelectContent>
        {filteredCharacters.map(c => (
          <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default CharacterSelector;
