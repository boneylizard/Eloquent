import React, { useEffect } from 'react';
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
    loadCharacters
  } = useApp();

  useEffect(() => { loadCharacters(); }, [loadCharacters]);

  return (
    <Select
      value={primaryCharacter?.id || ''}
      onValueChange={id =>
        setPrimaryCharacter(characters.find(c=>c.id===id) || null)
      }
    >
      <SelectTrigger className="w-[180px]">
        <SelectValue placeholder="Select Character" />
      </SelectTrigger>
      <SelectContent>
        {characters.map(c => (
          <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default CharacterSelector;
