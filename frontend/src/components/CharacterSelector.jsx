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
    secondaryCharacter,
    setPrimaryCharacter,
    setSecondaryCharacter,
    loadCharacters
  } = useApp();

  useEffect(() => { loadCharacters(); }, [loadCharacters]);

  return (
    <div className="flex flex-col gap-2">
      {/* Primary character picker */}
      <Select
        value={primaryCharacter?.id || ''}
        onValueChange={id =>
          setPrimaryCharacter(characters.find(c=>c.id===id) || null)
        }
      >
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Primary Character" />
        </SelectTrigger>
        <SelectContent>
          {characters.map(c => (
            <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Secondary character picker */}
      <Select
        value={secondaryCharacter?.id || ''}
        onValueChange={id =>
          setSecondaryCharacter(characters.find(c=>c.id===id) || null)
        }
      >
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Secondary Character" />
        </SelectTrigger>
        <SelectContent>
          {characters.map(c => (
            <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default CharacterSelector;
