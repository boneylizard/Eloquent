import React from 'react';
import { Select } from './ui/select';
import { useApp } from '../contexts/AppContext';

const CharacterSelector = () => {
  const { characters, activeCharacter, applyCharacter } = useApp();

  return (
    <div className="mb-4">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium flex-shrink-0">Character:</label>
        <Select
          value={activeCharacter?.id || ""}
          onValueChange={(value) => applyCharacter(value)}
          className="w-full"
        >
          <option value="">Default Assistant</option>
          {characters.map(character => (
            <option key={character.id} value={character.id}>
              {character.name}
            </option>
          ))}
        </Select>
      </div>
      {activeCharacter && (
        <p className="text-xs text-muted-foreground mt-1">
          {activeCharacter.persona}
        </p>
      )}
    </div>
  );
};

export default CharacterSelector;