import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Card, CardHeader, CardContent, CardTitle } from './ui/card';
import { useApp } from '../contexts/AppContext';

const CharacterEditor = () => {
  const { activeCharacter, saveCharacter, characters, setActiveCharacter } = useApp();
  const [character, setCharacter] = useState({
    id: null,
    name: '',
    persona: '',
    speakingStyle: '',
    knowledgeAreas: [],
    personalityTraits: [],
    systemPrompt: ''
  });

  useEffect(() => {
    // Load character data when activeCharacter changes
    if (activeCharacter) {
      setCharacter(activeCharacter);
    }
  }, [activeCharacter]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCharacter(prev => ({ ...prev, [name]: value }));
  };

  const handleArrayChange = (e, field) => {
    const values = e.target.value.split(',').map(item => item.trim());
    setCharacter(prev => ({ ...prev, [field]: values }));
  };

  const handleSubmit = () => {
    saveCharacter(character);
  };

  const handleClear = () => {
    setCharacter({
      id: null,
      name: '',
      persona: '',
      speakingStyle: '',
      knowledgeAreas: [],
      personalityTraits: [],
      systemPrompt: ''
    });
    setActiveCharacter(null);
  };

  const generateSystemPrompt = () => {
    const template = `You are ${character.name}, a ${character.persona}. You speak in a ${character.speakingStyle} manner. You have knowledge of ${character.knowledgeAreas.join(', ')}. You exhibit these traits: ${character.personalityTraits.join(', ')}.`;
    
    setCharacter(prev => ({ ...prev, systemPrompt: template }));
  };

  return (
    <div className="max-w-4xl mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>Character Editor</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Character Name</label>
              <Input name="name" value={character.name} onChange={handleChange} placeholder="e.g. Dr. Emma Richards" />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Persona/Identity</label>
              <Input name="persona" value={character.persona} onChange={handleChange} placeholder="e.g. Brilliant but eccentric quantum physicist" />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Speaking Style</label>
              <Input name="speakingStyle" value={character.speakingStyle} onChange={handleChange} placeholder="e.g. Technical but with unexpected humor" />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Knowledge Areas (comma separated)</label>
              <Input 
                name="knowledgeAreas" 
                value={character.knowledgeAreas.join(', ')} 
                onChange={(e) => handleArrayChange(e, 'knowledgeAreas')} 
                placeholder="e.g. quantum physics, mathematics, sci-fi literature"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Personality Traits (comma separated)</label>
              <Input 
                name="personalityTraits" 
                value={character.personalityTraits.join(', ')} 
                onChange={(e) => handleArrayChange(e, 'personalityTraits')} 
                placeholder="e.g. curious, analytical, slightly absent-minded"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">System Prompt</label>
              <Textarea 
                name="systemPrompt" 
                value={character.systemPrompt} 
                onChange={handleChange}
                className="h-32"
                placeholder="The system prompt used to define this character's behavior"
              />
            </div>
            
            <div className="flex space-x-4 pt-4">
              <Button onClick={generateSystemPrompt}>Generate Prompt</Button>
              <Button onClick={handleSubmit} className="bg-green-600 hover:bg-green-700">Save Character</Button>
              <Button onClick={handleClear} variant="outline">Clear</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="mt-8">
        <h3 className="text-xl font-bold mb-4">Saved Characters</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {characters.map(char => (
            <Card key={char.id} className="overflow-hidden">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">{char.name}</CardTitle>
                <p className="text-sm text-muted-foreground">{char.persona}</p>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1 mb-3">
                  {char.personalityTraits.map((trait, i) => (
                    <span key={i} className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                      {trait}
                    </span>
                  ))}
                </div>
                <div className="flex space-x-2 mt-4">
                  <Button 
                    size="sm" 
                    onClick={() => setActiveCharacter(char)}
                    variant="outline"
                  >
                    Edit
                  </Button>
                  <Button 
                    size="sm"
                    onClick={() => applyCharacter(char.id)}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Use Character
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
          
          {characters.length === 0 && (
            <p className="text-muted-foreground col-span-2 text-center py-8">
              No characters saved yet. Create your first character above!
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default CharacterEditor;