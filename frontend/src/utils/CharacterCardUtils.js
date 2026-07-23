// CharacterCardUtils.js - Import/Export utilities for TavernAI/SillyTavern character cards

import { getBackendUrl } from '../config/api.js';

/**
 * Utility functions for importing and exporting character cards in various formats
 */

// Helper function to convert base64 to bytes
function base64ToUint8Array(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// Helper function to convert bytes to base64
function uint8ArrayToBase64(uint8Array) {
  let binaryString = '';
  for (let i = 0; i < uint8Array.length; i++) {
    binaryString += String.fromCharCode(uint8Array[i]);
  }
  return btoa(binaryString);
}

// Extract JSON from PNG tEXt chunk (simplified browser implementation)
async function extractCharacterFromPNG(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    
    // Look for PNG signature
    if (uint8Array[0] !== 0x89 || uint8Array[1] !== 0x50 || 
        uint8Array[2] !== 0x4E || uint8Array[3] !== 0x47) {
      throw new Error('Not a valid PNG file');
    }
    
    let offset = 8; // Skip PNG signature
    
    while (offset < uint8Array.length) {
      // Read chunk length (4 bytes, big-endian)
      const length = (uint8Array[offset] << 24) | 
                    (uint8Array[offset + 1] << 16) | 
                    (uint8Array[offset + 2] << 8) | 
                    uint8Array[offset + 3];
      offset += 4;
      
      // Read chunk type (4 bytes)
      const type = String.fromCharCode(
        uint8Array[offset], 
        uint8Array[offset + 1], 
        uint8Array[offset + 2], 
        uint8Array[offset + 3]
      );
      offset += 4;
      
      // Check for tEXt chunks that might contain character data
      if (type === 'tEXt' || type === 'zTXt' || type === 'iTXt') {
        const chunkData = uint8Array.slice(offset, offset + length);
        
        // Find null separator for keyword
        let nullIndex = -1;
        for (let i = 0; i < chunkData.length; i++) {
          if (chunkData[i] === 0) {
            nullIndex = i;
            break;
          }
        }
        
        if (nullIndex > 0) {
          const keyword = String.fromCharCode(...chunkData.slice(0, nullIndex));
          
          // Common keywords used for character data
          if (keyword === 'chara' || keyword === 'ccv2' || keyword === 'Character') {
            let textData;
            
            if (type === 'tEXt') {
              // Uncompressed text
              textData = String.fromCharCode(...chunkData.slice(nullIndex + 1));
            } else if (type === 'zTXt') {
              // Compressed text - would need pako.js or similar for full implementation
              // For now, try to decode as uncompressed
              textData = String.fromCharCode(...chunkData.slice(nullIndex + 2));
            } else if (type === 'iTXt') {
              // International text - more complex, skip compression/language for now
              textData = String.fromCharCode(...chunkData.slice(nullIndex + 5));
            }
            
            try {
              // Try to parse as base64 first (common encoding)
              try {
                const decoded = atob(textData);
                return JSON.parse(decoded);
              } catch {
                // If not base64, try direct JSON parse
                return JSON.parse(textData);
              }
            } catch (parseError) {
              console.warn('Found character chunk but could not parse JSON:', parseError);
            }
          }
        }
      }
      
      // Skip chunk data and CRC
      offset += length + 4;
      
      // Stop at IEND chunk
      if (type === 'IEND') break;
    }
    
    throw new Error('No character data found in PNG file');
    
  } catch (error) {
    console.error('Error extracting character from PNG:', error);
    throw error;
  }
}

const KNOWN_CARD_FIELDS = new Set([
  'name',
  'description',
  'personality',
  'scenario',
  'first_mes',
  'first_message',
  'mes_example',
  'example_dialogue',
  'creator_notes',
  'system_prompt',
  'post_history_instructions',
  'alternate_greetings',
  'character_book',
  'tags',
  'creator',
  'character_version',
  'extensions',
]);

const asObject = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : {});
const asStringArray = (value) => (Array.isArray(value) ? value.filter((item) => typeof item === 'string') : []);

function parseExampleDialogue(value, characterName = '') {
  if (Array.isArray(value)) {
    return value
      .filter((entry) => entry && typeof entry === 'object')
      .map((entry) => ({
        role: entry.role === 'user' ? 'user' : 'character',
        content: String(entry.content || ''),
      }));
  }

  const lines = String(value || '').split('\n');
  const parsed = [];
  let fallbackRole = 'user';
  const escapedName = characterName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const characterPrefix = escapedName ? new RegExp(`^(${escapedName}):\\s*`, 'i') : null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || /^<START>$/i.test(line)) continue;

    if (/^({{user}}|<USER>|You):\s*/i.test(line)) {
      parsed.push({ role: 'user', content: line.replace(/^({{user}}|<USER>|You):\s*/i, '') });
      fallbackRole = 'character';
    } else if (/^({{char}}|<BOT>):\s*/i.test(line) || characterPrefix?.test(line)) {
      parsed.push({
        role: 'character',
        content: characterPrefix?.test(line)
          ? line.replace(characterPrefix, '')
          : line.replace(/^({{char}}|<BOT>):\s*/i, ''),
      });
      fallbackRole = 'user';
    } else {
      parsed.push({ role: fallbackRole, content: line });
      fallbackRole = fallbackRole === 'user' ? 'character' : 'user';
    }
  }

  return parsed;
}

// Convert TavernAI / SillyTavern V1 or V2 data to Mirid's internal format.
export function convertTavernToGinger(tavernData) {
  const source = asObject(tavernData);
  const data = asObject(source.data && typeof source.data === 'object' ? source.data : source);
  const extensions = asObject(data.extensions);
  const miridExt = asObject(extensions.mirid || extensions.eloquent || extensions.ginger_gui);
  const characterBook = asObject(data.character_book);
  const importedAvatars = Array.isArray(miridExt.avatars)
    ? miridExt.avatars.filter((url) => typeof url === 'string' && url.trim())
    : [];
  const cardDataPassthrough = Object.fromEntries(
    Object.entries(data).filter(([key]) => !KNOWN_CARD_FIELDS.has(key)),
  );
  const cardTopLevel = Object.fromEntries(
    Object.entries(source).filter(([key]) => !['spec', 'spec_version', 'data'].includes(key)),
  );

  const gingerCharacter = {
    id: null,
    name: data.name || '',
    description: data.description || '',
    personality: data.personality || '',
    background: miridExt.background || '',
    model_instructions: data.system_prompt || '',
    post_history_instructions: data.post_history_instructions || '',
    speech_style: miridExt.speech_style || '',
    scenario: data.scenario || '',
    first_message: data.first_mes || data.first_message || '',
    alternate_greetings: asStringArray(data.alternate_greetings),
    example_dialogue: parseExampleDialogue(data.mes_example || data.example_dialogue, data.name || ''),
    loreEntries: Array.isArray(characterBook.entries)
      ? characterBook.entries.map((entry) => ({
          content: entry?.content || entry?.value || '',
          keywords: Array.isArray(entry?.keys) ? entry.keys : (Array.isArray(entry?.key) ? entry.key : []),
          tavern_entry: asObject(entry),
        }))
      : [],
    creator_notes: data.creator_notes || '',
    tags: asStringArray(data.tags),
    creator: data.creator || '',
    character_version: data.character_version || '',
    avatar: null,
    avatars: importedAvatars,
    activeAvatarIndex: Number.isInteger(miridExt.activeAvatarIndex) ? miridExt.activeAvatarIndex : 0,
    chat_role: miridExt.chat_role === 'user' ? 'user' : 'npc',
    created_at: '',
    ethics_justification: typeof miridExt.ethics_justification === 'string'
      ? miridExt.ethics_justification
      : (typeof data.ethics_justification === 'string' ? data.ethics_justification : ''),
    card_extensions: extensions,
    card_data_passthrough: cardDataPassthrough,
    card_top_level: cardTopLevel,
    character_book_metadata: Object.fromEntries(
      Object.entries(characterBook).filter(([key]) => key !== 'entries'),
    ),
  };

  return gingerCharacter;
}

export function convertGingerToTavern(gingerCharacter, creatorName = 'Mirid') {
  const preservedExtensions = asObject(gingerCharacter.card_extensions);
  const preservedMirid = asObject(preservedExtensions.mirid);
  const extensions = {
    ...preservedExtensions,
    mirid: {
      ...preservedMirid,
      exported_at: new Date().toISOString(),
      original_format: preservedMirid.original_format || 'mirid',
      background: gingerCharacter.background || '',
      speech_style: gingerCharacter.speech_style || '',
      chat_role: gingerCharacter.chat_role === 'user' ? 'user' : 'npc',
      ethics_justification: (gingerCharacter.ethics_justification || '').trim(),
      avatars: Array.isArray(gingerCharacter.avatars) ? gingerCharacter.avatars : [],
      activeAvatarIndex: gingerCharacter.activeAvatarIndex ?? 0,
    },
  };
  const tavernData = {
    ...asObject(gingerCharacter.card_top_level),
    spec: 'chara_card_v2',
    spec_version: '2.0',
    data: {
      ...asObject(gingerCharacter.card_data_passthrough),
      name: gingerCharacter.name || '',
      description: gingerCharacter.description || '',
      personality: gingerCharacter.personality || '',
      scenario: gingerCharacter.scenario || '',
      first_mes: gingerCharacter.first_message || '',
      mes_example: '',
      creator_notes: gingerCharacter.creator_notes || '',
      system_prompt: gingerCharacter.model_instructions || '',
      post_history_instructions: gingerCharacter.post_history_instructions || '',
      alternate_greetings: asStringArray(gingerCharacter.alternate_greetings),
      tags: asStringArray(gingerCharacter.tags),
      creator: gingerCharacter.creator || creatorName,
      character_version: gingerCharacter.character_version || '1.0',
      extensions,
    }
  };
  
  // Convert example dialogue - ENSURE THIS WORKS
  if (gingerCharacter.example_dialogue && Array.isArray(gingerCharacter.example_dialogue) && gingerCharacter.example_dialogue.length > 0) {
    const exampleLines = [];
    for (const dialogue of gingerCharacter.example_dialogue) {
      if (dialogue.role === 'user' && dialogue.content) {
        exampleLines.push(`{{user}}: ${dialogue.content}`);
      } else if (dialogue.role === 'character' && dialogue.content) {
        exampleLines.push(`{{char}}: ${dialogue.content}`);
      }
    }
    if (exampleLines.length > 0) {
      tavernData.data.mes_example = exampleLines.join('\n');
    }
  }
  
  const loreEntries = Array.isArray(gingerCharacter.loreEntries) ? gingerCharacter.loreEntries : [];
  const bookMetadata = asObject(gingerCharacter.character_book_metadata);
  if (loreEntries.length > 0 || Object.keys(bookMetadata).length > 0) {
    tavernData.data.character_book = {
      ...bookMetadata,
      name: bookMetadata.name || `${gingerCharacter.name || 'Character'} Lorebook`,
      description: bookMetadata.description || '',
      extensions: asObject(bookMetadata.extensions),
      entries: loreEntries.map((entry, index) => ({
        ...asObject(entry.tavern_entry),
        id: entry.tavern_entry?.id ?? index,
        keys: Array.isArray(entry.keywords) ? entry.keywords : [], // FIX: Ensure keywords is array
        content: entry.content || '',
        extensions: asObject(entry.tavern_entry?.extensions),
        enabled: entry.tavern_entry?.enabled ?? true,
        insertion_order: entry.tavern_entry?.insertion_order ?? index,
      }))
    };
  }
  
  return tavernData;
}

// Main import function
export async function importCharacterCard(file, apiUrl = null) {
  apiUrl = apiUrl || getBackendUrl();
  try {
    let characterData;
    let avatarFile = null;
    
    if (file.type === 'application/json' || file.name.endsWith('.json')) {
      // Import JSON file
      const text = await file.text();
      characterData = JSON.parse(text);
    } else if (file.type.startsWith('image/') || file.name.endsWith('.png')) {
      // Import PNG character card
      characterData = await extractCharacterFromPNG(file);
      // Keep the PNG file itself for the avatar
      avatarFile = file;
    } else {
      throw new Error('Unsupported file type. Please upload a JSON file or PNG character card.');
    }
    
    // Convert to GingerGUI format
    const gingerCharacter = convertTavernToGinger(characterData);
    if (!gingerCharacter.name.trim()) {
      throw new Error('The file does not contain a named TavernAI character card.');
    }
    
    // If we have a PNG file, upload it as the avatar
    if (avatarFile) {
      try {
        // Create FormData to upload the PNG as avatar
        const formData = new FormData();
        formData.append("file", avatarFile);
        
        // Upload to your backend (you'll need to pass the API URL)
        const uploadUrl = `${apiUrl}/upload_avatar`;
        const response = await fetch(uploadUrl, { 
          method: 'POST', 
          body: formData 
        });
        
        if (response.ok) {
          const result = await response.json();
          if (result.status === 'success' && result.file_url) {
            gingerCharacter.avatar = result.file_url;
            gingerCharacter.avatars = [result.file_url];
          }
        }
      } catch (avatarError) {
        console.warn('Failed to upload avatar from PNG:', avatarError);
        // Continue with import even if avatar upload fails
      }
    }
    
    return gingerCharacter;
    
  } catch (error) {
    console.error('Import error:', error);
    throw new Error(`Failed to import character: ${error.message}`);
  }
}

export function isSupportedCharacterCardFile(file) {
  const fileName = String(file?.name || '').toLowerCase();
  return fileName.endsWith('.json') || fileName.endsWith('.png');
}

export async function importCharacterCardFiles(
  files,
  apiUrl = null,
  { importer = importCharacterCard, onProgress = null } = {},
) {
  const selectedFiles = Array.from(files || []);
  const supportedFiles = selectedFiles.filter(isSupportedCharacterCardFile);
  const skippedFiles = selectedFiles.filter((file) => !isSupportedCharacterCardFile(file));
  const imported = [];
  const failed = [];

  for (let index = 0; index < supportedFiles.length; index += 1) {
    const file = supportedFiles[index];
    onProgress?.({
      current: index + 1,
      total: supportedFiles.length,
      fileName: file.webkitRelativePath || file.name,
    });

    try {
      const character = await importer(file, apiUrl);
      imported.push({
        fileName: file.webkitRelativePath || file.name,
        character,
      });
    } catch (error) {
      failed.push({
        fileName: file.webkitRelativePath || file.name,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return {
    selectedCount: selectedFiles.length,
    supportedCount: supportedFiles.length,
    imported,
    failed,
    skipped: skippedFiles.map((file) => file.webkitRelativePath || file.name),
  };
}

// Export as JSON
export function exportAsJSON(gingerCharacter, format = 'tavern') {
  try {
    let exportData;
    
    if (format === 'tavern' || format === 'sillytavern') {
      exportData = convertGingerToTavern(gingerCharacter);
    } else if (format === 'ginger') {
      exportData = gingerCharacter;
    } else {
      throw new Error('Unsupported export format');
    }
    
    const jsonString = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const filename = `${gingerCharacter.name || 'character'}_${format}.json`;
    
    // Create download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    
    return { success: true, filename };
    
  } catch (error) {
    console.error('Export error:', error);
    throw new Error(`Failed to export character: ${error.message}`);
  }
}

// Note: PNG export with embedded JSON would require a more complex implementation
// involving canvas manipulation and proper tEXt chunk creation. This is complex
// in browser environments without additional libraries like pako.js for compression.

// Simple PNG export (creates JSON and suggests using external tools)
// Replace the exportAsPNGInstructions function in CharacterCardUtils.js with:
export async function exportAsPNG(gingerCharacter, apiUrl = null) {
  apiUrl = apiUrl || getBackendUrl();
  try {
    const response = await fetch(`${apiUrl}/export_character_png`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...gingerCharacter,
        tavern_card: convertGingerToTavern(gingerCharacter),
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown server error' }));
      throw new Error(`PNG export failed: ${response.status} - ${errorData.detail || response.statusText}`);
    }

    // Download the PNG file
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${gingerCharacter.name.replace(/[^a-z0-9]/gi, '_')}_character_card.png`;
    a.click();
    URL.revokeObjectURL(url);

    return { success: true };
  } catch (error) {
    console.error('PNG export error:', error);
    throw error;
  }
}

// Integration code for CharacterEditor component
export const CharacterCardIntegration = {
  importCharacterCard,
  importCharacterCardFiles,
  isSupportedCharacterCardFile,
  exportAsJSON,
  exportAsPNG,
  convertTavernToGinger,
  convertGingerToTavern
};
