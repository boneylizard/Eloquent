// CharacterCardUtils.js - Import/Export utilities for TavernAI/SillyTavern character cards

import { getBackendUrl } from '../config/api';

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

// Convert TavernAI format to GingerGUI format
export function convertTavernToGinger(tavernData) {
  // Handle both v1 and v2 formats
  const data = tavernData.data || tavernData;
  
  const gingerCharacter = {
    id: null, // Will be generated when saved
    name: data.name || '',
    description: data.description || '', // This becomes "persona" in your format
    model_instructions: data.system_prompt || '', // Map system_prompt to model_instructions
    scenario: data.scenario || '',
    first_message: data.first_mes || data.first_message || '',
    example_dialogue: [],
    loreEntries: [],
    avatar: null, // Will be set separately if importing from PNG
    created_at: ''
  };
  
  // Convert example messages
  if (data.mes_example || data.example_dialogue) {
    const examples = data.mes_example || data.example_dialogue || '';
    if (examples.trim()) {
      // Parse example dialogue - TavernAI uses various formats
      const dialogueLines = examples.split('\n').filter(line => line.trim());
      const parsedDialogue = [];
      
      for (const line of dialogueLines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('{{user}}:') || trimmed.startsWith('<USER>:') || trimmed.startsWith('You:')) {
          parsedDialogue.push({
            role: 'user',
            content: trimmed.replace(/^({{user}}|<USER>|You):\s*/, '')
          });
        } else if (trimmed.startsWith('{{char}}:') || trimmed.startsWith('<BOT>:') || 
                   trimmed.startsWith(data.name + ':')) {
          parsedDialogue.push({
            role: 'character', 
            content: trimmed.replace(new RegExp(`^({{char}}|<BOT>|${data.name}):\\s*`), '')
          });
        } else if (trimmed && parsedDialogue.length === 0) {
          // If no clear format, assume first line is user, second is character, etc.
          parsedDialogue.push({
            role: parsedDialogue.length % 2 === 0 ? 'user' : 'character',
            content: trimmed
          });
        }
      }
      
      gingerCharacter.example_dialogue = parsedDialogue.length > 0 ? parsedDialogue : [
        { role: 'user', content: '' },
        { role: 'character', content: '' }
      ];
    }
  }
  
  // Convert character book/world info to loreEntries
  if (data.character_book && data.character_book.entries) {
    gingerCharacter.loreEntries = data.character_book.entries.map(entry => ({
      content: entry.content || entry.value || '',
      keywords: entry.keys || entry.key || []
    }));
  }
  
  return gingerCharacter;
}

// Replace the convertGingerToTavern function with this:
export function convertGingerToTavern(gingerCharacter, creatorName = 'Eloquent') {
  const tavernData = {
    spec: 'chara_card_v2',
    spec_version: '2.0',
    data: {
      name: gingerCharacter.name || '',
      description: gingerCharacter.description || '', 
      personality: '', // Not used in Eloquent, but required by spec
      scenario: gingerCharacter.scenario || '',
      first_mes: gingerCharacter.first_message || '', // FIX: Ensure first_message maps to first_mes
      mes_example: '',
      creator_notes: 'Exported from Eloquent',
      system_prompt: gingerCharacter.model_instructions || '',
      post_history_instructions: '',
      alternate_greetings: [],
      tags: [],
      creator: creatorName,
      character_version: '1.0',
      extensions: {
        eloquent: { // FIX: Changed from ginger_gui to eloquent
          exported_at: new Date().toISOString(),
          original_format: 'eloquent'
        }
      }
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
  
  // FIX: Convert loreEntries to character_book - ENSURE PROPER STRUCTURE
  if (gingerCharacter.loreEntries && Array.isArray(gingerCharacter.loreEntries) && gingerCharacter.loreEntries.length > 0) {
    tavernData.data.character_book = {
      name: gingerCharacter.name + ' Lorebook',
      description: 'Lorebook for ' + gingerCharacter.name,
      scan_depth: 100,
      token_budget: 500,
      recursive_scanning: false,
      entries: gingerCharacter.loreEntries.map((entry, index) => ({
        id: index,
        keys: Array.isArray(entry.keywords) ? entry.keywords : [], // FIX: Ensure keywords is array
        content: entry.content || '',
        extensions: {},
        enabled: true,
        insertion_order: index,
        case_sensitive: false,
        name: `Entry ${index + 1}`,
        priority: 100,
        comment: '',
        selective: true,
        secondary_keys: [],
        constant: false,
        position: 'before_char'
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
      body: JSON.stringify(gingerCharacter)
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
  exportAsJSON,
  exportAsPNG,
  convertTavernToGinger,
  convertGingerToTavern
};