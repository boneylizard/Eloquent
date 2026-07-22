export const CHARACTER_DATASET_FIELDS = [
  ['name', 'Name'],
  ['description', 'Character core'],
  ['personality', 'Personality'],
  ['background', 'Background'],
  ['scenario', 'Scenario'],
  ['first_message', 'Opening message'],
  ['example_dialogue', 'Example dialogue'],
  ['model_instructions', 'Model instructions'],
  ['speech_style', 'Speech style'],
  ['tags', 'Tags'],
  ['creator', 'Creator'],
  ['license', 'Licence'],
  ['image', 'Image'],
  ['card', 'Complete card object'],
];

const ALIASES = {
  name: ['name', 'character_name', 'char_name', 'title'],
  description: ['description', 'character', 'character_summary', 'persona', 'persona_text', 'profile', 'character_card'],
  personality: ['personality', 'traits', 'temperament'],
  background: ['background', 'backstory', 'history'],
  scenario: ['scenario', 'setting', 'context', 'world'],
  first_message: ['first_message', 'first_mes', 'greeting', 'opening_message', 'story_introduction'],
  example_dialogue: ['example_dialogue', 'mes_example', 'examples', 'dialogue'],
  model_instructions: ['model_instructions', 'system_prompt', 'instructions'],
  speech_style: ['speech_style', 'voice', 'speaking_style'],
  tags: ['tags', 'genres', 'genre_tags'],
  creator: ['creator', 'author'],
  license: ['license', 'licence'],
  image: ['image', 'avatar', 'image_url', 'portrait'],
  card: ['card', 'card_json', 'character_json', 'tavern_card'],
};

export function autoMapCharacterColumns(columns = []) {
  const lookup = new Map(columns.map((column) => [String(column).toLowerCase(), String(column)]));
  return Object.fromEntries(CHARACTER_DATASET_FIELDS.map(([field]) => [
    field,
    (ALIASES[field] || []).map((alias) => lookup.get(alias)).find(Boolean) || '',
  ]));
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = '';
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === ',' && !quoted) {
      row.push(value);
      value = '';
    } else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && text[index + 1] === '\n') index += 1;
      row.push(value);
      if (row.some((cell) => cell.length > 0)) rows.push(row);
      row = [];
      value = '';
    } else {
      value += character;
    }
  }
  row.push(value);
  if (row.some((cell) => cell.length > 0)) rows.push(row);
  if (rows.length < 2) return [];
  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ''])));
}

export function parseCharacterDatasetText(text, filename = '') {
  const extension = filename.toLowerCase().split('.').pop();
  if (extension === 'csv') return parseCsv(text);
  if (extension === 'jsonl' || extension === 'ndjson') {
    return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line) => JSON.parse(line));
  }
  const parsed = JSON.parse(text);
  if (Array.isArray(parsed)) return parsed;
  for (const key of ['data', 'rows', 'characters', 'train']) {
    if (Array.isArray(parsed?.[key])) return parsed[key];
  }
  return parsed && typeof parsed === 'object' ? [parsed] : [];
}

function parsePossibleCard(value) {
  if (!value) return null;
  if (typeof value === 'object') return value;
  if (typeof value !== 'string' || !value.trim().startsWith('{')) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function readValue(row, mapping, field) {
  const column = mapping[field];
  return column ? row[column] : undefined;
}

function stringValue(value) {
  if (typeof value === 'string') return value.trim();
  if (value === null || value === undefined) return '';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

function dialogueValue(value) {
  if (Array.isArray(value)) return value;
  const text = stringValue(value);
  if (!text) return [];
  return text.split(/\r?\n/).filter(Boolean).map((line, index) => ({
    role: /^\s*(\{\{user\}\}|user|you)\s*:/i.test(line) || index % 2 === 0 ? 'user' : 'character',
    content: line.replace(/^\s*(\{\{user\}\}|\{\{char\}\}|user|character|you)\s*:\s*/i, ''),
  }));
}

export function characterFromDatasetRow(row, mapping, source = {}) {
  const mappedCard = parsePossibleCard(readValue(row, mapping, 'card'));
  const card = mappedCard?.data || mappedCard || {};
  const pick = (field, ...cardFields) => {
    const mapped = readValue(row, mapping, field);
    if (mapped !== undefined && mapped !== null && mapped !== '') return mapped;
    return cardFields.map((key) => card[key]).find((value) => value !== undefined && value !== null) ?? '';
  };
  const characterBook = card.character_book;
  const tags = pick('tags', 'tags');
  const character = {
    id: null,
    name: stringValue(pick('name', 'name')),
    description: stringValue(pick('description', 'description')),
    personality: stringValue(pick('personality', 'personality')),
    background: stringValue(pick('background', 'background')),
    scenario: stringValue(pick('scenario', 'scenario')),
    first_message: stringValue(pick('first_message', 'first_mes', 'first_message')),
    model_instructions: stringValue(pick('model_instructions', 'system_prompt')),
    speech_style: stringValue(pick('speech_style', 'speech_style')),
    example_dialogue: dialogueValue(pick('example_dialogue', 'mes_example', 'example_dialogue')),
    loreEntries: Array.isArray(characterBook?.entries) ? characterBook.entries.map((entry) => ({
      content: stringValue(entry.content || entry.value),
      keywords: Array.isArray(entry.keys) ? entry.keys : [],
    })) : [],
    tags: Array.isArray(tags) ? tags.map(String) : stringValue(tags).split(',').map((tag) => tag.trim()).filter(Boolean),
    creator: stringValue(pick('creator', 'creator')),
    license: stringValue(pick('license', 'license')),
    avatar: stringValue(pick('image', 'image', 'avatar')) || null,
    chat_role: 'npc',
    created_at: new Date().toISOString(),
    dataset_source: source,
  };
  if (!character.name && character.description) {
    const embeddedName = character.description.match(/(?:^|\n)\s*(?:\*\*)?Name:(?:\*\*)?\s*([^\n]+)/i);
    if (embeddedName) character.name = embeddedName[1].replace(/\*+/g, '').trim();
  }
  return {
    character,
    valid: Boolean(character.name && (character.description || character.model_instructions || character.first_message)),
  };
}
