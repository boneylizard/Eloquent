const asTrimmedString = (value) => (
  typeof value === 'string' ? value.trim() : ''
);

export function replaceCharacterCardTags(text, characterName = 'Character', userName = 'User') {
  return asTrimmedString(text)
    .replace(/{{char}}/gi, characterName || 'Character')
    .replace(/{{user}}/gi, userName || 'User');
}

export function resolveCharacterPromptOverride(
  override,
  original,
  characterName = 'Character',
  userName = 'User',
) {
  const fallback = replaceCharacterCardTags(original, characterName, userName);
  const custom = replaceCharacterCardTags(override, characterName, userName);
  if (!custom) return fallback;
  return custom.replace(/{{original}}/gi, fallback).trim();
}

export function resolveCharacterPostHistoryInstructions(
  character,
  userName = 'User',
  original = '',
) {
  if (!character) return '';
  return resolveCharacterPromptOverride(
    character.post_history_instructions,
    original,
    character.name || 'Character',
    userName,
  );
}

export function buildCharacterGreetingOptions(character, userName = 'User') {
  if (!character) return [];
  const characterName = character.name || 'Character';
  const rawGreetings = [
    character.first_message,
    ...(Array.isArray(character.alternate_greetings) ? character.alternate_greetings : []),
  ];
  const seen = new Set();
  const greetings = [];

  for (const rawGreeting of rawGreetings) {
    const greeting = replaceCharacterCardTags(rawGreeting, characterName, userName);
    if (!greeting || seen.has(greeting)) continue;
    seen.add(greeting);
    greetings.push(greeting);
  }

  return greetings;
}

export function createCharacterGreetingState(options, index = 0) {
  const greetings = Array.isArray(options)
    ? options.filter((value) => typeof value === 'string' && value.trim())
    : [];
  if (greetings.length < 2) return null;
  const normalizedIndex = ((Number(index) || 0) % greetings.length + greetings.length) % greetings.length;
  return {
    options: greetings,
    index: normalizedIndex,
  };
}

export function cycleCharacterGreetingMessage(message, direction = 'next') {
  const state = message?.characterGreeting;
  const options = Array.isArray(state?.options) ? state.options : [];
  if (options.length < 2) return message;
  const currentIndex = Number.isInteger(state.index) ? state.index : 0;
  const delta = direction === 'previous' || direction === 'prev' || direction === -1 ? -1 : 1;
  const nextIndex = (currentIndex + delta + options.length) % options.length;
  return {
    ...message,
    content: options[nextIndex],
    characterGreeting: {
      ...state,
      index: nextIndex,
    },
  };
}

function normaliseLoreKeys(value) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item) => typeof item === 'string')
    .map((item) => item.trim())
    .filter(Boolean);
}

function loreKeyMatches(text, key, caseSensitive) {
  if (!key) return false;
  if (caseSensitive) return text.includes(key);
  return text.toLocaleLowerCase().includes(key.toLocaleLowerCase());
}

export function resolveCharacterLoreEntries(character, text = '') {
  if (!character || !Array.isArray(character.loreEntries)) return [];

  const sourceText = typeof text === 'string' ? text : '';
  return character.loreEntries
    .map((entry, index) => {
      if (!entry || typeof entry !== 'object' || !asTrimmedString(entry.content)) return null;
      const cardEntry = entry.tavern_entry && typeof entry.tavern_entry === 'object'
        ? entry.tavern_entry
        : {};
      if (cardEntry.enabled === false) return null;

      const primaryKeys = normaliseLoreKeys(
        entry.keywords?.length ? entry.keywords : (cardEntry.keys || cardEntry.key),
      );
      const secondaryKeys = normaliseLoreKeys(
        cardEntry.secondary_keys || cardEntry.secondaryKeys,
      );
      const caseSensitive = cardEntry.case_sensitive === true || cardEntry.caseSensitive === true;
      const primaryMatch = primaryKeys.some((key) => loreKeyMatches(sourceText, key, caseSensitive));
      const secondaryMatch = secondaryKeys.some((key) => loreKeyMatches(sourceText, key, caseSensitive));
      const constant = cardEntry.constant === true;
      const selective = cardEntry.selective === true;
      const triggered = constant || (primaryMatch && (!selective || secondaryMatch));
      if (!triggered) return null;

      return {
        keyword: constant
          ? null
          : primaryKeys.find((key) => loreKeyMatches(sourceText, key, caseSensitive)) || null,
        content: asTrimmedString(entry.content),
        importance: entry.importance || cardEntry.priority || 0.8,
        source: 'loreEntries',
        insertionOrder: Number.isFinite(Number(cardEntry.insertion_order))
          ? Number(cardEntry.insertion_order)
          : index,
      };
    })
    .filter(Boolean)
    .sort((left, right) => left.insertionOrder - right.insertionOrder);
}
