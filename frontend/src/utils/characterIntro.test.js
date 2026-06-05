import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  parseCharacterIntroResponse,
  repairTruncatedJsonBlob,
  salvageIntroFields,
  introJsonIsPartialUsable,
  introJsonIsFullyUsable,
  isCharacterIntroReady,
  buildCharacterIntroRepairPrompt,
  formatCharacterIntroAsMarkdown,
  buildCharacterIntroSeedMessages,
  deriveIntroChatTitle,
  conversationAcceptsIntroTitle,
  DEFAULT_CHAT_TITLE,
} from './characterIntro.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const validFixture = JSON.parse(
  readFileSync(join(__dirname, '__fixtures__', 'characterIntroValid.json'), 'utf8')
);

describe('characterIntro parse/repair', () => {
  it('parses valid intro JSON', () => {
    const raw = JSON.stringify(validFixture);
    const result = parseCharacterIntroResponse(raw);
    assert.equal(result.structured, true);
    assert.equal(result.partial, false);
    assert.equal(isCharacterIntroReady(result), true);
    assert.equal(introJsonIsFullyUsable(result.data), true);
    assert.equal(result.data.headline, validFixture.headline);
  });

  it('parses fenced JSON', () => {
    const raw = '```json\n' + JSON.stringify(validFixture) + '\n```';
    const result = parseCharacterIntroResponse(raw);
    assert.equal(isCharacterIntroReady(result), true);
  });

  it('repairs truncated JSON blob', () => {
    const broken =
      '{"headline": "Edge of the map", "who_they_are": "A wandering smith who speaks bluntly", '
      + '"how_they_engage": "Direct and practical';
    const repaired = repairTruncatedJsonBlob(broken);
    const parsed = JSON.parse(repaired);
    assert.equal(parsed.headline, 'Edge of the map');
    assert.ok(parsed.who_they_are);
  });

  it('salvages fields from truncated stream', () => {
    const blob =
      '{"headline": "Lantern light", "who_they_are": "Keeps secrets in plain sight", '
      + '"how_they_engage": "Warm but watchful';
    const salvaged = salvageIntroFields(blob);
    assert.equal(salvaged.headline, 'Lantern light');
    assert.equal(salvaged.who_they_are, 'Keeps secrets in plain sight');
    assert.equal(salvaged._salvaged, true);
    const result = parseCharacterIntroResponse(blob);
    assert.equal(isCharacterIntroReady(result), true);
    assert.equal(result.data.headline, 'Lantern light');
    assert.ok(result.salvaged || result.partial || introJsonIsFullyUsable(result.data));
  });

  it('marks partial when only headline and one section', () => {
    const minimal = {
      headline: 'First light',
      who_they_are: 'Calm and deliberate.',
    };
    const result = parseCharacterIntroResponse(JSON.stringify(minimal));
    assert.equal(isCharacterIntroReady(result), true);
    assert.equal(introJsonIsPartialUsable(result.data), true);
    assert.equal(introJsonIsFullyUsable(result.data), false);
    assert.equal(result.partial, true);
  });

  it('returns non-ready for garbage with no fields', () => {
    const result = parseCharacterIntroResponse('Thanks for chatting! No JSON here.');
    assert.equal(isCharacterIntroReady(result), false);
    assert.equal(result.structured, false);
  });

  it('repair prompt includes schema and broken JSON', () => {
    const prompt = buildCharacterIntroRepairPrompt('{"headline": "broken');
    assert.match(prompt, /headline/);
    assert.match(prompt, /BROKEN_JSON/);
    assert.match(prompt, /who_they_are/);
  });

  it('formats intro JSON as markdown for chat seed', () => {
    const md = formatCharacterIntroAsMarkdown(validFixture);
    assert.match(md, /^## A quiet threshold/);
    assert.match(md, /### Who they are/);
    assert.match(md, /Mira is a sharp-witted archivist/);
    assert.match(md, /### In their voice/);
    assert.match(md, /wrong shelf again/);
    assert.match(md, /archives.*trust/);
  });

  it('derives sidebar title from headline', () => {
    const result = parseCharacterIntroResponse(JSON.stringify(validFixture));
    assert.equal(deriveIntroChatTitle(result, { characterName: 'Mira' }), validFixture.headline);
    const longHeadline = 'A'.repeat(50);
    const longResult = parseCharacterIntroResponse(JSON.stringify({
      ...validFixture,
      headline: longHeadline,
    }));
    assert.equal(deriveIntroChatTitle(longResult), `${'A'.repeat(37)}...`);
    assert.equal(deriveIntroChatTitle(parseCharacterIntroResponse('not json')), null);
  });

  it('conversationAcceptsIntroTitle respects manual titles', () => {
    assert.equal(conversationAcceptsIntroTitle({ name: DEFAULT_CHAT_TITLE, requiresTitle: true }), true);
    assert.equal(conversationAcceptsIntroTitle({ name: 'Custom', titleSource: 'manual' }), false);
    assert.equal(
      conversationAcceptsIntroTitle({ name: 'A quiet threshold', titleSource: 'intro', requiresTitle: false }),
      true
    );
    assert.equal(
      conversationAcceptsIntroTitle({ name: 'My thread', requiresTitle: false, titleSource: 'first_message' }),
      false
    );
  });

  it('builds a single bot seed message from intro result', () => {
    const result = parseCharacterIntroResponse(JSON.stringify(validFixture));
    const msgs = buildCharacterIntroSeedMessages(result, {
      character: { id: 'c1', name: 'Mira' },
      generateId: () => 'intro-msg-1',
    });
    assert.equal(msgs.length, 1);
    assert.equal(msgs[0].role, 'bot');
    assert.equal(msgs[0].id, 'intro-msg-1');
    assert.equal(msgs[0].isCharacterIntro, true);
    assert.match(msgs[0].content, /## A quiet threshold/);
  });
});
