import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/models/get-settings', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', settings: {} }),
    });
  });
  await page.route('**/models/save-custom-endpoints', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success' }),
    });
  });
});

async function dismissProviderSetup(page) {
  await page.addInitScript(() => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
    }));
  });
  await page.reload();
}

const primaryPanels = [
  { title: 'Documents', heading: 'Documents' },
  { title: 'Characters', heading: 'Character Library' },
  { title: 'User Profiles', heading: 'User Profiles' },
  { title: 'Audio', heading: 'Audio' },
  { title: 'Memory tools', heading: 'Memory tools' },
  { title: 'Help and guides', heading: 'Learn what you need. Leave the plumbing to us.' },
];

const settingsTabs = [
  'General',
  'Models',
  'Styles',
  'LLM Settings',
  'Image Generation',
  'Characters',
  'Memory Intent',
  'Character review',
  'Memory Browser',
  'About',
];

test('primary navigation and Settings tabs render', async ({ page }, testInfo) => {
  const pageErrors = [];
  const offlineErrors = [];
  const reactErrors = [];
  page.on('pageerror', (error) => {
    if (error.message === 'Failed to fetch') offlineErrors.push(error.message);
    else pageErrors.push(error.message);
  });
  page.on('console', (message) => {
    const text = message.text();
    const knownChessboardWarning =
      text.includes('Legacy context API') && text.includes('chessboardjsx');
    if (message.type() === 'error' && text.startsWith('Warning:') && !knownChessboardWarning) {
      reactErrors.push(text);
    }
  });

  await page.goto('/');
  await expect(page).toHaveTitle('Mirid');
  await dismissProviderSetup(page);

  for (const retiredTitle of ['Code Editor', 'Forensic Linguistics', 'Chess', 'Market Simulator', 'Pool', 'Watch']) {
    await expect(page.getByTitle(retiredTitle, { exact: true })).toHaveCount(0);
  }

  await expect(page.getByTitle('Settings', { exact: true })).toHaveCount(1);

  const inventory = [];
  for (const panel of primaryPanels) {
    const button = page.getByTitle(panel.title, { exact: true });
    await expect(button).toHaveCount(1);
    await button.click();
    await expect(button).toHaveClass(/bg-secondary/);

    await expect(page.getByRole('heading', { name: panel.heading, exact: true })).toBeVisible();

    inventory.push({
      panel: panel.title,
      visibleButtons: await page.locator('button:visible').count(),
    });
  }

  await page.getByTitle('Settings', { exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Open in new window', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();

  for (const name of settingsTabs) {
    const tab = page.getByRole('tab', { name, exact: true });
    await expect(tab).toHaveCount(1);
    await tab.click();
    await expect(tab).toHaveAttribute('aria-selected', 'true');
  }

  await testInfo.attach('control-inventory', {
    body: JSON.stringify(inventory, null, 2),
    contentType: 'application/json',
  });
  await testInfo.attach('expected-offline-errors', {
    body: JSON.stringify(offlineErrors, null, 2),
    contentType: 'application/json',
  });

  expect(reactErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test('Focus and Call modes work without Dual Overlay', async ({ page }) => {
  const endpoint = {
    id: 'endpoint-focus-mode',
    name: 'Focus test model',
    url: 'https://example.test/v1',
    apiKey: 'test-key',
    model: 'provider/focus-test',
    enabled: true,
    rotate_enabled: true,
  };
  await page.addInitScript(({ selectedEndpoint }) => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
      sttEnabled: true,
      ttsEnabled: true,
      customApiEndpoints: [selectedEndpoint],
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', selectedEndpoint.id);
  }, { selectedEndpoint: endpoint });

  await page.goto('/');

  await expect(page.getByText('Dual Overlay', { exact: true })).toHaveCount(0);
  await expect(page.locator('#dual-overlay')).toHaveCount(0);

  await page.getByTitle('Enter Focus Mode', { exact: true }).click();
  await expect(page.getByLabel('Change model', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'TTS', exact: true })).toBeVisible();
  await expect(page.getByTitle('Read new replies automatically', { exact: true })).toBeVisible();
  await expect(page.getByTitle('Turn voice input off', { exact: true })).toBeVisible();

  await page.getByLabel('Change model', { exact: true }).click();
  await expect(page.getByLabel('Model selector', { exact: true })).toBeVisible();
  await expect(page.getByPlaceholder('Search models…', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByLabel('Exit Focus Mode', { exact: true })).toBeVisible();
  await page.getByLabel('Exit Focus Mode', { exact: true }).click();

  await page.getByTitle('Start Call Mode', { exact: true }).click();
  await expect(page.getByRole('button', { name: 'Exit call mode', exact: true })).toBeVisible();
});

test('Memory tools explains its workflows and automates character review', async ({ page }) => {
  const endpoint = {
    id: 'endpoint-memory-review',
    name: 'Memory review',
    url: 'https://example.test/v1',
    apiKey: 'test-key',
    model: 'provider/memory-review',
    enabled: true,
    rotate_enabled: false,
  };
  const character = {
    id: 'character-memory-review',
    name: 'Archivist',
    chat_role: 'npc',
    description: 'A careful keeper of shared history.',
    personality: 'Patient and exact.',
    model_instructions: 'Keep replies factual.',
  };
  const secondCharacter = {
    id: 'character-memory-target',
    name: 'Cartographer',
    chat_role: 'npc',
    description: 'A mapper of unfamiliar places.',
    personality: 'Curious and composed.',
  };
  await page.addInitScript(({ selectedEndpoint, savedCharacters }) => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
      customApiEndpoints: [selectedEndpoint],
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', selectedEndpoint.id);
    localStorage.setItem('llm-characters', JSON.stringify(savedCharacters));
    localStorage.setItem('user-profiles', JSON.stringify({
      profiles: [{ id: 'profile-memory-review', name: 'Test User', preferences: {} }],
      activeProfileId: 'profile-memory-review',
    }));
  }, { selectedEndpoint: endpoint, savedCharacters: [character, secondCharacter] });

  let promptRequest;
  let generateRequest;
  await page.route('**/memory/persona_realignment/prompt_pack', async (route) => {
    promptRequest = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', combined: 'Review this character from the selected context.' }),
    });
  });
  await page.route('**/generate', async (route) => {
    generateRequest = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ text: '{"revised_model_instructions":"Remember established details and avoid repeated questions."}' }),
    });
  });
  await page.route('**/memory/persona_realignment/parse_response', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'success',
        delta_vs_current_instructions: ['Preserve established details across replies.'],
        revised_character_instructions: 'You are Archivist. Preserve established details across replies.',
        revised_model_instructions: 'Remember established details and avoid repeated questions.',
        revised_user_profile_memories: [],
      }),
    });
  });
  await page.route('**/memory/curator/prompt_pack', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', prompt_pack: { combined: 'Review these profile memories.' } }),
    });
  });
  await page.route('**/memory/curator/parse_response', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'success',
        summary: 'Merged one repeated preference.',
        memories: [{ content: 'The user prefers concise explanations.', category: 'preferences', importance: 0.8 }],
      }),
    });
  });
  let profileApplyRequest;
  await page.route('**/memory/curator/apply_profile', async (route) => {
    profileApplyRequest = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', saved: 1 }),
    });
  });
  await page.route('**/memory/agentic/list?*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'success',
        profiles: [{ character_id: character.id, count: 2, insights: [{ id: 'one' }, { id: 'two' }] }],
      }),
    });
  });
  let transferRequest;
  await page.route('**/memory/agentic/copy_to_character', async (route) => {
    transferRequest = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', mode: 'merge', added: 2, target_count: 2 }),
    });
  });

  await page.goto('/');
  await page.getByTitle('Memory tools', { exact: true }).click();

  await expect(page.getByText('Mirid keeps facts about you with your user profile', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: /Refresh a character/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: /Move character memories/ })).toBeVisible();

  await page.getByRole('button', { name: 'Review this character', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Proposed character update', exact: true })).toBeVisible();
  await expect(page.getByLabel('Proposed model instructions')).toHaveValue('Remember established details and avoid repeated questions.');
  expect(promptRequest.character_id).toBe(character.id);
  expect(promptRequest.user_id).toBe('profile-memory-review');
  expect(generateRequest.model_name).toBe(endpoint.id);
  expect(generateRequest.memoryEnabled).toBe(false);

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Save to Archivist', exact: true }).click();
  await expect.poll(async () => page.evaluate(() => {
    const saved = JSON.parse(localStorage.getItem('llm-characters') || '[]');
    return saved.find((item) => item.id === 'character-memory-review')?.model_instructions;
  })).toBe('Remember established details and avoid repeated questions.');

  await page.getByRole('button', { name: /Move character memories/ }).click();
  await expect(page.getByRole('heading', { name: 'Move memories between characters', exact: true })).toBeVisible();
  await page.getByText('Choose a different character', { exact: true }).click();
  await page.getByRole('option', { name: 'Cartographer', exact: true }).click();
  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Copy memories', exact: true }).click();
  await expect(page.getByText('2 memories were added to Cartographer; duplicates were left out.', { exact: true })).toBeVisible();
  expect(transferRequest).toMatchObject({
    user_id: 'profile-memory-review',
    source_character_id: character.id,
    target_character_id: secondCharacter.id,
    mode: 'merge',
  });

  await page.getByRole('button', { name: /Clean my memories/ }).click();
  await expect(page.getByText('This keeps the review in a familiar voice.', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Review memories', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Reviewed memory list', exact: true })).toBeVisible();
  await expect(page.getByText('Merged one repeated preference.', { exact: true })).toBeVisible();
  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Save reviewed memories', exact: true }).click();
  await expect(page.getByText('1 reviewed memory was saved.', { exact: true })).toBeVisible();
  expect(profileApplyRequest).toMatchObject({
    user_id: 'profile-memory-review',
    memories: [expect.objectContaining({ content: 'The user prefers concise explanations.' })],
  });
});

test('chat reveals user profile controls only when requested', async ({ page }) => {
  await page.goto('/');
  await dismissProviderSetup(page);

  const profileToggle = page.getByRole('checkbox', { name: 'Show user profiles', exact: true });
  await expect(profileToggle).not.toBeChecked();
  await expect(page.getByText('No user profile found.', { exact: true })).toHaveCount(0);

  await profileToggle.click();
  await expect(profileToggle).toBeChecked();
  await expect(page.getByText('No user profile found.', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Create profile now', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'User Profiles', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'New Profile', exact: true }).click();
  await page.getByPlaceholder('Profile name').fill('Test User');
  await page.getByRole('button', { name: 'Create', exact: true }).click();

  await page.getByTitle('Chat', { exact: true }).click();
  await expect(page.getByRole('combobox').filter({ hasText: 'Test User' })).toBeVisible();
});

test('chat header opens recent chat history', async ({ page }) => {
  await page.goto('/');
  await dismissProviderSetup(page);

  await page.getByRole('button', { name: 'Recent Chat History', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Chat history', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '+ New Chat', exact: true })).toBeVisible();
});

test('Documents owns local search and exposes optional agent document search', async ({ page }) => {
  await page.route('**/document/list', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'success',
        file_list: [{
          id: 'doc-1',
          filename: 'notes.txt',
          upload_date: '2026-07-21T00:00:00Z',
        }],
      }),
    });
  });

  await page.goto('/');
  await dismissProviderSetup(page);

  await expect(page.getByTitle('Transcript Search', { exact: true })).toHaveCount(0);
  await page.getByTitle('Documents', { exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Documents', exact: true })).toBeVisible();

  const documentContext = page.getByRole('checkbox', { name: 'Enable Document Context', exact: true });
  await page.getByText('Enable Document Context', { exact: true }).click();
  await expect(documentContext).toBeChecked();

  const agentSearch = page.getByRole('checkbox', { name: 'Agent document search', exact: true });
  await expect(agentSearch).toBeDisabled();
  await page.getByRole('checkbox', { name: 'Include notes.txt in document context', exact: true }).click();
  await expect(agentSearch).toBeEnabled();
  await page.getByText('Agent document search', { exact: true }).click();
  await expect(agentSearch).toBeChecked();
});

test('image generation directs an empty local setup into image model installation', async ({ page }) => {
  await page.addInitScript(() => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
      customApiEndpoints: [{
        id: 'image-test-endpoint',
        name: 'Image test chat model',
        url: 'https://example.test/v1',
        apiKey: 'test-key',
        model: 'test/chat-model',
        enabled: true,
      }],
      imageEngine: 'EloDiffusion',
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', 'image-test-endpoint');
  });
  await page.route('**/sd-local/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ available: true, loaded_models: {} }),
    });
  });
  await page.route('**/sd-local/list-models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', models: [] }),
    });
  });
  await page.route('**/model-library/destinations', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        destinations: [
          { type: 'text', label: 'Text / GGUF', path: 'C:\\Users\\Test\\models\\gguf', setting_key: 'modelDirectory', custom: false },
          { type: 'image', label: 'Image generation', path: 'C:\\Users\\Test\\models\\stable-diffusion', setting_key: 'sdModelDirectory', custom: false },
        ],
      }),
    });
  });
  await page.route('**/model-library/recommendations', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ models: [] }) });
  });
  await page.route('**/model-library/huggingface/search?*', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ models: [] }) });
  });
  await page.route('**/system/gpu_info', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ gpus: [] }) });
  });

  await page.goto('/');
  await dismissProviderSetup(page);
  await page.getByTitle('Generate Image', { exact: true }).click();

  await expect(page.getByRole('heading', { name: 'Image generation', exact: true })).toBeVisible();
  await expect(page.getByText('No local image model found', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate Image', exact: true }).last()).toBeDisabled();

  await page.getByRole('button', { name: 'Find an image model', exact: true }).click();
  await expect(page.getByText('Set up local image generation', { exact: true })).toBeVisible();
  await expect(page.getByPlaceholder('Search image models — e.g. SDXL checkpoint')).toHaveValue('stable diffusion checkpoint');
  await expect(page.getByRole('button', { name: 'Civitai', exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Civitai', exact: true }).click();
  await expect(page.getByText("Mirid uses Civitai's official API", { exact: false })).toBeVisible();
});

test('Character Studio opens from the library', async ({ page }) => {
  await page.goto('/');
  await dismissProviderSetup(page);
  await page.getByTitle('Characters', { exact: true }).click();
  await page.getByRole('button', { name: 'Import Dataset', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Bring a character collection into focus.' })).toBeVisible();
  await page.getByRole('button', { name: 'Character Library', exact: true }).click();
  await page.getByRole('button', { name: /Build with Mirid/ }).click();

  await expect(page.getByRole('heading', { name: 'Build a first draft with Mirid' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Build together' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Preview', exact: true })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Review card', exact: true })).toBeVisible();
  await expect(page.getByPlaceholder('Describe the character you have in mind…')).toBeVisible();
});

test('Character Library makes manual creation primary and explains card fields', async ({ page }) => {
  await page.goto('/');
  await dismissProviderSetup(page);
  await page.getByTitle('Characters', { exact: true }).click();

  await expect(page.getByText('Import supports TavernAI and SillyTavern V1 or V2 character cards in JSON or PNG format.', { exact: false })).toBeVisible();
  await expect(page.getByText('Assistant', { exact: true })).toBeVisible();
  await expect(page.getByText('Plain chat. No character card or roleplay instructions.', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'New Character', exact: true }).click();
  await expect(page.getByText('Create New Character', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Description', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Personality', { exact: true })).toBeVisible();
  await expect(page.getByText('The broad character definition. This is the standard card field sometimes called persona or character description.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Suggest', exact: true })).toHaveCount(0);

  await page.getByText('Writing help', { exact: true }).click();
  await expect(page.getByRole('button', { name: 'Suggest', exact: true }).first()).toBeVisible();

  const messageFieldsBefore = await page.locator('textarea[id^="dialogue-"][id$="-content"]').count();
  await page.getByRole('button', { name: 'Add exchange', exact: true }).click();
  await expect(page.locator('textarea[id^="dialogue-"][id$="-content"]')).toHaveCount(messageFieldsBefore + 2);
});

test('manual character writing help is opt-in and never overwrites a draft automatically', async ({ page }) => {
  const endpoint = {
    id: 'endpoint-character-writing-help',
    name: 'Character writing help',
    url: 'https://example.test/v1',
    apiKey: 'test-key',
    model: 'provider/character-writing-help',
    enabled: true,
    rotate_enabled: false,
  };
  await page.addInitScript(({ selectedEndpoint }) => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
      customApiEndpoints: [selectedEndpoint],
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', selectedEndpoint.id);
  }, { selectedEndpoint: endpoint });

  let refinementPayload;
  await page.route('**/character/refine-generated', async (route) => {
    refinementPayload = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'success',
        character_json: {
          ...refinementPayload.character_json,
          description: 'A cartographer who maps places that should not exist.',
        },
      }),
    });
  });

  await page.goto('/');
  await page.getByTitle('Characters', { exact: true }).click();
  await page.getByRole('button', { name: 'New Character', exact: true }).click();
  await page.getByLabel('Description', { exact: true }).fill('A cartographer.');

  expect(refinementPayload).toBeUndefined();
  await page.getByText('Writing help', { exact: true }).click();
  await page.getByLabel('Description', { exact: true }).locator('..').getByRole('button', { name: 'Suggest', exact: true }).click();

  await expect.poll(() => refinementPayload?.character_json?.description).toBe('A cartographer.');
  expect(refinementPayload.model_name).toBe(endpoint.id);
  await expect(page.getByLabel('Description', { exact: true })).toHaveValue('A cartographer.');
  await expect(page.getByText('A cartographer who maps places that should not exist.', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Use suggestion', exact: true }).click();
  await expect(page.getByLabel('Description', { exact: true })).toHaveValue('A cartographer who maps places that should not exist.');
});

test('Help Centre owns /docs and fits a narrow window', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/docs');
  await dismissProviderSetup(page);

  await expect(page).toHaveURL(/\/docs$/);
  await expect(page.getByRole('heading', { name: 'Learn what you need. Leave the plumbing to us.' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Ask an AI to teach me Mirid' })).toBeVisible();
  await expect(page.getByText('NanoGPT Pro', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Subscribe or manage', exact: true })).toHaveAttribute(
    'href',
    /utm_source=mirid.*utm_campaign=mirid-provider-partners/,
  );
  await expect(page.getByText('Mirid partner offer', { exact: true })).toHaveCount(0);

  const hasHorizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  expect(hasHorizontalOverflow).toBe(false);
});

test('NanoGPT subscription filter keeps the standard API endpoint', async ({ page }) => {
  await page.route('**/model-library/nanogpt/subscription-models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: [{
          id: 'zhipu/glm-roleplay-test',
          name: 'GLM Roleplay Test',
          provider: 'zhipu',
          description: 'Fixture model included by this account.',
          maxInputTokens: 131072,
          capabilities: ['reasoning'],
        }],
      }),
    });
  });

  await page.goto('/');
  await dismissProviderSetup(page);
  await page.getByTitle('Settings', { exact: true }).click();
  await page.getByRole('tab', { name: 'Models', exact: true }).click();
  await page.getByRole('button', { name: 'NanoGPT', exact: true }).click();
  await page.getByPlaceholder('Paste your NanoGPT key').fill('nano-subscription-test-key');
  await page.getByRole('button', { name: 'Show subscription-included models', exact: true }).click();

  await expect(page.getByText('GLM Roleplay Test', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Use subscription-covered model', exact: true }).click();

  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('Eloquent-settings') || '{}'));
  expect(stored.nanoGptBillingMode).toBe('subscription');
  expect(stored.customApiEndpoints).toEqual(expect.arrayContaining([
    expect.objectContaining({
      model: 'zhipu/glm-roleplay-test',
      url: 'https://nano-gpt.com/api/v1',
      billing_mode: 'subscription',
    }),
  ]));
});

test('first run offers local and hosted models without assuming a GPU', async ({ page }) => {
  await page.addInitScript(() => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: false,
    }));
  });
  await page.route('**/system/gpu_info', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        gpu_count: 0,
        single_gpu_mode: true,
        gpus: [],
        cuda_available: false,
        compute_mode: 'cpu',
        hosted_models_recommended: true,
        local_gguf_available: true,
      }),
    });
  });
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'You need a model to start' })).toBeVisible();
  await expect(page.getByRole('button', { name: /Download a local model/ })).toBeVisible();
  await expect(page.getByRole('button', { name: /Connect a remote model/ })).toBeVisible();
});

test('chat queues multiple ordinary image attachments', async ({ page }) => {
  await page.addInitScript(() => {
    const endpoint = {
      id: 'endpoint-vision-attachment-test',
      name: 'Vision attachment test',
      url: 'https://example.test/v1',
      apiKey: 'test-key',
      model: 'provider/vision-test',
      enabled: true,
      rotate_enabled: false,
    };
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
      visionModel: 'LFM2.5-VL-450M-Extract',
      customApiEndpoints: [endpoint],
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', endpoint.id);
  });
  await page.goto('/');
  await dismissProviderSetup(page);

  const attachButton = page.getByRole('button', { name: 'Attach images', exact: true });
  await expect(attachButton).toBeEnabled();

  const imageInput = page.locator('form input[type="file"][accept="image/*"][multiple]').first();
  await imageInput.setInputFiles([
    { name: 'first.png', mimeType: 'image/png', buffer: Buffer.from('first') },
    { name: 'second.png', mimeType: 'image/png', buffer: Buffer.from('second') },
  ]);

  await expect(page.getByText('first.png', { exact: true })).toBeVisible();
  await expect(page.getByText('second.png', { exact: true })).toBeVisible();
});


test('a selected model opens a plain chat without asking again', async ({ page }) => {
  const endpoint = {
    id: 'endpoint-clean-chat-test',
    name: 'Clean chat test',
    url: 'https://example.test/v1',
    apiKey: 'test-key',
    model: 'provider/clean-chat-test',
    enabled: true,
    rotate_enabled: false,
  };
  await page.addInitScript(({ selectedEndpoint }) => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
      customApiEndpoints: [selectedEndpoint],
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', selectedEndpoint.id);
  }, { selectedEndpoint: endpoint });

  await page.goto('/');

  await expect(page.getByPlaceholder('Message...', { exact: true })).toBeVisible();
  await expect(page.getByTitle('Change model', { exact: true }).first()).toContainText('clean-chat-test');
  for (const removedText of [
    'What are you working on, Default User?',
    'Choose how you want to work, then write the first line.',
    'Make sense of it',
    'Make something',
    'Talk it through',
    'Read closely',
  ]) {
    await expect(page.getByText(removedText, { exact: true })).toHaveCount(0);
  }
});

test('saving a Character Studio draft selects it visibly in chat', async ({ page }) => {
  const endpoint = {
    id: 'endpoint-character-test',
    name: 'Character test',
    url: 'https://example.test/v1',
    apiKey: 'test-key',
    model: 'provider/character-test',
    enabled: true,
    rotate_enabled: false,
  };
  await page.addInitScript(({ selectedEndpoint }) => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      modelSetupRequired: false,
      customApiEndpoints: [selectedEndpoint],
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', selectedEndpoint.id);
  }, { selectedEndpoint: endpoint });
  await page.route('**/character/generate-from-conversation', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'success',
        character_json: {
          name: 'Rhys Test Character',
          description: 'A test character with a clearly visible selection state.',
          personality: 'Direct and observant.',
          scenario: 'Testing Mirid.',
          first_message: 'I am ready.',
        },
      }),
    });
  });

  await page.goto('/');
  await page.getByTitle('Characters', { exact: true }).click();
  await page.getByRole('button', { name: /Build with Mirid/ }).click();
  await page.getByPlaceholder(/Describe the character you have in mind/).fill('Create a test character.');
  await page.getByRole('button', { name: 'Send to character builder', exact: true }).click();
  await expect(page.getByText('Rhys Test Character', { exact: true }).first()).toBeVisible();
  await page.getByRole('button', { name: 'Save and chat', exact: true }).click();

  await expect(page.getByRole('button', { name: 'Character Library', exact: true })).toBeVisible();
  await expect(page.locator('button[role="combobox"]').filter({ hasText: 'Rhys Test Character' })).toBeVisible();
  await expect.poll(async () => page.evaluate(() => {
    const saved = JSON.parse(localStorage.getItem('llm-characters') || '[]');
    return saved.some((character) => character.name === 'Rhys Test Character');
  })).toBe(true);

  await page.locator('button[role="combobox"]').filter({ hasText: 'Rhys Test Character' }).click();
  await page.getByRole('option', { name: 'Assistant · plain chat', exact: true }).click();
  await expect(page.locator('button[role="combobox"]').filter({ hasText: 'Assistant · plain chat' })).toBeVisible();
});

test('empty auto-routing pool falls back to the selected API model', async ({ page }) => {
  const selectedEndpoint = {
    id: 'endpoint-selected-fallback',
    name: 'Selected fallback',
    url: 'https://example.test/v1',
    apiKey: 'test-key',
    model: 'provider/selected-fallback',
    enabled: true,
    rotate_enabled: false,
  };

  await page.addInitScript(({ endpoint }) => {
    const settings = JSON.parse(localStorage.getItem('Eloquent-settings') || '{}');
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      ...settings,
      providerSetupCompleted: true,
      streamResponses: false,
      apiEndpointRoundRobinEnabled: true,
      customApiEndpoints: [endpoint],
    }));
    localStorage.setItem('Eloquent-last-primary-api-model', endpoint.id);
  }, { endpoint: selectedEndpoint });

  await page.route('**/models/get-settings', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', settings: {} }),
    });
  });

  let generatePayload;
  await page.route('**/generate', async (route) => {
    generatePayload = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ text: 'Routing works.' }),
    });
  });

  await page.goto('/');

  const modelButton = page.getByTitle('Change model', { exact: true }).first();
  await expect(modelButton).toContainText('selected-fallback');
  await modelButton.click();
  await expect(page.getByText('Paused · selected model will be used', { exact: true })).toBeVisible();

  await page.getByPlaceholder('Message...', { exact: true }).fill('Use the selected endpoint.');
  await page.locator('form button[type="submit"]').click();
  await expect.poll(() => generatePayload).toBeTruthy();

  expect(generatePayload).toEqual(expect.objectContaining({
    model_name: selectedEndpoint.id,
    selected_model: selectedEndpoint.id,
    round_robin_enabled: false,
  }));
  await expect(page.getByText('Routing works.', { exact: true })).toBeVisible();
});
