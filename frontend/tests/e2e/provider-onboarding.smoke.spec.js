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
  await page.route('https://openrouter.ai/api/v1/models*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        data: [
          {
            id: 'example/chat:free',
            name: 'Example Chat',
            architecture: { output_modalities: ['text'] },
            pricing: { prompt: '0', completion: '0' },
          },
          {
            id: 'example/music',
            name: 'Example Music',
            architecture: { output_modalities: ['text', 'audio'] },
            pricing: { prompt: '0', completion: '0' },
          },
        ],
      }),
    });
  });
});

test('first launch explains the missing chat model and offers two clear paths', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'You need a model to start' })).toBeVisible();
  await expect(page.getByText('It did not include a chat model', { exact: false })).toBeVisible();
  await expect(page.getByText('about 3.3 GB', { exact: false })).toBeVisible();
  await expect(page.getByText('about 9 GB', { exact: false })).toBeVisible();
  await expect(page.getByText('Those files live in Windows app data, not beside the app.', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'View source', exact: true })).toHaveAttribute('href', 'https://github.com/boneylizard/Eloquent');
  await expect(page.getByRole('button', { name: /Download a local model/ })).toBeVisible();
  await expect(page.getByRole('button', { name: /Connect a remote model/ })).toBeVisible();
  await expect(page.getByText('What are you working on, Default User?', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Choose how you want to work, then write the first line.', { exact: true })).toHaveCount(0);
});

test('OpenRouter first launch selects only its free router automatically', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Connect a remote model/ }).click();

  await expect(page.getByRole('heading', { name: 'Connect a model provider' })).toBeVisible();
  await page.getByText('Other providers', { exact: true }).click();
  for (const provider of ['OpenAI', 'Anthropic', 'Google Gemini', 'Mistral', 'xAI', 'Meta Model API']) {
    await expect(page.getByText(provider, { exact: true })).toBeVisible();
  }

  await page.getByLabel('OpenRouter').fill('sk-or-v1-browser-smoke');
  await page.locator('#first-run-openai-key').fill('sk-openai-browser-smoke');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Connect a model provider' })).toHaveCount(0);

  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('Eloquent-settings') || '{}'));
  expect(stored.providerSetupCompleted).toBe(true);
  expect(stored.openRouterApiKey).toBe('sk-or-v1-browser-smoke');
  expect(stored.openAiApiKey).toBe('sk-openai-browser-smoke');
  expect(stored.customApiEndpoints).toEqual(expect.arrayContaining([
    expect.objectContaining({ model: 'openrouter/free', url: 'https://openrouter.ai/api/v1' }),
  ]));
  expect(stored.customApiEndpoints).toHaveLength(1);
});

test('local first launch opens model setup without locking navigation', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Download a local model/ }).click();

  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Models', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('Choose one model to start', { exact: true })).toBeVisible();
  await expect(page.getByText(/Mirid.*Picks/)).toBeVisible();

  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('Eloquent-settings') || '{}'));
  expect(stored.providerSetupCompleted).toBe(true);
  expect(stored.modelSetupRequired).toBe(true);
  expect(stored.modelSetupSource).toBe('huggingface');

  await page.getByTitle('Chat', { exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: /Select model/ })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Choose a model to start', exact: true })).toBeVisible();
});

test('first launch exposes persistent interface zoom', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Make interface larger', exact: true }).click();

  await expect.poll(() => page.evaluate(() => localStorage.getItem('mirid-interface-zoom'))).toBe('1.2');
  await expect.poll(() => page.evaluate(() => document.documentElement.style.zoom)).toBe('1.2');

  await page.keyboard.press('Control+=');
  await expect.poll(() => page.evaluate(() => localStorage.getItem('mirid-interface-zoom'))).toBe('1.3');

  await page.reload();
  await expect.poll(() => page.evaluate(() => document.documentElement.style.zoom)).toBe('1.3');
});
