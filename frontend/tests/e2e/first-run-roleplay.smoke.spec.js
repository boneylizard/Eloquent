import { expect, test } from '@playwright/test';

test('first run offers three unrestricted starting paths before runtime install', async ({ page }) => {
  await page.goto('/?preview=first-run-purpose');

  await expect(page.getByRole('heading', { name: 'What will you mostly use Mirid for?' })).toBeVisible();
  await expect(page.getByRole('radio')).toHaveCount(3);
  await expect(page.getByRole('button', { name: 'Install Mirid’s engine' })).toBeDisabled();

  await page.getByRole('radio', { name: /Use Mirid with SillyTavern/ }).click();
  await expect(page.getByText('Keep port 8000 free for Mirid.', { exact: true })).toBeVisible();
  await expect(page.getByText('Your choice changes the welcome, not your access.', { exact: true })).toBeVisible();
  await expect(page.getByText('Advanced first-run settings', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Install Mirid’s engine' })).toBeEnabled();
});

test('SillyTavern primary use opens its guide without restricting Mirid', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      providerSetupCompleted: true,
      modelSetupRequired: false,
      primaryUse: 'sillytavern',
      sillyTavernSetupCompleted: false,
    }));
  });
  await page.route('**/integrations/sillytavern/capabilities', (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ready' }),
  }));
  await page.route('**/v1/models', (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ data: [{ id: 'test-model' }] }),
  }));
  await page.route('**/tts/voices', (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ kokoro_voices: [{ id: 'af_heart' }] }),
  }));
  await page.route('**/stt/available-engines', (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ available_engines: ['whisper'] }),
  }));
  await page.route('**/sdapi/v1/sd-models', (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify([]),
  }));

  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Connect SillyTavern to Mirid' })).toBeVisible();
  await expect(page.getByText('Keep port 8000 free for Mirid', { exact: true })).toBeVisible();
  await expect(page.getByTitle('SillyTavern setup')).toBeVisible();

  await page.getByRole('button', { name: 'Use Mirid normally' }).click();
  await expect(page.getByPlaceholder('Choose a model to start')).toBeVisible();

  await page.getByTitle('SillyTavern setup').click();
  await page.getByRole('button', { name: 'Mark guide complete' }).click();
  await expect(page.getByRole('button', { name: 'Guide complete' })).toBeDisabled();
  await expect.poll(() => page.evaluate(() => (
    JSON.parse(localStorage.getItem('Eloquent-settings') || '{}').sillyTavernSetupCompleted
  ))).toBe(true);
});

test('character-room welcome stays spacious at the default interface zoom', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('mirid-interface-zoom', '1.1'));
  await page.goto('/?preview=roleplay-welcome');

  const dialog = page.getByRole('dialog');
  await expect(page.getByRole('heading', { name: 'Welcome back to character-first AI.' })).toBeVisible();
  const firstPageBox = await dialog.boundingBox();
  expect(firstPageBox?.y).toBeGreaterThanOrEqual(0);
  expect((firstPageBox?.y || 0) + (firstPageBox?.height || 0)).toBeLessThanOrEqual(720);

  await page.getByRole('button', { name: 'Show me around' }).click();
  await expect(page.getByRole('heading', { name: 'Three ideas organise the room.' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Open character library' })).toBeVisible();
  const guideBox = await dialog.boundingBox();
  expect(guideBox?.y).toBeGreaterThanOrEqual(0);
  expect((guideBox?.y || 0) + (guideBox?.height || 0)).toBeLessThanOrEqual(720);
});
