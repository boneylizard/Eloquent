import { expect, test } from '@playwright/test';

const TEST_PROMPT = 'A blue glass lighthouse under a quiet moon';
const GENERATED_PATH = '/static/generated_images/mirid-image-smoke.png';
const ONE_PIXEL_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
);

function jsonResponse(route, body) {
  return route.fulfill({
    status: 200,
    contentType: 'application/json',
    headers: { 'access-control-allow-origin': '*' },
    body: JSON.stringify(body),
  });
}

async function prepareLocalImageEngine(page) {
  await page.addInitScript(() => {
    localStorage.setItem('Eloquent-settings', JSON.stringify({
      providerSetupCompleted: true,
      modelSetupRequired: false,
      imageEngine: 'EloDiffusion',
    }));
  });
  await page.route('**/models/get-settings', (route) => jsonResponse(route, {
    status: 'success',
    settings: {},
  }));
  await page.route('**/models/save-custom-endpoints', (route) => jsonResponse(route, {
    status: 'success',
  }));
  await page.route('**/sd-local/status', (route) => jsonResponse(route, {
    available: true,
    loaded_models: { 0: 'smoke-checkpoint.safetensors' },
  }));
  await page.route('**/sd-local/list-models', (route) => jsonResponse(route, {
    status: 'success',
    models: ['smoke-checkpoint.safetensors'],
  }));
  await page.route('**/sd-local/adetailer-status', (route) => jsonResponse(route, {
    available: false,
  }));
  await page.route('**/sd-local/adetailer-models', (route) => jsonResponse(route, {
    available: false,
    models: [],
  }));
}

function generatedImageResponse(path = GENERATED_PATH) {
  return {
    status: 'success',
    image_urls: [path],
    parameters: {
      width: 512,
      height: 512,
      steps: 20,
      cfg_scale: 7,
      sampler: 'dpmpp2m',
      seed: 42,
      sd_model_checkpoint: 'smoke-checkpoint.safetensors',
    },
  };
}

test('generated image loads from the backend, retries visibly, and survives immediate reload', async ({ page }) => {
  let generatedMediaRequests = 0;
  const requestedMediaUrls = [];

  await prepareLocalImageEngine(page);
  await page.route('**/sd-local/txt2img', (route) => jsonResponse(route, {
    ...generatedImageResponse(),
  }));
  await page.route(`**${GENERATED_PATH}*`, async (route) => {
    generatedMediaRequests += 1;
    requestedMediaUrls.push(route.request().url());
    if (generatedMediaRequests === 1) {
      await route.fulfill({
        status: 404,
        headers: { 'cache-control': 'no-store' },
        body: 'missing once for retry coverage',
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'image/png',
      headers: { 'cache-control': 'no-store' },
      body: ONE_PIXEL_PNG,
    });
  });

  await page.goto('/');
  await page.getByRole('button', { name: /Generate an image/ }).click();
  await expect(page.getByRole('heading', { name: 'Image generation', exact: true })).toBeVisible();

  await page.getByLabel('Prompt', { exact: true }).fill(TEST_PROMPT);
  await page.getByRole('button', { name: 'Generate Image', exact: true }).last().click();

  const loadAlert = page.getByRole('alert').filter({
    hasText: 'couldn’t load this image',
  });
  await expect(loadAlert).toBeVisible();
  await expect(loadAlert.getByRole('button', { name: 'Try loading again', exact: true })).toBeVisible();

  expect(requestedMediaUrls[0]).toBe(`http://localhost:8000${GENERATED_PATH}`);
  await loadAlert.getByRole('button', { name: 'Try loading again', exact: true }).click();
  await expect(page.getByRole('img', { name: TEST_PROMPT, exact: true })).toBeVisible();
  expect(requestedMediaUrls[1]).toContain(`http://localhost:8000${GENERATED_PATH}?mirid_retry=1`);

  await page.getByTitle('Chat history', { exact: true }).click();
  let historyDialog = page.getByRole('dialog', { name: 'Chat history', exact: true });
  await expect(historyDialog.locator('button[title="New Chat"]')).toHaveCount(1);

  // Do not wait for the ordinary two-second save indicator. The generation
  // completion itself must durably commit the chat before it becomes visible.
  await page.reload();
  await page.getByTitle('Chat history', { exact: true }).click();
  historyDialog = page.getByRole('dialog', { name: 'Chat history', exact: true });
  const savedConversation = historyDialog.locator('button[title="New Chat"]');
  await expect(savedConversation).toHaveCount(1);
  await savedConversation.click();

  await expect(page.getByRole('img', { name: TEST_PROMPT, exact: true })).toBeVisible();
  expect(requestedMediaUrls.at(-1)).toBe(`http://localhost:8000${GENERATED_PATH}`);
});

test('a long image job stays with the chat that started it when the user switches chats', async ({ page }) => {
  let startGeneration;
  let finishGeneration;
  const generationStarted = new Promise((resolve) => { startGeneration = resolve; });
  const generationGate = new Promise((resolve) => { finishGeneration = resolve; });

  await prepareLocalImageEngine(page);
  await page.route('**/sd-local/txt2img', async (route) => {
    startGeneration();
    await generationGate;
    await jsonResponse(route, generatedImageResponse());
  });
  await page.route(`**${GENERATED_PATH}*`, (route) => route.fulfill({
    status: 200,
    contentType: 'image/png',
    headers: { 'cache-control': 'no-store' },
    body: ONE_PIXEL_PNG,
  }));

  await page.goto('/');
  await page.getByRole('button', { name: /Generate an image/ }).click();
  await page.getByLabel('Prompt', { exact: true }).fill(TEST_PROMPT);
  await page.getByRole('button', { name: 'Generate Image', exact: true }).last().click();
  await generationStarted;

  await page.getByRole('button', { name: 'Close image generation', exact: true }).click();
  await page.getByRole('main').getByRole('button', { name: 'New Chat', exact: true }).last().click();
  await page.getByTitle('Chat history', { exact: true }).click();

  let historyDialog = page.getByRole('dialog', { name: 'Chat history', exact: true });
  await expect(historyDialog.locator('button[title="New Chat"]')).toHaveCount(2);

  finishGeneration();
  await expect.poll(async () => {
    const debug = await page.evaluate(() => window.eloquentChatStorage.debug());
    return debug.catalogIds.some((entry) => entry.messageCount === 1)
      && debug.shardCount === 1;
  }, { timeout: 10_000 }).toBe(true);

  // The currently open second chat must remain untouched.
  await expect(page.getByRole('img', { name: TEST_PROMPT, exact: true })).toHaveCount(0);

  // History is newest-first: the second row is the older chat that launched
  // the image job.
  await historyDialog.locator('button[title="New Chat"]').nth(1).click();
  await expect(page.getByRole('img', { name: TEST_PROMPT, exact: true })).toBeVisible();

  await page.reload();
  await page.getByTitle('Chat history', { exact: true }).click();
  historyDialog = page.getByRole('dialog', { name: 'Chat history', exact: true });
  await expect(historyDialog.locator('button[title="New Chat"]')).toHaveCount(1);
  await historyDialog.locator('button[title="New Chat"]').click();
  await expect(page.getByRole('img', { name: TEST_PROMPT, exact: true })).toBeVisible();
});

test('deleting the owning chat during generation does not resurrect it or contaminate another chat', async ({ page }) => {
  let startGeneration;
  let finishGeneration;
  const generationStarted = new Promise((resolve) => { startGeneration = resolve; });
  const generationGate = new Promise((resolve) => { finishGeneration = resolve; });

  await prepareLocalImageEngine(page);
  await page.route('**/sd-local/txt2img', async (route) => {
    startGeneration();
    await generationGate;
    await jsonResponse(route, generatedImageResponse());
  });
  await page.route(`**${GENERATED_PATH}*`, (route) => route.fulfill({
    status: 200,
    contentType: 'image/png',
    body: ONE_PIXEL_PNG,
  }));

  await page.goto('/');
  await page.getByRole('button', { name: /Generate an image/ }).click();
  await page.getByLabel('Prompt', { exact: true }).fill(TEST_PROMPT);
  await page.getByRole('button', { name: 'Generate Image', exact: true }).last().click();
  await generationStarted;

  await page.getByRole('button', { name: 'Close image generation', exact: true }).click();
  await page.getByRole('main').getByRole('button', { name: 'New Chat', exact: true }).last().click();
  await page.getByTitle('Chat history', { exact: true }).click();

  const historyDialog = page.getByRole('dialog', { name: 'Chat history', exact: true });
  const olderChatButton = historyDialog.locator('button[title="New Chat"]').nth(1);
  await expect(olderChatButton).toBeVisible();
  await olderChatButton.locator('..').locator('button').nth(1).click();
  await page.getByRole('button', { name: 'Delete', exact: true }).click();
  await expect(historyDialog.locator('button[title="New Chat"]')).toHaveCount(1);

  finishGeneration();
  await expect.poll(async () => {
    const debug = await page.evaluate(() => window.eloquentChatStorage.debug());
    return debug.bannedIds.length === 1
      && debug.catalogIds.length === 1
      && debug.shardCount === 0;
  }, { timeout: 10_000 }).toBe(true);
  await expect(page.getByRole('img', { name: TEST_PROMPT, exact: true })).toHaveCount(0);

  await page.reload();
  await page.getByTitle('Chat history', { exact: true }).click();
  const reloadedHistory = page.getByRole('dialog', { name: 'Chat history', exact: true });
  await expect(reloadedHistory.locator('button[title="New Chat"]')).toHaveCount(1);
  await expect(page.getByRole('img', { name: TEST_PROMPT, exact: true })).toHaveCount(0);
});
