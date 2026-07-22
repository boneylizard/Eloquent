import { expect, test } from '@playwright/test';

test('purpose selection leads with roleplay before the runtime install', async ({ page }) => {
  await page.goto('/?preview=first-run-purpose');

  await expect(page.getByRole('heading', { name: 'What will you mostly use Mirid for?' })).toBeVisible();
  await expect(page.getByRole('radio')).toHaveCount(6);
  await expect(page.getByRole('button', { name: 'Install Mirid’s engine' })).toBeDisabled();

  await page.getByRole('radio', { name: /Roleplay & characters/ }).click();
  await expect(page.getByText('A character-first welcome.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Advanced first-run settings' }).click();
  await expect(page.getByRole('button', { name: 'Make interface smaller' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Make interface larger' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Install Mirid’s engine' })).toBeEnabled();
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
