import { test, expect } from '@playwright/test';

test('renders Mermaid only after generation finishes', async ({ page }) => {
  await page.goto('/tests/e2e/fixtures/mermaid-smoke.html');

  await expect(page.getByText('Finishing the diagram before rendering it.')).toBeVisible();
  await expect(page.locator('svg')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('Mirid could not render this diagram.')).toHaveCount(0);
  await expect(page.locator('svg')).toContainText('Start');
  await expect(page.locator('svg')).toContainText('Done');
});
