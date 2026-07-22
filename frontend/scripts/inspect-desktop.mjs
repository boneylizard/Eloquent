import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from '@playwright/test';

function readArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith('--')) continue;
    const key = value.slice(2);
    const next = values[index + 1];
    if (!next || next.startsWith('--')) {
      result[key] = true;
    } else {
      result[key] = next;
      index += 1;
    }
  }
  return result;
}

const args = readArguments(process.argv.slice(2));
const port = Number(args.port || 9229);
const outputPath = args.output ? path.resolve(args.output) : null;
const screenshotPath = args.screenshot ? path.resolve(args.screenshot) : null;
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
const pages = browser.contexts().flatMap((context) => context.pages());
const page = pages.find((candidate) => candidate.url().startsWith('http://tauri.localhost'));

if (!page) {
  throw new Error('Mirid was not found among the WebView2 debugging targets.');
}

await page.waitForLoadState('domcontentloaded');
const firstRunHeading = page.getByRole('heading', {
  name: /Start with a model that already works|This laptop is ready for hosted models/,
});
if (args['expect-first-run']) {
  await firstRunHeading.waitFor({ state: 'visible', timeout: 30_000 });
}

const state = await page.evaluate(() => {
  const text = document.body?.innerText || '';
  return {
    capturedAt: new Date().toISOString(),
    title: document.title,
    url: window.location.href,
    theme: document.documentElement.dataset.theme || '',
    bodyTextSample: text.slice(0, 2_000),
    firstRunSetupVisible:
      text.includes('Start with a model that already works')
      || text.includes('This laptop is ready for hosted models'),
  };
});

if (args['expect-dark'] && state.theme !== 'dark') {
  throw new Error(`Expected Mirid to begin in dark mode; the active theme is '${state.theme || 'unset'}'.`);
}

if (screenshotPath) {
  await mkdir(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: false });
}

if (outputPath) {
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
}

await new Promise((resolve) => {
  process.stdout.write(`${JSON.stringify(state, null, 2)}\n`, resolve);
});
process.exit(0);
