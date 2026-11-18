import { existsSync } from 'node:fs';
import { defineConfig } from '@playwright/test';

const chrome = process.env.CHROME_PATH || (existsSync('/usr/bin/google-chrome') ? '/usr/bin/google-chrome' : undefined);
const port = process.env.KT_PREVIEW_PORT || '4321';
export default defineConfig({
  testDir: './tests/browser',
  outputDir: './test-results',
  fullyParallel: true,
  workers: 2,
  reporter: 'list',
  use: {
    baseURL: `http://127.0.0.1:${port}/knowledge-tree/`,
    browserName: 'chromium',
    launchOptions: { executablePath: chrome, args: ['--disable-gpu'] },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    // Keep Astro in the foreground so Playwright owns and stops this process.
    command: `npm run preview -- --host 127.0.0.1 --port ${port} --ignore-lock`,
    url: `http://127.0.0.1:${port}/knowledge-tree/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
