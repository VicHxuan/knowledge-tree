import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
const contentMap = JSON.parse(readFileSync(new URL('../../src/data/content-map.json', import.meta.url), 'utf8'));
const published = new Set(contentMap.entries.filter(e => e.status === 'published').map(e => `/knowledge-tree/${e.slug}/`));
test('论文筛选、空结果、重置和固定链接定位', async ({ page }) => {
  await page.goto('papers/');
  await expect(page.locator('.paper-card').first()).toBeVisible();
  const total = await page.locator('.paper-card').count();
  expect(total).toBeGreaterThan(50);
  await page.getByRole('searchbox').fill('Schulman Proximal');
  await expect(page.locator('.paper-card:visible')).toHaveCount(1);
  await expect(page.locator('#ppo')).toBeVisible();
  await page.getByRole('searchbox').fill('不存在的论文xyz');
  await expect(page.locator('.paper-empty')).toBeVisible();
  await page.getByRole('button', {name:'重置筛选'}).click();
  await expect(page.locator('.paper-card:visible')).toHaveCount(total);
  await page.goto('papers/?query=nonexistent#ppo');
  await expect(page.locator('#ppo')).toBeVisible();
  await expect(page.getByRole('searchbox')).toHaveValue('');
});
test('文章引用跳转文献卡片，关联目录可返回知识文章', async ({ page }) => {
  await page.goto('learning-based-planning/rl-algorithms/model-free-rl/ppo/');
  const citation = page.locator('a.kt-paper-catalog-link[href$="#ppo"]').first();
  await expect(citation).toBeVisible();
  await citation.click();
  await expect(page.locator('#ppo')).toBeVisible();
  await page.goto('papers/?article=learning-based-planning%2Frl-algorithms%2Fmodel-free-rl%2Fppo');
  await expect(page.locator('.paper-card:visible')).toHaveCount(3);
  await expect(page.locator('#ppo a[href="/knowledge-tree/learning-based-planning/rl-algorithms/model-free-rl/ppo/"]')).toBeVisible();
});
test('移动端可阅读，论文关联仅链接已发布文章，关闭脚本仍有原文', async ({ page, browser }) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('papers/');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await expect(page.locator('#supermimic a[href$="/model-based-rl/supermimic/"]')).toBeVisible();
  const articleLinks = await page.locator('.paper-articles a').evaluateAll(links => links.map(a => a.getAttribute('href')));
  expect(articleLinks.length).toBeGreaterThan(0);
  expect(articleLinks.every(href => published.has(href))).toBe(true);
  const papersURL = page.url();
  const context = await browser.newContext({javaScriptEnabled:false});
  try {
    const p = await context.newPage();
    await p.goto(`${papersURL}#ppo`);
    await expect(p.locator('#ppo')).toBeVisible();
    await expect(p.locator('#ppo a').filter({hasText:'原文'})).toHaveAttribute('href', /^https:/);
  } finally { await context.close(); }
});
