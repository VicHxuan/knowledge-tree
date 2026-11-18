import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const routes = ['', 'learning-based-planning/rl-algorithms/model-free-rl/ppo/', 'learning-based-planning/rl-algorithms/model-free-rl/sac/', 'learning-based-planning/rl-algorithms/model-free-rl/dqn/', 'classical-planning-control/', 'learning-paths/', 'papers/', 'bookmarks/', 'knowledge-map/'];

for (const theme of ['light', 'dark']) {
  test(`${theme}：核心页面保持可读、无横向溢出、满足 WCAG AA`, async ({ page }) => {
    test.setTimeout(120_000);
    await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
    for (const route of routes) {
      await page.goto(route || './');
      await expect(page.locator('main h1')).toHaveCount(1);
      await page.evaluate(() => document.fonts.ready);
      const a11y = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      expect.soft(a11y.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), route).toEqual([]);
      for (const width of [1440, 1024, 768, 390, 320]) {
        await page.setViewportSize({ width, height: 900 });
        await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), { message: `${route} at ${width}px` }).toBe(true);
      }
      await page.setViewportSize({ width: 1440, height: 900 });
    }
  });
}

test('首页目录、阅读入口与搜索可通过键盘使用', async ({ page }) => {
  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('无人机规划与控制');
  await expect(page.locator('.kt-domain-index li')).toHaveCount(6);
  await page.getByRole('link', { name: '阅读算法解析' }).focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/ppo\/$/);
  await page.locator('site-search button[data-open-modal]').click();
  const search = page.locator('.pagefind-ui__search-input');
  await expect(search).toBeVisible();
  await search.fill('PPO');
  await expect(page.locator('.pagefind-ui__result-link').first()).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('site-search dialog')).not.toBeVisible();
});

test('200% 缩放后的阅读栏仍可操作，主题选择保持一致', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('learning-based-planning/rl-algorithms/model-free-rl/ppo/');
  // 200% text-size zoom stresses CJK wrapping and controls without changing device pixels.
  await page.addStyleTag({ content: 'html { font-size: 200% !important; }' });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.getByRole('button', { name: '展开全部细节', exact: true }).click();
  expect(await page.locator('details.kt-derivation:not([open])').count()).toBe(0);
  await page.locator('starlight-theme-select select:visible').selectOption('dark');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});
