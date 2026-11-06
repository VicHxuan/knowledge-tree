import { test, expect } from '@playwright/test';

const route = 'learning-based-planning/rl-algorithms/model-free-rl/ppo/';
const derivations = '.sl-markdown-content details.kt-derivation';
const gae = '#ppo-gae-derivation';
const opened = (page) => page.locator(derivations).evaluateAll((details) => details.map((detail) => detail.open));

test('原生键盘开合、批量控制、数学与 Mermaid 渲染没有客户端异常', async ({ page }) => {
	test.setTimeout(60_000);
	const errors = [];
	page.on('pageerror', (error) => errors.push(error.message));
	page.on('console', (message) => {
		if (message.type() === 'warning' && message.text().includes('[knowledge-tree] mermaid render failed:')) {
			errors.push(message.text());
		}
	});
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto(route);
	const details = page.locator(derivations);
	expect(await details.count()).toBeGreaterThan(0);
	await expect(page.getByRole('button', { name: '展开全部细节', exact: true })).toBeVisible();
	const first = details.first();
	const firstSummary = first.locator(':scope > summary');
	const markerTransform = () => firstSummary.evaluate((summary) => getComputedStyle(summary, '::before').transform);
	const closedMarker = await markerTransform();
	await firstSummary.focus();
	await page.keyboard.press('Enter');
	await expect(first).toHaveAttribute('open', '');
	await expect.poll(markerTransform).not.toBe(closedMarker);
	await page.keyboard.press('Space');
	await expect(first).not.toHaveAttribute('open');
	await expect.poll(markerTransform).toBe(closedMarker);
	await page.getByRole('button', { name: '展开全部细节', exact: true }).click();
	expect((await opened(page)).every(Boolean)).toBe(true);
	expect(await page.locator(`${derivations} .katex`).count()).toBeGreaterThan(0);
	await expect(page.locator('.katex-error')).toHaveCount(0);
	const diagrams = page.locator('.kt-mermaid');
	const diagramCount = await diagrams.count();
	if (diagramCount) {
		await expect(page.locator('.kt-mermaid-out svg')).toHaveCount(diagramCount, { timeout: 30_000 });
		await expect(page.locator('.kt-mermaid-error')).toHaveCount(0);
		const clippedLabels = () => page.locator('.kt-mermaid-out foreignObject').evaluateAll((labels) =>
			labels.filter((label) => label.firstElementChild &&
				label.firstElementChild.scrollHeight > label.height.baseVal.value + 1)
				.map((label) => ({ text: label.textContent, available: label.height.baseVal.value,
					content: label.firstElementChild.scrollHeight })),
		);
		expect(await clippedLabels()).toEqual([]);
		for (let switchCount = 0; switchCount < 2; switchCount++) {
			const firstSvg = page.locator('.kt-mermaid-out svg').first();
			const previousId = await firstSvg.getAttribute('id');
			const nextTheme = await page.evaluate(() => document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
			await page.locator('starlight-theme-select select:visible').selectOption(nextTheme);
			await expect(page.locator('html')).toHaveAttribute('data-theme', nextTheme);
			await expect.poll(() => firstSvg.getAttribute('id'), { timeout: 30_000 }).not.toBe(previousId);
			await expect(page.locator('.kt-mermaid-error')).toHaveCount(0);
			expect(errors).toEqual([]);
			expect(await clippedLabels()).toEqual([]);
		}
	}
	await page.getByRole('button', { name: '收起全部细节', exact: true }).click();
	expect((await opened(page)).every((open) => !open)).toBe(true);
	await expect(page.locator('kt-table-of-contents')).toHaveCount(1);
	expect(errors).toEqual([]);
});

test('初始 hash、相同 hash 重复点击与历史前后退都打开对应推导', async ({ page }) => {
	await page.goto(`${route}${gae}`);
	const target = page.locator(gae);
	await expect(target).toHaveAttribute('open', '');
	await target.locator(':scope > summary').click();
	await expect(target).not.toHaveAttribute('open');
	await page.evaluate((hash) => {
		const anchor = document.createElement('a');
		anchor.id = 'disclosure-test-link';
		anchor.href = hash;
		anchor.textContent = '测试原生片段链接';
		document.body.append(anchor);
	}, gae);
	await page.locator('#disclosure-test-link').focus();
	await page.keyboard.press('Enter');
	await expect(target).toHaveAttribute('open', '');
	const secondId = await page.locator(`${derivations}[id]`).evaluateAll((details) => details.find((detail) => detail.id !== 'ppo-gae-derivation')?.id);
	expect(secondId).toBeTruthy();
	await page.evaluate((id) => { location.hash = id; }, secondId);
	const second = page.locator(`[id="${secondId}"]`);
	await expect(second).toHaveAttribute('open', '');
	await target.evaluate((detail) => { detail.open = false; });
	await page.goBack();
	await expect(target).toHaveAttribute('open', '');
	await second.evaluate((detail) => { detail.open = false; });
	await page.goForward();
	await expect(second).toHaveAttribute('open', '');
});

test('打印包含展开内容，退出打印恢复原阅读状态', async ({ page }) => {
	await page.goto(route);
	await page.locator(gae).locator(':scope > summary').click();
	const original = await opened(page);
	await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
	expect((await opened(page)).every(Boolean)).toBe(true);
	await page.emulateMedia({ media: 'print' });
	await expect(page.locator('.kt-disclosure-controls')).toBeHidden();
	await expect(page.locator(`${gae} .katex`).first()).toBeVisible();
	await page.evaluate(() => window.dispatchEvent(new Event('afterprint')));
	await page.emulateMedia({ media: 'screen' });
	expect(await opened(page)).toEqual(original);
});

test('搜索关闭推导中的特征词，点击结果后能直接读到命中内容', async ({ page }) => {
	await page.goto('learning-paths/');
	await page.locator('site-search button[data-open-modal]').click();
	await page.locator('.pagefind-ui__search-input').fill('有限混合式');
	const result = page.locator('.pagefind-ui__result-link[href*="/model-free-rl/ppo/"]').first();
	await expect(result).toBeVisible({ timeout: 15_000 });
	await result.click();
	await expect(page).toHaveURL(/highlight=/);
	await expect(page.locator(gae)).toHaveAttribute('open', '');
	await expect(page.locator(gae).getByText(/正确的有限混合式/)).toBeVisible();
});

test('窄屏及浅色主题下，展开推导不撑破正文', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto(`${route}${gae}`);
	await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
	await expect(page.locator(gae)).toHaveAttribute('open', '');
	const summary = page.locator(`${gae} > summary`);
	await summary.scrollIntoViewIfNeeded();
	const box = await summary.boundingBox();
	expect(box.height).toBeGreaterThanOrEqual(44);
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
	await summary.focus();
	await expect(summary).toBeFocused();
});

test('关闭 JavaScript 后仍能打开原生推导并阅读静态公式', async ({ browser, baseURL }) => {
	const context = await browser.newContext({ javaScriptEnabled: false });
	try {
		const page = await context.newPage();
		await page.goto(new URL(route, baseURL).href);
		await expect(page.locator('.kt-disclosure-controls')).toHaveCount(0);
		await page.locator(`${gae} > summary`).click();
		await expect(page.locator(gae)).toHaveAttribute('open', '');
		await expect(page.locator(`${gae} .katex`).first()).toBeVisible();
	} finally { await context.close(); }
});

test('移动端长公式在自身容器横滚，左右两端均可完整到达', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto(route);
	await page.getByRole('button', { name: '展开全部细节', exact: true }).click();
	const scrollable = await page.locator('.sl-markdown-content .katex-display').evaluateAll((formulas) =>
		formulas.filter((formula) => formula.scrollWidth > formula.clientWidth + 1).map((formula) => {
			const bounds = formula.getBoundingClientRect();
			const parts = [...formula.querySelectorAll('.katex-html > .base')];
			formula.scrollLeft = 0;
			const left = Math.min(...parts.map((part) => part.getBoundingClientRect().left));
			formula.scrollLeft = formula.scrollWidth;
			const maxScroll = formula.scrollLeft;
			const right = Math.max(...parts.map((part) => part.getBoundingClientRect().right));
			formula.scrollLeft = 0;
			return { maxScroll, leftHidden: bounds.left - left, rightHidden: right - bounds.right,
				overflow: getComputedStyle(formula).overflowX };
		}),
	);
	expect(scrollable.length).toBeGreaterThan(0);
	for (const formula of scrollable) {
		expect(formula.maxScroll).toBeGreaterThan(0);
		expect(['auto', 'scroll']).toContain(formula.overflow);
		expect(formula.leftHidden).toBeLessThanOrEqual(1);
		expect(formula.rightHidden).toBeLessThanOrEqual(1);
	}
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
});
