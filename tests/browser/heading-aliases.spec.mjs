import { test, expect } from '@playwright/test';

test('合并文章的静态跳转保留查询参数和旧章节片段', async ({ page }) => {
	const fragment = '1-任务设计';
	await page.goto(`learning-based-planning/engineering-practice/work-pipeline/?source=bookmark#${encodeURIComponent(fragment)}`);
	await expect(page).toHaveURL(/learning-based-trajectory-planning\/learning-based-planning-overview\//);
	const url = new URL(page.url());
	expect(url.search).toBe('?source=bookmark');
	expect(decodeURIComponent(url.hash.slice(1))).toBe(fragment);
	await expect(page.locator('h3[id="21-系统接口"] .kt-heading-alias[id="1-任务设计"]')).toHaveCount(1);
});

test('关闭 JavaScript 时合并文章仍可跳转', async ({ browser }) => {
	const context = await browser.newContext({ javaScriptEnabled: false });
	try {
		const page = await context.newPage();
		await page.goto('http://127.0.0.1:4321/knowledge-tree/learning-based-planning/engineering-practice/work-pipeline/');
		await expect(page).toHaveURL(/learning-based-trajectory-planning\/learning-based-planning-overview\/?$/);
		await expect(page.locator('h1')).toHaveText('学习式无人机规划');
	} finally {
		await context.close();
	}
});

test('旧标题片段保持原章节语义，新目录使用稳定章节锚点', async ({ page }) => {
	const ppo = 'learning-based-planning/rl-algorithms/model-free-rl/ppo/';
	await page.goto(`${ppo}#${encodeURIComponent('1-概述overview')}`);
	const overview = page.locator('h2#kt-section-1');
	await expect(overview).toHaveText('1. 概述');
	await expect(overview.locator('.kt-heading-alias[id="1-概述overview"]')).toHaveCount(1);
	await expect(page.locator('kt-table-of-contents a[href="#kt-section-1"]')).toHaveText('1. 概述');
	await expect(page.locator('kt-table-of-contents a[href="#1-概述overview"]')).toHaveCount(0);

	// The historic “1. 概述” subsection now belongs to network structure.
	// Its fragment must reach that definition rather than the overview module.
	await page.goto(`learning-based-planning/neural-network-architectures/encoder-decoder/#${encodeURIComponent('1-概述')}`);
	const subsection = page.locator('h3[id="21-网络结构"]');
	await expect(subsection.locator('.kt-heading-alias[id="1-概述"]')).toHaveCount(1);
	await expect(page.locator('h2#kt-section-1 .kt-heading-alias[id="1-概述"]')).toHaveCount(0);
	const positions = await subsection.evaluate((heading) => ({
		heading: heading.getBoundingClientRect().top,
		header: document.querySelector('header').getBoundingClientRect().bottom,
		viewport: innerHeight,
	}));
	expect(positions.heading).toBeGreaterThanOrEqual(positions.header - 1);
	expect(positions.heading).toBeLessThan(positions.viewport);
});
