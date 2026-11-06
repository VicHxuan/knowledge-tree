import { test, expect } from '@playwright/test';

test('旧标题片段保持原章节语义，新目录使用稳定章节锚点', async ({ page }) => {
	const ppo = 'learning-based-planning/rl-algorithms/model-free-rl/ppo/';
	await page.goto(`${ppo}#${encodeURIComponent('1-概述overview')}`);
	const overview = page.locator('h2#kt-section-1');
	await expect(overview).toHaveText('1. 概述');
	await expect(overview.locator('.kt-heading-alias[id="1-概述overview"]')).toHaveCount(1);
	await expect(page.locator('kt-table-of-contents a[href="#kt-section-1"]')).toHaveText('1. 概述');
	await expect(page.locator('kt-table-of-contents a[href="#1-概述overview"]')).toHaveCount(0);

	// The old subsection and the newly shortened module title share the text
	// “1. 概述”. The historic fragment must still reach the original subsection.
	await page.goto(`learning-based-planning/neural-network-architectures/encoder-decoder/#${encodeURIComponent('1-概述')}`);
	const subsection = page.locator('h3[id="1-架构定义"]');
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
