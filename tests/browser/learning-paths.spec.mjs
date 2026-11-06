import { test, expect } from '@playwright/test';

const pathId = 'reinforcement-learning';
const route = `learning-paths/${pathId}/`;
const key = `kt-lp:${pathId}`;
const progress = (page) => page.evaluate((key) => JSON.parse(localStorage.getItem(key)), key);
const stateOf = (page, id) => page.locator(`.lp-step[data-step-id="${id}"] .lp-toggle`);
const upload = (page, payload) => page.locator('.lp-file').setInputFiles({
	name: 'progress.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(payload)),
});

test('旧浏览器数组迁移为稳定 ID，首页总览兼容且键盘可以重置', async ({ page }) => {
	const errors = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.addInitScript(({ key }) => {
		if (!localStorage.getItem(key)) localStorage.setItem(key, JSON.stringify([2, 1, 0, 0, 0, 2, 0, 0, 0, 0]));
	}, { key });
	await page.goto(route);
	await expect(stateOf(page, 'optimization-theory')).toHaveAttribute('data-state', '2');
	await expect(stateOf(page, 'probability-and-gaussian')).toHaveAttribute('data-state', '1');
	await expect(stateOf(page, 'ppo')).toHaveAttribute('data-state', '2');
	const saved = await progress(page);
	expect(saved.version).toBe(2);
	expect(saved.pathId).toBe(pathId);
	expect(Array.isArray(saved.steps)).toBe(false);
	expect(saved.steps.ppo).toBe(2);
	await page.goto('learning-paths/');
	await expect(page.locator(`.lpo-row[data-lp-id="${pathId}"] .lpo-pct`)).toContainText('2/10');
	await page.goto(route);
	await page.getByRole('button', { name: '重置进度', exact: true }).focus();
	await page.keyboard.press('Enter');
	await expect(page.locator('.lp-done')).toHaveText('0');
	expect(Object.values((await progress(page)).steps).every((state) => state === 0)).toBe(true);
	expect(errors).toEqual([]);
});

test('v1 可导入，v2 仅按 ID 更新匹配步骤，导出携带 v2 标识', async ({ page }) => {
	await page.goto(route);
	await upload(page, {
		format: 'kt-learning-progress', version: 1, pathId,
		steps: [2, 1, 0, 0, 0, 1, 0, 0, 0, 0],
	});
	await expect(page.locator('.lp-msg')).toContainText('导入成功');
	await expect(stateOf(page, 'optimization-theory')).toHaveAttribute('data-state', '2');
	await expect(stateOf(page, 'ppo')).toHaveAttribute('data-state', '1');
	await upload(page, {
		format: 'kt-learning-progress', version: 2, pathId,
		steps: { ppo: 2, 'removed-step': 1 },
	});
	await expect(page.locator('.lp-msg')).toContainText('更新 1 个匹配步骤');
	await expect(stateOf(page, 'optimization-theory')).toHaveAttribute('data-state', '2');
	await expect(stateOf(page, 'ppo')).toHaveAttribute('data-state', '2');
	const before = await progress(page);
	await upload(page, { format: 'kt-learning-progress', version: 2, pathId: 'other', steps: { ppo: 0 } });
	await expect(page.locator('.lp-msg')).toContainText('学习路径不匹配');
	expect(await progress(page)).toEqual(before);
	const downloading = page.waitForEvent('download');
	await page.getByRole('button', { name: '导出进度', exact: true }).click();
	const download = await downloading;
	const chunks = [];
	for await (const chunk of await download.createReadStream()) chunks.push(chunk);
	const payload = JSON.parse(Buffer.concat(chunks).toString());
	expect(payload.version).toBe(2);
	expect(payload.steps.ppo).toBe(2);
	expect(payload.steps['optimization-theory']).toBe(2);
	await download.delete();
});

for (const [id, count] of [['reinforcement-learning', 10], ['classical-trajectory-planning', 7], ['learning-based-uav-navigation', 8]]) {
	test(`${id} 使用五段结构，草稿步骤不产生不可用链接`, async ({ page }) => {
		await page.goto(`learning-paths/${id}/`);
		const headings = await page.locator('.sl-markdown-content h2').allTextContents();
		expect(headings.map((text) => text.trim())).toEqual(['学习目标', '开始之前', '学习步骤', '完成标准', '延伸方向']);
		await expect(page.locator('.lp-step')).toHaveCount(count);
		await expect(page.locator('.lp-step[data-status="draft"] a.lp-title')).toHaveCount(0);
		await expect(page.getByRole('button', { name: '重置进度', exact: true })).toBeVisible();
	});
}
