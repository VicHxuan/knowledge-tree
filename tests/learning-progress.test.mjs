import test from 'node:test';
import assert from 'node:assert/strict';
import {
	emptyProgress, importProgress, progressPayload, restoreProgress, summarizeProgress,
} from '../src/data/learning-progress.mjs';

// 历史顺序为 A、B、C；当前插入新步骤并重排，C 已从路径移除。
const schema = { pathId: 'rl', stepIds: ['new', 'b', 'a'], legacyStepIds: ['a', 'b', 'c'] };

test('旧 localStorage 数组只按冻结历史 ID 迁移，重排与新增步骤不串位', () => {
	assert.deepEqual(restoreProgress([2, 1, 2], schema), { new: 0, b: 1, a: 2 });
	assert.deepEqual(restoreProgress([2], schema), { new: 0, b: 0, a: 2 });
});

test('v2 本地进度按 ID 恢复，忽略移除步骤和非法状态', () => {
	const saved = progressPayload('rl', { a: 2, b: 1, c: 2, new: 7 });
	assert.deepEqual(restoreProgress(saved, schema), { new: 0, b: 1, a: 2 });
	assert.deepEqual(restoreProgress({ ...saved, pathId: 'other' }, schema), emptyProgress(schema.stepIds));
	assert.deepEqual(restoreProgress({ ...saved, version: 99 }, schema), emptyProgress(schema.stepIds));
});

test('v1 导入用历史顺序映射，保留后来新增步骤已有进度', () => {
	const current = { new: 2, b: 0, a: 0 };
	const result = importProgress({ format: 'kt-learning-progress', version: 1, pathId: 'rl', steps: [2, 1, 0] }, current, schema);
	assert.deepEqual(result, { state: { new: 2, b: 1, a: 2 }, matched: 2 });
	assert.deepEqual(current, { new: 2, b: 0, a: 0 });
});

test('v2 导入按 ID 合并，文件顺序、缺少步骤和移除步骤不影响匹配', () => {
	const result = importProgress(progressPayload('rl', { a: 2, c: 1 }), { new: 1, b: 2, a: 0 }, schema);
	assert.deepEqual(result, { state: { new: 1, b: 2, a: 2 }, matched: 1 });
	const reordered = { ...schema, stepIds: ['a', 'new', 'b'] };
	assert.deepEqual(importProgress(progressPayload('rl', { a: 2, b: 1 }), { new: 2 }, reordered).state,
		{ a: 2, new: 2, b: 1 });
});

test('错误文件拒绝导入且不修改原进度', () => {
	const current = { new: 1, b: 2, a: 0 };
	const invalid = [
		progressPayload('other', { a: 2 }),
		{ ...progressPayload('rl', { a: 2 }), version: 3 },
		progressPayload('rl', { a: -1 }),
		progressPayload('rl', { a: 1.5 }),
		progressPayload('rl', { a: '2' }),
		progressPayload('rl', { retired: 2 }),
		{ format: 'kt-learning-progress', version: 1, pathId: 'rl', steps: [0, 1, 2, 2] },
	];
	for (const payload of invalid) assert.throws(() => importProgress(payload, current, schema));
	assert.deepEqual(current, { new: 1, b: 2, a: 0 });
});

test('导出往返不依赖顺序，payload 与原对象互不修改', () => {
	const state = { a: 2, b: 1, new: 0 };
	const payload = progressPayload('rl', state);
	payload.steps.a = 0;
	assert.equal(state.a, 2);
	assert.deepEqual(restoreProgress(JSON.parse(JSON.stringify(progressPayload('rl', state))), schema),
		{ new: 0, b: 1, a: 2 });
});

test('总览兼容历史数组与 v2 对象', () => {
	assert.deepEqual(summarizeProgress([2, 1, 0], 'rl'), { total: 3, done: 1, started: true });
	assert.deepEqual(summarizeProgress(progressPayload('rl', { a: 2, b: 2 }), 'rl'),
		{ total: 2, done: 2, started: true });
	assert.deepEqual(summarizeProgress(null, 'rl'), { total: 0, done: 0, started: false });
});
