/**
 * 学习进度存储格式。旧数组的位置只允许通过冻结的 legacyStepIds 解释，
 * 当前步骤的展示顺序不参与迁移；v2 始终用步骤 ID 关联进度。
 */
const FORMAT = 'kt-learning-progress';
const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const isState = (value) => value === 0 || value === 1 || value === 2;
const hasOwn = (value, key) => Object.prototype.hasOwnProperty.call(value, key);

/** @param {string[]} stepIds @returns {Record<string, number>} */
export function emptyProgress(stepIds) {
	return Object.fromEntries(stepIds.map((id) => [id, 0]));
}

/**
 * 损坏的本地值按未开始处理。只读取当前路径的已知 ID，不把旧位置赋给新增步骤。
 * @param {unknown} stored
 * @param {{pathId: string, stepIds: string[], legacyStepIds: string[]}} schema
 * @returns {Record<string, number>}
 */
export function restoreProgress(stored, { pathId, stepIds, legacyStepIds }) {
	const state = emptyProgress(stepIds);
	let legacy;
	if (Array.isArray(stored)) legacy = stored;
	else if (isRecord(stored) && stored.format === FORMAT && stored.pathId === pathId) {
		if (stored.version === 1 && Array.isArray(stored.steps)) legacy = stored.steps;
		else if (stored.version === 2 && isRecord(stored.steps)) {
			for (const id of stepIds) {
				if (hasOwn(stored.steps, id) && isState(stored.steps[id])) state[id] = stored.steps[id];
			}
		}
	}
	if (legacy) {
		legacyStepIds.forEach((id, index) => {
			if (hasOwn(state, id) && isState(legacy[index])) state[id] = legacy[index];
		});
	}
	return state;
}

/** @param {string} pathId @param {Record<string, number>} state */
export function progressPayload(pathId, state) {
	return { format: FORMAT, version: 2, pathId, steps: { ...state } };
}

/**
 * 导入只覆盖文件与当前路径共有的 ID；保留文件中没有的当前步骤。
 * 校验失败直接抛错，调用方继续使用原进度。
 * @param {unknown} payload
 * @param {Record<string, number>} current
 * @param {{pathId: string, stepIds: string[], legacyStepIds: string[]}} schema
 * @returns {{state: Record<string, number>, matched: number}}
 */
export function importProgress(payload, current, { pathId, stepIds, legacyStepIds }) {
	if (!isRecord(payload) || payload.format !== FORMAT || ![1, 2].includes(payload.version)) {
		throw new Error('进度文件格式或版本不受支持');
	}
	if (payload.pathId !== pathId) throw new Error('学习路径不匹配');
	let incoming;
	if (payload.version === 1) {
		if (!Array.isArray(payload.steps) || payload.steps.length > legacyStepIds.length) {
			throw new Error('旧版步骤数超出历史映射');
		}
		if (!payload.steps.every(isState)) throw new Error('学习状态必须是 0、1 或 2');
		incoming = Object.fromEntries(payload.steps.map((state, index) => [legacyStepIds[index], state]));
	} else {
		if (!isRecord(payload.steps)) throw new Error('新版进度必须使用步骤 ID');
		if (!Object.values(payload.steps).every(isState)) throw new Error('学习状态必须是 0、1 或 2');
		incoming = payload.steps;
	}
	const state = emptyProgress(stepIds);
	let matched = 0;
	for (const id of stepIds) {
		state[id] = isState(current[id]) ? current[id] : 0;
		if (hasOwn(incoming, id)) {
			state[id] = incoming[id];
			matched++;
		}
	}
	if (!matched) throw new Error('文件中没有当前路径的可匹配步骤');
	return { state, matched };
}

/** 总览兼容旧数组与新版对象，统计最近一次路径页保存的步骤。 */
export function summarizeProgress(stored, pathId) {
	let values;
	if (Array.isArray(stored)) values = stored;
	else if (isRecord(stored) && stored.format === FORMAT && stored.pathId === pathId) {
		if (stored.version === 1 && Array.isArray(stored.steps)) values = stored.steps;
		if (stored.version === 2 && isRecord(stored.steps)) values = Object.values(stored.steps);
	}
	if (!values?.length) return { total: 0, done: 0, started: false };
	return {
		total: values.length,
		done: values.filter((state) => state === 2).length,
		started: values.some((state) => state === 1 || state === 2),
	};
}
