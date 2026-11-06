/**
 * relations-view.mjs — 知识关联的统一读取层（元数据兼容层）。
 *
 * knowledge-relations.json 只维护边；全部节点与展示信息从 content-map 派生。
 * 无关联边的新文章也会进入图谱，标题、状态、顺序只维护一处。
 *
 * 消费方：src/components/knowledge-map/KnowledgeMap.tsx、
 *        src/components/TableOfContents.astro。
 */
import relations from './knowledge-relations.json';
import contentMap from './content-map.json';
import { DOMAIN_BY_ID } from './domains.mjs';

export const NODES = contentMap.entries.map((m) => {
	const domain = m.slug.split('/')[0];
	return {
		id: m.slug,
		label: m.label,
		title: m.title,
		domain,
		domainLabel: DOMAIN_BY_ID[domain]?.label ?? domain,
		status: m.status,
		order: m.order,
	};
});

export const EDGES = relations.edges;

export const NODE_BY_ID = new Map(NODES.map((n) => [n.id, n]));

/** id 的直接前置知识（prerequisite 入边来源） */
export function prerequisitesOf(id) {
	return EDGES.filter((e) => e.type === 'prerequisite' && e.to === id).map((e) => e.from);
}

/** id 的后续依赖（prerequisite 出边目标）——「谁用到它」 */
export function dependentsOf(id) {
	return EDGES.filter((e) => e.type === 'prerequisite' && e.from === id).map((e) => e.to);
}

/** id 的相关主题（related 边对端，无方向） */
export function relatedOf(id) {
	return EDGES.filter((e) => e.type === 'related' && (e.from === id || e.to === id)).map((e) =>
		e.from === id ? e.to : e.from,
	);
}
